/* Copyright (c) 2017 Hans-Kristian Arntzen
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "device.hpp"
#include <string.h>
#include <algorithm>
#include "format.hpp"

using namespace std;
using namespace Util;

namespace Vulkan {
Device::Device()
    : framebuffer_allocator(this),
      transient_allocator(this),
      physical_allocator(this),
      shader_manager(this),
      texture_manager(this) {}

Semaphore Device::request_semaphore() {
  auto semaphore = semaphore_manager.request_cleared_semaphore();
  auto ptr = make_handle<SemaphoreHolder>(this, semaphore, false);
  return ptr;
}

#ifndef _WIN32
Semaphore Device::request_imported_semaphore(
    int fd,
    VkExternalSemaphoreHandleTypeFlagBitsKHR handle_type) {
  if (!supports_external)
    return {};

  VkExternalSemaphorePropertiesKHR props = {
      VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES_KHR};
  VkPhysicalDeviceExternalSemaphoreInfoKHR info = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO_KHR};
  info.handleType = handle_type;

  vkGetPhysicalDeviceExternalSemaphorePropertiesKHR(gpu, &info, &props);
  if ((props.externalSemaphoreFeatures &
       VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT_KHR) == 0)
    return nullptr;

  auto semaphore = semaphore_manager.request_cleared_semaphore();

  VkImportSemaphoreFdInfoKHR import = {
      VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR};
  import.fd = fd;
  import.semaphore = semaphore;
  import.handleType = handle_type;
  import.flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT_KHR;
  auto ptr = make_handle<SemaphoreHolder>(this, semaphore, false);

  if (vkImportSemaphoreFdKHR(device, &import) != VK_SUCCESS)
    return nullptr;

  ptr->signal_external();
  ptr->destroy_on_consume();
  return ptr;
}
#endif

void Device::add_wait_semaphore(CommandBuffer::Type type,
                                Semaphore semaphore,
                                VkPipelineStageFlags stages) {
  flush_frame(type);
  auto& data = get_queue_data(type);
  data.wait_semaphores.push_back(semaphore);
  data.wait_stages.push_back(stages);
}

void* Device::map_host_buffer(Buffer& buffer, MemoryAccessFlags access) {
  void* host = allocator.map_memory(&buffer.get_allocation(), access);
  return host;
}

void Device::unmap_host_buffer(const Buffer& buffer) {
  allocator.unmap_memory(buffer.get_allocation());
}

ShaderHandle Device::create_shader(ShaderStage stage,
                                   const uint32_t* data,
                                   size_t size) {
  return make_handle<Shader>(device, stage, data, size);
}

ProgramHandle Device::create_program(const uint32_t* compute_data,
                                     size_t compute_size) {
  auto compute = make_handle<Shader>(device, ShaderStage::Compute, compute_data,
                                     compute_size);
  auto program = make_handle<Program>(this);
  program->set_shader(compute);
  bake_program(*program);
  return program;
}

ProgramHandle Device::create_program(const uint32_t* vertex_data,
                                     size_t vertex_size,
                                     const uint32_t* fragment_data,
                                     size_t fragment_size) {
  auto vertex = make_handle<Shader>(device, ShaderStage::Vertex, vertex_data,
                                    vertex_size);
  auto fragment = make_handle<Shader>(device, ShaderStage::Fragment,
                                      fragment_data, fragment_size);
  auto program = make_handle<Program>(this);
  program->set_shader(vertex);
  program->set_shader(fragment);
  bake_program(*program);
  return program;
}

PipelineLayout* Device::request_pipeline_layout(
    const CombinedResourceLayout& layout) {
  Hasher h;
  h.data(reinterpret_cast<const uint32_t*>(layout.sets), sizeof(layout.sets));
  h.data(reinterpret_cast<const uint32_t*>(layout.ranges),
         sizeof(layout.ranges));
  h.u32(layout.attribute_mask);

  auto hash = h.get();
  auto itr = pipeline_layouts.find(hash);
  if (itr != end(pipeline_layouts))
    return itr->second.get();

  auto* pipe = new PipelineLayout(this, layout);
  pipeline_layouts.insert(make_pair(hash, unique_ptr<PipelineLayout>(pipe)));
  return pipe;
}

DescriptorSetAllocator* Device::request_descriptor_set_allocator(
    const DescriptorSetLayout& layout) {
  Hasher h;
  h.data(reinterpret_cast<const uint32_t*>(&layout), sizeof(layout));
  auto hash = h.get();
  auto itr = descriptor_set_allocators.find(hash);
  if (itr != end(descriptor_set_allocators))
    return itr->second.get();

  auto* allocator = new DescriptorSetAllocator(this, layout);
  descriptor_set_allocators.insert(
      make_pair(hash, unique_ptr<DescriptorSetAllocator>(allocator)));
  return allocator;
}

void Device::bake_program(Program& program) {
  CombinedResourceLayout layout;
  if (program.get_shader(ShaderStage::Vertex))
    layout.attribute_mask =
        program.get_shader(ShaderStage::Vertex)->get_layout().attribute_mask;
  if (program.get_shader(ShaderStage::Fragment))
    layout.render_target_mask = program.get_shader(ShaderStage::Fragment)
                                    ->get_layout()
                                    .render_target_mask;

  layout.descriptor_set_mask = 0;

  for (unsigned i = 0; i < static_cast<unsigned>(ShaderStage::Count); i++) {
    auto* shader = program.get_shader(static_cast<ShaderStage>(i));
    if (!shader)
      continue;

    auto& shader_layout = shader->get_layout();
    for (unsigned set = 0; set < VULKAN_NUM_DESCRIPTOR_SETS; set++) {
      layout.sets[set].sampled_image_mask |=
          shader_layout.sets[set].sampled_image_mask;
      layout.sets[set].storage_image_mask |=
          shader_layout.sets[set].storage_image_mask;
      layout.sets[set].uniform_buffer_mask |=
          shader_layout.sets[set].uniform_buffer_mask;
      layout.sets[set].storage_buffer_mask |=
          shader_layout.sets[set].storage_buffer_mask;
      layout.sets[set].sampled_buffer_mask |=
          shader_layout.sets[set].sampled_buffer_mask;
      layout.sets[set].input_attachment_mask |=
          shader_layout.sets[set].input_attachment_mask;
      layout.sets[set].fp_mask |= shader_layout.sets[set].fp_mask;
      layout.sets[set].stages |= shader_layout.sets[set].stages;
    }

    layout.ranges[i].stageFlags = 1u << i;
    layout.ranges[i].offset = shader_layout.push_constant_offset;
    layout.ranges[i].size = shader_layout.push_constant_range;
  }

  for (unsigned i = 0; i < VULKAN_NUM_DESCRIPTOR_SETS; i++) {
    if (layout.sets[i].stages != 0)
      layout.descriptor_set_mask |= 1u << i;
  }

  Hasher h;
  h.data(reinterpret_cast<uint32_t*>(layout.ranges), sizeof(layout.ranges));
  layout.push_constant_layout_hash = h.get();

  program.set_pipeline_layout(request_pipeline_layout(layout));

  if (program.get_shader(ShaderStage::Compute)) {
    auto& shader = *program.get_shader(ShaderStage::Compute);
    VkComputePipelineCreateInfo info = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.layout = program.get_pipeline_layout()->get_layout();
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.module = shader.get_module();
    info.stage.pName = "main";
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;

    VkPipeline compute_pipeline;
    if (vkCreateComputePipelines(device, pipeline_cache, 1, &info, nullptr,
                                 &compute_pipeline) != VK_SUCCESS)
      LOGE("Failed to create compute pipeline!\n");
    program.set_compute_pipeline(compute_pipeline);
  }
}

void Device::init_pipeline_cache() {
  auto file =
      Filesystem::get().open("cache://pipeline_cache.bin", FileMode::ReadOnly);
  VkPipelineCacheCreateInfo info = {
      VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};

  if (file) {
    auto size = file->get_size();
    static const auto uuid_size = sizeof(gpu_props.pipelineCacheUUID);
    if (size >= uuid_size) {
      uint8_t* mapped = static_cast<uint8_t*>(file->map());
      if (mapped) {
        if (memcmp(gpu_props.pipelineCacheUUID, mapped, uuid_size) == 0) {
          info.initialDataSize = size - uuid_size;
          info.pInitialData = mapped + uuid_size;
        }
      }
    }
  }

  vkCreatePipelineCache(device, &info, nullptr, &pipeline_cache);
}

void Device::flush_pipeline_cache() {
  static const auto uuid_size = sizeof(gpu_props.pipelineCacheUUID);
  size_t size = 0;
  if (vkGetPipelineCacheData(device, pipeline_cache, &size, nullptr) !=
      VK_SUCCESS) {
    LOGE("Failed to get pipeline cache data.\n");
    return;
  }

  auto file =
      Filesystem::get().open("cache://pipeline_cache.bin", FileMode::WriteOnly);
  if (!file) {
    LOGE("Failed to get pipeline cache data.\n");
    return;
  }

  uint8_t* data = static_cast<uint8_t*>(file->map_write(uuid_size + size));
  if (!data) {
    LOGE("Failed to get pipeline cache data.\n");
    return;
  }

  memcpy(data, gpu_props.pipelineCacheUUID, uuid_size);
  if (vkGetPipelineCacheData(device, pipeline_cache, &size, data + uuid_size) !=
      VK_SUCCESS) {
    LOGE("Failed to get pipeline cache data.\n");
    return;
  }
}

void Device::set_context(const Context& context) {
  instance = context.get_instance();
  gpu = context.get_gpu();
  device = context.get_device();

  graphics_queue_family_index = context.get_graphics_queue_family();
  graphics_queue = context.get_graphics_queue();
  compute_queue_family_index = context.get_compute_queue_family();
  compute_queue = context.get_compute_queue();
  transfer_queue_family_index = context.get_transfer_queue_family();
  transfer_queue = context.get_transfer_queue();

  mem_props = context.get_mem_props();
  gpu_props = context.get_gpu_props();

  allocator.init(gpu, device);
  init_stock_samplers();

  init_pipeline_cache();
  semaphore_manager.init(device);
  event_manager.init(device);

  supports_external = context.supports_external_memory_and_sync();
  supports_dedicated = context.supports_dedicated_allocation();
  allocator.set_supports_dedicated_allocation(supports_dedicated);
}

void Device::init_stock_samplers() {
  SamplerCreateInfo info = {};
  info.maxLod = VK_LOD_CLAMP_NONE;
  info.maxAnisotropy = 1.0f;

  for (unsigned i = 0; i < static_cast<unsigned>(StockSampler::Count); i++) {
    auto mode = static_cast<StockSampler>(i);

    switch (mode) {
      case StockSampler::NearestShadow:
      case StockSampler::LinearShadow:
        info.compareEnable = true;
        info.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        break;

      default:
        info.compareEnable = false;
        break;
    }

    switch (mode) {
      case StockSampler::TrilinearClamp:
      case StockSampler::TrilinearWrap:
        info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        break;

      default:
        info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        break;
    }

    switch (mode) {
      case StockSampler::LinearClamp:
      case StockSampler::LinearWrap:
      case StockSampler::TrilinearClamp:
      case StockSampler::TrilinearWrap:
      case StockSampler::LinearShadow:
        info.magFilter = VK_FILTER_LINEAR;
        info.minFilter = VK_FILTER_LINEAR;
        break;

      default:
        info.magFilter = VK_FILTER_NEAREST;
        info.minFilter = VK_FILTER_NEAREST;
        break;
    }

    switch (mode) {
      default:
      case StockSampler::LinearWrap:
      case StockSampler::NearestWrap:
      case StockSampler::TrilinearWrap:
        info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        break;

      case StockSampler::LinearClamp:
      case StockSampler::NearestClamp:
      case StockSampler::TrilinearClamp:
      case StockSampler::NearestShadow:
      case StockSampler::LinearShadow:
        info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        break;
    }
    samplers[i] = create_sampler(info);
  }
}

void Device::submit(CommandBufferHandle cmd,
                    Fence* fence,
                    Semaphore* semaphore) {
  auto type = cmd->get_command_buffer_type();
  auto& data = get_queue_data(type);
  auto& pool = get_command_pool(type);
  auto& submissions = get_queue_submissions(type);

  if (data.staging_cmd) {
    pool.signal_submitted(data.staging_cmd->get_command_buffer());
    vkEndCommandBuffer(data.staging_cmd->get_command_buffer());
    submissions.push_back(data.staging_cmd);
    data.staging_cmd.reset();
  }

  pool.signal_submitted(cmd->get_command_buffer());
  vkEndCommandBuffer(cmd->get_command_buffer());
  submissions.push_back(move(cmd));

  if (fence || semaphore)
    submit_queue(type, fence, semaphore);
}

void Device::submit_empty(CommandBuffer::Type type,
                          Fence* fence,
                          Semaphore* semaphore) {
  auto& data = get_queue_data(type);
  VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO};

  // Add external wait semaphores.
  vector<VkSemaphore> waits;
  vector<VkSemaphore> signals;
  auto stages = move(data.wait_stages);

  for (auto& semaphore : data.wait_semaphores) {
    auto wait = semaphore->consume();
    if (semaphore->can_recycle())
      frame().recycled_semaphores.push_back(wait);
    else
      frame().destroyed_semaphores.push_back(wait);
    waits.push_back(wait);
  }
  data.wait_stages.clear();
  data.wait_semaphores.clear();

  // Add external signal semaphores.
  VkSemaphore cleared_semaphore = VK_NULL_HANDLE;
  if (semaphore) {
    cleared_semaphore = semaphore_manager.request_cleared_semaphore();
    signals.push_back(cleared_semaphore);
  }

  submit.signalSemaphoreCount = signals.size();
  submit.waitSemaphoreCount = waits.size();
  if (!signals.empty())
    submit.pSignalSemaphores = signals.data();
  if (!stages.empty())
    submit.pWaitDstStageMask = stages.data();
  if (!waits.empty())
    submit.pWaitSemaphores = waits.data();

  VkQueue queue;
  switch (type) {
    default:
    case CommandBuffer::Type::Graphics:
      queue = graphics_queue;
      break;
    case CommandBuffer::Type::Compute:
      queue = compute_queue;
      break;
    case CommandBuffer::Type::Transfer:
      queue = transfer_queue;
      break;
  }

  VkFence cleared_fence = frame().fence_manager.request_cleared_fence();
  if (queue_lock_callback)
    queue_lock_callback();
  VkResult result = vkQueueSubmit(queue, 1, &submit, cleared_fence);
  if (queue_unlock_callback)
    queue_unlock_callback();

  if (result != VK_SUCCESS)
    LOGE("vkQueueSubmit failed.\n");

  if (fence) {
    auto ptr = make_shared<FenceHolder>(this, cleared_fence);
    *fence = ptr;
    frame().fences.push_back(move(ptr));
  }

  if (semaphore) {
    auto ptr = make_handle<SemaphoreHolder>(this, cleared_semaphore, true);
    *semaphore = ptr;
  }
}

void Device::add_queue_dependency(CommandBuffer::Type consumer,
                                  VkPipelineStageFlags stages,
                                  CommandBuffer::Type producer) {
  VK_ASSERT(consumer != producer);
  auto& dst = get_queue_data(consumer);
  auto& src = get_queue_data(producer);

  VkPipelineStageFlags* dst_stages;
  VkPipelineStageFlags* src_stages;

  switch (producer) {
    default:
    case CommandBuffer::Type::Graphics:
      dst_stages = &dst.wait_for_graphics;
      break;
    case CommandBuffer::Type::Compute:
      dst_stages = &dst.wait_for_compute;
      break;
    case CommandBuffer::Type::Transfer:
      dst_stages = &dst.wait_for_transfer;
      break;
  }

  switch (consumer) {
    default:
    case CommandBuffer::Type::Graphics:
      src_stages = &src.wait_for_graphics;
      break;
    case CommandBuffer::Type::Compute:
      src_stages = &src.wait_for_compute;
      break;
    case CommandBuffer::Type::Transfer:
      src_stages = &src.wait_for_transfer;
      break;
  }

  // If the stage we need to wait for, in turn waits for us, we have a problem,
  // we need some kind of early flushing to make this work, but it will probably
  // mess with staging command buffers because we can randomly flush those ...
  // For now, just assert on this case, it should never happen for internal
  // stuff.
  VK_ASSERT(*src_stages == 0);
  (void)src_stages;

  // This way of dealing with the queue dependencies is very lazy, for optimal
  // theoretical overlap we would need to flush producer here and inject a wait
  // for consumer. This however probably isn't a good idea when we have a large
  // amount of staging transfers. This function is generally only used by the
  // internal APIs to inject dependencies between staging command buffers.
  *dst_stages |= stages;
}

void Device::add_staging_transfer_queue_dependency(const Buffer& dst,
                                                   VkBufferUsageFlags usage) {
  if (transfer_queue == graphics_queue && transfer_queue == compute_queue) {
    // For single-queue systems, just use a pipeline barrier,
    // this is more efficient than semaphores because with semaphores we will
    // end up draining the entire graphics queue.
    transfer.staging_cmd->buffer_barrier(
        dst, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
        buffer_usage_to_possible_stages(usage),
        buffer_usage_to_possible_access(usage));
  } else {
    if (transfer_queue == graphics_queue) {
      transfer.staging_cmd->buffer_barrier(
          dst, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
          buffer_usage_to_possible_stages(usage),
          buffer_usage_to_possible_access(usage));
    } else {
      add_queue_dependency(CommandBuffer::Type::Graphics,
                           buffer_usage_to_possible_stages(usage),
                           CommandBuffer::Type::Transfer);
    }

    if (transfer_queue == compute_queue) {
      transfer.staging_cmd->buffer_barrier(
          dst, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
          buffer_usage_to_possible_stages(usage) &
              (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
               VK_PIPELINE_STAGE_TRANSFER_BIT),
          buffer_usage_to_possible_access(usage) &
              (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
               VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT));
    } else {
      add_queue_dependency(CommandBuffer::Type::Compute,
                           buffer_usage_to_possible_stages(usage),
                           CommandBuffer::Type::Transfer);
    }
  }
}

void Device::sync_buffer_to_gpu(const Buffer& dst,
                                const Buffer& src,
                                VkDeviceSize offset,
                                VkDeviceSize size) {
  begin_staging(CommandBuffer::Type::Transfer);
  transfer.staging_cmd->copy_buffer(dst, offset, src, offset, size);
  add_staging_transfer_queue_dependency(dst, dst.get_create_info().usage);
}

void Device::submit_queue(CommandBuffer::Type type,
                          Fence* fence,
                          Semaphore* semaphore) {
  // Always check if we need to flush pending transfers.
  if (type != CommandBuffer::Type::Transfer)
    flush_frame(CommandBuffer::Type::Transfer);

  auto& data = get_queue_data(type);
  auto& submissions = get_queue_submissions(type);

  if (data.wait_for_graphics && type != CommandBuffer::Type::Graphics) {
    // Avoid recursive waits just in case.
    data.wait_stages.push_back(data.wait_for_graphics);
    data.wait_for_graphics = 0;

    Semaphore transition;
    flush_frame(CommandBuffer::Type::Graphics);
    submit_queue(CommandBuffer::Type::Graphics, nullptr, &transition);
    data.wait_semaphores.push_back(move(transition));
  }

  if (data.wait_for_compute && type != CommandBuffer::Type::Compute) {
    // Avoid recursive waits just in case.
    data.wait_stages.push_back(data.wait_for_compute);
    data.wait_for_compute = 0;

    Semaphore transition;
    flush_frame(CommandBuffer::Type::Compute);
    submit_queue(CommandBuffer::Type::Compute, nullptr, &transition);
    data.wait_semaphores.push_back(move(transition));
  }

  if (data.wait_for_transfer && type != CommandBuffer::Type::Transfer) {
    // Avoid recursive waits just in case.
    data.wait_stages.push_back(data.wait_for_transfer);
    data.wait_for_transfer = 0;

    Semaphore transition;
    flush_frame(CommandBuffer::Type::Transfer);
    submit_queue(CommandBuffer::Type::Transfer, nullptr, &transition);
    data.wait_semaphores.push_back(move(transition));
  }

  if (submissions.empty()) {
    if (fence || semaphore)
      submit_empty(type, fence, semaphore);
    return;
  }

  vector<VkCommandBuffer> cmds;
  cmds.reserve(submissions.size());

  vector<VkSubmitInfo> submits;
  submits.reserve(2);
  size_t last_cmd = 0;

  vector<VkSemaphore> waits[2];
  vector<VkSemaphore> signals[2];
  vector<VkFlags> stages[2];

  // Add external wait semaphores.
  stages[0] = move(data.wait_stages);

  for (auto& semaphore : data.wait_semaphores) {
    auto wait = semaphore->consume();
    if (semaphore->can_recycle())
      frame().recycled_semaphores.push_back(wait);
    else
      frame().destroyed_semaphores.push_back(wait);
    waits[0].push_back(wait);
  }
  data.wait_stages.clear();
  data.wait_semaphores.clear();

  for (auto& cmd : submissions) {
    if (cmd->swapchain_touched() && !frame().swapchain_touched &&
        !frame().swapchain_consumed) {
      if (!cmds.empty()) {
        // Push all pending cmd buffers to their own submission.
        submits.emplace_back();

        auto& submit = submits.back();
        memset(&submit, 0, sizeof(submit));
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.pNext = nullptr;
        submit.commandBufferCount = cmds.size() - last_cmd;
        submit.pCommandBuffers = cmds.data() + last_cmd;
        last_cmd = cmds.size();
      }
      frame().swapchain_touched = true;
    }

    cmds.push_back(cmd->get_command_buffer());
  }

  if (cmds.size() > last_cmd) {
    unsigned index = submits.size();

    // Push all pending cmd buffers to their own submission.
    submits.emplace_back();

    auto& submit = submits.back();
    memset(&submit, 0, sizeof(submit));
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.pNext = nullptr;
    submit.commandBufferCount = cmds.size() - last_cmd;
    submit.pCommandBuffers = cmds.data() + last_cmd;
    if (frame().swapchain_touched && !frame().swapchain_consumed) {
      static const VkFlags wait = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      if (wsi_acquire != VK_NULL_HANDLE) {
        waits[index].push_back(wsi_acquire);
        stages[index].push_back(wait);
      }

      VK_ASSERT(wsi_release != VK_NULL_HANDLE);
      signals[index].push_back(wsi_release);
      frame().swapchain_consumed = true;
    }
    last_cmd = cmds.size();
  }

  VkFence cleared_fence = frame().fence_manager.request_cleared_fence();
  VkSemaphore cleared_semaphore = VK_NULL_HANDLE;
  if (semaphore) {
    cleared_semaphore = semaphore_manager.request_cleared_semaphore();
    signals[submits.size() - 1].push_back(cleared_semaphore);
  }

  for (unsigned i = 0; i < submits.size(); i++) {
    auto& submit = submits[i];
    submit.waitSemaphoreCount = waits[i].size();
    if (!waits[i].empty()) {
      submit.pWaitSemaphores = waits[i].data();
      submit.pWaitDstStageMask = stages[i].data();
    }

    submit.signalSemaphoreCount = signals[i].size();
    if (!signals[i].empty())
      submit.pSignalSemaphores = signals[i].data();
  }

  VkQueue queue;
  switch (type) {
    default:
    case CommandBuffer::Type::Graphics:
      queue = graphics_queue;
      break;
    case CommandBuffer::Type::Compute:
      queue = compute_queue;
      break;
    case CommandBuffer::Type::Transfer:
      queue = transfer_queue;
      break;
  }

  if (queue_lock_callback)
    queue_lock_callback();
  VkResult result =
      vkQueueSubmit(queue, submits.size(), submits.data(), cleared_fence);
  if (queue_unlock_callback)
    queue_unlock_callback();
  if (result != VK_SUCCESS)
    LOGE("vkQueueSubmit failed.\n");
  submissions.clear();

  if (fence) {
    auto ptr = make_shared<FenceHolder>(this, cleared_fence);
    *fence = ptr;
    frame().fences.push_back(move(ptr));
  }

  if (semaphore) {
    auto ptr = make_handle<SemaphoreHolder>(this, cleared_semaphore, true);
    *semaphore = ptr;
  }
}

void Device::flush_frame(CommandBuffer::Type type) {
  auto& data = get_queue_data(type);
  auto& pool = get_command_pool(type);
  auto& submissions = get_queue_submissions(type);

  if (type == CommandBuffer::Type::Transfer) {
    // Flush any copies for pending chain allocators.
    frame().sync_to_gpu();
  }

  if (data.staging_cmd) {
    pool.signal_submitted(data.staging_cmd->get_command_buffer());
    vkEndCommandBuffer(data.staging_cmd->get_command_buffer());
    submissions.push_back(data.staging_cmd);
    data.staging_cmd.reset();
  }

  submit_queue(type, nullptr, nullptr);
}

void Device::flush_frame() {
  // The order here is important. We need to flush transfers first, because
  // create_buffer() and create_image() can relax semaphores into pipeline
  // barriers if the transfer queue and compute/graphics queues alias.
  flush_frame(CommandBuffer::Type::Transfer);

  flush_frame(CommandBuffer::Type::Compute);
  flush_frame(CommandBuffer::Type::Graphics);
}

void Device::begin_staging(CommandBuffer::Type type) {
  switch (type) {
    default:
    case CommandBuffer::Type::Graphics:
      if (!graphics.staging_cmd)
        graphics.staging_cmd = request_command_buffer(type);
      break;
    case CommandBuffer::Type::Compute:
      if (!compute.staging_cmd)
        compute.staging_cmd = request_command_buffer(type);
      break;
    case CommandBuffer::Type::Transfer:
      if (!transfer.staging_cmd)
        transfer.staging_cmd = request_command_buffer(type);
      break;
  }
}

Device::QueueData& Device::get_queue_data(CommandBuffer::Type type) {
  switch (type) {
    default:
    case CommandBuffer::Type::Graphics:
      return graphics;
    case CommandBuffer::Type::Compute:
      return compute;
    case CommandBuffer::Type::Transfer:
      return transfer;
  }
}

CommandPool& Device::get_command_pool(CommandBuffer::Type type) {
  switch (type) {
    default:
    case CommandBuffer::Type::Graphics:
      return frame().graphics_cmd_pool;
    case CommandBuffer::Type::Compute:
      return frame().compute_cmd_pool;
    case CommandBuffer::Type::Transfer:
      return frame().transfer_cmd_pool;
  }
}

vector<CommandBufferHandle>& Device::get_queue_submissions(
    CommandBuffer::Type type) {
  switch (type) {
    default:
    case CommandBuffer::Type::Graphics:
      return frame().graphics_submissions;
    case CommandBuffer::Type::Compute:
      return frame().compute_submissions;
    case CommandBuffer::Type::Transfer:
      return frame().transfer_submissions;
  }
}

CommandBufferHandle Device::request_command_buffer(CommandBuffer::Type type) {
  auto cmd = get_command_pool(type).request_command_buffer();

  VkCommandBufferBeginInfo info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &info);
  return make_handle<CommandBuffer>(this, cmd, pipeline_cache, type);
}

VkSemaphore Device::set_acquire(VkSemaphore acquire) {
  swap(acquire, wsi_acquire);
  return acquire;
}

VkSemaphore Device::set_release(VkSemaphore release) {
  swap(release, wsi_release);
  return release;
}

const Sampler& Device::get_stock_sampler(StockSampler sampler) const {
  return *samplers[static_cast<unsigned>(sampler)];
}

bool Device::swapchain_touched() const {
  return frame().swapchain_touched;
}

Device::~Device() {
  wait_idle();

  if (wsi_acquire != VK_NULL_HANDLE) {
    vkDestroySemaphore(device, wsi_acquire, nullptr);
    wsi_acquire = VK_NULL_HANDLE;
  }

  if (wsi_release != VK_NULL_HANDLE) {
    vkDestroySemaphore(device, wsi_release, nullptr);
    wsi_release = VK_NULL_HANDLE;
  }

  if (pipeline_cache != VK_NULL_HANDLE) {
    flush_pipeline_cache();
    vkDestroyPipelineCache(device, pipeline_cache, nullptr);
  }

  framebuffer_allocator.clear();
  transient_allocator.clear();
  for (auto& sampler : samplers)
    sampler.reset();

  for (auto& frame : per_frame)
    frame->release_owned_resources();
}

void Device::init_external_swapchain(
    const vector<ImageHandle>& swapchain_images) {
  wait_idle();

  // Clear out caches which might contain stale data from now on.
  framebuffer_allocator.clear();
  transient_allocator.clear();

  for (auto& frame : per_frame)
    frame->release_owned_resources();
  per_frame.clear();

  for (auto& image : swapchain_images) {
    auto frame = unique_ptr<PerFrame>(
        new PerFrame(this, allocator, semaphore_manager, event_manager,
                     graphics_queue_family_index, compute_queue_family_index,
                     transfer_queue_family_index));

    frame->backbuffer = image;
    per_frame.emplace_back(move(frame));
  }
}

void Device::init_swapchain(const vector<VkImage>& swapchain_images,
                            unsigned width,
                            unsigned height,
                            VkFormat format) {
  wait_idle();

  // Clear out caches which might contain stale data from now on.
  framebuffer_allocator.clear();
  transient_allocator.clear();

  for (auto& frame : per_frame)
    frame->release_owned_resources();
  per_frame.clear();

  const auto info = ImageCreateInfo::render_target(width, height, format);

  for (auto& image : swapchain_images) {
    auto frame = unique_ptr<PerFrame>(
        new PerFrame(this, allocator, semaphore_manager, event_manager,
                     graphics_queue_family_index, compute_queue_family_index,
                     transfer_queue_family_index));

    VkImageViewCreateInfo view_info = {
        VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_info.image = image;
    view_info.format = format;
    view_info.components.r = VK_COMPONENT_SWIZZLE_R;
    view_info.components.g = VK_COMPONENT_SWIZZLE_G;
    view_info.components.b = VK_COMPONENT_SWIZZLE_B;
    view_info.components.a = VK_COMPONENT_SWIZZLE_A;
    view_info.subresourceRange.aspectMask = format_to_aspect_mask(format);
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.layerCount = 1;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;

    VkImageView image_view;
    if (vkCreateImageView(device, &view_info, nullptr, &image_view) !=
        VK_SUCCESS)
      LOGE("Failed to create view for backbuffer.");

    frame->backbuffer =
        make_handle<Image>(this, image, image_view, DeviceAllocation{}, info);
    frame->backbuffer->set_swapchain_layout(VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    per_frame.emplace_back(move(frame));
  }
}

Device::PerFrame::PerFrame(Device* device,
                           DeviceAllocator& global,
                           SemaphoreManager& semaphore_manager,
                           EventManager& event_manager,
                           uint32_t graphics_queue_family_index,
                           uint32_t compute_queue_family_index,
                           uint32_t transfer_queue_family_index)
    : device(device->get_device()),
      global_allocator(global),
      semaphore_manager(semaphore_manager),
      event_manager(event_manager),
      graphics_cmd_pool(device->get_device(), graphics_queue_family_index),
      compute_cmd_pool(device->get_device(), compute_queue_family_index),
      transfer_cmd_pool(device->get_device(), transfer_queue_family_index),
      fence_manager(device->get_device()),
      vbo_chain(device, 1024 * 1024, 64, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT),
      ibo_chain(device, 1024 * 1024, 64, VK_BUFFER_USAGE_INDEX_BUFFER_BIT),
      ubo_chain(
          device,
          1024 * 1024,
          device->get_gpu_properties().limits.minUniformBufferOffsetAlignment,
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT),
      staging_chain(device,
                    4 * 1024 * 1024,
                    64,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT) {}

void Device::free_memory(const DeviceAllocation& alloc) {
  frame().allocations.push_back(alloc);
}

#ifdef VULKAN_DEBUG

template <typename T, typename U>
static inline bool exists(const T& container, const U& value) {
  return find(begin(container), end(container), value) != end(container);
}

#endif

void Device::destroy_pipeline(VkPipeline pipeline) {
  VK_ASSERT(!exists(frame().destroyed_pipelines, pipeline));
  frame().destroyed_pipelines.push_back(pipeline);
}

void Device::destroy_image_view(VkImageView view) {
  VK_ASSERT(!exists(frame().destroyed_image_views, view));
  frame().destroyed_image_views.push_back(view);
}

void Device::destroy_buffer_view(VkBufferView view) {
  VK_ASSERT(!exists(frame().destroyed_buffer_views, view));
  frame().destroyed_buffer_views.push_back(view);
}

void Device::destroy_semaphore(VkSemaphore semaphore) {
  VK_ASSERT(!exists(frame().destroyed_semaphores, semaphore));
  frame().destroyed_semaphores.push_back(semaphore);
}

void Device::destroy_event(VkEvent event) {
  VK_ASSERT(!exists(frame().recycled_events, event));
  frame().recycled_events.push_back(event);
}

PipelineEvent Device::request_pipeline_event() {
  return Util::make_handle<EventHolder>(this,
                                        event_manager.request_cleared_event());
}

void Device::destroy_image(VkImage image) {
  VK_ASSERT(!exists(frame().destroyed_images, image));
  frame().destroyed_images.push_back(image);
}

void Device::destroy_buffer(VkBuffer buffer) {
  VK_ASSERT(!exists(frame().destroyed_buffers, buffer));
  frame().destroyed_buffers.push_back(buffer);
}

void Device::destroy_sampler(VkSampler sampler) {
  VK_ASSERT(!exists(frame().destroyed_samplers, sampler));
  frame().destroyed_samplers.push_back(sampler);
}

void Device::destroy_framebuffer(VkFramebuffer framebuffer) {
  VK_ASSERT(!exists(frame().destroyed_framebuffers, framebuffer));
  frame().destroyed_framebuffers.push_back(framebuffer);
}

void Device::clear_wait_semaphores() {
  for (auto& sem : graphics.wait_semaphores)
    vkDestroySemaphore(device, sem->consume(), nullptr);
  for (auto& sem : compute.wait_semaphores)
    vkDestroySemaphore(device, sem->consume(), nullptr);
  for (auto& sem : transfer.wait_semaphores)
    vkDestroySemaphore(device, sem->consume(), nullptr);

  graphics.wait_semaphores.clear();
  graphics.wait_stages.clear();
  graphics.wait_for_graphics = 0;
  graphics.wait_for_transfer = 0;
  graphics.wait_for_compute = 0;
  compute.wait_semaphores.clear();
  compute.wait_stages.clear();
  compute.wait_for_graphics = 0;
  compute.wait_for_transfer = 0;
  compute.wait_for_compute = 0;
  transfer.wait_semaphores.clear();
  transfer.wait_stages.clear();
  transfer.wait_for_graphics = 0;
  transfer.wait_for_transfer = 0;
  transfer.wait_for_compute = 0;
}

void Device::wait_idle() {
  if (!per_frame.empty())
    flush_frame();

  if (device != VK_NULL_HANDLE) {
    if (queue_lock_callback)
      queue_lock_callback();
    vkDeviceWaitIdle(device);
    if (queue_unlock_callback)
      queue_unlock_callback();
  }

  clear_wait_semaphores();

  framebuffer_allocator.clear();
  transient_allocator.clear();
  for (auto& allocator : descriptor_set_allocators)
    allocator.second->clear();

  for (auto& frame : per_frame) {
    // Avoid double-wait-on-semaphore scenarios.
    bool touched_swapchain = frame->swapchain_touched;
    frame->cleanup();
    frame->begin();
    frame->swapchain_touched = touched_swapchain;
  }
}

void Device::begin_frame(unsigned index) {
  // Flush the frame here as we might have pending staging command buffers from
  // init stage.
  flush_frame();

  current_swapchain_index = index;

  frame().begin();
  framebuffer_allocator.begin_frame();
  transient_allocator.begin_frame();
  physical_allocator.begin_frame();
  for (auto& allocator : descriptor_set_allocators)
    allocator.second->begin_frame();
}

void Device::PerFrame::sync_to_gpu() {
  ubo_chain.sync_to_gpu();
  staging_chain.sync_to_gpu();
  vbo_chain.sync_to_gpu();
  ibo_chain.sync_to_gpu();
}

void Device::PerFrame::begin() {
  ubo_chain.discard();
  staging_chain.discard();
  vbo_chain.discard();
  ibo_chain.discard();
  fence_manager.begin();
  graphics_cmd_pool.begin();
  compute_cmd_pool.begin();
  transfer_cmd_pool.begin();

  for (auto& framebuffer : destroyed_framebuffers)
    vkDestroyFramebuffer(device, framebuffer, nullptr);
  for (auto& sampler : destroyed_samplers)
    vkDestroySampler(device, sampler, nullptr);
  for (auto& pipeline : destroyed_pipelines)
    vkDestroyPipeline(device, pipeline, nullptr);
  for (auto& view : destroyed_image_views)
    vkDestroyImageView(device, view, nullptr);
  for (auto& view : destroyed_buffer_views)
    vkDestroyBufferView(device, view, nullptr);
  for (auto& image : destroyed_images)
    vkDestroyImage(device, image, nullptr);
  for (auto& buffer : destroyed_buffers)
    vkDestroyBuffer(device, buffer, nullptr);
  for (auto& semaphore : destroyed_semaphores)
    vkDestroySemaphore(device, semaphore, nullptr);
  for (auto& semaphore : recycled_semaphores)
    semaphore_manager.recycle(semaphore);
  for (auto& event : recycled_events)
    event_manager.recycle(event);
  for (auto& alloc : allocations)
    alloc.free_immediate(global_allocator);

  destroyed_framebuffers.clear();
  destroyed_samplers.clear();
  destroyed_pipelines.clear();
  destroyed_image_views.clear();
  destroyed_buffer_views.clear();
  destroyed_images.clear();
  destroyed_buffers.clear();
  destroyed_semaphores.clear();
  recycled_semaphores.clear();
  recycled_events.clear();
  allocations.clear();
  fences.clear();

  swapchain_touched = false;
  swapchain_consumed = false;
}

void Device::PerFrame::cleanup() {
  vbo_chain.reset();
  ibo_chain.reset();
  ubo_chain.reset();
  staging_chain.reset();
}

void Device::PerFrame::release_owned_resources() {
  cleanup();
  backbuffer.reset();
}

Device::PerFrame::~PerFrame() {
  cleanup();
  begin();
}

ChainDataAllocation Device::allocate_constant_data(VkDeviceSize size) {
  return frame().ubo_chain.allocate(size);
}

ChainDataAllocation Device::allocate_vertex_data(VkDeviceSize size) {
  return frame().vbo_chain.allocate(size);
}

ChainDataAllocation Device::allocate_index_data(VkDeviceSize size) {
  return frame().ibo_chain.allocate(size);
}

ChainDataAllocation Device::allocate_staging_data(VkDeviceSize size) {
  return frame().staging_chain.allocate(size);
}

uint32_t Device::find_memory_type(BufferDomain domain, uint32_t mask) {
  uint32_t desired = 0, fallback = 0;
  switch (domain) {
    case BufferDomain::Device:
      desired = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
      fallback = 0;
      break;

    case BufferDomain::LinkedDeviceHost:
      desired = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
      fallback = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
      break;

    case BufferDomain::Host:
      desired = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
      fallback = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
      break;

    case BufferDomain::CachedHost:
      desired = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
      fallback = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
      break;
  }

  for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
    if ((1u << i) & mask) {
      uint32_t flags = mem_props.memoryTypes[i].propertyFlags;
      if ((flags & desired) == desired)
        return i;
    }
  }

  for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
    if ((1u << i) & mask) {
      uint32_t flags = mem_props.memoryTypes[i].propertyFlags;
      if ((flags & fallback) == fallback)
        return i;
    }
  }

  throw runtime_error("Couldn't find memory type.");
}

uint32_t Device::find_memory_type(ImageDomain domain, uint32_t mask) {
  uint32_t desired = 0, fallback = 0;
  switch (domain) {
    case ImageDomain::Physical:
      desired = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
      fallback = 0;
      break;

    case ImageDomain::Transient:
      desired = VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;
      fallback = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
      break;
  }

  for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
    if ((1u << i) & mask) {
      uint32_t flags = mem_props.memoryTypes[i].propertyFlags;
      if ((flags & desired) == desired)
        return i;
    }
  }

  for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
    if ((1u << i) & mask) {
      uint32_t flags = mem_props.memoryTypes[i].propertyFlags;
      if ((flags & fallback) == fallback)
        return i;
    }
  }

  throw runtime_error("Couldn't find memory type.");
}

static inline VkImageViewType get_image_view_type(
    const ImageCreateInfo& create_info,
    const ImageViewCreateInfo* view) {
  unsigned layers = view ? view->layers : create_info.layers;
  unsigned levels = view ? view->levels : create_info.levels;
  unsigned base_level = view ? view->base_level : 0;
  unsigned base_layer = view ? view->base_layer : 0;

  if (layers == VK_REMAINING_ARRAY_LAYERS)
    layers = create_info.layers - base_layer;
  if (levels == VK_REMAINING_MIP_LEVELS)
    levels = create_info.levels - base_level;

  bool force_array = view ? (view->misc & IMAGE_VIEW_MISC_FORCE_ARRAY_BIT)
                          : (create_info.misc & IMAGE_MISC_FORCE_ARRAY_BIT);

  switch (create_info.type) {
    case VK_IMAGE_TYPE_1D:
      VK_ASSERT(create_info.width >= 1);
      VK_ASSERT(create_info.height == 1);
      VK_ASSERT(create_info.depth == 1);
      VK_ASSERT(create_info.samples == VK_SAMPLE_COUNT_1_BIT);

      if (layers > 1 || force_array)
        return VK_IMAGE_VIEW_TYPE_1D_ARRAY;
      else
        return VK_IMAGE_VIEW_TYPE_1D;

    case VK_IMAGE_TYPE_2D:
      VK_ASSERT(create_info.width >= 1);
      VK_ASSERT(create_info.height >= 1);
      VK_ASSERT(create_info.depth == 1);

      if ((create_info.flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) &&
          (layers % 6) == 0) {
        VK_ASSERT(create_info.width == create_info.height);

        if (layers > 6 || force_array)
          return VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
        else
          return VK_IMAGE_VIEW_TYPE_CUBE;
      } else {
        if (layers > 1 || force_array)
          return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        else
          return VK_IMAGE_VIEW_TYPE_2D;
      }

    case VK_IMAGE_TYPE_3D:
      VK_ASSERT(create_info.width >= 1);
      VK_ASSERT(create_info.height >= 1);
      VK_ASSERT(create_info.depth >= 1);
      return VK_IMAGE_VIEW_TYPE_3D;

    default:
      VK_ASSERT(0 && "bogus");
      return VK_IMAGE_VIEW_TYPE_RANGE_SIZE;
  }
}

BufferViewHandle Device::create_buffer_view(
    const BufferViewCreateInfo& view_info) {
  VkBufferViewCreateInfo info = {VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO};
  info.buffer = view_info.buffer->get_buffer();
  info.format = view_info.format;
  info.offset = view_info.offset;
  info.range = view_info.range;

  VkBufferView view;
  auto res = vkCreateBufferView(device, &info, nullptr, &view);
  if (res != VK_SUCCESS)
    return nullptr;

  return make_handle<BufferView>(this, view, view_info);
}

ImageViewHandle Device::create_image_view(
    const ImageViewCreateInfo& create_info) {
  auto& image_create_info = create_info.image->get_create_info();

  VkFormat format = create_info.format != VK_FORMAT_UNDEFINED
                        ? create_info.format
                        : image_create_info.format;

  VkImageViewCreateInfo view_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  view_info.image = create_info.image->get_image();
  view_info.format = format;
  view_info.components = create_info.swizzle;
  view_info.subresourceRange.aspectMask = format_to_aspect_mask(format);
  view_info.subresourceRange.baseMipLevel = create_info.base_level;
  view_info.subresourceRange.baseArrayLayer = create_info.base_layer;
  view_info.subresourceRange.levelCount = create_info.levels;
  view_info.subresourceRange.layerCount = create_info.layers;
  view_info.viewType = get_image_view_type(image_create_info, &create_info);

  unsigned num_levels;
  if (view_info.subresourceRange.levelCount == VK_REMAINING_MIP_LEVELS)
    num_levels = create_info.image->get_create_info().levels -
                 view_info.subresourceRange.baseMipLevel;
  else
    num_levels = view_info.subresourceRange.levelCount;

  VkImageView image_view = VK_NULL_HANDLE;
  VkImageView depth_view = VK_NULL_HANDLE;
  VkImageView stencil_view = VK_NULL_HANDLE;
  VkImageView base_level_view = VK_NULL_HANDLE;
  if (vkCreateImageView(device, &view_info, nullptr, &image_view) != VK_SUCCESS)
    return nullptr;

  if (num_levels > 1) {
    view_info.subresourceRange.levelCount = 1;
    if (vkCreateImageView(device, &view_info, nullptr, &base_level_view) !=
        VK_SUCCESS) {
      vkDestroyImageView(device, image_view, nullptr);
      return nullptr;
    }
    view_info.subresourceRange.levelCount = create_info.levels;
  }

  // If the image has multiple aspects, make split up images.
  if (view_info.subresourceRange.aspectMask ==
      (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (vkCreateImageView(device, &view_info, nullptr, &depth_view) !=
        VK_SUCCESS) {
      vkDestroyImageView(device, image_view, nullptr);
      vkDestroyImageView(device, base_level_view, nullptr);
      return nullptr;
    }

    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
    if (vkCreateImageView(device, &view_info, nullptr, &stencil_view) !=
        VK_SUCCESS) {
      vkDestroyImageView(device, image_view, nullptr);
      vkDestroyImageView(device, depth_view, nullptr);
      vkDestroyImageView(device, base_level_view, nullptr);
      return nullptr;
    }
  }

  ImageViewCreateInfo tmp = create_info;
  tmp.format = format;
  auto ret = make_handle<ImageView>(this, image_view, tmp);
  ret->set_alt_views(depth_view, stencil_view);
  ret->set_base_level_view(base_level_view);
  return ret;
}

#ifndef _WIN32
ImageHandle Device::create_imported_image(
    int fd,
    VkDeviceSize size,
    uint32_t memory_type,
    VkExternalMemoryHandleTypeFlagBitsKHR handle_type,
    const ImageCreateInfo& create_info) {
  if (!supports_external)
    return {};

  VkImageCreateInfo info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  info.format = create_info.format;
  info.extent.width = create_info.width;
  info.extent.height = create_info.height;
  info.extent.depth = create_info.depth;
  info.imageType = create_info.type;
  info.mipLevels = create_info.levels;
  info.arrayLayers = create_info.layers;
  info.samples = create_info.samples;
  info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.usage = create_info.usage;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  info.flags = create_info.flags;
  VK_ASSERT(create_info.domain != ImageDomain::Transient);

  VkExternalMemoryImageCreateInfoKHR externalInfo = {
      VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO_KHR};
  externalInfo.handleTypes = handle_type;
  info.pNext = &externalInfo;

  VK_ASSERT(format_is_supported(create_info.format,
                                image_usage_to_features(info.usage)));

  VkImage image;
  if (vkCreateImage(device, &info, nullptr, &image) != VK_SUCCESS)
    return nullptr;

  VkMemoryAllocateInfo alloc_info = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  alloc_info.allocationSize = size;
  alloc_info.memoryTypeIndex = memory_type;

  VkMemoryDedicatedAllocateInfoKHR dedicated_info = {
      VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_KHR};
  dedicated_info.image = image;
  alloc_info.pNext = &dedicated_info;

  VkImportMemoryFdInfoKHR fd_info = {
      VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR};
  fd_info.handleType = handle_type;
  fd_info.fd = fd;
  dedicated_info.pNext = &fd_info;

  VkDeviceMemory memory;

  VkMemoryRequirements reqs;
  vkGetImageMemoryRequirements(device, image, &reqs);
  if (reqs.size > size) {
    vkDestroyImage(device, image, nullptr);
    return nullptr;
  }

  if (((1u << memory_type) & reqs.memoryTypeBits) == 0) {
    vkDestroyImage(device, image, nullptr);
    return nullptr;
  }

  if (vkAllocateMemory(device, &alloc_info, nullptr, &memory) != VK_SUCCESS) {
    vkDestroyImage(device, image, nullptr);
    return nullptr;
  }

  if (vkBindImageMemory(device, image, memory, 0) != VK_SUCCESS) {
    vkDestroyImage(device, image, nullptr);
    vkFreeMemory(device, memory, nullptr);
    return nullptr;
  }

  // Create a default image view.
  VkImageView image_view = VK_NULL_HANDLE;
  VkImageView depth_view = VK_NULL_HANDLE;
  VkImageView stencil_view = VK_NULL_HANDLE;
  VkImageView base_level_view = VK_NULL_HANDLE;
  if (info.usage & (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                    VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT)) {
    VkImageViewCreateInfo view_info = {
        VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_info.image = image;
    view_info.format = create_info.format;
    view_info.components.r = VK_COMPONENT_SWIZZLE_R;
    view_info.components.g = VK_COMPONENT_SWIZZLE_G;
    view_info.components.b = VK_COMPONENT_SWIZZLE_B;
    view_info.components.a = VK_COMPONENT_SWIZZLE_A;
    view_info.subresourceRange.aspectMask =
        format_to_aspect_mask(view_info.format);
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.levelCount = info.mipLevels;
    view_info.subresourceRange.layerCount = info.arrayLayers;
    view_info.viewType = get_image_view_type(create_info, nullptr);

    if (vkCreateImageView(device, &view_info, nullptr, &image_view) !=
        VK_SUCCESS) {
      vkFreeMemory(device, memory, nullptr);
      vkDestroyImage(device, image, nullptr);
      return nullptr;
    }

    if (info.mipLevels > 1) {
      view_info.subresourceRange.levelCount = 1;
      if (vkCreateImageView(device, &view_info, nullptr, &base_level_view) !=
          VK_SUCCESS) {
        vkFreeMemory(device, memory, nullptr);
        vkDestroyImage(device, image, nullptr);
        vkDestroyImageView(device, image_view, nullptr);
        return nullptr;
      }
      view_info.subresourceRange.levelCount = info.mipLevels;
    }

    // If the image has multiple aspects, make split up images.
    if (view_info.subresourceRange.aspectMask ==
        (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {
      view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
      if (vkCreateImageView(device, &view_info, nullptr, &depth_view) !=
          VK_SUCCESS) {
        vkFreeMemory(device, memory, nullptr);
        vkDestroyImageView(device, image_view, nullptr);
        vkDestroyImageView(device, base_level_view, nullptr);
        vkDestroyImage(device, image, nullptr);
        return nullptr;
      }

      view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
      if (vkCreateImageView(device, &view_info, nullptr, &stencil_view) !=
          VK_SUCCESS) {
        vkFreeMemory(device, memory, nullptr);
        vkDestroyImageView(device, image_view, nullptr);
        vkDestroyImageView(device, depth_view, nullptr);
        vkDestroyImageView(device, base_level_view, nullptr);
        vkDestroyImage(device, image, nullptr);
        return nullptr;
      }
    }
  }

  auto allocation =
      DeviceAllocation::make_imported_allocation(memory, size, memory_type);
  auto handle =
      make_handle<Image>(this, image, image_view, allocation, create_info);
  handle->get_view().set_alt_views(depth_view, stencil_view);
  handle->get_view().set_base_level_view(base_level_view);

  // Set possible dstStage and dstAccess.
  handle->set_stage_flags(image_usage_to_possible_stages(info.usage));
  handle->set_access_flags(image_usage_to_possible_access(info.usage));
  return handle;
}
#endif

ImageHandle Device::create_image(const ImageCreateInfo& create_info,
                                 const ImageInitialData* initial) {
  VkImage image;
  VkMemoryRequirements reqs;
  DeviceAllocation allocation;

  VkImageCreateInfo info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  info.format = create_info.format;
  info.extent.width = create_info.width;
  info.extent.height = create_info.height;
  info.extent.depth = create_info.depth;
  info.imageType = create_info.type;
  info.mipLevels = create_info.levels;
  info.arrayLayers = create_info.layers;
  info.samples = create_info.samples;
  info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.usage = create_info.usage;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (create_info.domain == ImageDomain::Transient)
    info.usage |= VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
  if (initial)
    info.usage |=
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

  info.flags = create_info.flags;
  if (create_info.usage & VK_IMAGE_USAGE_STORAGE_BIT)
    info.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;

  if (info.mipLevels == 0)
    info.mipLevels = image_num_miplevels(info.extent);

  // Only do this conditionally.
  // On AMD, using CONCURRENT with async compute disables compression.
  uint32_t sharing_indices[3];
  bool concurrent_queue =
      (create_info.misc & IMAGE_MISC_CONCURRENT_QUEUE_BIT) != 0;
  if (concurrent_queue &&
      (graphics_queue_family_index != compute_queue_family_index ||
       graphics_queue_family_index != transfer_queue_family_index)) {
    info.sharingMode = VK_SHARING_MODE_CONCURRENT;
    sharing_indices[info.queueFamilyIndexCount++] = graphics_queue_family_index;

    if (graphics_queue_family_index != compute_queue_family_index)
      sharing_indices[info.queueFamilyIndexCount++] =
          compute_queue_family_index;

    if (graphics_queue_family_index != transfer_queue_family_index &&
        compute_queue_family_index != transfer_queue_family_index) {
      sharing_indices[info.queueFamilyIndexCount++] =
          transfer_queue_family_index;
    }

    info.pQueueFamilyIndices = sharing_indices;
  }

  VK_ASSERT(format_is_supported(create_info.format,
                                image_usage_to_features(info.usage)));

  if (vkCreateImage(device, &info, nullptr, &image) != VK_SUCCESS)
    return nullptr;

  vkGetImageMemoryRequirements(device, image, &reqs);
  uint32_t memory_type =
      find_memory_type(create_info.domain, reqs.memoryTypeBits);
  if (!allocator.allocate_image_memory(reqs.size, reqs.alignment, memory_type,
                                       ALLOCATION_TILING_OPTIMAL, &allocation,
                                       image)) {
    vkDestroyImage(device, image, nullptr);
    return nullptr;
  }

  if (vkBindImageMemory(device, image, allocation.get_memory(),
                        allocation.get_offset()) != VK_SUCCESS) {
    allocation.free_immediate(allocator);
    vkDestroyImage(device, image, nullptr);
    return nullptr;
  }

  auto tmpinfo = create_info;
  tmpinfo.usage = info.usage;
  tmpinfo.levels = info.mipLevels;

  // Create a default image view.
  VkImageView image_view = VK_NULL_HANDLE;
  VkImageView depth_view = VK_NULL_HANDLE;
  VkImageView stencil_view = VK_NULL_HANDLE;
  VkImageView base_level_view = VK_NULL_HANDLE;
  if (info.usage & (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                    VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT)) {
    VkImageViewCreateInfo view_info = {
        VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_info.image = image;
    view_info.format = create_info.format;
    view_info.components.r = VK_COMPONENT_SWIZZLE_R;
    view_info.components.g = VK_COMPONENT_SWIZZLE_G;
    view_info.components.b = VK_COMPONENT_SWIZZLE_B;
    view_info.components.a = VK_COMPONENT_SWIZZLE_A;
    view_info.subresourceRange.aspectMask =
        format_to_aspect_mask(view_info.format);
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.levelCount = info.mipLevels;
    view_info.subresourceRange.layerCount = info.arrayLayers;
    view_info.viewType = get_image_view_type(tmpinfo, nullptr);

    if (vkCreateImageView(device, &view_info, nullptr, &image_view) !=
        VK_SUCCESS) {
      allocation.free_immediate(allocator);
      vkDestroyImage(device, image, nullptr);
      return nullptr;
    }

    if (info.mipLevels > 1) {
      view_info.subresourceRange.levelCount = 1;
      if (vkCreateImageView(device, &view_info, nullptr, &base_level_view) !=
          VK_SUCCESS) {
        allocation.free_immediate(allocator);
        vkDestroyImage(device, image, nullptr);
        vkDestroyImageView(device, image_view, nullptr);
        return nullptr;
      }
      view_info.subresourceRange.levelCount = info.mipLevels;
    }

    // If the image has multiple aspects, make split up images.
    if (view_info.subresourceRange.aspectMask ==
        (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {
      view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
      if (vkCreateImageView(device, &view_info, nullptr, &depth_view) !=
          VK_SUCCESS) {
        allocation.free_immediate(allocator);
        vkDestroyImageView(device, image_view, nullptr);
        vkDestroyImageView(device, base_level_view, nullptr);
        vkDestroyImage(device, image, nullptr);
        return nullptr;
      }

      view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
      if (vkCreateImageView(device, &view_info, nullptr, &stencil_view) !=
          VK_SUCCESS) {
        allocation.free_immediate(allocator);
        vkDestroyImageView(device, image_view, nullptr);
        vkDestroyImageView(device, depth_view, nullptr);
        vkDestroyImageView(device, base_level_view, nullptr);
        vkDestroyImage(device, image, nullptr);
        return nullptr;
      }
    }
  }

  auto handle =
      make_handle<Image>(this, image, image_view, allocation, tmpinfo);
  handle->get_view().set_alt_views(depth_view, stencil_view);
  handle->get_view().set_base_level_view(base_level_view);

  // Set possible dstStage and dstAccess.
  handle->set_stage_flags(image_usage_to_possible_stages(info.usage));
  handle->set_access_flags(image_usage_to_possible_access(info.usage));

  // Copy initial data to texture.
  if (initial) {
    begin_staging(CommandBuffer::Type::Transfer);
    begin_staging(CommandBuffer::Type::Graphics);

    VK_ASSERT(create_info.domain != ImageDomain::Transient);
    VK_ASSERT(create_info.initial_layout != VK_IMAGE_LAYOUT_UNDEFINED);
    bool generate_mips = (create_info.misc & IMAGE_MISC_GENERATE_MIPS_BIT) != 0;
    unsigned copy_levels = generate_mips ? 1u : info.mipLevels;

    transfer.staging_cmd->image_barrier(
        *handle, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);

    handle->set_layout(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    VkExtent3D extent = {create_info.width, create_info.height,
                         create_info.depth};

    VkImageSubresourceLayers subresource = {
        format_to_aspect_mask(info.format),
        0,
        0,
        1,
    };

    unsigned index = 0;
    for (unsigned level = 0; level < copy_levels; level++) {
      for (unsigned layer = 0; layer < create_info.layers; layer++, index++) {
        subresource.baseArrayLayer = layer;
        subresource.mipLevel = level;

        uint32_t row_length = initial[index].row_length
                                  ? initial[index].row_length
                                  : extent.width;
        uint32_t array_height = initial[index].array_height
                                    ? initial[index].array_height
                                    : extent.height;

        uint32_t blocks_x = row_length;
        uint32_t blocks_y = array_height;
        format_num_blocks(create_info.format, blocks_x, blocks_y);
        format_align_dim(create_info.format, row_length, array_height);

        VkDeviceSize size = format_block_size(create_info.format) *
                            extent.depth * blocks_x * blocks_y;

        auto* ptr = transfer.staging_cmd->update_image(
            *handle, {0, 0, 0}, extent, row_length, array_height, subresource);
        VK_ASSERT(ptr);
        memcpy(ptr, initial[index].data, size);
      }

      extent.width = max(extent.width >> 1u, 1u);
      extent.height = max(extent.height >> 1u, 1u);
      extent.depth = max(extent.depth >> 1u, 1u);
    }

    // If graphics_queue != transfer_queue, we will use a semaphore, so no
    // srcAccess mask is necessary.
    VkAccessFlags final_transition_src_access = 0;
    if (generate_mips)
      final_transition_src_access =
          VK_ACCESS_TRANSFER_READ_BIT;  // Validation complains otherwise.
    else if (graphics_queue == transfer_queue)
      final_transition_src_access = VK_ACCESS_TRANSFER_WRITE_BIT;

    VkAccessFlags prepare_src_access =
        graphics_queue == transfer_queue ? VK_ACCESS_TRANSFER_WRITE_BIT : 0;
    bool need_mipmap_barrier = true;
    bool need_initial_barrier = true;

    // Now we've used the TRANSFER queue to copy data over to the GPU.
    // For mipmapping, we're now moving over to graphics,
    // the transfer queue is designed for CPU <-> GPU and that's it.

    // For concurrent queue mode, we just need to inject a semaphore.
    // For non-concurrent queue mode, we will have to inject ownership transfer
    // barrier if the queue families do not match.

    if (transfer_queue != graphics_queue) {
      VkPipelineStageFlags dst_stages =
          generate_mips ? VkPipelineStageFlags(VK_PIPELINE_STAGE_TRANSFER_BIT)
                        : handle->get_stage_flags();

      // We can't just use semaphores, we will also need a release + acquire
      // barrier to marshal ownership from transfer queue over to graphics ...
      if (!concurrent_queue &&
          transfer_queue_family_index != graphics_queue_family_index) {
        VkImageMemoryBarrier release = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        release.image = handle->get_image();
        release.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        release.dstAccessMask = 0;
        release.srcQueueFamilyIndex = transfer_queue_family_index;
        release.dstQueueFamilyIndex = graphics_queue_family_index;
        release.oldLayout = handle->get_layout();

        if (generate_mips) {
          release.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
          release.subresourceRange.levelCount = 1;
          need_mipmap_barrier = false;
        } else {
          release.newLayout = create_info.initial_layout;
          release.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
          need_initial_barrier = false;
        }

        handle->set_layout(release.newLayout);
        release.subresourceRange.aspectMask =
            format_to_aspect_mask(info.format);
        release.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;

        VkImageMemoryBarrier acquire = release;
        acquire.srcAccessMask = 0;

        if (generate_mips)
          acquire.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        else
          acquire.dstAccessMask =
              handle->get_access_flags() &
              image_layout_to_possible_access(create_info.initial_layout);

        transfer.staging_cmd->barrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,
                                      nullptr, 0, nullptr, 1, &release);

        graphics.staging_cmd->barrier(dst_stages, dst_stages, 0, nullptr, 0,
                                      nullptr, 1, &acquire);

        add_queue_dependency(CommandBuffer::Type::Graphics, dst_stages,
                             CommandBuffer::Type::Transfer);
      } else {
        add_queue_dependency(CommandBuffer::Type::Graphics,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             CommandBuffer::Type::Transfer);
      }
    }

    if (generate_mips) {
      graphics.staging_cmd->barrier_prepare_generate_mipmap(
          *handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
          VK_PIPELINE_STAGE_TRANSFER_BIT, prepare_src_access,
          need_mipmap_barrier);
      handle->set_layout(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
      graphics.staging_cmd->generate_mipmap(*handle);
    }

    if (need_initial_barrier) {
      graphics.staging_cmd->image_barrier(
          *handle, handle->get_layout(), create_info.initial_layout,
          VK_PIPELINE_STAGE_TRANSFER_BIT, final_transition_src_access,
          handle->get_stage_flags(),
          handle->get_access_flags() &
              image_layout_to_possible_access(create_info.initial_layout));
    }

    // For concurrent queue, make sure that compute can see the final image as
    // well.
    if (concurrent_queue && graphics_queue != compute_queue) {
      add_queue_dependency(
          CommandBuffer::Type::Compute,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
          CommandBuffer::Type::Graphics);
    }
  } else if (create_info.initial_layout != VK_IMAGE_LAYOUT_UNDEFINED) {
    begin_staging(CommandBuffer::Type::Graphics);

    VK_ASSERT(create_info.domain != ImageDomain::Transient);
    graphics.staging_cmd->image_barrier(
        *handle, VK_IMAGE_LAYOUT_UNDEFINED, create_info.initial_layout,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, handle->get_stage_flags(),
        handle->get_access_flags() &
            image_layout_to_possible_access(create_info.initial_layout));

    // For concurrent queue, make sure that compute can see the final image as
    // well.
    if (concurrent_queue && graphics_queue != compute_queue) {
      add_queue_dependency(
          CommandBuffer::Type::Compute,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
          CommandBuffer::Type::Graphics);
    }
  }

  handle->set_layout(create_info.initial_layout);
  return handle;
}

SamplerHandle Device::create_sampler(const SamplerCreateInfo& sampler_info) {
  VkSamplerCreateInfo info = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};

  info.magFilter = sampler_info.magFilter;
  info.minFilter = sampler_info.minFilter;
  info.mipmapMode = sampler_info.mipmapMode;
  info.addressModeU = sampler_info.addressModeU;
  info.addressModeV = sampler_info.addressModeV;
  info.addressModeW = sampler_info.addressModeW;
  info.mipLodBias = sampler_info.mipLodBias;
  info.anisotropyEnable = sampler_info.anisotropyEnable;
  info.maxAnisotropy = sampler_info.maxAnisotropy;
  info.compareEnable = sampler_info.compareEnable;
  info.compareOp = sampler_info.compareOp;
  info.minLod = sampler_info.minLod;
  info.maxLod = sampler_info.maxLod;
  info.borderColor = sampler_info.borderColor;
  info.unnormalizedCoordinates = sampler_info.unnormalizedCoordinates;

  VkSampler sampler;
  if (vkCreateSampler(device, &info, nullptr, &sampler) != VK_SUCCESS)
    return nullptr;
  return make_handle<Sampler>(this, sampler, sampler_info);
}

BufferHandle Device::create_buffer(const BufferCreateInfo& create_info,
                                   const void* initial) {
  VkBuffer buffer;
  VkMemoryRequirements reqs;
  DeviceAllocation allocation;

  VkBufferCreateInfo info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  info.size = create_info.size;
  info.usage = create_info.usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  uint32_t sharing_indices[3];
  if (graphics_queue_family_index != compute_queue_family_index ||
      graphics_queue_family_index != transfer_queue_family_index) {
    // For buffers, always just use CONCURRENT access modes,
    // so we don't have to deal with acquire/release barriers in async compute.
    info.sharingMode = VK_SHARING_MODE_CONCURRENT;

    sharing_indices[info.queueFamilyIndexCount++] = graphics_queue_family_index;

    if (graphics_queue_family_index != compute_queue_family_index)
      sharing_indices[info.queueFamilyIndexCount++] =
          compute_queue_family_index;

    if (graphics_queue_family_index != transfer_queue_family_index &&
        compute_queue_family_index != transfer_queue_family_index) {
      sharing_indices[info.queueFamilyIndexCount++] =
          transfer_queue_family_index;
    }

    info.pQueueFamilyIndices = sharing_indices;
  }

  if (vkCreateBuffer(device, &info, nullptr, &buffer) != VK_SUCCESS)
    return nullptr;

  vkGetBufferMemoryRequirements(device, buffer, &reqs);

  uint32_t memory_type =
      find_memory_type(create_info.domain, reqs.memoryTypeBits);

  if (!allocator.allocate(reqs.size, reqs.alignment, memory_type,
                          ALLOCATION_TILING_LINEAR, &allocation)) {
    vkDestroyBuffer(device, buffer, nullptr);
    return nullptr;
  }

  if (vkBindBufferMemory(device, buffer, allocation.get_memory(),
                         allocation.get_offset()) != VK_SUCCESS) {
    allocation.free_immediate(allocator);
    vkDestroyBuffer(device, buffer, nullptr);
    return nullptr;
  }

  auto tmpinfo = create_info;
  tmpinfo.usage |=
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  auto handle = make_handle<Buffer>(this, buffer, allocation, tmpinfo);

  if (create_info.domain == BufferDomain::Device && initial &&
      !memory_type_is_host_visible(memory_type)) {
    begin_staging(CommandBuffer::Type::Transfer);
    auto* ptr =
        transfer.staging_cmd->update_buffer(*handle, 0, create_info.size);
    VK_ASSERT(ptr);
    memcpy(ptr, initial, create_info.size);
    add_staging_transfer_queue_dependency(*handle, info.usage);
  } else if (initial) {
    void* ptr = allocator.map_memory(&allocation, MEMORY_ACCESS_WRITE);
    if (!ptr)
      return nullptr;
    memcpy(ptr, initial, create_info.size);
    allocator.unmap_memory(allocation);
  }
  return handle;
}

bool Device::memory_type_is_device_optimal(uint32_t type) const {
  return (mem_props.memoryTypes[type].propertyFlags &
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
}

bool Device::memory_type_is_host_visible(uint32_t type) const {
  return (mem_props.memoryTypes[type].propertyFlags &
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
}

bool Device::format_is_supported(VkFormat format,
                                 VkFormatFeatureFlags required) const {
  VkFormatProperties props;
  vkGetPhysicalDeviceFormatProperties(gpu, format, &props);
  auto flags = props.optimalTilingFeatures;
  return (flags & required) == required;
}

VkFormat Device::get_default_depth_stencil_format() const {
  if (format_is_supported(VK_FORMAT_D24_UNORM_S8_UINT,
                          VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT))
    return VK_FORMAT_D24_UNORM_S8_UINT;
  if (format_is_supported(VK_FORMAT_D32_SFLOAT_S8_UINT,
                          VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT))
    return VK_FORMAT_D32_SFLOAT_S8_UINT;

  return VK_FORMAT_UNDEFINED;
}

VkFormat Device::get_default_depth_format() const {
  if (format_is_supported(VK_FORMAT_D32_SFLOAT,
                          VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT))
    return VK_FORMAT_D32_SFLOAT;
  if (format_is_supported(VK_FORMAT_X8_D24_UNORM_PACK32,
                          VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT))
    return VK_FORMAT_X8_D24_UNORM_PACK32;
  if (format_is_supported(VK_FORMAT_D16_UNORM,
                          VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT))
    return VK_FORMAT_D16_UNORM;

  return VK_FORMAT_UNDEFINED;
}

const RenderPass& Device::request_render_pass(const RenderPassInfo& info) {
  Hasher h;
  VkFormat formats[VULKAN_NUM_ATTACHMENTS];
  VkFormat depth_stencil;
  uint32_t lazy = 0;

  for (unsigned i = 0; i < info.num_color_attachments; i++) {
    VK_ASSERT(info.color_attachments[i]);
    formats[i] = info.color_attachments[i]->get_format();
    if (info.color_attachments[i]->get_image().get_create_info().domain ==
        ImageDomain::Transient)
      lazy |= 1u << i;

    h.u32(info.color_attachments[i]->get_image().get_swapchain_layout());
  }

  if (info.depth_stencil &&
      info.depth_stencil->get_image().get_create_info().domain ==
          ImageDomain::Transient)
    lazy |= 1u << info.num_color_attachments;

  h.u32(info.num_subpasses);
  for (unsigned i = 0; i < info.num_subpasses; i++) {
    h.u32(info.subpasses[i].num_color_attachments);
    h.u32(info.subpasses[i].num_input_attachments);
    h.u32(info.subpasses[i].num_resolve_attachments);
    h.u32(static_cast<uint32_t>(info.subpasses[i].depth_stencil_mode));
    for (unsigned j = 0; j < info.subpasses[i].num_color_attachments; j++)
      h.u32(info.subpasses[i].color_attachments[j]);
    for (unsigned j = 0; j < info.subpasses[i].num_input_attachments; j++)
      h.u32(info.subpasses[i].input_attachments[j]);
    for (unsigned j = 0; j < info.subpasses[i].num_resolve_attachments; j++)
      h.u32(info.subpasses[i].resolve_attachments[j]);
  }

  depth_stencil = info.depth_stencil ? info.depth_stencil->get_format()
                                     : VK_FORMAT_UNDEFINED;
  h.data(formats, info.num_color_attachments * sizeof(VkFormat));
  h.u32(info.num_color_attachments);
  h.u32(depth_stencil);
  h.u32(info.op_flags);
  h.u32(info.clear_attachments);
  h.u32(info.load_attachments);
  h.u32(info.store_attachments);
  h.u32(lazy);

  auto hash = h.get();
  auto itr = render_passes.find(hash);
  if (itr != end(render_passes))
    return *itr->second.get();
  else {
    RenderPass* pass = new RenderPass(this, info);
    render_passes.insert(make_pair(hash, unique_ptr<RenderPass>(pass)));
    return *pass;
  }
}

const Framebuffer& Device::request_framebuffer(const RenderPassInfo& info) {
  return framebuffer_allocator.request_framebuffer(info);
}

ImageView& Device::get_transient_attachment(unsigned width,
                                            unsigned height,
                                            VkFormat format,
                                            unsigned index,
                                            unsigned samples) {
  return transient_allocator.request_attachment(width, height, format, index,
                                                samples);
}

ImageView& Device::get_physical_attachment(unsigned width,
                                           unsigned height,
                                           VkFormat format,
                                           unsigned index,
                                           unsigned samples) {
  return physical_allocator.request_attachment(width, height, format, index,
                                               samples);
}

ImageView& Device::get_swapchain_view() {
  return frame().backbuffer->get_view();
}

RenderPassInfo Device::get_swapchain_render_pass(SwapchainRenderPass style) {
  RenderPassInfo info;
  info.num_color_attachments = 1;
  info.color_attachments[0] = &frame().backbuffer->get_view();
  info.op_flags = RENDER_PASS_OP_COLOR_OPTIMAL_BIT;
  info.clear_attachments = ~0u;
  info.store_attachments = 1u << 0;

  switch (style) {
    case SwapchainRenderPass::Depth: {
      info.op_flags |= RENDER_PASS_OP_DEPTH_STENCIL_OPTIMAL_BIT |
                       RENDER_PASS_OP_CLEAR_DEPTH_STENCIL_BIT;
      info.depth_stencil = &get_transient_attachment(
          frame().backbuffer->get_create_info().width,
          frame().backbuffer->get_create_info().height,
          get_default_depth_format());
      break;
    }

    case SwapchainRenderPass::DepthStencil: {
      info.op_flags |= RENDER_PASS_OP_DEPTH_STENCIL_OPTIMAL_BIT |
                       RENDER_PASS_OP_CLEAR_DEPTH_STENCIL_BIT;
      info.depth_stencil = &get_transient_attachment(
          frame().backbuffer->get_create_info().width,
          frame().backbuffer->get_create_info().height,
          get_default_depth_stencil_format());
      break;
    }

    default:
      break;
  }
  return info;
}

void Device::wait_for_fence(const Fence& fence) {
  auto locked_fence = fence.lock();
  if (locked_fence)
    vkWaitForFences(device, 1, &locked_fence->get_fence(), true, UINT64_MAX);
}

void Device::set_queue_lock(std::function<void()> lock_callback,
                            std::function<void()> unlock_callback) {
  queue_lock_callback = move(lock_callback);
  queue_unlock_callback = move(unlock_callback);
}
}  // namespace Vulkan
