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

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "device.hpp"
#include "stack_allocator.hpp"
#include "vulkan.hpp"
#include "vulkan_events.hpp"

namespace Granite {
class RenderGraph;
class RenderPass;

enum SizeClass { Absolute, SwapchainRelative, InputRelative };

struct AttachmentInfo {
  SizeClass size_class = SizeClass::SwapchainRelative;
  float size_x = 1.0f;
  float size_y = 1.0f;
  VkFormat format = VK_FORMAT_UNDEFINED;
  std::string size_relative_name;
  unsigned samples = 1;
  unsigned levels = 1;
  unsigned layers = 1;
  bool persistent = true;
};

struct BufferInfo {
  VkDeviceSize size = 0;
  VkBufferUsageFlags usage = 0;
  bool persistent = true;

  bool operator==(const BufferInfo& other) const {
    return size == other.size && usage == other.usage &&
           persistent == other.persistent;
  }

  bool operator!=(const BufferInfo& other) const { return !(*this == other); }
};

struct ResourceDimensions {
  VkFormat format = VK_FORMAT_UNDEFINED;
  BufferInfo buffer_info;
  unsigned width = 0;
  unsigned height = 0;
  unsigned depth = 1;
  unsigned layers = 1;
  unsigned levels = 1;
  unsigned samples = 1;
  bool transient = false;
  bool persistent = true;
  bool storage = false;
  VkPipelineStageFlags stages = 0;

  bool operator==(const ResourceDimensions& other) const {
    return format == other.format && width == other.width &&
           height == other.height && depth == other.depth &&
           layers == other.layers && levels == other.levels &&
           buffer_info == buffer_info && transient == other.transient &&
           persistent == other.persistent && storage == other.storage;
    // stages is deliberately not part of this test.
  }

  bool operator!=(const ResourceDimensions& other) const {
    return !(*this == other);
  }

  bool uses_semaphore() const {
    static const VkPipelineStageFlags concurrent =
        VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT |
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    return (stages & concurrent) == concurrent;
  }

  std::string name;
};

class RenderResource {
 public:
  enum class Type { Buffer, Texture };

  enum { Unused = ~0u };

  RenderResource(Type type, unsigned index)
      : resource_type(type), index(index) {}

  virtual ~RenderResource() = default;

  Type get_type() const { return resource_type; }

  void written_in_pass(unsigned index) { written_in_passes.insert(index); }

  void read_in_pass(unsigned index) { read_in_passes.insert(index); }

  const std::unordered_set<unsigned>& get_read_passes() const {
    return read_in_passes;
  }

  const std::unordered_set<unsigned>& get_write_passes() const {
    return written_in_passes;
  }

  unsigned get_index() const { return index; }

  void set_physical_index(unsigned index) { physical_index = index; }

  unsigned get_physical_index() const { return physical_index; }

  void set_name(const std::string& name) { this->name = name; }

  const std::string& get_name() const { return name; }

  void add_stages(VkPipelineStageFlags stages) { used_stages |= stages; }

  VkPipelineStageFlags get_used_stages() const { return used_stages; }

 private:
  Type resource_type;
  unsigned index;
  unsigned physical_index = Unused;
  std::unordered_set<unsigned> written_in_passes;
  std::unordered_set<unsigned> read_in_passes;
  std::string name;
  VkPipelineStageFlags used_stages = 0;
};

class RenderBufferResource : public RenderResource {
 public:
  RenderBufferResource(unsigned index)
      : RenderResource(RenderResource::Type::Buffer, index) {}

  void set_buffer_info(const BufferInfo& info) { this->info = info; }

  const BufferInfo& get_buffer_info() const { return info; }

 private:
  BufferInfo info;
};

class RenderTextureResource : public RenderResource {
 public:
  RenderTextureResource(unsigned index)
      : RenderResource(RenderResource::Type::Texture, index) {}

  void set_attachment_info(const AttachmentInfo& info) { this->info = info; }

  const AttachmentInfo& get_attachment_info() const { return info; }

  void set_transient_state(bool enable) { transient = enable; }

  bool get_transient_state() const { return transient; }

  void set_storage_state(bool enable) { storage = enable; }

  bool get_storage_state() const { return storage; }

 private:
  AttachmentInfo info;
  bool transient = false;
  bool storage = false;
};

class RenderPass {
 public:
  RenderPass(RenderGraph& graph, unsigned index, VkPipelineStageFlags stages)
      : graph(graph), index(index), stages(stages) {}

  enum { Unused = ~0u };

  VkPipelineStageFlags get_stages() const { return stages; }

  RenderGraph& get_graph() { return graph; }

  unsigned get_index() const { return index; }

  RenderTextureResource& set_depth_stencil_input(const std::string& name);
  RenderTextureResource& set_depth_stencil_output(const std::string& name,
                                                  const AttachmentInfo& info);
  RenderTextureResource& add_color_output(const std::string& name,
                                          const AttachmentInfo& info,
                                          const std::string& input = "");
  RenderTextureResource& add_resolve_output(const std::string& name,
                                            const AttachmentInfo& info);
  RenderTextureResource& add_texture_input(const std::string& name);
  RenderTextureResource& add_attachment_input(const std::string& name);
  RenderTextureResource& add_history_input(const std::string& name);

  RenderBufferResource& add_uniform_input(const std::string& name);
  RenderBufferResource& add_storage_output(const std::string& name,
                                           const BufferInfo& info,
                                           const std::string& input = "");
  RenderBufferResource& add_storage_read_only_input(const std::string& name);

  RenderTextureResource& add_storage_texture_output(
      const std::string& name,
      const AttachmentInfo& info,
      const std::string& input = "");

  void set_texture_inputs(Vulkan::CommandBuffer& cmd,
                          unsigned set,
                          unsigned start_binding,
                          Vulkan::StockSampler sampler);

  void make_color_input_scaled(unsigned index) {
    std::swap(color_scale_inputs[index], color_inputs[index]);
  }

  const std::vector<RenderTextureResource*>& get_color_outputs() const {
    return color_outputs;
  }

  const std::vector<RenderTextureResource*>& get_resolve_outputs() const {
    return resolve_outputs;
  }

  const std::vector<RenderTextureResource*>& get_color_inputs() const {
    return color_inputs;
  }

  const std::vector<RenderTextureResource*>& get_color_scale_inputs() const {
    return color_scale_inputs;
  }

  const std::vector<RenderTextureResource*>& get_texture_inputs() const {
    return texture_inputs;
  }

  const std::vector<RenderTextureResource*>& get_storage_texture_outputs()
      const {
    return storage_texture_outputs;
  }

  const std::vector<RenderTextureResource*>& get_storage_texture_inputs()
      const {
    return storage_texture_inputs;
  }

  const std::vector<RenderTextureResource*>& get_attachment_inputs() const {
    return attachments_inputs;
  }

  const std::vector<RenderTextureResource*>& get_history_inputs() const {
    return history_inputs;
  }

  const std::vector<RenderBufferResource*>& get_uniform_inputs() const {
    return uniform_inputs;
  }

  const std::vector<RenderBufferResource*>& get_storage_inputs() const {
    return storage_inputs;
  }

  const std::vector<RenderBufferResource*>& get_storage_read_inputs() const {
    return storage_read_inputs;
  }

  const std::vector<RenderBufferResource*>& get_storage_outputs() const {
    return storage_outputs;
  }

  RenderTextureResource* get_depth_stencil_input() const {
    return depth_stencil_input;
  }

  RenderTextureResource* get_depth_stencil_output() const {
    return depth_stencil_output;
  }

  unsigned get_physical_pass_index() const { return physical_pass; }

  void set_physical_pass_index(unsigned index) { physical_pass = index; }

  bool need_render_pass() {
    if (need_render_pass_cb)
      return need_render_pass_cb();
    else
      return true;
  }

  bool get_clear_color(unsigned index, VkClearColorValue* value = nullptr) {
    if (get_clear_color_cb)
      return get_clear_color_cb(index, value);
    else
      return false;
  }

  bool get_clear_depth_stencil(VkClearDepthStencilValue* value = nullptr) {
    if (get_clear_depth_stencil_cb)
      return get_clear_depth_stencil_cb(value);
    else
      return false;
  }

  void build_render_pass(Vulkan::CommandBuffer& cmd) {
    build_render_pass_cb(cmd);
  }

  void set_need_render_pass(std::function<bool()> func) {
    need_render_pass_cb = std::move(func);
  }

  void set_build_render_pass(std::function<void(Vulkan::CommandBuffer&)> func) {
    build_render_pass_cb = std::move(func);
  }

  void set_get_clear_depth_stencil(
      std::function<bool(VkClearDepthStencilValue*)> func) {
    get_clear_depth_stencil_cb = std::move(func);
  }

  void set_get_clear_color(
      std::function<bool(unsigned, VkClearColorValue*)> func) {
    get_clear_color_cb = std::move(func);
  }

  void set_name(const std::string& name) { this->name = name; }

  const std::string& get_name() const { return name; }

 private:
  RenderGraph& graph;
  unsigned index;
  unsigned physical_pass = Unused;
  VkPipelineStageFlags stages;

  std::function<void(Vulkan::CommandBuffer&)> build_render_pass_cb;
  std::function<bool()> need_render_pass_cb;
  std::function<bool(VkClearDepthStencilValue*)> get_clear_depth_stencil_cb;
  std::function<bool(unsigned, VkClearColorValue*)> get_clear_color_cb;

  std::vector<RenderTextureResource*> color_outputs;
  std::vector<RenderTextureResource*> resolve_outputs;
  std::vector<RenderTextureResource*> color_inputs;
  std::vector<RenderTextureResource*> color_scale_inputs;
  std::vector<RenderTextureResource*> texture_inputs;
  std::vector<RenderTextureResource*> storage_texture_inputs;
  std::vector<RenderTextureResource*> storage_texture_outputs;
  std::vector<RenderTextureResource*> attachments_inputs;
  std::vector<RenderTextureResource*> history_inputs;
  std::vector<RenderBufferResource*> uniform_inputs;
  std::vector<RenderBufferResource*> storage_outputs;
  std::vector<RenderBufferResource*> storage_read_inputs;
  std::vector<RenderBufferResource*> storage_inputs;
  RenderTextureResource* depth_stencil_input = nullptr;
  RenderTextureResource* depth_stencil_output = nullptr;
  std::string name;
};

class RenderGraph : public Vulkan::NoCopyNoMove, public EventHandler {
 public:
  RenderGraph();

  void set_device(Vulkan::Device* device) { this->device = device; }

  Vulkan::Device& get_device() {
    assert(device);
    return *device;
  }

  RenderPass& add_pass(const std::string& name, VkPipelineStageFlags stages);
  void set_backbuffer_source(const std::string& name);
  void set_backbuffer_dimensions(const ResourceDimensions& dim) {
    swapchain_dimensions = dim;
  }

  void bake();
  void reset();
  void log();
  void setup_attachments(Vulkan::Device& device, Vulkan::ImageView* swapchain);
  void enqueue_render_passes(Vulkan::Device& device);

  RenderTextureResource& get_texture_resource(const std::string& name);
  RenderBufferResource& get_buffer_resource(const std::string& name);

  Vulkan::ImageView& get_physical_texture_resource(unsigned index) {
    assert(physical_attachments[index]);
    return *physical_attachments[index];
  }

  Vulkan::ImageView* get_physical_history_texture_resource(unsigned index) {
    if (!physical_history_image_attachments[index])
      return nullptr;
    return &physical_history_image_attachments[index]->get_view();
  }

  Vulkan::Buffer& get_physical_buffer_resource(unsigned index) {
    assert(physical_buffers[index]);
    return *physical_buffers[index];
  }

  // For keeping feed-back resources alive during rebaking.
  Vulkan::BufferHandle consume_persistent_physical_buffer_resource(
      unsigned index) const;
  void install_persistent_physical_buffer_resource(unsigned index,
                                                   Vulkan::BufferHandle buffer);

  // Utility to consume all physical buffer handles and install them.
  std::vector<Vulkan::BufferHandle> consume_physical_buffers() const;
  void install_physical_buffers(std::vector<Vulkan::BufferHandle> buffers);

 private:
  Vulkan::Device* device = nullptr;
  std::vector<std::unique_ptr<RenderPass>> passes;
  std::vector<std::unique_ptr<RenderResource>> resources;
  std::unordered_map<std::string, unsigned> pass_to_index;
  std::unordered_map<std::string, unsigned> resource_to_index;
  std::string backbuffer_source;

  std::vector<unsigned> pass_stack;

  struct Barrier {
    unsigned resource_index;
    VkImageLayout layout;
    VkAccessFlags access;
    VkPipelineStageFlags stages;
    bool history;
  };

  struct Barriers {
    std::vector<Barrier> invalidate;
    std::vector<Barrier> flush;
  };

  std::vector<Barriers> pass_barriers;

  void filter_passes(std::vector<unsigned>& list);
  void validate_passes();
  void build_barriers();

  ResourceDimensions get_resource_dimensions(
      const RenderBufferResource& resource) const;
  ResourceDimensions get_resource_dimensions(
      const RenderTextureResource& resource) const;
  ResourceDimensions swapchain_dimensions;

  struct ColorClearRequest {
    RenderPass* pass;
    VkClearColorValue* target;
    unsigned index;
  };

  struct DepthClearRequest {
    RenderPass* pass;
    VkClearDepthStencilValue* target;
  };

  struct ScaledClearRequests {
    unsigned target;
    unsigned physical_resource;
  };

  struct MipmapRequests {
    unsigned physical_resource;
    VkPipelineStageFlags stages;
    VkAccessFlags access;
    VkImageLayout layout;
  };

  struct PhysicalPass {
    std::vector<unsigned> passes;
    std::vector<unsigned> discards;
    std::vector<Barrier> invalidate;
    std::vector<Barrier> flush;
    std::vector<Barrier> history;
    std::vector<std::pair<unsigned, unsigned>> alias_transfer;

    Vulkan::RenderPassInfo render_pass_info;
    std::vector<Vulkan::RenderPassInfo::Subpass> subpasses;
    std::vector<unsigned> physical_color_attachments;
    unsigned physical_depth_stencil_attachment = RenderResource::Unused;

    std::vector<ColorClearRequest> color_clear_requests;
    DepthClearRequest depth_clear_request;

    std::vector<std::vector<ScaledClearRequests>> scaled_clear_requests;
    std::vector<MipmapRequests> mipmap_requests;
  };
  std::vector<PhysicalPass> physical_passes;
  void build_physical_passes();
  void build_transients();
  void build_physical_resources();
  void build_physical_barriers();
  void build_render_pass_info();
  void build_aliases();

  std::vector<ResourceDimensions> physical_dimensions;
  std::vector<Vulkan::ImageView*> physical_attachments;
  std::vector<Vulkan::BufferHandle> physical_buffers;
  std::vector<Vulkan::ImageHandle> physical_image_attachments;
  std::vector<Vulkan::ImageHandle> physical_history_image_attachments;

  struct PipelineEvent {
    Vulkan::PipelineEvent event;
    // Need two separate semaphores so we can wait in both queues independently.
    // Waiting for a semaphore resets it.
    Vulkan::Semaphore wait_graphics_semaphore;
    Vulkan::Semaphore wait_compute_semaphore;

    // Stages to wait for are stored inside the events.
    VkAccessFlags to_flush_access = 0;

    VkAccessFlags invalidated_in_stage[32] = {};
  };

  std::vector<PipelineEvent> physical_events;
  std::vector<PipelineEvent> physical_history_events;
  std::vector<bool> physical_image_has_history;
  std::vector<unsigned> physical_aliases;

  Vulkan::ImageView* swapchain_attachment = nullptr;
  unsigned swapchain_physical_index = RenderResource::Unused;

  void enqueue_scaled_requests(
      Vulkan::CommandBuffer& cmd,
      const std::vector<ScaledClearRequests>& requests);
  void enqueue_mipmap_requests(Vulkan::CommandBuffer& cmd,
                               const std::vector<MipmapRequests>& requests);

  void on_swapchain_changed(const Vulkan::SwapchainParameterEvent& e);
  void on_swapchain_destroyed(const Vulkan::SwapchainParameterEvent& e);
  void on_device_created(const Vulkan::DeviceCreatedEvent& e);
  void on_device_destroyed(const Vulkan::DeviceCreatedEvent& e);

  void setup_physical_buffer(Vulkan::Device& device, unsigned attachment);
  void setup_physical_image(Vulkan::Device& device,
                            unsigned attachment,
                            bool storage);

  void depend_passes_recursive(const RenderPass& pass,
                               const std::unordered_set<unsigned>& passes,
                               unsigned stack_count,
                               bool no_check,
                               bool ignore_self,
                               bool merge_dependency);

  void traverse_dependencies(const RenderPass& pass, unsigned stack_count);

  std::vector<std::unordered_set<unsigned>> pass_dependencies;
  std::vector<std::unordered_set<unsigned>> pass_merge_dependencies;
  bool depends_on_pass(unsigned dst_pass, unsigned src_pass);

  void reorder_passes(std::vector<unsigned>& passes);
  static bool need_invalidate(const Barrier& barrier,
                              const PipelineEvent& event);
};
}  // namespace Granite
