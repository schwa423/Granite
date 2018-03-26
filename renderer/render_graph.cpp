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

#include "render_graph.hpp"
#include <algorithm>
#include "format.hpp"
#include "type_to_string.hpp"

using namespace std;

namespace Granite {
void RenderPass::set_texture_inputs(Vulkan::CommandBuffer& cmd,
                                    unsigned set,
                                    unsigned start_binding,
                                    Vulkan::StockSampler sampler) {
  for (auto& tex : texture_inputs_) {
    cmd.set_texture(
        set, start_binding,
        graph_.get_physical_texture_resource(tex->get_physical_index()),
        sampler);
    start_binding++;
  }
}

RenderTextureResource& RenderPass::add_attachment_input(
    const std::string& name) {
  auto& res = graph_.get_texture_resource(name);
  res.add_stages(stages_);
  res.read_in_pass(index_);
  attachments_inputs_.push_back(&res);
  return res;
}

RenderTextureResource& RenderPass::add_history_input(const std::string& name) {
  auto& res = graph_.get_texture_resource(name);
  res.add_stages(stages_);
  // History inputs are not used in any particular pass, but next frame.
  history_inputs_.push_back(&res);
  return res;
}

RenderBufferResource& RenderPass::add_uniform_input(const std::string& name) {
  auto& res = graph_.get_buffer_resource(name);
  res.add_stages(stages_);
  res.read_in_pass(index_);
  uniform_inputs_.push_back(&res);
  return res;
}

RenderBufferResource& RenderPass::add_storage_read_only_input(
    const std::string& name) {
  auto& res = graph_.get_buffer_resource(name);
  res.add_stages(stages_);
  res.read_in_pass(index_);
  storage_read_inputs_.push_back(&res);
  return res;
}

RenderBufferResource& RenderPass::add_storage_output(const std::string& name,
                                                     const BufferInfo& info,
                                                     const std::string& input) {
  auto& res = graph_.get_buffer_resource(name);
  res.add_stages(stages_);
  res.set_buffer_info(info);
  res.written_in_pass(index_);
  storage_outputs_.push_back(&res);

  if (!input.empty()) {
    auto& input_res = graph_.get_buffer_resource(input);
    input_res.read_in_pass(index_);
    storage_inputs_.push_back(&input_res);
  } else
    storage_inputs_.push_back(nullptr);

  return res;
}

RenderTextureResource& RenderPass::add_texture_input(const std::string& name) {
  auto& res = graph_.get_texture_resource(name);
  res.add_stages(stages_);
  res.read_in_pass(index_);
  texture_inputs_.push_back(&res);
  return res;
}

RenderTextureResource& RenderPass::add_resolve_output(
    const std::string& name,
    const AttachmentInfo& info) {
  auto& res = graph_.get_texture_resource(name);
  res.add_stages(stages_);
  res.written_in_pass(index_);
  res.set_attachment_info(info);
  resolve_outputs_.push_back(&res);
  return res;
}

RenderTextureResource& RenderPass::add_color_output(const std::string& name,
                                                    const AttachmentInfo& info,
                                                    const std::string& input) {
  auto& res = graph_.get_texture_resource(name);
  res.add_stages(stages_);
  res.written_in_pass(index_);
  res.set_attachment_info(info);
  color_outputs_.push_back(&res);

  if (!input.empty()) {
    auto& input_res = graph_.get_texture_resource(input);
    input_res.read_in_pass(index_);
    color_inputs_.push_back(&input_res);
    color_scale_inputs_.push_back(nullptr);
  } else {
    color_inputs_.push_back(nullptr);
    color_scale_inputs_.push_back(nullptr);
  }

  return res;
}

RenderTextureResource& RenderPass::add_storage_texture_output(
    const std::string& name,
    const AttachmentInfo& info,
    const std::string& input) {
  auto& res = graph_.get_texture_resource(name);
  res.add_stages(stages_);
  res.written_in_pass(index_);
  res.set_attachment_info(info);
  res.set_storage_state(true);
  storage_texture_outputs_.push_back(&res);

  if (!input.empty()) {
    auto& input_res = graph_.get_texture_resource(input);
    input_res.read_in_pass(index_);
    storage_texture_inputs_.push_back(&input_res);
  } else
    storage_texture_inputs_.push_back(nullptr);

  return res;
}

RenderTextureResource& RenderPass::set_depth_stencil_output(
    const std::string& name,
    const AttachmentInfo& info) {
  auto& res = graph_.get_texture_resource(name);
  res.add_stages(stages_);
  res.written_in_pass(index_);
  res.set_attachment_info(info);
  depth_stencil_output_ = &res;
  return res;
}

RenderTextureResource& RenderPass::set_depth_stencil_input(
    const std::string& name) {
  auto& res = graph_.get_texture_resource(name);
  res.add_stages(stages_);
  res.read_in_pass(index_);
  depth_stencil_input_ = &res;
  return res;
}

RenderGraph::RenderGraph() {
  EVENT_MANAGER_REGISTER_LATCH(RenderGraph, on_swapchain_changed,
                               on_swapchain_destroyed,
                               Vulkan::SwapchainParameterEvent);
  EVENT_MANAGER_REGISTER_LATCH(RenderGraph, on_device_created,
                               on_device_destroyed, Vulkan::DeviceCreatedEvent);
}

void RenderGraph::on_swapchain_destroyed(
    const Vulkan::SwapchainParameterEvent&) {
  physical_image_attachments_.clear();
  physical_history_image_attachments_.clear();
  physical_events_.clear();
  physical_history_events_.clear();
}

void RenderGraph::on_swapchain_changed(const Vulkan::SwapchainParameterEvent&) {
}

void RenderGraph::on_device_created(const Vulkan::DeviceCreatedEvent&) {}

void RenderGraph::on_device_destroyed(const Vulkan::DeviceCreatedEvent&) {
  physical_buffers_.clear();
}

RenderTextureResource& RenderGraph::get_texture_resource(
    const std::string& name) {
  auto itr = resource_to_index_.find(name);
  if (itr != end(resource_to_index_)) {
    assert(resources_[itr->second]->get_type() ==
           RenderResource::Type::Texture);
    return static_cast<RenderTextureResource&>(*resources_[itr->second]);
  } else {
    unsigned index = resources_.size();
    resources_.emplace_back(new RenderTextureResource(index));
    resources_.back()->set_name(name);
    resource_to_index_[name] = index;
    return static_cast<RenderTextureResource&>(*resources_.back());
  }
}

RenderBufferResource& RenderGraph::get_buffer_resource(
    const std::string& name) {
  auto itr = resource_to_index_.find(name);
  if (itr != end(resource_to_index_)) {
    assert(resources_[itr->second]->get_type() == RenderResource::Type::Buffer);
    return static_cast<RenderBufferResource&>(*resources_[itr->second]);
  } else {
    unsigned index = resources_.size();
    resources_.emplace_back(new RenderBufferResource(index));
    resources_.back()->set_name(name);
    resource_to_index_[name] = index;
    return static_cast<RenderBufferResource&>(*resources_.back());
  }
}

std::vector<Vulkan::BufferHandle> RenderGraph::consume_physical_buffers()
    const {
  return physical_buffers_;
}

void RenderGraph::install_physical_buffers(
    std::vector<Vulkan::BufferHandle> buffers) {
  physical_buffers_ = move(buffers);
}

Vulkan::BufferHandle RenderGraph::consume_persistent_physical_buffer_resource(
    unsigned index) const {
  if (index >= physical_buffers_.size())
    return {};
  if (!physical_buffers_[index])
    return {};

  return physical_buffers_[index];
}

void RenderGraph::install_persistent_physical_buffer_resource(
    unsigned index,
    Vulkan::BufferHandle buffer) {
  if (index >= physical_buffers_.size())
    throw logic_error("Out of range.");
  physical_buffers_[index] = buffer;
}

RenderPass& RenderGraph::add_pass(const std::string& name,
                                  VkPipelineStageFlags stages) {
  auto itr = pass_to_index_.find(name);
  if (itr != end(pass_to_index_)) {
    return *passes_[itr->second];
  } else {
    unsigned index = passes_.size();
    passes_.emplace_back(new RenderPass(*this, index, stages));
    passes_.back()->set_name(name);
    pass_to_index_[name] = index;
    return *passes_.back();
  }
}

void RenderGraph::set_backbuffer_source(const std::string& name) {
  backbuffer_source_ = name;
}

void RenderGraph::validate_passes() {
  for (auto& pass_ptr : passes_) {
    auto& pass = *pass_ptr;

    if (pass.get_color_inputs().size() != pass.get_color_outputs().size())
      throw logic_error("Size of color inputs must match color outputs.");

    if (pass.get_storage_inputs().size() != pass.get_storage_outputs().size())
      throw logic_error("Size of storage inputs must match storage outputs.");

    if (pass.get_storage_texture_inputs().size() !=
        pass.get_storage_texture_outputs().size())
      throw logic_error(
          "Size of storage texture inputs must match storage texture outputs.");

    if (!pass.get_resolve_outputs().empty() &&
        pass.get_resolve_outputs().size() != pass.get_color_outputs().size())
      throw logic_error("Must have one resolve output for each color output.");

    unsigned num_inputs = pass.get_color_inputs().size();
    for (unsigned i = 0; i < num_inputs; i++) {
      if (!pass.get_color_inputs()[i])
        continue;

      if (get_resource_dimensions(*pass.get_color_inputs()[i]) !=
          get_resource_dimensions(*pass.get_color_outputs()[i]))
        pass.make_color_input_scaled(i);
    }

    if (!pass.get_storage_outputs().empty()) {
      unsigned num_outputs = pass.get_storage_outputs().size();
      for (unsigned i = 0; i < num_outputs; i++) {
        if (!pass.get_storage_inputs()[i])
          continue;

        if (pass.get_storage_outputs()[i]->get_buffer_info() !=
            pass.get_storage_inputs()[i]->get_buffer_info())
          throw logic_error(
              "Doing RMW on a storage buffer, but usage and sizes do not "
              "match.");
      }
    }

    if (!pass.get_storage_texture_outputs().empty()) {
      unsigned num_outputs = pass.get_storage_texture_outputs().size();
      for (unsigned i = 0; i < num_outputs; i++) {
        if (!pass.get_storage_texture_inputs()[i])
          continue;

        if (get_resource_dimensions(*pass.get_storage_texture_outputs()[i]) !=
            get_resource_dimensions(*pass.get_storage_texture_inputs()[i]))
          throw logic_error(
              "Doing RMW on a storage texture image, but sizes do not match.");
      }
    }

    if (pass.get_depth_stencil_input() && pass.get_depth_stencil_output()) {
      if (get_resource_dimensions(*pass.get_depth_stencil_input()) !=
          get_resource_dimensions(*pass.get_depth_stencil_output()))
        throw logic_error("Dimension mismatch.");
    }
  }
}

void RenderGraph::build_physical_resources() {
  unsigned phys_index = 0;

  // Find resources which can alias safely.
  for (auto& pass_index : pass_stack_) {
    auto& pass = *passes_[pass_index];

    for (auto* input : pass.get_texture_inputs()) {
      if (input->get_physical_index() == RenderResource::Unused) {
        physical_dimensions_.push_back(get_resource_dimensions(*input));
        input->set_physical_index(phys_index++);
      } else
        physical_dimensions_[input->get_physical_index()].stages |=
            input->get_used_stages();
    }

    for (auto* input : pass.get_uniform_inputs()) {
      if (input->get_physical_index() == RenderResource::Unused) {
        physical_dimensions_.push_back(get_resource_dimensions(*input));
        input->set_physical_index(phys_index++);
      } else
        physical_dimensions_[input->get_physical_index()].stages |=
            input->get_used_stages();
    }

    for (auto* input : pass.get_storage_read_inputs()) {
      if (input->get_physical_index() == RenderResource::Unused) {
        physical_dimensions_.push_back(get_resource_dimensions(*input));
        input->set_physical_index(phys_index++);
      } else
        physical_dimensions_[input->get_physical_index()].stages |=
            input->get_used_stages();
    }

    for (auto* input : pass.get_color_scale_inputs()) {
      if (input && input->get_physical_index() == RenderResource::Unused) {
        physical_dimensions_.push_back(get_resource_dimensions(*input));
        input->set_physical_index(phys_index++);
      } else if (input)
        physical_dimensions_[input->get_physical_index()].stages |=
            input->get_used_stages();
    }

    if (!pass.get_color_inputs().empty()) {
      unsigned size = pass.get_color_inputs().size();
      for (unsigned i = 0; i < size; i++) {
        auto* input = pass.get_color_inputs()[i];
        if (input) {
          if (input->get_physical_index() == RenderResource::Unused) {
            physical_dimensions_.push_back(get_resource_dimensions(*input));
            input->set_physical_index(phys_index++);
          } else
            physical_dimensions_[input->get_physical_index()].stages |=
                input->get_used_stages();

          if (pass.get_color_outputs()[i]->get_physical_index() ==
              RenderResource::Unused)
            pass.get_color_outputs()[i]->set_physical_index(
                input->get_physical_index());
          else if (pass.get_color_outputs()[i]->get_physical_index() !=
                   input->get_physical_index())
            throw logic_error("Cannot alias resources. Index already claimed.");
        }
      }
    }

    if (!pass.get_storage_inputs().empty()) {
      unsigned size = pass.get_storage_inputs().size();
      for (unsigned i = 0; i < size; i++) {
        auto* input = pass.get_storage_inputs()[i];
        if (input) {
          if (input->get_physical_index() == RenderResource::Unused) {
            physical_dimensions_.push_back(get_resource_dimensions(*input));
            input->set_physical_index(phys_index++);
          } else
            physical_dimensions_[input->get_physical_index()].stages |=
                input->get_used_stages();

          if (pass.get_storage_outputs()[i]->get_physical_index() ==
              RenderResource::Unused)
            pass.get_storage_outputs()[i]->set_physical_index(
                input->get_physical_index());
          else if (pass.get_storage_outputs()[i]->get_physical_index() !=
                   input->get_physical_index())
            throw logic_error("Cannot alias resources. Index already claimed.");
        }
      }
    }

    if (!pass.get_storage_texture_inputs().empty()) {
      unsigned size = pass.get_storage_texture_inputs().size();
      for (unsigned i = 0; i < size; i++) {
        auto* input = pass.get_storage_texture_inputs()[i];
        if (input) {
          if (input->get_physical_index() == RenderResource::Unused) {
            physical_dimensions_.push_back(get_resource_dimensions(*input));
            input->set_physical_index(phys_index++);
          } else
            physical_dimensions_[input->get_physical_index()].stages |=
                input->get_used_stages();

          if (pass.get_storage_texture_outputs()[i]->get_physical_index() ==
              RenderResource::Unused)
            pass.get_storage_texture_outputs()[i]->set_physical_index(
                input->get_physical_index());
          else if (pass.get_storage_texture_outputs()[i]
                       ->get_physical_index() != input->get_physical_index())
            throw logic_error("Cannot alias resources. Index already claimed.");
        }
      }
    }

    for (auto* output : pass.get_color_outputs()) {
      if (output->get_physical_index() == RenderResource::Unused) {
        physical_dimensions_.push_back(get_resource_dimensions(*output));
        output->set_physical_index(phys_index++);
      } else
        physical_dimensions_[output->get_physical_index()].stages |=
            output->get_used_stages();
    }

    for (auto* output : pass.get_resolve_outputs()) {
      if (output->get_physical_index() == RenderResource::Unused) {
        physical_dimensions_.push_back(get_resource_dimensions(*output));
        output->set_physical_index(phys_index++);
      } else
        physical_dimensions_[output->get_physical_index()].stages |=
            output->get_used_stages();
    }

    for (auto* output : pass.get_storage_outputs()) {
      if (output->get_physical_index() == RenderResource::Unused) {
        physical_dimensions_.push_back(get_resource_dimensions(*output));
        output->set_physical_index(phys_index++);
      } else
        physical_dimensions_[output->get_physical_index()].stages |=
            output->get_used_stages();
    }

    for (auto* output : pass.get_storage_texture_outputs()) {
      if (output->get_physical_index() == RenderResource::Unused) {
        physical_dimensions_.push_back(get_resource_dimensions(*output));
        output->set_physical_index(phys_index++);
      } else
        physical_dimensions_[output->get_physical_index()].stages |=
            output->get_used_stages();
    }

    auto* ds_output = pass.get_depth_stencil_output();
    auto* ds_input = pass.get_depth_stencil_input();
    if (ds_input) {
      if (ds_input->get_physical_index() == RenderResource::Unused) {
        physical_dimensions_.push_back(get_resource_dimensions(*ds_input));
        ds_input->set_physical_index(phys_index++);
      } else
        physical_dimensions_[ds_input->get_physical_index()].stages |=
            ds_input->get_used_stages();

      if (ds_output) {
        if (ds_output->get_physical_index() == RenderResource::Unused)
          ds_output->set_physical_index(ds_input->get_physical_index());
        else if (ds_output->get_physical_index() !=
                 ds_input->get_physical_index())
          throw logic_error("Cannot alias resources. Index already claimed.");
        else
          physical_dimensions_[ds_output->get_physical_index()].stages |=
              ds_output->get_used_stages();
      }
    } else if (ds_output) {
      if (ds_output->get_physical_index() == RenderResource::Unused) {
        physical_dimensions_.push_back(get_resource_dimensions(*ds_output));
        ds_output->set_physical_index(phys_index++);
      } else
        physical_dimensions_[ds_output->get_physical_index()].stages |=
            ds_output->get_used_stages();
    }

    // Assign input attachments last so they can alias properly with existing
    // color/depth attachments in the same subpass.
    for (auto* input : pass.get_attachment_inputs()) {
      if (input->get_physical_index() == RenderResource::Unused) {
        physical_dimensions_.push_back(get_resource_dimensions(*input));
        input->set_physical_index(phys_index++);
      } else
        physical_dimensions_[input->get_physical_index()].stages |=
            input->get_used_stages();
    }
  }

  // Figure out which physical resources need to have history.
  physical_image_has_history_.clear();
  physical_image_has_history_.resize(physical_dimensions_.size());

  for (auto& pass_index : pass_stack_) {
    auto& pass = *passes_[pass_index];
    for (auto& history : pass.get_history_inputs()) {
      unsigned phys_index = history->get_physical_index();
      if (phys_index == RenderResource::Unused)
        throw logic_error(
            "History input is used, but it was never written to.");
      physical_image_has_history_[phys_index] = true;
    }
  }
}

void RenderGraph::build_transients() {
  vector<unsigned> physical_pass_used(physical_dimensions_.size());
  for (auto& u : physical_pass_used)
    u = RenderPass::Unused;

  for (auto& dim : physical_dimensions_) {
    // Buffers are never transient.
    if (dim.buffer_info.size)
      dim.transient = false;
    else
      dim.transient = true;

    unsigned index = unsigned(&dim - physical_dimensions_.data());
    if (physical_image_has_history_[index])
      dim.transient = false;
  }

  for (auto& resource : resources_) {
    if (resource->get_type() != RenderResource::Type::Texture)
      continue;

    unsigned physical_index = resource->get_physical_index();
    if (physical_index == RenderResource::Unused)
      continue;

    for (auto& pass : resource->get_write_passes()) {
      unsigned phys = passes_[pass]->get_physical_pass_index();
      if (phys != RenderPass::Unused) {
        if (physical_pass_used[physical_index] != RenderPass::Unused &&
            phys != physical_pass_used[physical_index]) {
          physical_dimensions_[physical_index].transient = false;
          break;
        }
        physical_pass_used[physical_index] = phys;
      }
    }

    for (auto& pass : resource->get_read_passes()) {
      unsigned phys = passes_[pass]->get_physical_pass_index();
      if (phys != RenderPass::Unused) {
        if (physical_pass_used[physical_index] != RenderPass::Unused &&
            phys != physical_pass_used[physical_index]) {
          physical_dimensions_[physical_index].transient = false;
          break;
        }
        physical_pass_used[physical_index] = phys;
      }
    }
  }
}

void RenderGraph::build_render_pass_info() {
  for (auto& physical_pass : physical_passes_) {
    auto& rp = physical_pass.render_pass_info;
    physical_pass.subpasses.resize(physical_pass.passes.size());
    rp.subpasses = physical_pass.subpasses.data();
    rp.num_subpasses = physical_pass.subpasses.size();
    rp.clear_attachments = 0;
    rp.load_attachments = 0;
    rp.store_attachments = ~0u;
    rp.op_flags = Vulkan::RENDER_PASS_OP_COLOR_OPTIMAL_BIT;
    physical_pass.color_clear_requests.clear();
    physical_pass.depth_clear_request = {};

    auto& colors = physical_pass.physical_color_attachments;
    colors.clear();

    const auto add_unique_color = [&](unsigned index) -> pair<unsigned, bool> {
      auto itr = find(begin(colors), end(colors), index);
      if (itr != end(colors))
        return make_pair(unsigned(itr - begin(colors)), false);
      else {
        unsigned ret = colors.size();
        colors.push_back(index);
        return make_pair(ret, true);
      }
    };

    const auto add_unique_input_attachment =
        [&](unsigned index) -> pair<unsigned, bool> {
      if (index == physical_pass.physical_depth_stencil_attachment)
        return make_pair(unsigned(colors.size()),
                         false);  // The N + 1 attachment refers to depth.
      else
        return add_unique_color(index);
    };

    for (auto& subpass : physical_pass.passes) {
      vector<ScaledClearRequests> scaled_clear_requests;

      auto& pass = *passes_[subpass];
      unsigned subpass_index = unsigned(&subpass - physical_pass.passes.data());

      // Add color attachments.
      unsigned num_color_attachments = pass.get_color_outputs().size();
      physical_pass.subpasses[subpass_index].num_color_attachments =
          num_color_attachments;
      for (unsigned i = 0; i < num_color_attachments; i++) {
        auto res =
            add_unique_color(pass.get_color_outputs()[i]->get_physical_index());
        physical_pass.subpasses[subpass_index].color_attachments[i] = res.first;

        if (res.second)  // This is the first time the color attachment is used,
                         // check if we need LOAD, or if we can clear it.
        {
          bool has_color_input =
              !pass.get_color_inputs().empty() && pass.get_color_inputs()[i];
          bool has_scaled_color_input =
              !pass.get_color_scale_inputs().empty() &&
              pass.get_color_scale_inputs()[i];

          if (!has_color_input && !has_scaled_color_input) {
            if (pass.get_clear_color(i)) {
              rp.clear_attachments |= 1u << res.first;
              physical_pass.color_clear_requests.push_back(
                  {&pass, &rp.clear_color[res.first], i});
            }
          } else {
            if (has_scaled_color_input)
              scaled_clear_requests.push_back(
                  {i, pass.get_color_scale_inputs()[i]->get_physical_index()});
            else
              rp.load_attachments |= 1u << res.first;
          }
        }
      }

      if (!pass.get_resolve_outputs().empty()) {
        physical_pass.subpasses[subpass_index].num_resolve_attachments =
            num_color_attachments;
        for (unsigned i = 0; i < num_color_attachments; i++) {
          auto res = add_unique_color(
              pass.get_resolve_outputs()[i]->get_physical_index());
          physical_pass.subpasses[subpass_index].resolve_attachments[i] =
              res.first;
          // Resolve attachments are don't care always.
        }
      }

      physical_pass.scaled_clear_requests.push_back(
          move(scaled_clear_requests));

      auto* ds_input = pass.get_depth_stencil_input();
      auto* ds_output = pass.get_depth_stencil_output();

      const auto add_unique_ds = [&](unsigned index) -> pair<unsigned, bool> {
        assert(physical_pass.physical_depth_stencil_attachment ==
                   RenderResource::Unused ||
               physical_pass.physical_depth_stencil_attachment == index);

        bool new_attachment = physical_pass.physical_depth_stencil_attachment ==
                              RenderResource::Unused;
        physical_pass.physical_depth_stencil_attachment = index;
        return make_pair(index, new_attachment);
      };

      if (ds_output && ds_input) {
        auto res = add_unique_ds(ds_output->get_physical_index());
        // If this is the first subpass the attachment is used, we need to load
        // it.
        if (res.second)
          rp.load_attachments |= 1u << res.first;

        rp.op_flags |= Vulkan::RENDER_PASS_OP_DEPTH_STENCIL_OPTIMAL_BIT |
                       Vulkan::RENDER_PASS_OP_STORE_DEPTH_STENCIL_BIT;
        physical_pass.subpasses[subpass_index].depth_stencil_mode =
            Vulkan::RenderPassInfo::DepthStencil::ReadWrite;
      } else if (ds_output) {
        auto res = add_unique_ds(ds_output->get_physical_index());
        // If this is the first subpass the attachment is used, we need to
        // either clear or discard.
        if (res.second && pass.get_clear_depth_stencil()) {
          rp.op_flags |= Vulkan::RENDER_PASS_OP_CLEAR_DEPTH_STENCIL_BIT;
          physical_pass.depth_clear_request.pass = &pass;
          physical_pass.depth_clear_request.target = &rp.clear_depth_stencil;
        }

        rp.op_flags |= Vulkan::RENDER_PASS_OP_DEPTH_STENCIL_OPTIMAL_BIT |
                       Vulkan::RENDER_PASS_OP_STORE_DEPTH_STENCIL_BIT;
        physical_pass.subpasses[subpass_index].depth_stencil_mode =
            Vulkan::RenderPassInfo::DepthStencil::ReadWrite;

        assert(physical_pass.physical_depth_stencil_attachment ==
                   RenderResource::Unused ||
               physical_pass.physical_depth_stencil_attachment ==
                   ds_output->get_physical_index());
        physical_pass.physical_depth_stencil_attachment =
            ds_output->get_physical_index();
      } else if (ds_input) {
        auto res = add_unique_ds(ds_input->get_physical_index());

        // If this is the first subpass the attachment is used, we need to load.
        if (res.second) {
          rp.op_flags |= Vulkan::RENDER_PASS_OP_DEPTH_STENCIL_READ_ONLY_BIT |
                         Vulkan::RENDER_PASS_OP_LOAD_DEPTH_STENCIL_BIT;

          bool preserve_depth = false;
          for (auto& read_pass : ds_input->get_read_passes()) {
            if (passes_[read_pass]->get_physical_pass_index() >
                unsigned(&physical_pass - physical_passes_.data())) {
              preserve_depth = true;
              break;
            }
          }

          if (preserve_depth) {
            // Have to store here, or the attachment becomes undefined in future
            // passes.
            rp.op_flags |= Vulkan::RENDER_PASS_OP_STORE_DEPTH_STENCIL_BIT;
          }
        }

        physical_pass.subpasses[subpass_index].depth_stencil_mode =
            Vulkan::RenderPassInfo::DepthStencil::ReadOnly;
      } else {
        physical_pass.subpasses[subpass_index].depth_stencil_mode =
            Vulkan::RenderPassInfo::DepthStencil::None;
      }
    }

    for (auto& subpass : physical_pass.passes) {
      auto& pass = *passes_[subpass];
      unsigned subpass_index = unsigned(&subpass - physical_pass.passes.data());

      // Add input attachments.
      // Have to do these in a separate loop so we can pick up depth stencil
      // input attachments properly.
      unsigned num_input_attachments = pass.get_attachment_inputs().size();
      physical_pass.subpasses[subpass_index].num_input_attachments =
          num_input_attachments;
      for (unsigned i = 0; i < num_input_attachments; i++) {
        auto res = add_unique_input_attachment(
            pass.get_attachment_inputs()[i]->get_physical_index());
        physical_pass.subpasses[subpass_index].input_attachments[i] = res.first;

        // If this is the first subpass the attachment is used, we need to load
        // it.
        if (res.second)
          rp.load_attachments |= 1u << res.first;
      }
    }

    physical_pass.render_pass_info.num_color_attachments =
        physical_pass.physical_color_attachments.size();
  }
}

void RenderGraph::build_physical_passes() {
  physical_passes_.clear();
  PhysicalPass physical_pass;

  const auto find_attachment =
      [](const vector<RenderTextureResource*>& resources,
         const RenderTextureResource* resource) -> bool {
    auto itr = find(begin(resources), end(resources), resource);
    return itr != end(resources);
  };

  const auto find_buffer = [](const vector<RenderBufferResource*>& resources,
                              const RenderBufferResource* resource) -> bool {
    auto itr = find(begin(resources), end(resources), resource);
    return itr != end(resources);
  };

  const auto should_merge = [&](const RenderPass& prev,
                                const RenderPass& next) -> bool {
    // Can only merge graphics.
    if (prev.get_stages() != VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT ||
        next.get_stages() != VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT)
      return false;

    for (auto* output : prev.get_color_outputs()) {
      // Need to mip-map after this pass, so cannot merge.
      if (physical_dimensions_[output->get_physical_index()].levels > 1)
        return false;
    }

    // Need non-local dependency, cannot merge.
    for (auto* input : next.get_texture_inputs()) {
      if (find_attachment(prev.get_color_outputs(), input))
        return false;
      if (find_attachment(prev.get_resolve_outputs(), input))
        return false;
      if (find_attachment(prev.get_storage_texture_outputs(), input))
        return false;
      if (input && prev.get_depth_stencil_output() == input)
        return false;
    }

    // Need non-local dependency, cannot merge.
    for (auto* input : next.get_uniform_inputs())
      if (find_buffer(prev.get_storage_outputs(), input))
        return false;

    // Need non-local dependency, cannot merge.
    for (auto* input : next.get_storage_read_inputs())
      if (find_buffer(prev.get_storage_outputs(), input))
        return false;

    // Need non-local dependency, cannot merge.
    for (auto* input : next.get_storage_inputs())
      if (find_buffer(prev.get_storage_outputs(), input))
        return false;

    // Need non-local dependency, cannot merge.
    for (auto* input : next.get_storage_texture_inputs())
      if (find_attachment(prev.get_storage_texture_outputs(), input))
        return false;

    // Need non-local dependency, cannot merge.
    for (auto* input : next.get_color_scale_inputs()) {
      if (find_attachment(prev.get_storage_texture_outputs(), input))
        return false;
      if (find_attachment(prev.get_color_outputs(), input))
        return false;
      if (find_attachment(prev.get_resolve_outputs(), input))
        return false;
    }

    // Keep color on tile.
    for (auto* input : next.get_color_inputs()) {
      if (!input)
        continue;
      if (find_attachment(prev.get_storage_texture_outputs(), input))
        return false;
      if (find_attachment(prev.get_color_outputs(), input))
        return true;
      if (find_attachment(prev.get_resolve_outputs(), input))
        return true;
    }

    const auto different_attachment = [](const RenderResource* a,
                                         const RenderResource* b) {
      return a && b && a->get_physical_index() != b->get_physical_index();
    };

    // Need a different depth attachment, break up the pass.
    if (different_attachment(next.get_depth_stencil_input(),
                             prev.get_depth_stencil_input()))
      return false;
    if (different_attachment(next.get_depth_stencil_output(),
                             prev.get_depth_stencil_input()))
      return false;
    if (different_attachment(next.get_depth_stencil_input(),
                             prev.get_depth_stencil_output()))
      return false;
    if (different_attachment(next.get_depth_stencil_output(),
                             prev.get_depth_stencil_output()))
      return false;

    // Keep depth on tile.
    if (next.get_depth_stencil_input() &&
        next.get_depth_stencil_input() == prev.get_depth_stencil_output())
      return true;

    // Keep depth attachment or color on-tile.
    for (auto* input : next.get_attachment_inputs()) {
      if (find_attachment(prev.get_color_outputs(), input))
        return true;
      if (find_attachment(prev.get_resolve_outputs(), input))
        return true;
      if (input && prev.get_depth_stencil_output() == input)
        return true;
    }

    // No reason to merge, so don't.
    return false;
  };

  for (unsigned index = 0; index < pass_stack_.size();) {
    unsigned merge_end = index + 1;
    for (; merge_end < pass_stack_.size(); merge_end++) {
      bool merge = true;
      for (unsigned merge_start = index; merge_start < merge_end;
           merge_start++) {
        if (!should_merge(*passes_[pass_stack_[merge_start]],
                          *passes_[pass_stack_[merge_end]])) {
          merge = false;
          break;
        }
      }

      if (!merge)
        break;
    }

    physical_pass.passes.insert(end(physical_pass.passes),
                                begin(pass_stack_) + index,
                                begin(pass_stack_) + merge_end);
    physical_passes_.push_back(move(physical_pass));
    index = merge_end;
  }

  for (auto& physical_pass : physical_passes_) {
    unsigned index = &physical_pass - physical_passes_.data();
    for (auto& pass : physical_pass.passes)
      passes_[pass]->set_physical_pass_index(index);
  }
}

void RenderGraph::log() {
  for (auto& resource : physical_dimensions_) {
    if (resource.buffer_info.size) {
      LOGI("Resource #%u (%s): size: %u\n",
           unsigned(&resource - physical_dimensions_.data()),
           resource.name.c_str(), unsigned(resource.buffer_info.size));
    } else {
      LOGI(
          "Resource #%u (%s): %u x %u (fmt: %u), samples: %u, transient: "
          "%s%s\n",
          unsigned(&resource - physical_dimensions_.data()),
          resource.name.c_str(), resource.width, resource.height,
          unsigned(resource.format), resource.samples,
          resource.transient ? "yes" : "no",
          unsigned(&resource - physical_dimensions_.data()) ==
                  swapchain_physical_index_
              ? " (swapchain)"
              : "");
    }
  }

  auto barrier_itr = begin(pass_barriers_);

  const auto swap_str = [this](const Barrier& barrier) -> const char* {
    return barrier.resource_index == swapchain_physical_index_ ? " (swapchain)"
                                                               : "";
  };

  for (auto& passes : physical_passes_) {
    LOGI("Physical pass #%u:\n", unsigned(&passes - physical_passes_.data()));

    for (auto& barrier : passes.invalidate) {
      LOGI("  Invalidate: %u%s, layout: %s, access: %s, stages: %s\n",
           barrier.resource_index, swap_str(barrier),
           Vulkan::layout_to_string(barrier.layout),
           Vulkan::access_flags_to_string(barrier.access).c_str(),
           Vulkan::stage_flags_to_string(barrier.stages).c_str());
    }

    for (auto& subpass : passes.passes) {
      LOGI("    Subpass #%u (%s):\n", unsigned(&subpass - passes.passes.data()),
           this->passes_[subpass]->get_name().c_str());
      auto& pass = *this->passes_[subpass];

      auto& barriers = *barrier_itr;
      for (auto& barrier : barriers.invalidate) {
        if (!physical_dimensions_[barrier.resource_index].transient) {
          LOGI("      Invalidate: %u%s, layout: %s, access: %s, stages: %s\n",
               barrier.resource_index, swap_str(barrier),
               Vulkan::layout_to_string(barrier.layout),
               Vulkan::access_flags_to_string(barrier.access).c_str(),
               Vulkan::stage_flags_to_string(barrier.stages).c_str());
        }
      }

      if (pass.get_depth_stencil_output())
        LOGI("        DepthStencil RW: %u\n",
             pass.get_depth_stencil_output()->get_physical_index());
      else if (pass.get_depth_stencil_input())
        LOGI("        DepthStencil ReadOnly: %u\n",
             pass.get_depth_stencil_input()->get_physical_index());

      for (auto& output : pass.get_color_outputs())
        LOGI("        ColorAttachment #%u: %u\n",
             unsigned(&output - pass.get_color_outputs().data()),
             output->get_physical_index());
      for (auto& output : pass.get_resolve_outputs())
        LOGI("        ResolveAttachment #%u: %u\n",
             unsigned(&output - pass.get_resolve_outputs().data()),
             output->get_physical_index());
      for (auto& input : pass.get_attachment_inputs())
        LOGI("        InputAttachment #%u: %u\n",
             unsigned(&input - pass.get_attachment_inputs().data()),
             input->get_physical_index());
      for (auto& input : pass.get_texture_inputs())
        LOGI("        Texture #%u: %u\n",
             unsigned(&input - pass.get_texture_inputs().data()),
             input->get_physical_index());

      for (auto& input : pass.get_color_scale_inputs()) {
        if (input) {
          LOGI("        ColorScaleInput #%u: %u\n",
               unsigned(&input - pass.get_color_scale_inputs().data()),
               input->get_physical_index());
        }
      }

      for (auto& barrier : barriers.flush) {
        if (!physical_dimensions_[barrier.resource_index].transient &&
            barrier.resource_index != swapchain_physical_index_) {
          LOGI("      Flush: %u, layout: %s, access: %s, stages: %s\n",
               barrier.resource_index, Vulkan::layout_to_string(barrier.layout),
               Vulkan::access_flags_to_string(barrier.access).c_str(),
               Vulkan::stage_flags_to_string(barrier.stages).c_str());
        }
      }

      ++barrier_itr;
    }

    for (auto& barrier : passes.flush) {
      LOGI("  Flush: %u%s, layout: %s, access: %s, stages: %s\n",
           barrier.resource_index, swap_str(barrier),
           Vulkan::layout_to_string(barrier.layout),
           Vulkan::access_flags_to_string(barrier.access).c_str(),
           Vulkan::stage_flags_to_string(barrier.stages).c_str());
    }
  }
}

void RenderGraph::enqueue_mipmap_requests(
    Vulkan::CommandBuffer& cmd,
    const std::vector<MipmapRequests>& requests) {
  if (requests.empty())
    return;

  for (auto& req : requests) {
    auto& image = physical_attachments_[req.physical_resource]->get_image();
    auto old_layout = image.get_layout();
    cmd.barrier_prepare_generate_mipmap(image, req.layout, req.stages,
                                        req.access);

    image.set_layout(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    cmd.generate_mipmap(image);

    // Keep the old layout so that the flush barriers can detect that a
    // transition has happened.
    image.set_layout(old_layout);
  }
}

void RenderGraph::enqueue_scaled_requests(
    Vulkan::CommandBuffer& cmd,
    const std::vector<ScaledClearRequests>& requests) {
  if (requests.empty())
    return;

  vector<pair<string, int>> defines;
  defines.reserve(requests.size());

  for (auto& req : requests) {
    defines.push_back({string("HAVE_TARGET_") + to_string(req.target), 1});
    cmd.set_texture(0, req.target,
                    *physical_attachments_[req.physical_resource],
                    Vulkan::StockSampler::LinearClamp);
  }

  Vulkan::CommandBufferUtil::draw_quad(cmd, "builtin://shaders/quad.vert",
                                       "builtin://shaders/scaled_readback.frag",
                                       defines);
}

void RenderGraph::build_aliases() {
  struct Range {
    unsigned first_write_pass = ~0u;
    unsigned last_write_pass = 0;
    unsigned first_read_pass = ~0u;
    unsigned last_read_pass = 0;

    bool has_writer() const { return first_write_pass <= last_write_pass; }

    bool has_reader() const { return first_read_pass <= last_read_pass; }

    bool is_used() const { return has_writer() || has_reader(); }

    bool can_alias() const {
      // If we read before we have completely written to a resource we need to
      // preserve it, so no alias is possible.
      if (has_reader() && has_writer() && first_read_pass <= first_write_pass)
        return false;
      return true;
    }

    unsigned last_used_pass() const {
      unsigned last_pass = 0;
      if (has_writer())
        last_pass = std::max(last_pass, last_write_pass);
      if (has_reader())
        last_pass = std::max(last_pass, last_read_pass);
      return last_pass;
    }

    unsigned first_used_pass() const {
      unsigned first_pass = ~0u;
      if (has_writer())
        first_pass = std::min(first_pass, first_write_pass);
      if (has_reader())
        first_pass = std::min(first_pass, first_read_pass);
      return first_pass;
    }

    bool disjoint_lifetime(const Range& range) const {
      if (!is_used() || !range.is_used())
        return false;
      if (!can_alias() || !range.can_alias())
        return false;

      bool left = last_used_pass() < range.first_used_pass();
      bool right = range.last_used_pass() < first_used_pass();
      return left || right;
    }
  };

  vector<Range> pass_range(physical_dimensions_.size());

  const auto register_reader = [this, &pass_range](
                                   const RenderTextureResource* resource,
                                   unsigned pass_index) {
    if (resource && pass_index != RenderPass::Unused) {
      unsigned phys = resource->get_physical_index();
      if (phys != RenderResource::Unused) {
        auto& range = pass_range[phys];
        range.last_read_pass = std::max(range.last_read_pass, pass_index);
        range.first_read_pass = std::min(range.first_read_pass, pass_index);
      }
    }
  };

  const auto register_writer = [this, &pass_range](
                                   const RenderTextureResource* resource,
                                   unsigned pass_index) {
    if (resource && pass_index != RenderPass::Unused) {
      unsigned phys = resource->get_physical_index();
      if (phys != RenderResource::Unused) {
        auto& range = pass_range[phys];
        range.last_write_pass = std::max(range.last_write_pass, pass_index);
        range.first_write_pass = std::min(range.first_write_pass, pass_index);
      }
    }
  };

  for (auto& pass : pass_stack_) {
    auto& subpass = *passes_[pass];

    for (auto* input : subpass.get_color_inputs())
      register_reader(input, subpass.get_physical_pass_index());
    for (auto* input : subpass.get_color_scale_inputs())
      register_reader(input, subpass.get_physical_pass_index());
    for (auto* input : subpass.get_attachment_inputs())
      register_reader(input, subpass.get_physical_pass_index());
    for (auto* input : subpass.get_texture_inputs())
      register_reader(input, subpass.get_physical_pass_index());
    if (subpass.get_depth_stencil_input())
      register_reader(subpass.get_depth_stencil_input(),
                      subpass.get_physical_pass_index());

    if (subpass.get_depth_stencil_output())
      register_writer(subpass.get_depth_stencil_output(),
                      subpass.get_physical_pass_index());
    for (auto* output : subpass.get_color_outputs())
      register_writer(output, subpass.get_physical_pass_index());
    for (auto* output : subpass.get_resolve_outputs())
      register_writer(output, subpass.get_physical_pass_index());
  }

  vector<vector<unsigned>> alias_chains(physical_dimensions_.size());

  physical_aliases_.resize(physical_dimensions_.size());
  for (auto& v : physical_aliases_)
    v = RenderResource::Unused;

  for (unsigned i = 0; i < physical_dimensions_.size(); i++) {
    // No aliases for buffers.
    if (physical_dimensions_[i].buffer_info.size)
      continue;

    // Only try to alias with lower-indexed resources, because we allocate them
    // one-by-one starting from index 0.
    for (unsigned j = 0; j < i; j++) {
      if (physical_image_has_history_[j])
        continue;

      if (physical_dimensions_[i] == physical_dimensions_[j]) {
        if (pass_range[i].disjoint_lifetime(pass_range[j]))  // We can alias.
        {
          physical_aliases_[i] = j;
          if (alias_chains[j].empty())
            alias_chains[j].push_back(j);
          alias_chains[j].push_back(i);

          break;
        }
      }
    }
  }

  // Now we've found the aliases, so set up the transfer barriers in order of
  // use.
  for (auto& chain : alias_chains) {
    if (chain.empty())
      continue;

    sort(begin(chain), end(chain), [&](unsigned a, unsigned b) -> bool {
      return pass_range[a].last_used_pass() < pass_range[b].first_used_pass();
    });

    for (unsigned i = 0; i < chain.size(); i++) {
      if (i + 1 < chain.size())
        physical_passes_[pass_range[chain[i]].last_used_pass()]
            .alias_transfer.push_back(make_pair(chain[i], chain[i + 1]));
      else
        physical_passes_[pass_range[chain[i]].last_used_pass()]
            .alias_transfer.push_back(make_pair(chain[i], chain[0]));
    }
  }
}

bool RenderGraph::need_invalidate(const Barrier& barrier,
                                  const PipelineEvent& event) {
  bool need_invalidate = false;
  Util::for_each_bit(barrier.stages, [&](uint32_t bit) {
    if (barrier.access & ~event.invalidated_in_stage[bit])
      need_invalidate = true;
  });
  return need_invalidate;
}

void RenderGraph::enqueue_render_passes(Vulkan::Device& device) {
  vector<VkBufferMemoryBarrier> buffer_barriers;
  vector<VkImageMemoryBarrier> image_barriers;

  // Immediate buffer barriers are useless because they don't need any layout
  // transition, and the API guarantees that submitting a batch makes memory
  // visible to GPU resources. Immediate image barriers are purely for doing
  // layout transitions without waiting (srcStage = TOP_OF_PIPE).
  vector<VkImageMemoryBarrier> immediate_image_barriers;

  // Barriers which are used when waiting for a semaphore, and then doing a
  // transition. We need to use pipeline barriers here so we can have srcStage =
  // dstStage, and hand over while not breaking the pipeline.
  vector<VkImageMemoryBarrier> semaphore_handover_barriers;

  vector<VkEvent> events;

  const auto transfer_ownership = [this](PhysicalPass& pass) {
    // Need to wait on this event before we can transfer ownership to another
    // alias.
    for (auto& transfer : pass.alias_transfer) {
      auto& events = physical_events_[transfer.second];
      events = physical_events_[transfer.first];
      for (auto& e : events.invalidated_in_stage)
        e = 0;

      // If we have pending writes, we have a problem. We cannot safely alias
      // unless we first flush caches, but we cannot flush caches from UNDEFINED
      // layout. "Write-only" resources should be transient to begin with, and
      // not hit this path. If required, we could inject a pipeline barrier here
      // which flushes caches. Generally, the last pass a resource is used, it
      // will be *read*, not written to.
      assert(events.to_flush_access == 0);

      events.to_flush_access = 0;

      physical_attachments_[transfer.second]->get_image().set_layout(
          VK_IMAGE_LAYOUT_UNDEFINED);
    }
  };

  for (auto& physical_pass : physical_passes_) {
    bool require_pass = false;
    for (auto& pass : physical_pass.passes) {
      if (passes_[pass]->need_render_pass())
        require_pass = true;
    }

    if (!require_pass) {
      transfer_ownership(physical_pass);
      continue;
    }

    bool graphics = (passes_[physical_pass.passes.front()]->get_stages() &
                     VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT) != 0;
    auto queue_type = graphics ? Vulkan::CommandBuffer::Type::Graphics
                               : Vulkan::CommandBuffer::Type::Compute;
    auto cmd = device.request_command_buffer(queue_type);

    const auto wait_for_semaphore_in_queue = [&](Vulkan::Semaphore sem,
                                                 VkPipelineStageFlags stages) {
      if (sem->get_semaphore() != VK_NULL_HANDLE)
        device.add_wait_semaphore(queue_type, sem, stages);
    };

    VkPipelineStageFlags dst_stages = 0;
    VkPipelineStageFlags immediate_dst_stages = 0;
    VkPipelineStageFlags src_stages = 0;
    VkPipelineStageFlags handover_stages = 0;
    buffer_barriers.clear();
    image_barriers.clear();
    immediate_image_barriers.clear();
    semaphore_handover_barriers.clear();

    events.clear();

    const auto add_unique_event = [&](VkEvent event) {
      assert(event != VK_NULL_HANDLE);
      auto itr = find(begin(events), end(events), event);
      if (itr == end(events))
        events.push_back(event);
    };

    // Before invalidating, force the layout to UNDEFINED.
    // This will be required for resource aliasing later.
    for (auto& discard : physical_pass.discards)
      if (!physical_dimensions_[discard].buffer_info.size)
        physical_attachments_[discard]->get_image().set_layout(
            VK_IMAGE_LAYOUT_UNDEFINED);

    // Queue up invalidates and change layouts.
    for (auto& barrier : physical_pass.invalidate) {
      auto& event = barrier.history
                        ? physical_history_events_[barrier.resource_index]
                        : physical_events_[barrier.resource_index];

      bool need_event_barrier = false;
      bool layout_change = false;
      bool need_wait_semaphore = false;
      auto& wait_semaphore = graphics ? event.wait_graphics_semaphore
                                      : event.wait_compute_semaphore;

      if (physical_dimensions_[barrier.resource_index].buffer_info.size) {
        // Buffers.
        bool need_sync =
            (event.to_flush_access != 0) || need_invalidate(barrier, event);

        if (need_sync) {
          need_event_barrier = bool(event.event);
          // Signalling and waiting for a semaphore satisfies the memory barrier
          // automatically.
          need_wait_semaphore = bool(wait_semaphore);
        }

        if (need_event_barrier) {
          auto& buffer = *physical_buffers_[barrier.resource_index];
          VkBufferMemoryBarrier b = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};

          b.srcAccessMask = event.to_flush_access;
          b.dstAccessMask = barrier.access;
          b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
          b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
          b.buffer = buffer.get_buffer();
          b.offset = 0;
          b.size = VK_WHOLE_SIZE;
          buffer_barriers.push_back(b);
        }
      } else {
        // Images.
        Vulkan::Image* image =
            barrier.history
                ? physical_history_image_attachments_[barrier.resource_index]
                      .get()
                : &physical_attachments_[barrier.resource_index]->get_image();

        if (!image) {
          // Can happen for history inputs if this is the first frame.
          continue;
        }

        VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout = image->get_layout();
        b.newLayout = barrier.layout;
        b.srcAccessMask = event.to_flush_access;
        b.dstAccessMask = barrier.access;

        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = image->get_image();
        b.subresourceRange.aspectMask =
            Vulkan::format_to_aspect_mask(image->get_format());
        b.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
        b.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
        image->set_layout(barrier.layout);

        layout_change = b.oldLayout != b.newLayout;

        bool need_sync = layout_change || (event.to_flush_access != 0) ||
                         need_invalidate(barrier, event);

        if (need_sync) {
          if (event.event) {
            // Either we wait for a VkEvent ...
            image_barriers.push_back(b);
            need_event_barrier = true;
          } else if (wait_semaphore) {
            // We wait for a semaphore ...
            if (layout_change) {
              // When the semaphore was signalled, caches were flushed, so we
              // don't need to do that again. We still need dstAccessMask
              // however, because layout changes may perform writes.
              b.srcAccessMask = 0;
              semaphore_handover_barriers.push_back(b);
              handover_stages |= barrier.stages;
            }
            // If we don't need a layout transition, signalling and waiting for
            // semaphores satisfies all requirements we have of
            // srcAccessMask/dstAccessMask.
            need_wait_semaphore = true;
          } else {
            // ... or vkCmdPipelineBarrier from TOP_OF_PIPE_BIT if this is the
            // first time we use the resource.
            immediate_image_barriers.push_back(b);
            if (b.oldLayout != VK_NULL_HANDLE)
              throw logic_error(
                  "Cannot do immediate image barriers from a layout other than "
                  "UNDEFINED.");
            immediate_dst_stages |= barrier.stages;
          }
        }
      }

      // Any pending writes or layout changes means we have to invalidate
      // caches.
      if (event.to_flush_access || layout_change) {
        for (auto& e : event.invalidated_in_stage)
          e = 0;
      }
      event.to_flush_access = 0;

      if (need_event_barrier) {
        dst_stages |= barrier.stages;

        assert(event.event);
        src_stages |= event.event->get_stages();
        add_unique_event(event.event->get_event());

        // Mark appropriate caches as invalidated now.
        Util::for_each_bit(barrier.stages, [&](uint32_t bit) {
          event.invalidated_in_stage[bit] |= barrier.access;
        });
      } else if (need_wait_semaphore) {
        assert(wait_semaphore);

        // Wait for a semaphore, unless it has already been waited for ...
        wait_for_semaphore_in_queue(wait_semaphore, barrier.stages);

        // Waiting for a semaphore makes data visible to all access bits in
        // relevant stages. The exception is if we perform a layout change ...
        // In this case we only invalidate the access bits which we placed in
        // the vkCmdPipelineBarrier.
        Util::for_each_bit(barrier.stages, [&](uint32_t bit) {
          if (layout_change)
            event.invalidated_in_stage[bit] |= barrier.access;
          else
            event.invalidated_in_stage[bit] |= ~0u;
        });
      }
    }

    // Submit barriers.
    if (!semaphore_handover_barriers.empty()) {
      cmd->barrier(handover_stages, handover_stages, 0, nullptr, 0, nullptr,
                   semaphore_handover_barriers.size(),
                   semaphore_handover_barriers.empty()
                       ? nullptr
                       : semaphore_handover_barriers.data());
    }

    if (!image_barriers.empty() || !buffer_barriers.empty()) {
      cmd->wait_events(
          events.size(), events.data(), src_stages, dst_stages, 0, nullptr,
          buffer_barriers.size(),
          buffer_barriers.empty() ? nullptr : buffer_barriers.data(),
          image_barriers.size(),
          image_barriers.empty() ? nullptr : image_barriers.data());
    }

    if (!immediate_image_barriers.empty()) {
      cmd->barrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, immediate_dst_stages, 0,
                   nullptr, 0, nullptr, immediate_image_barriers.size(),
                   immediate_image_barriers.data());
    }

    if (graphics) {
      for (auto& clear_req : physical_pass.color_clear_requests)
        clear_req.pass->get_clear_color(clear_req.index, clear_req.target);

      if (physical_pass.depth_clear_request.pass) {
        physical_pass.depth_clear_request.pass->get_clear_depth_stencil(
            physical_pass.depth_clear_request.target);
      }

      cmd->begin_render_pass(physical_pass.render_pass_info);

      for (auto& subpass : physical_pass.passes) {
        unsigned subpass_index =
            unsigned(&subpass - physical_pass.passes.data());
        auto& scaled_requests =
            physical_pass.scaled_clear_requests[subpass_index];
        enqueue_scaled_requests(*cmd, scaled_requests);

        auto& pass = *passes_[subpass];

        // If we have started the render pass, we have to do it, even if a lone
        // subpass might not be required, due to clearing and so on. This should
        // be an extremely unlikely scenario. Either you need all subpasses or
        // none.
        pass.build_render_pass(*cmd);

        if (&subpass != &physical_pass.passes.back())
          cmd->next_subpass();
      }

      cmd->end_render_pass();
      enqueue_mipmap_requests(*cmd, physical_pass.mipmap_requests);
    } else {
      assert(physical_pass.passes.size() == 1);
      auto& pass = *passes_[physical_pass.passes.front()];
      pass.build_render_pass(*cmd);
    }

    VkPipelineStageFlags wait_stages = 0;
    for (auto& barrier : physical_pass.flush)
      if (!physical_dimensions_[barrier.resource_index].uses_semaphore())
        wait_stages |= barrier.stages;

    Vulkan::PipelineEvent pipeline_event;
    if (wait_stages != 0)
      pipeline_event = cmd->signal_event(wait_stages);

    bool need_submission_semaphore = false;

    for (auto& barrier : physical_pass.flush) {
      auto& event = barrier.history
                        ? physical_history_events_[barrier.resource_index]
                        : physical_events_[barrier.resource_index];

      // A render pass might have changed the final layout.
      if (!physical_dimensions_[barrier.resource_index].buffer_info.size) {
        auto* image =
            barrier.history
                ? physical_history_image_attachments_[barrier.resource_index]
                      .get()
                : &physical_attachments_[barrier.resource_index]->get_image();

        if (!image)
          continue;

        image->set_layout(barrier.layout);
      }

      // Mark if there are pending writes from this pass.
      event.to_flush_access = barrier.access;

      if (physical_dimensions_[barrier.resource_index].uses_semaphore()) {
        need_submission_semaphore = true;
        // Actual semaphore will be set on submission.
      } else
        event.event = pipeline_event;
    }

    Vulkan::Semaphore graphics_semaphore;
    Vulkan::Semaphore compute_semaphore;
    if (need_submission_semaphore) {
      // TODO: Add support for signalling multiple semaphores in one submit?
      device.submit(cmd, nullptr, &graphics_semaphore);
      device.submit_empty(queue_type, nullptr, &compute_semaphore);
    } else
      device.submit(cmd);

    // Assign semaphores to resources which are cross-queue.
    if (need_submission_semaphore) {
      for (auto& barrier : physical_pass.flush) {
        auto& event = barrier.history
                          ? physical_history_events_[barrier.resource_index]
                          : physical_events_[barrier.resource_index];

        if (physical_dimensions_[barrier.resource_index].uses_semaphore()) {
          event.wait_graphics_semaphore = graphics_semaphore;
          event.wait_compute_semaphore = compute_semaphore;
        }
      }
    }

    // Hand over aliases to some future pass.
    transfer_ownership(physical_pass);
  }

  // Scale to swapchain.
  if (swapchain_physical_index_ == RenderResource::Unused) {
    auto cmd = device.request_command_buffer();

    unsigned index = this->resources_[resource_to_index_[backbuffer_source_]]
                         ->get_physical_index();
    auto& image = physical_attachments_[index]->get_image();

    if (physical_events_[index].event) {
      VkEvent event = physical_events_[index].event->get_event();
      VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
      barrier.image = image.get_image();
      barrier.oldLayout =
          physical_attachments_[index]->get_image().get_layout();
      barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      barrier.srcAccessMask = physical_events_[index].to_flush_access;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
      barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
      barrier.subresourceRange.aspectMask = Vulkan::format_to_aspect_mask(
          physical_attachments_[index]->get_format());

      cmd->wait_events(1, &event, physical_events_[index].event->get_stages(),
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, nullptr, 0,
                       nullptr, 1, &barrier);

      physical_attachments_[index]->get_image().set_layout(
          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    } else if (physical_events_[index].wait_graphics_semaphore) {
      if (physical_events_[index].wait_graphics_semaphore->get_semaphore() !=
          VK_NULL_HANDLE) {
        device.add_wait_semaphore(
            Vulkan::CommandBuffer::Type::Graphics,
            physical_events_[index].wait_graphics_semaphore,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
      }

      if (image.get_layout() != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        cmd->image_barrier(
            image, image.get_layout(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
        image.set_layout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
      }
    } else {
      throw logic_error("Swapchain resource was not written to.");
    }

    auto rp_info = device.get_swapchain_render_pass(
        Vulkan::SwapchainRenderPass::ColorOnly);
    rp_info.clear_attachments = 0;
    cmd->begin_render_pass(rp_info);
    enqueue_scaled_requests(*cmd, {{0, index}});
    cmd->end_render_pass();

    // Set a write-after-read barrier on this resource.
    physical_events_[index].to_flush_access = 0;
    for (auto& e : physical_events_[index].invalidated_in_stage)
      e = 0;
    physical_events_[index].invalidated_in_stage[trailing_zeroes(
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)] = VK_ACCESS_SHADER_READ_BIT;

    if (physical_dimensions_[index].uses_semaphore()) {
      Vulkan::Semaphore graphics_semaphore;
      Vulkan::Semaphore compute_semaphore;

      // TODO: Add support for signalling multiple semaphores in one submit?
      device.submit(cmd, nullptr, &graphics_semaphore);
      device.submit_empty(Vulkan::CommandBuffer::Type::Graphics, nullptr,
                          &compute_semaphore);
      physical_events_[index].wait_graphics_semaphore = graphics_semaphore;
      physical_events_[index].wait_compute_semaphore = compute_semaphore;
    } else {
      physical_events_[index].event =
          cmd->signal_event(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
      device.submit(cmd);
    }
  }
}

void RenderGraph::setup_physical_buffer(Vulkan::Device& device,
                                        unsigned attachment) {
  auto& att = physical_dimensions_[attachment];

  Vulkan::BufferCreateInfo info = {};
  info.size = att.buffer_info.size;
  info.usage = att.buffer_info.usage;
  info.domain = Vulkan::BufferDomain::Device;

  bool need_buffer = true;
  if (physical_buffers_[attachment]) {
    if (att.persistent &&
        physical_buffers_[attachment]->get_create_info().size == info.size &&
        (physical_buffers_[attachment]->get_create_info().usage & info.usage) ==
            info.usage) {
      need_buffer = false;
    }
  }

  if (need_buffer) {
    // Zero-initialize buffers. TODO: Make this configurable.
    vector<uint8_t> blank(info.size);
    physical_buffers_[attachment] = device.create_buffer(info, blank.data());
    physical_events_[attachment] = {};
  }
}

void RenderGraph::setup_physical_image(Vulkan::Device& device,
                                       unsigned attachment,
                                       bool storage) {
  auto& att = physical_dimensions_[attachment];

  if (physical_aliases_[attachment] != RenderResource::Unused) {
    physical_image_attachments_[attachment] =
        physical_image_attachments_[physical_aliases_[attachment]];
    physical_attachments_[attachment] =
        &physical_image_attachments_[attachment]->get_view();
    physical_events_[attachment] = {};
    return;
  }

  bool need_image = true;
  VkImageUsageFlags usage =
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
  VkImageCreateFlags flags = 0;

  if (storage) {
    usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
  }

  if (att.levels > 1)
    usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

  if (Vulkan::format_is_stencil(att.format) ||
      Vulkan::format_is_depth_stencil(att.format) ||
      Vulkan::format_is_depth(att.format)) {
    usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  } else {
    usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  }

  if (physical_image_attachments_[attachment]) {
    if (att.persistent &&
        physical_image_attachments_[attachment]->get_create_info().format ==
            att.format &&
        physical_image_attachments_[attachment]->get_create_info().width ==
            att.width &&
        physical_image_attachments_[attachment]->get_create_info().height ==
            att.height &&
        physical_image_attachments_[attachment]->get_create_info().samples ==
            att.samples &&
        (physical_image_attachments_[attachment]->get_create_info().usage &
         usage) == usage &&
        (physical_image_attachments_[attachment]->get_create_info().flags &
         flags) == flags) {
      need_image = false;
    }
  }

  if (need_image) {
    Vulkan::ImageCreateInfo info;
    info.format = att.format;
    info.width = att.width;
    info.height = att.height;
    info.domain = Vulkan::ImageDomain::Physical;
    info.levels = att.levels;
    info.layers = att.layers;
    info.usage = usage;
    info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    info.samples = static_cast<VkSampleCountFlagBits>(att.samples);
    info.flags = flags;

    // For resources which are accessed in both compute and graphics, we will
    // use async compute queue, so make sure the image can be accessed
    // concurrently.
    static const VkPipelineStageFlags concurrent =
        VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT |
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    info.misc = (att.stages & concurrent) == concurrent
                    ? Vulkan::IMAGE_MISC_CONCURRENT_QUEUE_BIT
                    : 0;

    physical_image_attachments_[attachment] =
        device.create_image(info, nullptr);
    physical_events_[attachment] = {};
  }

  physical_attachments_[attachment] =
      &physical_image_attachments_[attachment]->get_view();
}

void RenderGraph::setup_attachments(Vulkan::Device& device,
                                    Vulkan::ImageView* swapchain) {
  physical_attachments_.clear();
  physical_attachments_.resize(physical_dimensions_.size());

  // Try to reuse the buffers if possible.
  physical_buffers_.resize(physical_dimensions_.size());

  // Try to reuse render targets if possible.
  physical_image_attachments_.resize(physical_dimensions_.size());
  physical_history_image_attachments_.resize(physical_dimensions_.size());
  physical_events_.resize(physical_dimensions_.size());
  physical_history_events_.resize(physical_dimensions_.size());

  swapchain_attachment_ = swapchain;

  unsigned num_attachments = physical_dimensions_.size();
  for (unsigned i = 0; i < num_attachments; i++) {
    // Move over history attachments and events.
    if (physical_image_has_history_[i]) {
      swap(physical_history_image_attachments_[i],
           physical_image_attachments_[i]);
      swap(physical_history_events_[i], physical_events_[i]);
    }

    auto& att = physical_dimensions_[i];
    if (att.buffer_info.size != 0) {
      setup_physical_buffer(device, i);
    } else {
      if (att.storage)
        setup_physical_image(device, i, true);
      else if (i == swapchain_physical_index_)
        physical_attachments_[i] = swapchain;
      else if (att.transient)
        physical_attachments_[i] = &device.get_transient_attachment(
            att.width, att.height, att.format, i, att.samples);
      else
        setup_physical_image(device, i, false);
    }
  }

  // Assign concrete ImageViews to the render pass.
  for (auto& physical_pass : physical_passes_) {
    unsigned num_attachments = physical_pass.physical_color_attachments.size();
    for (unsigned i = 0; i < num_attachments; i++)
      physical_pass.render_pass_info.color_attachments[i] =
          physical_attachments_[physical_pass.physical_color_attachments[i]];

    if (physical_pass.physical_depth_stencil_attachment !=
        RenderResource::Unused)
      physical_pass.render_pass_info.depth_stencil =
          physical_attachments_[physical_pass
                                    .physical_depth_stencil_attachment];
    else
      physical_pass.render_pass_info.depth_stencil = nullptr;
  }
}

void RenderGraph::traverse_dependencies(const RenderPass& pass,
                                        unsigned stack_count) {
  // For these kinds of resources,
  // make sure that we pull in the dependency right away so we can merge render
  // passes if possible.
  if (pass.get_depth_stencil_input()) {
    depend_passes_recursive(pass,
                            pass.get_depth_stencil_input()->get_write_passes(),
                            stack_count, false, false, true);
  }

  for (auto* input : pass.get_attachment_inputs()) {
    bool self_dependency = pass.get_depth_stencil_output() == input;
    if (find(begin(pass.get_color_outputs()), end(pass.get_color_outputs()),
             input) != end(pass.get_color_outputs()))
      self_dependency = true;

    if (!self_dependency)
      depend_passes_recursive(pass, input->get_write_passes(), stack_count,
                              false, false, true);
  }

  for (auto* input : pass.get_color_inputs()) {
    if (input)
      depend_passes_recursive(pass, input->get_write_passes(), stack_count,
                              false, false, true);
  }

  for (auto* input : pass.get_texture_inputs())
    depend_passes_recursive(pass, input->get_write_passes(), stack_count, false,
                            false, false);

  for (auto* input : pass.get_storage_inputs()) {
    if (input) {
      // There might be no writers of this resource if it's used in a feedback
      // fashion.
      depend_passes_recursive(pass, input->get_write_passes(), stack_count,
                              true, false, false);
      // Deal with write-after-read hazards if a storage buffer is read in other
      // passes (feedback) before being updated.
      depend_passes_recursive(pass, input->get_read_passes(), stack_count, true,
                              true, false);
    }
  }

  for (auto* input : pass.get_storage_texture_inputs()) {
    if (input)
      depend_passes_recursive(pass, input->get_write_passes(), stack_count,
                              false, false, false);
  }

  for (auto* input : pass.get_uniform_inputs()) {
    // There might be no writers of this resource if it's used in a feedback
    // fashion.
    depend_passes_recursive(pass, input->get_write_passes(), stack_count, true,
                            false, false);
  }

  for (auto* input : pass.get_storage_read_inputs()) {
    // There might be no writers of this resource if it's used in a feedback
    // fashion.
    depend_passes_recursive(pass, input->get_write_passes(), stack_count, true,
                            false, false);
  }
}

void RenderGraph::depend_passes_recursive(
    const RenderPass& self,
    const std::unordered_set<unsigned>& passes,
    unsigned stack_count,
    bool no_check,
    bool ignore_self,
    bool merge_dependency) {
  if (!no_check && passes.empty())
    throw logic_error("No pass exists which writes to resource.");

  if (stack_count > this->passes_.size())
    throw logic_error("Cycle detected.");

  for (auto& pass : passes)
    if (pass != self.get_index())
      pass_dependencies_[self.get_index()].insert(pass);

  if (merge_dependency)
    for (auto& pass : passes)
      if (pass != self.get_index())
        pass_merge_dependencies_[self.get_index()].insert(pass);

  stack_count++;

  for (auto& pushed_pass : passes) {
    if (ignore_self && pushed_pass == self.get_index())
      continue;
    else if (pushed_pass == self.get_index())
      throw logic_error("Pass depends on itself.");

    pass_stack_.push_back(pushed_pass);
    auto& pass = *this->passes_[pushed_pass];
    traverse_dependencies(pass, stack_count);
  }
}

void RenderGraph::reorder_passes(std::vector<unsigned>& passes) {
  // If a pass depends on an earlier pass via merge dependencies,
  // copy over dependencies to the dependees to avoid cases which can break
  // subpass merging. This is a "soft" dependency. If we ignore it, it's not a
  // real problem.
  for (auto& pass_merge_deps : pass_merge_dependencies_) {
    auto pass_index =
        unsigned(&pass_merge_deps - pass_merge_dependencies_.data());
    auto& pass_deps = pass_dependencies_[pass_index];

    for (auto& merge_dep : pass_merge_deps) {
      for (auto& dependee : pass_deps) {
        // Avoid cycles.
        if (depends_on_pass(dependee, merge_dep))
          continue;

        if (merge_dep != dependee)
          pass_dependencies_[merge_dep].insert(dependee);
      }
    }
  }

  // TODO: This is very inefficient, but should work okay for a reasonable
  // amount of passes ... But, reasonable amounts are always one more than what
  // you'd think ... Clarity in the algorithm is pretty important, because these
  // things tend to be very annoying to debug.

  if (passes.size() <= 2)
    return;

  vector<unsigned> unscheduled_passes;
  unscheduled_passes.reserve(passes.size());
  swap(passes, unscheduled_passes);

  const auto schedule = [&](unsigned index) {
    // Need to preserve the order of remaining elements.
    passes.push_back(unscheduled_passes[index]);
    move(unscheduled_passes.begin() + index + 1, unscheduled_passes.end(),
         unscheduled_passes.begin() + index);
    unscheduled_passes.pop_back();
  };

  schedule(0);
  while (!unscheduled_passes.empty()) {
    // Find the next pass to schedule.
    // We can pick any pass N, if the pass does not depend on anything left in
    // unscheduled_passes. unscheduled_passes[0] is always okay as a fallback,
    // so unless we find something better, we will at least pick that.

    // Ideally, we pick a pass which does not introduce any hard barrier.
    // A "hard barrier" here is where a pass depends directly on the pass before
    // it forcing something ala vkCmdPipelineBarrier, we would like to avoid
    // this if possible.

    // Find the pass which has the optimal overlap factor which means the number
    // of passes can be scheduled in-between the depender, and the dependee.

    unsigned best_candidate = 0;
    unsigned best_overlap_factor = 0;

    for (unsigned i = 0; i < unscheduled_passes.size(); i++) {
      unsigned overlap_factor = 0;

      // Always try to merge passes if possible on tilers.
      // This might not make sense on desktop however,
      // so we can conditionally enable this path depending on our GPU.
      if (pass_merge_dependencies_[unscheduled_passes[i]].count(
              passes.back())) {
        overlap_factor = ~0u;
      } else {
        for (auto itr = passes.rbegin(); itr != passes.rend(); ++itr) {
          if (depends_on_pass(unscheduled_passes[i], *itr))
            break;
          overlap_factor++;
        }
      }

      if (overlap_factor <= best_overlap_factor)
        continue;

      bool possible_candidate = true;
      for (unsigned j = 0; j < i; j++) {
        if (depends_on_pass(unscheduled_passes[i], unscheduled_passes[j])) {
          possible_candidate = false;
          break;
        }
      }

      if (!possible_candidate)
        continue;

      best_candidate = i;
      best_overlap_factor = overlap_factor;
    }

    schedule(best_candidate);
  }
}

bool RenderGraph::depends_on_pass(unsigned dst_pass, unsigned src_pass) {
  if (dst_pass == src_pass)
    return true;

  for (auto& dep : pass_dependencies_[dst_pass]) {
    if (depends_on_pass(dep, src_pass))
      return true;
  }

  return false;
}

void RenderGraph::bake() {
  // First, validate that the graph is sane.
  validate_passes();

  auto itr = resource_to_index_.find(backbuffer_source_);
  if (itr == end(resource_to_index_))
    throw logic_error("Backbuffer source does not exist.");

  pass_stack_.clear();

  pass_dependencies_.clear();
  pass_merge_dependencies_.clear();
  pass_dependencies_.resize(passes_.size());
  pass_merge_dependencies_.resize(passes_.size());

  // Work our way back from the backbuffer, and sort out all the dependencies.
  auto& backbuffer_resource = *resources_[itr->second];

  if (backbuffer_resource.get_write_passes().empty())
    throw logic_error("No pass exists which writes to resource.");

  for (auto& pass : backbuffer_resource.get_write_passes())
    pass_stack_.push_back(pass);

  auto tmp_pass_stack = pass_stack_;
  for (auto& pushed_pass : tmp_pass_stack) {
    auto& pass = *passes_[pushed_pass];
    traverse_dependencies(pass, 0);
  }

  reverse(begin(pass_stack_), end(pass_stack_));
  filter_passes(pass_stack_);

  // Now, reorder passes to extract better pipelining.
  reorder_passes(pass_stack_);

  // Now, we have a linear list of passes to submit in-order which would obey
  // the dependencies.

  // Figure out which physical resources we need. Here we will alias resources
  // which can trivially alias via renaming. E.g. depth input -> depth output is
  // just one physical attachment, similar with color.
  build_physical_resources();

  // Next, try to merge adjacent passes together.
  build_physical_passes();

  // After merging physical passes and resources, if an image resource is only
  // used in a single physical pass, make it transient.
  build_transients();

  // Now that we are done, we can make render passes.
  build_render_pass_info();

  // For each render pass in isolation, figure out the barriers required.
  build_barriers();

  // Check if the swapchain needs to be blitted to (in case the geometry does
  // not match the backbuffer).
  swapchain_physical_index_ =
      resources_[resource_to_index_[backbuffer_source_]]->get_physical_index();
  physical_dimensions_[swapchain_physical_index_].transient = false;
  physical_dimensions_[swapchain_physical_index_].persistent =
      swapchain_dimensions_.persistent;
  if (physical_dimensions_[swapchain_physical_index_] != swapchain_dimensions_)
    swapchain_physical_index_ = RenderResource::Unused;
  else
    physical_dimensions_[swapchain_physical_index_].transient = true;

  // Based on our render graph, figure out the barriers we actually need.
  // Some barriers are implicit (transients), and some are redundant, i.e. same
  // texture read in multiple passes.
  build_physical_barriers();

  // Figure out which images can alias with each other.
  // Also build virtual "transfer" barriers. These things only copy events over
  // to other physical resources.
  build_aliases();
}

ResourceDimensions RenderGraph::get_resource_dimensions(
    const RenderBufferResource& resource) const {
  ResourceDimensions dim;
  auto& info = resource.get_buffer_info();
  dim.buffer_info = info;
  dim.persistent = info.persistent;
  dim.name = resource.get_name();
  return dim;
}

ResourceDimensions RenderGraph::get_resource_dimensions(
    const RenderTextureResource& resource) const {
  ResourceDimensions dim;
  auto& info = resource.get_attachment_info();
  dim.layers = info.layers;
  dim.samples = info.samples;
  dim.format = info.format;
  dim.transient = resource.get_transient_state();
  dim.persistent = info.persistent;
  dim.storage = resource.get_storage_state();
  dim.stages = resource.get_used_stages();
  dim.name = resource.get_name();

  switch (info.size_class) {
    case SizeClass::SwapchainRelative:
      dim.width = unsigned(info.size_x * swapchain_dimensions_.width);
      dim.height = unsigned(info.size_y * swapchain_dimensions_.height);
      break;

    case SizeClass::Absolute:
      dim.width = unsigned(info.size_x);
      dim.height = unsigned(info.size_y);
      break;

    case SizeClass::InputRelative: {
      auto itr = resource_to_index_.find(info.size_relative_name);
      if (itr == end(resource_to_index_))
        throw logic_error("Resource does not exist.");
      auto& input =
          static_cast<RenderTextureResource&>(*resources_[itr->second]);
      auto input_dim = get_resource_dimensions(input);

      dim.width = unsigned(input_dim.width * info.size_x);
      dim.height = unsigned(input_dim.height * info.size_y);
      dim.depth = input_dim.depth;
      break;
    }
  }

  if (dim.format == VK_FORMAT_UNDEFINED)
    dim.format = swapchain_dimensions_.format;

  const auto num_levels = [](unsigned width, unsigned height) -> unsigned {
    unsigned levels = 0;
    unsigned max_dim = std::max(width, height);
    while (max_dim) {
      levels++;
      max_dim >>= 1;
    }
    return levels;
  };

  dim.levels = std::min(num_levels(dim.width, dim.height),
                        info.levels == 0 ? ~0u : info.levels);
  return dim;
}

void RenderGraph::build_physical_barriers() {
  auto barrier_itr = begin(pass_barriers_);

  const auto flush_access_to_invalidate =
      [](VkAccessFlags flags) -> VkAccessFlags {
    if (flags & VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT)
      flags |= VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
    if (flags & VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)
      flags |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    if (flags & VK_ACCESS_SHADER_WRITE_BIT)
      flags |= VK_ACCESS_SHADER_READ_BIT;
    return flags;
  };

  struct ResourceState {
    VkImageLayout initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout final_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkAccessFlags invalidated_types = 0;
    VkAccessFlags flushed_types = 0;

    VkPipelineStageFlags invalidated_stages = 0;
    VkPipelineStageFlags flushed_stages = 0;
  };

  // To handle state inside a physical pass.
  vector<ResourceState> resource_state;
  resource_state.reserve(physical_dimensions_.size());

  for (auto& physical_pass : physical_passes_) {
    resource_state.clear();
    resource_state.resize(physical_dimensions_.size());

    // Go over all physical passes, and observe their use of barriers.
    // In multipass, only the first and last barriers need to be considered
    // externally. Compute never has multipass.
    unsigned subpasses = physical_pass.passes.size();
    for (unsigned i = 0; i < subpasses; i++, ++barrier_itr) {
      auto& barriers = *barrier_itr;
      auto& invalidates = barriers.invalidate;
      auto& flushes = barriers.flush;

      for (auto& invalidate : invalidates) {
        // Transients and swapchain images are handled implicitly.
        if (physical_dimensions_[invalidate.resource_index].transient ||
            invalidate.resource_index == swapchain_physical_index_) {
          continue;
        }

        if (invalidate.history) {
          auto itr = find_if(
              begin(physical_pass.invalidate), end(physical_pass.invalidate),
              [&](const Barrier& b) -> bool {
                return b.resource_index == invalidate.resource_index &&
                       b.history;
              });

          if (itr == end(physical_pass.invalidate)) {
            // Special case history barriers. They are a bit different from
            // other barriers. We just need to ensure the layout is right and
            // that we avoid write-after-read. Even if we see these barriers in
            // multiple render passes, they will not emit multiple barriers.
            physical_pass.invalidate.push_back(
                {invalidate.resource_index, invalidate.layout,
                 invalidate.access, invalidate.stages, true});
            physical_pass.flush.push_back({invalidate.resource_index,
                                           invalidate.layout, 0,
                                           invalidate.stages, true});
          }

          continue;
        }

        // Only the first use of a resource in a physical pass needs to be
        // handled externally.
        if (resource_state[invalidate.resource_index].initial_layout ==
            VK_IMAGE_LAYOUT_UNDEFINED) {
          resource_state[invalidate.resource_index].invalidated_types |=
              invalidate.access;
          resource_state[invalidate.resource_index].invalidated_stages |=
              invalidate.stages;
          resource_state[invalidate.resource_index].initial_layout =
              invalidate.layout;
        }

        // All pending flushes have been invalidated in the appropriate stages
        // already. This is relevant if the invalidate happens in subpass #1 and
        // beyond.
        resource_state[invalidate.resource_index].flushed_types = 0;
        resource_state[invalidate.resource_index].flushed_stages = 0;
      }

      for (auto& flush : flushes) {
        // Transients are handled implicitly.
        if (physical_dimensions_[flush.resource_index].transient ||
            flush.resource_index == swapchain_physical_index_) {
          continue;
        }

        // The last use of a resource in a physical pass needs to be handled
        // externally.
        resource_state[flush.resource_index].flushed_types |= flush.access;
        resource_state[flush.resource_index].flushed_stages |= flush.stages;
        resource_state[flush.resource_index].final_layout = flush.layout;

        // If we didn't have an invalidation before first flush, we must
        // invalidate first. Only first flush in a render pass needs a matching
        // invalidation.
        if (resource_state[flush.resource_index].initial_layout ==
            VK_IMAGE_LAYOUT_UNDEFINED) {
          // If we end in TRANSFER_SRC_OPTIMAL, we actually start in
          // COLOR_ATTACHMENT_OPTIMAL.
          if (flush.layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
            resource_state[flush.resource_index].initial_layout =
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            resource_state[flush.resource_index].invalidated_stages =
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            resource_state[flush.resource_index].invalidated_types =
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
          } else {
            resource_state[flush.resource_index].initial_layout = flush.layout;
            resource_state[flush.resource_index].invalidated_stages =
                flush.stages;
            resource_state[flush.resource_index].invalidated_types =
                flush_access_to_invalidate(flush.access);
          }

          // We're not reading the resource in this pass, so we might as well
          // transition from UNDEFINED to discard the resource.
          physical_pass.discards.push_back(flush.resource_index);
        }
      }
    }

    // Now that the render pass has been studied, look at each resource
    // individually and see how we need to deal with the physical render pass as
    // a whole.
    for (auto& resource : resource_state) {
      // Resource was not touched in this pass.
      if (resource.final_layout == VK_IMAGE_LAYOUT_UNDEFINED &&
          resource.initial_layout == VK_IMAGE_LAYOUT_UNDEFINED)
        continue;

      unsigned index = unsigned(&resource - resource_state.data());

      if (resource.final_layout == VK_IMAGE_LAYOUT_UNDEFINED) {
        // If there are only invalidations in this pass it is read-only, and the
        // final layout becomes the initial one. Promote the last initial layout
        // to the final layout.
        resource.final_layout = resource.initial_layout;
      }

      physical_pass.invalidate.push_back({index, resource.initial_layout,
                                          resource.invalidated_types,
                                          resource.invalidated_stages, false});

      if (resource.flushed_types) {
        // Did the pass write anything in this pass which needs to be flushed?
        physical_pass.flush.push_back({index, resource.final_layout,
                                       resource.flushed_types,
                                       resource.flushed_stages, false});
      } else if (resource.invalidated_types) {
        // Did the pass read anything in this pass which needs to be protected
        // before it can be written? Implement this as a flush with 0 access
        // bits. This is how Vulkan essentially implements a write-after-read
        // hazard. The only purpose of this flush barrier is to set the last
        // pass which the resource was used as a stage. Do not clear
        // last_invalidate_pass, because we can still keep tacking on new access
        // flags, etc.
        physical_pass.flush.push_back({index, resource.final_layout, 0,
                                       resource.invalidated_stages, false});
      }

      // If we end in TRANSFER_SRC_OPTIMAL, this is a sentinel for needing
      // mipmapping, so enqueue that up here.
      if (resource.final_layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        physical_pass.mipmap_requests.push_back(
            {index, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
      }
    }
  }
}

void RenderGraph::build_barriers() {
  pass_barriers_.clear();
  pass_barriers_.reserve(pass_stack_.size());

  const auto get_access = [&](vector<Barrier>& barriers, unsigned index,
                              bool history) -> Barrier& {
    auto itr = find_if(
        begin(barriers), end(barriers), [index, history](const Barrier& b) {
          return index == b.resource_index && history == b.history;
        });
    if (itr != end(barriers))
      return *itr;
    else {
      barriers.push_back({index, VK_IMAGE_LAYOUT_UNDEFINED, 0, 0, history});
      return barriers.back();
    }
  };

  for (auto& index : pass_stack_) {
    auto& pass = *passes_[index];
    Barriers barriers;

    const auto get_invalidate_access = [&](unsigned index,
                                           bool history) -> Barrier& {
      return get_access(barriers.invalidate, index, history);
    };

    const auto get_flush_access = [&](unsigned index) -> Barrier& {
      return get_access(barriers.flush, index, false);
    };

    for (auto* input : pass.get_uniform_inputs()) {
      auto& barrier = get_invalidate_access(input->get_physical_index(), false);
      barrier.access |= VK_ACCESS_UNIFORM_READ_BIT;
      if (pass.get_stages() == VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT)
        barrier.stages |= VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                          VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;  // TODO: Pick
                                                                // appropriate
                                                                // stage.
      else
        barrier.stages |= pass.get_stages();

      if (barrier.layout != VK_IMAGE_LAYOUT_UNDEFINED)
        throw logic_error("Layout mismatch.");
      barrier.layout = VK_IMAGE_LAYOUT_GENERAL;  // It's a buffer, but use this
                                                 // as a sentinel.
    }

    for (auto* input : pass.get_storage_read_inputs()) {
      auto& barrier = get_invalidate_access(input->get_physical_index(), false);
      barrier.access |= VK_ACCESS_SHADER_READ_BIT;
      if (pass.get_stages() == VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT)
        barrier.stages |= VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;  // TODO: Pick
                                                                  // appropriate
                                                                  // stage.
      else
        barrier.stages |= pass.get_stages();

      if (barrier.layout != VK_IMAGE_LAYOUT_UNDEFINED)
        throw logic_error("Layout mismatch.");
      barrier.layout = VK_IMAGE_LAYOUT_GENERAL;  // It's a buffer, but use this
                                                 // as a sentinel.
    }

    for (auto* input : pass.get_texture_inputs()) {
      auto& barrier = get_invalidate_access(input->get_physical_index(), false);
      barrier.access |= VK_ACCESS_SHADER_READ_BIT;

      if (pass.get_stages() == VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT)
        barrier.stages |=
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;  // TODO: VERTEX_SHADER can
                                                    // also read textures!
      else
        barrier.stages |= pass.get_stages();

      if (barrier.layout != VK_IMAGE_LAYOUT_UNDEFINED)
        throw logic_error("Layout mismatch.");
      barrier.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    for (auto* input : pass.get_history_inputs()) {
      auto& barrier = get_invalidate_access(input->get_physical_index(), true);
      barrier.access |= VK_ACCESS_SHADER_READ_BIT;

      if (pass.get_stages() == VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT)
        barrier.stages |= VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;  // TODO: Pick
                                                                  // appropriate
                                                                  // stage.
      else
        barrier.stages |= pass.get_stages();

      if (barrier.layout != VK_IMAGE_LAYOUT_UNDEFINED)
        throw logic_error("Layout mismatch.");
      barrier.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    for (auto* input : pass.get_attachment_inputs()) {
      if (pass.get_stages() != VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT)
        throw logic_error("Only graphics passes can have input attachments.");

      auto& barrier = get_invalidate_access(input->get_physical_index(), false);
      barrier.access |= VK_ACCESS_INPUT_ATTACHMENT_READ_BIT;
      barrier.stages |= VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
      if (barrier.layout != VK_IMAGE_LAYOUT_UNDEFINED)
        throw logic_error("Layout mismatch.");
      barrier.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    for (auto* input : pass.get_storage_inputs()) {
      if (!input)
        continue;

      auto& barrier = get_invalidate_access(input->get_physical_index(), false);
      barrier.access |= VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      if (pass.get_stages() == VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT)
        barrier.stages |= VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;  // TODO: Pick
                                                                  // appropriate
                                                                  // stage.
      else
        barrier.stages |= pass.get_stages();

      if (barrier.layout != VK_IMAGE_LAYOUT_UNDEFINED)
        throw logic_error("Layout mismatch.");
      barrier.layout = VK_IMAGE_LAYOUT_GENERAL;
    }

    for (auto* input : pass.get_storage_texture_inputs()) {
      if (!input)
        continue;

      auto& barrier = get_invalidate_access(input->get_physical_index(), false);
      barrier.access |= VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      if (pass.get_stages() == VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT)
        barrier.stages |= VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;  // TODO: Pick
                                                                  // appropriate
                                                                  // stage.
      else
        barrier.stages |= pass.get_stages();

      if (barrier.layout != VK_IMAGE_LAYOUT_UNDEFINED)
        throw logic_error("Layout mismatch.");
      barrier.layout = VK_IMAGE_LAYOUT_GENERAL;
    }

    for (auto* input : pass.get_color_inputs()) {
      if (!input)
        continue;

      if (pass.get_stages() != VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT)
        throw logic_error("Only graphics passes can have color inputs.");

      auto& barrier = get_invalidate_access(input->get_physical_index(), false);
      barrier.access |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                        VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
      barrier.stages |= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

      // If the attachment is also bound as an input attachment (programmable
      // blending) we need VK_IMAGE_LAYOUT_GENERAL.
      if (barrier.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        barrier.layout = VK_IMAGE_LAYOUT_GENERAL;
      else if (barrier.layout != VK_IMAGE_LAYOUT_UNDEFINED)
        throw logic_error("Layout mismatch.");
      else
        barrier.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }

    for (auto* input : pass.get_color_scale_inputs()) {
      if (!input)
        continue;

      if (pass.get_stages() != VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT)
        throw logic_error("Only graphics passes can have scaled color inputs.");

      auto& barrier = get_invalidate_access(input->get_physical_index(), false);
      barrier.access |= VK_ACCESS_SHADER_READ_BIT;
      barrier.stages |= VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
      if (barrier.layout != VK_IMAGE_LAYOUT_UNDEFINED)
        throw logic_error("Layout mismatch.");
      barrier.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    for (auto* output : pass.get_color_outputs()) {
      if (pass.get_stages() != VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT)
        throw logic_error(
            "Only graphics passes can have scaled color outputs.");

      auto& barrier = get_flush_access(output->get_physical_index());

      if (physical_dimensions_[output->get_physical_index()].levels > 1) {
        // access should be 0 here. generate_mipmaps will take care of
        // invalidation needed.
        barrier.access |= VK_ACCESS_TRANSFER_READ_BIT;  // Validation layers
                                                        // complain without
                                                        // this.
        barrier.stages |= VK_PIPELINE_STAGE_TRANSFER_BIT;
        if (barrier.layout != VK_IMAGE_LAYOUT_UNDEFINED)
          throw logic_error("Layout mismatch.");
        barrier.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      } else {
        barrier.access |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.stages |= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

        // If the attachment is also bound as an input attachment (programmable
        // blending) we need VK_IMAGE_LAYOUT_GENERAL.
        if (barrier.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL ||
            barrier.layout == VK_IMAGE_LAYOUT_GENERAL) {
          barrier.layout = VK_IMAGE_LAYOUT_GENERAL;
        } else if (barrier.layout != VK_IMAGE_LAYOUT_UNDEFINED)
          throw logic_error("Layout mismatch.");
        else
          barrier.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      }
    }

    for (auto* output : pass.get_resolve_outputs()) {
      if (pass.get_stages() != VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT)
        throw logic_error("Only graphics passes can resolve outputs.");

      auto& barrier = get_flush_access(output->get_physical_index());
      barrier.access |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      barrier.stages |= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      if (barrier.layout != VK_IMAGE_LAYOUT_UNDEFINED)
        throw logic_error("Layout mismatch.");
      barrier.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }

    for (auto* output : pass.get_storage_outputs()) {
      auto& barrier = get_flush_access(output->get_physical_index());
      barrier.access |= VK_ACCESS_SHADER_WRITE_BIT;

      if (pass.get_stages() == VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT)
        barrier.stages |= VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;  // TODO: Pick
                                                                  // appropriate
                                                                  // stage.
      else
        barrier.stages |= pass.get_stages();

      if (barrier.layout != VK_IMAGE_LAYOUT_UNDEFINED)
        throw logic_error("Layout mismatch.");
      barrier.layout = VK_IMAGE_LAYOUT_GENERAL;
    }

    for (auto* output : pass.get_storage_texture_outputs()) {
      auto& barrier = get_flush_access(output->get_physical_index());
      barrier.access |= VK_ACCESS_SHADER_WRITE_BIT;

      if (pass.get_stages() == VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT)
        barrier.stages |= VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;  // TODO: Pick
                                                                  // appropriate
                                                                  // stage.
      else
        barrier.stages |= pass.get_stages();

      if (barrier.layout != VK_IMAGE_LAYOUT_UNDEFINED)
        throw logic_error("Layout mismatch.");
      barrier.layout = VK_IMAGE_LAYOUT_GENERAL;
    }

    auto* output = pass.get_depth_stencil_output();
    auto* input = pass.get_depth_stencil_input();

    if ((output || input) &&
        pass.get_stages() != VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT)
      throw logic_error("Only graphics passes can have depth attachments.");

    if (output && input) {
      auto& dst_barrier =
          get_invalidate_access(input->get_physical_index(), false);
      auto& src_barrier = get_flush_access(output->get_physical_index());

      if (dst_barrier.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        dst_barrier.layout = VK_IMAGE_LAYOUT_GENERAL;
      else if (dst_barrier.layout != VK_IMAGE_LAYOUT_UNDEFINED)
        throw logic_error("Layout mismatch.");
      else
        dst_barrier.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

      dst_barrier.access |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
      dst_barrier.stages |= VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;

      src_barrier.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      src_barrier.access |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
      src_barrier.stages |= VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    } else if (input) {
      auto& dst_barrier =
          get_invalidate_access(input->get_physical_index(), false);

      if (dst_barrier.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        dst_barrier.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
      else if (dst_barrier.layout != VK_IMAGE_LAYOUT_UNDEFINED)
        throw logic_error("Layout mismatch.");
      else
        dst_barrier.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

      dst_barrier.access |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
      dst_barrier.stages |= VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    } else if (output) {
      auto& src_barrier = get_flush_access(output->get_physical_index());

      if (src_barrier.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        src_barrier.layout = VK_IMAGE_LAYOUT_GENERAL;
      else if (src_barrier.layout != VK_IMAGE_LAYOUT_UNDEFINED)
        throw logic_error("Layout mismatch.");
      else
        src_barrier.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

      src_barrier.access |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
      src_barrier.stages |= VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    }

    pass_barriers_.push_back(move(barriers));
  }
}

void RenderGraph::filter_passes(std::vector<unsigned>& list) {
  unordered_set<unsigned> seen;

  auto output_itr = begin(list);
  for (auto itr = begin(list); itr != end(list); ++itr) {
    if (!seen.count(*itr)) {
      *output_itr = *itr;
      seen.insert(*itr);
      ++output_itr;
    }
  }
  list.erase(output_itr, end(list));
}

void RenderGraph::reset() {
  passes_.clear();
  resources_.clear();
  pass_to_index_.clear();
  resource_to_index_.clear();
  physical_passes_.clear();
  physical_dimensions_.clear();
  physical_attachments_.clear();
  physical_buffers_.clear();
  physical_image_attachments_.clear();
  physical_events_.clear();
  physical_history_events_.clear();
  physical_history_image_attachments_.clear();
}

}  // namespace Granite
