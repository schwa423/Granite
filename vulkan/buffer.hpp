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

#include "cookie.hpp"
#include "intrusive.hpp"
#include "memory_allocator.hpp"

namespace Vulkan {
class Device;

static inline VkPipelineStageFlags buffer_usage_to_possible_stages(
    VkBufferUsageFlags usage) {
  VkPipelineStageFlags flags = 0;
  if (usage &
      (VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT))
    flags |= VK_PIPELINE_STAGE_TRANSFER_BIT;
  if (usage &
      (VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT))
    flags |= VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
  if (usage & VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT)
    flags |= VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
  if (usage & (VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
               VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT |
               VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT))
    flags |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
             VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  if (usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
    flags |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

  return flags;
}

static inline VkAccessFlags buffer_usage_to_possible_access(
    VkBufferUsageFlags usage) {
  VkAccessFlags flags = 0;
  if (usage &
      (VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT))
    flags |= VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  if (usage & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)
    flags |= VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
  if (usage & VK_BUFFER_USAGE_INDEX_BUFFER_BIT)
    flags |= VK_ACCESS_INDEX_READ_BIT;
  if (usage & VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT)
    flags |= VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
  if (usage & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)
    flags |= VK_ACCESS_UNIFORM_READ_BIT;
  if (usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
    flags |= VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

  return flags;
}

enum class BufferDomain {
  Device,            // Device local. Probably not visible from CPU.
  LinkedDeviceHost,  // On desktop, directly mapped VRAM over PCI.
  Host,  // Host-only, needs to be synced to GPU. Might be device local as well
         // on iGPUs.
  CachedHost  // Host-only, used for readbacks.
};

struct BufferCreateInfo {
  BufferDomain domain;
  VkDeviceSize size;
  VkBufferUsageFlags usage;
};

class Buffer : public Util::IntrusivePtrEnabled<Buffer>, public Cookie {
 public:
  Buffer(Device* device,
         VkBuffer buffer,
         const DeviceAllocation& alloc,
         const BufferCreateInfo& info);
  ~Buffer();

  VkBuffer get_buffer() const { return buffer; }

  const BufferCreateInfo& get_create_info() const { return info; }

  DeviceAllocation& get_allocation() { return alloc; }

  const DeviceAllocation& get_allocation() const { return alloc; }

 private:
  Device* device;
  VkBuffer buffer;
  DeviceAllocation alloc;
  BufferCreateInfo info;
};
using BufferHandle = Util::IntrusivePtr<Buffer>;

struct BufferViewCreateInfo {
  const Buffer* buffer;
  VkFormat format;
  VkDeviceSize offset;
  VkDeviceSize range;
};

class BufferView : public Util::IntrusivePtrEnabled<BufferView>, public Cookie {
 public:
  BufferView(Device* device,
             VkBufferView view,
             const BufferViewCreateInfo& info);
  ~BufferView();

  VkBufferView get_view() const { return view; }

  const BufferViewCreateInfo& get_create_info() { return info; }

  const Buffer& get_buffer() const { return *info.buffer; }

 private:
  Device* device;
  VkBufferView view;
  BufferViewCreateInfo info;
};
using BufferViewHandle = Util::IntrusivePtr<BufferView>;
}  // namespace Vulkan
