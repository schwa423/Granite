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

#include "buffer.hpp"
#include "device.hpp"

namespace Vulkan {
Buffer::Buffer(Device* device,
               VkBuffer buffer,
               const DeviceAllocation& alloc,
               const BufferCreateInfo& info)
    : Cookie(device),
      device(device),
      buffer(buffer),
      alloc(alloc),
      info(info) {}

Buffer::~Buffer() {
  device->destroy_buffer(buffer);
  device->free_memory(alloc);
}

BufferView::BufferView(Device* device,
                       VkBufferView view,
                       const BufferViewCreateInfo& create_info)
    : Cookie(device), device(device), view(view), info(create_info) {}

BufferView::~BufferView() {
  if (view != VK_NULL_HANDLE)
    device->destroy_buffer_view(view);
}
}  // namespace Vulkan
