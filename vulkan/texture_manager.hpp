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

#include "image.hpp"
#include "volatile_source.hpp"

namespace Vulkan {
class Texture : public Util::VolatileSource<Texture> {
 public:
  friend class Util::VolatileSource<Texture>;

  Texture(Device* device,
          const std::string& path,
          VkFormat format = VK_FORMAT_UNDEFINED);
  Texture(Device* device);
  void set_path(const std::string& path);

  ImageHandle get_image() {
    VK_ASSERT(handle);
    return handle;
  }

  void replace_image(ImageHandle handle);

 private:
  Device* device;
  ImageHandle handle;
  VkFormat format;
  void update_stb(const void* data, size_t size);
  void update_hdr(const void* data, size_t size);
  void update_gli(const void* data, size_t size);

  void load();
  void unload();
  void update(const void* data, size_t size);
};

class TextureManager {
 public:
  TextureManager(Device* device);
  Texture* request_texture(const std::string& path,
                           VkFormat format = VK_FORMAT_UNDEFINED);
  Texture* register_deferred_texture(const std::string& path);

  void register_texture_update_notification(const std::string& modified_path,
                                            std::function<void(Texture&)> func);

  void notify_updated_texture(const std::string& path,
                              Vulkan::Texture& texture);

 private:
  Device* device;
  std::unordered_map<std::string, std::unique_ptr<Texture>> textures;
  std::unordered_map<std::string, std::vector<std::function<void(Texture&)>>>
      notifications;
};
}  // namespace Vulkan
