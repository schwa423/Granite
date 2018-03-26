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

#include <memory>
#include <stdexcept>
#include "util.hpp"
#include "vulkan_symbol_wrapper.h"

#define V_S(x) #x
#define V_S_(x) V_S(x)
#define S__LINE__ V_S_(__LINE__)

#define V(x)                                                                   \
  do {                                                                         \
    VkResult err = x;                                                          \
    if (err != VK_SUCCESS && err != VK_INCOMPLETE)                             \
      throw std::runtime_error("Vulkan call failed at " __FILE__ ":" S__LINE__ \
                               ".\n");                                         \
  } while (0)

#ifdef VULKAN_DEBUG
#define VK_ASSERT(x)                                        \
  do {                                                      \
    if (!(x)) {                                             \
      LOGE("Vulkan error at %s:%d.\n", __FILE__, __LINE__); \
      std::terminate();                                     \
    }                                                       \
  } while (0)
#else
#define VK_ASSERT(x) ((void)0)
#endif

namespace Vulkan {
struct NoCopyNoMove {
  NoCopyNoMove() = default;
  NoCopyNoMove(const NoCopyNoMove&) = delete;
  void operator=(const NoCopyNoMove&) = delete;
};
}  // namespace Vulkan

namespace Vulkan {
class Context {
 public:
  Context(const char** instance_ext,
          uint32_t instance_ext_count,
          const char** device_ext,
          uint32_t device_ext_count);
  Context(VkInstance instance,
          VkPhysicalDevice gpu,
          VkDevice device,
          VkQueue queue,
          uint32_t queue_family);
  Context(VkInstance instance,
          VkPhysicalDevice gpu,
          VkSurfaceKHR surface,
          const char** required_device_extensions,
          unsigned num_required_device_extensions,
          const char** required_device_layers,
          unsigned num_required_device_layers,
          const VkPhysicalDeviceFeatures* required_features);

  Context(const Context&) = delete;
  void operator=(const Context&) = delete;
  static bool init_loader(PFN_vkGetInstanceProcAddr addr);

  ~Context();

  VkInstance get_instance() const { return instance; }

  VkPhysicalDevice get_gpu() const { return gpu; }

  VkDevice get_device() const { return device; }

  VkQueue get_graphics_queue() const { return graphics_queue; }

  VkQueue get_compute_queue() const { return compute_queue; }

  VkQueue get_transfer_queue() const { return transfer_queue; }

  const VkPhysicalDeviceProperties& get_gpu_props() const { return gpu_props; }

  const VkPhysicalDeviceMemoryProperties& get_mem_props() const {
    return mem_props;
  }

  uint32_t get_graphics_queue_family() const { return graphics_queue_family; }

  uint32_t get_compute_queue_family() const { return compute_queue_family; }

  uint32_t get_transfer_queue_family() const { return transfer_queue_family; }

  void release_instance() { owned_instance = false; }

  void release_device() { owned_device = false; }

  bool supports_external_memory_and_sync() const { return supports_external; }

  bool supports_dedicated_allocation() const { return supports_dedicated; }

  static const VkApplicationInfo& get_application_info();

 private:
  VkDevice device = VK_NULL_HANDLE;
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice gpu = VK_NULL_HANDLE;

  VkPhysicalDeviceProperties gpu_props;
  VkPhysicalDeviceMemoryProperties mem_props;

  VkQueue graphics_queue = VK_NULL_HANDLE;
  VkQueue compute_queue = VK_NULL_HANDLE;
  VkQueue transfer_queue = VK_NULL_HANDLE;
  uint32_t graphics_queue_family = VK_QUEUE_FAMILY_IGNORED;
  uint32_t compute_queue_family = VK_QUEUE_FAMILY_IGNORED;
  uint32_t transfer_queue_family = VK_QUEUE_FAMILY_IGNORED;

  bool create_instance(const char** instance_ext, uint32_t instance_ext_count);
  bool create_device(VkPhysicalDevice gpu,
                     VkSurfaceKHR surface,
                     const char** required_device_extensions,
                     unsigned num_required_device_extensions,
                     const char** required_device_layers,
                     unsigned num_required_device_layers,
                     const VkPhysicalDeviceFeatures* required_features);

  bool owned_instance = false;
  bool owned_device = false;
  bool supports_external = false;
  bool supports_dedicated = false;

#ifdef VULKAN_DEBUG
  VkDebugReportCallbackEXT debug_callback = VK_NULL_HANDLE;
#endif

  void destroy();
};
}  // namespace Vulkan
