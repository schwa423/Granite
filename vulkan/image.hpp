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

#include <algorithm>
#include "cookie.hpp"
#include "format.hpp"
#include "intrusive.hpp"
#include "memory_allocator.hpp"
#include "vulkan.hpp"

namespace Vulkan {
class Device;

static inline VkPipelineStageFlags image_usage_to_possible_stages(
    VkImageUsageFlags usage) {
  VkPipelineStageFlags flags = 0;

  if (usage &
      (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT))
    flags |= VK_PIPELINE_STAGE_TRANSFER_BIT;
  if (usage & VK_IMAGE_USAGE_SAMPLED_BIT)
    flags |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
             VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  if (usage & VK_IMAGE_USAGE_STORAGE_BIT)
    flags |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
    flags |= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
    flags |= VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
             VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
  if (usage & VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT)
    flags |= VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

  if (usage & VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT) {
    VkPipelineStageFlags possible =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;

    if (usage & VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT)
      possible |= VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

    flags &= possible;
  }

  return flags;
}

static inline VkAccessFlags image_layout_to_possible_access(
    VkImageLayout layout) {
  switch (layout) {
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
      return VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INPUT_ATTACHMENT_READ_BIT;
    case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
      return VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
             VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
      return VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
      return VK_ACCESS_INPUT_ATTACHMENT_READ_BIT |
             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
      return VK_ACCESS_TRANSFER_READ_BIT;
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
      return VK_ACCESS_TRANSFER_WRITE_BIT;
    default:
      return ~0u;
  }
}

static inline VkAccessFlags image_usage_to_possible_access(
    VkImageUsageFlags usage) {
  VkAccessFlags flags = 0;

  if (usage &
      (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT))
    flags |= VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  if (usage & VK_IMAGE_USAGE_SAMPLED_BIT)
    flags |= VK_ACCESS_SHADER_READ_BIT;
  if (usage & VK_IMAGE_USAGE_STORAGE_BIT)
    flags |= VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
  if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
    flags |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
             VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
  if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
    flags |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
  if (usage & VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT)
    flags |= VK_ACCESS_INPUT_ATTACHMENT_READ_BIT;

  // Transient attachments can only be attachments, and never other resources.
  if (usage & VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT) {
    flags &= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
             VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
             VK_ACCESS_INPUT_ATTACHMENT_READ_BIT;
  }

  return flags;
}

static inline uint32_t image_num_miplevels(const VkExtent3D& extent) {
  uint32_t size = std::max(std::max(extent.width, extent.height), extent.depth);
  uint32_t levels = 0;
  while (size) {
    levels++;
    size >>= 1;
  }
  return levels;
}

static inline VkFormatFeatureFlags image_usage_to_features(
    VkImageUsageFlags usage) {
  VkFormatFeatureFlags flags = 0;
  if (usage & VK_IMAGE_USAGE_SAMPLED_BIT)
    flags |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
  if (usage & VK_IMAGE_USAGE_STORAGE_BIT)
    flags |= VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
  if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
    flags |= VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;
  if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
    flags |= VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;

  return flags;
}

struct ImageInitialData {
  const void* data;
  unsigned row_length;
  unsigned array_height;
};

enum ImageMiscFlagBits {
  IMAGE_MISC_GENERATE_MIPS_BIT = 1 << 0,
  IMAGE_MISC_FORCE_ARRAY_BIT = 1 << 1,
  IMAGE_MISC_CONCURRENT_QUEUE_BIT = 1 << 2
};
using ImageMiscFlags = uint32_t;

enum ImageViewMiscFlagBits { IMAGE_VIEW_MISC_FORCE_ARRAY_BIT = 1 << 0 };
using ImageViewMiscFlags = uint32_t;

class Image;
struct ImageViewCreateInfo {
  Image* image = nullptr;
  VkFormat format = VK_FORMAT_UNDEFINED;
  unsigned base_level = 0;
  unsigned levels = VK_REMAINING_MIP_LEVELS;
  unsigned base_layer = 0;
  unsigned layers = VK_REMAINING_ARRAY_LAYERS;
  ImageViewMiscFlags misc = 0;
  VkComponentMapping swizzle = {
      VK_COMPONENT_SWIZZLE_R,
      VK_COMPONENT_SWIZZLE_G,
      VK_COMPONENT_SWIZZLE_B,
      VK_COMPONENT_SWIZZLE_A,
  };
};

class ImageView : public Util::IntrusivePtrEnabled<ImageView>, public Cookie {
 public:
  ImageView(Device* device, VkImageView view, const ImageViewCreateInfo& info);
  ~ImageView();

  void set_alt_views(VkImageView depth, VkImageView stencil) {
    VK_ASSERT(depth_view == VK_NULL_HANDLE);
    VK_ASSERT(stencil_view == VK_NULL_HANDLE);
    depth_view = depth;
    stencil_view = stencil;
  }

  void set_base_level_view(VkImageView view) { base_level_view = view; }

  // By default, gets a combined view which includes all aspects in the image.
  // This would be used mostly for render targets.
  VkImageView get_view() const { return view; }

  VkImageView get_base_level_view() const {
    return base_level_view != VK_NULL_HANDLE ? base_level_view : view;
  }

  // Gets an image view which only includes floating point domains.
  // Takes effect when we want to sample from an image which is Depth/Stencil,
  // but we only want to sample depth.
  VkImageView get_float_view() const {
    return depth_view != VK_NULL_HANDLE ? depth_view : view;
  }

  // Gets an image view which only includes integer domains.
  // Takes effect when we want to sample from an image which is Depth/Stencil,
  // but we only want to sample stencil.
  VkImageView get_integer_view() const {
    return stencil_view != VK_NULL_HANDLE ? stencil_view : view;
  }

  VkFormat get_format() const { return info.format; }

  const Image& get_image() const { return *info.image; }

  Image& get_image() { return *info.image; }

  const ImageViewCreateInfo& get_create_info() const { return info; }

 private:
  Device* device;
  VkImageView view;
  VkImageView base_level_view = VK_NULL_HANDLE;
  VkImageView depth_view = VK_NULL_HANDLE;
  VkImageView stencil_view = VK_NULL_HANDLE;
  ImageViewCreateInfo info;
};
using ImageViewHandle = Util::IntrusivePtr<ImageView>;

enum class ImageDomain { Physical, Transient };

struct ImageCreateInfo {
  ImageDomain domain = ImageDomain::Physical;
  unsigned width = 0;
  unsigned height = 0;
  unsigned depth = 1;
  unsigned levels = 1;
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkImageType type = VK_IMAGE_TYPE_2D;
  unsigned layers = 1;
  VkImageUsageFlags usage = 0;
  VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
  VkImageCreateFlags flags = 0;
  ImageMiscFlags misc = 0;
  VkImageLayout initial_layout = VK_IMAGE_LAYOUT_GENERAL;

  static ImageCreateInfo immutable_2d_image(unsigned width,
                                            unsigned height,
                                            VkFormat format,
                                            bool mipmapped = false) {
    ImageCreateInfo info;
    info.width = width;
    info.height = height;
    info.depth = 1;
    info.levels = mipmapped ? 0u : 1u;
    info.format = format;
    info.type = VK_IMAGE_TYPE_2D;
    info.layers = 1;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.flags = 0;
    info.misc = mipmapped ? unsigned(IMAGE_MISC_GENERATE_MIPS_BIT) : 0u;
    info.initial_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    return info;
  }

  static ImageCreateInfo render_target(unsigned width,
                                       unsigned height,
                                       VkFormat format) {
    ImageCreateInfo info;
    info.width = width;
    info.height = height;
    info.depth = 1;
    info.levels = 1;
    info.format = format;
    info.type = VK_IMAGE_TYPE_2D;
    info.layers = 1;
    info.usage = (format_is_depth_stencil(format)
                      ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                      : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) |
                 VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                 VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.flags = 0;
    info.misc = 0;
    info.initial_layout = VK_IMAGE_LAYOUT_GENERAL;
    return info;
  }

  static ImageCreateInfo transient_render_target(unsigned width,
                                                 unsigned height,
                                                 VkFormat format) {
    ImageCreateInfo info;
    info.domain = ImageDomain::Transient;
    info.width = width;
    info.height = height;
    info.depth = 1;
    info.levels = 1;
    info.format = format;
    info.type = VK_IMAGE_TYPE_2D;
    info.layers = 1;
    info.usage = (format_is_depth_stencil(format)
                      ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                      : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) |
                 VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.flags = 0;
    info.misc = 0;
    info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    return info;
  }
};

class Image : public Util::IntrusivePtrEnabled<Image>, public Cookie {
 public:
  Image(Device* device,
        VkImage image,
        VkImageView default_view,
        const DeviceAllocation& alloc,
        const ImageCreateInfo& info);
  ~Image();
  Image(Image&&) = delete;
  Image& operator=(Image&&) = delete;

  const ImageView& get_view() const {
    VK_ASSERT(view);
    return *view;
  }

  ImageView& get_view() {
    VK_ASSERT(view);
    return *view;
  }

  VkImage get_image() const { return image; }

  VkFormat get_format() const { return create_info.format; }

  uint32_t get_width(uint32_t lod = 0) const {
    return std::max(1u, create_info.width >> lod);
  }

  uint32_t get_height(uint32_t lod = 0) const {
    return std::max(1u, create_info.height >> lod);
  }

  uint32_t get_depth(uint32_t lod = 0) const {
    return std::max(1u, create_info.depth >> lod);
  }

  const ImageCreateInfo& get_create_info() const { return create_info; }

  VkImageLayout get_layout() const { return layout; }

  void set_layout(VkImageLayout new_layout) { layout = new_layout; }

  bool is_swapchain_image() const {
    return swapchain_layout != VK_IMAGE_LAYOUT_UNDEFINED;
  }

  VkImageLayout get_swapchain_layout() const { return swapchain_layout; }

  void set_swapchain_layout(VkImageLayout layout) { swapchain_layout = layout; }

  void set_stage_flags(VkPipelineStageFlags flags) { stage_flags = flags; }

  void set_access_flags(VkAccessFlags flags) { access_flags = flags; }

  VkPipelineStageFlags get_stage_flags() const { return stage_flags; }

  VkAccessFlags get_access_flags() const { return access_flags; }

  const DeviceAllocation& get_allocation() const { return alloc; }

 private:
  Device* device;
  VkImage image;
  ImageViewHandle view;
  DeviceAllocation alloc;
  ImageCreateInfo create_info;

  VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;
  VkPipelineStageFlags stage_flags = 0;
  VkAccessFlags access_flags = 0;
  VkImageLayout swapchain_layout = VK_IMAGE_LAYOUT_UNDEFINED;
};

using ImageHandle = Util::IntrusivePtr<Image>;
}  // namespace Vulkan
