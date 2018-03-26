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

#include "mesh_util.hpp"
#include <string.h>
#include "application_events.hpp"
#include "device.hpp"
#include "material_manager.hpp"
#include "material_util.hpp"
#include "render_context.hpp"
#include "render_graph.hpp"
#include "renderer.hpp"
#include "shader_suite.hpp"
#include "utils/image_utils.hpp"

using namespace Vulkan;
using namespace Util;
using namespace Granite::Importer;

namespace Granite {
ImportedSkinnedMesh::ImportedSkinnedMesh(const Mesh& mesh,
                                         const MaterialInfo& info)
    : mesh(mesh), info(info) {
  topology = mesh.topology;
  index_type = mesh.index_type;

  position_stride = mesh.position_stride;
  attribute_stride = mesh.attribute_stride;
  memcpy(attributes, mesh.attribute_layout, sizeof(mesh.attribute_layout));

  count = mesh.count;
  vertex_offset = 0;
  ibo_offset = 0;

  material = Util::make_abstract_handle<Material, MaterialFile>(info);
  static_aabb = mesh.static_aabb;

  EVENT_MANAGER_REGISTER_LATCH(ImportedSkinnedMesh, on_device_created,
                               on_device_destroyed, DeviceCreatedEvent);
}

void ImportedSkinnedMesh::on_device_created(const DeviceCreatedEvent& created) {
  auto& device = created.get_device();

  BufferCreateInfo info = {};
  info.domain = BufferDomain::Device;
  info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

  info.size = mesh.positions.size();
  vbo_position = device.create_buffer(info, mesh.positions.data());

  if (!mesh.attributes.empty()) {
    info.size = mesh.attributes.size();
    vbo_attributes = device.create_buffer(info, mesh.attributes.data());
  }

  if (!mesh.indices.empty()) {
    info.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    info.size = mesh.indices.size();
    ibo = device.create_buffer(info, mesh.indices.data());
  }

  bake();
}

void ImportedSkinnedMesh::on_device_destroyed(const DeviceCreatedEvent&) {
  vbo_attributes.reset();
  vbo_position.reset();
  ibo.reset();
}

ImportedMesh::ImportedMesh(const Mesh& mesh, const MaterialInfo& info)
    : mesh(mesh), info(info) {
  topology = mesh.topology;
  index_type = mesh.index_type;

  position_stride = mesh.position_stride;
  attribute_stride = mesh.attribute_stride;
  memcpy(attributes, mesh.attribute_layout, sizeof(mesh.attribute_layout));

  count = mesh.count;
  vertex_offset = 0;
  ibo_offset = 0;

  material = Util::make_abstract_handle<Material, MaterialFile>(info);
  static_aabb = mesh.static_aabb;

  EVENT_MANAGER_REGISTER_LATCH(ImportedMesh, on_device_created,
                               on_device_destroyed, DeviceCreatedEvent);
}

void ImportedMesh::on_device_created(const DeviceCreatedEvent& created) {
  auto& device = created.get_device();

  BufferCreateInfo info = {};
  info.domain = BufferDomain::Device;
  info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

  info.size = mesh.positions.size();
  vbo_position = device.create_buffer(info, mesh.positions.data());

  if (!mesh.attributes.empty()) {
    info.size = mesh.attributes.size();
    vbo_attributes = device.create_buffer(info, mesh.attributes.data());
  }

  if (!mesh.indices.empty()) {
    info.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    info.size = mesh.indices.size();
    ibo = device.create_buffer(info, mesh.indices.data());
  }

  bake();
}

void ImportedMesh::on_device_destroyed(const DeviceCreatedEvent&) {
  vbo_attributes.reset();
  vbo_position.reset();
  ibo.reset();
}

SphereMesh::SphereMesh(unsigned density) : density(density) {
  static_aabb = AABB(vec3(-1.0f), vec3(1.0f));
  material = StockMaterials::get().get_checkerboard();
  EVENT_MANAGER_REGISTER_LATCH(SphereMesh, on_device_created,
                               on_device_destroyed, DeviceCreatedEvent);
}

void SphereMesh::on_device_created(const DeviceCreatedEvent& event) {
  auto& device = event.get_device();

  struct Attribute {
    vec3 normal;
    vec2 uv;
  };

  std::vector<vec3> positions;
  std::vector<Attribute> attributes;
  std::vector<uint16_t> indices;

  positions.reserve(6 * density * density);
  attributes.reserve(6 * density * density);
  indices.reserve(2 * density * density * 6);

  float density_mod = 1.0f / float(density - 1);
  const auto to_uv = [&](unsigned x, unsigned y) -> vec2 {
    return vec2(density_mod * x, density_mod * y);
  };

  static const vec3 base_pos[6] = {
      vec3(1.0f, 1.0f, 1.0f),   vec3(-1.0f, 1.0f, -1.0f),
      vec3(-1.0f, 1.0f, -1.0f), vec3(-1.0f, -1.0f, +1.0f),
      vec3(-1.0f, 1.0f, +1.0f), vec3(+1.0f, 1.0f, -1.0f),
  };

  static const vec3 dx[6] = {
      vec3(0.0f, 0.0f, -2.0f), vec3(0.0f, 0.0f, +2.0f), vec3(2.0f, 0.0f, 0.0f),
      vec3(2.0f, 0.0f, 0.0f),  vec3(2.0f, 0.0f, 0.0f),  vec3(-2.0f, 0.0f, 0.0f),
  };

  static const vec3 dy[6] = {
      vec3(0.0f, -2.0f, 0.0f), vec3(0.0f, -2.0f, 0.0f), vec3(0.0f, 0.0f, +2.0f),
      vec3(0.0f, 0.0f, -2.0f), vec3(0.0f, -2.0f, 0.0f), vec3(0.0f, -2.0f, 0.0f),
  };

  // I don't know how many times I've written this exact code in different
  // projects by now. :)
  for (unsigned face = 0; face < 6; face++) {
    unsigned index_offset = face * density * density;
    for (unsigned y = 0; y < density; y++) {
      for (unsigned x = 0; x < density; x++) {
        vec2 uv = to_uv(x, y);
        vec3 pos =
            normalize(base_pos[face] + dx[face] * uv.x + dy[face] * uv.y);
        positions.push_back(pos);
        attributes.push_back({pos, uv});
      }
    }

    unsigned strips = density - 1;
    for (unsigned y = 0; y < strips; y++) {
      unsigned base_index = index_offset + y * density;
      for (unsigned x = 0; x < density; x++) {
        indices.push_back(base_index + x);
        indices.push_back(base_index + x + density);
      }
      indices.push_back(0xffff);
    }
  }

  BufferCreateInfo info = {};
  info.size = positions.size() * sizeof(vec3);
  info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  info.domain = BufferDomain::Device;
  vbo_position = device.create_buffer(info, positions.data());

  info.size = attributes.size() * sizeof(Attribute);
  vbo_attributes = device.create_buffer(info, attributes.data());

  this->attributes[ecast(MeshAttribute::Position)].format =
      VK_FORMAT_R32G32B32_SFLOAT;
  this->attributes[ecast(MeshAttribute::Position)].offset = 0;
  this->attributes[ecast(MeshAttribute::Normal)].format =
      VK_FORMAT_R32G32B32_SFLOAT;
  this->attributes[ecast(MeshAttribute::Normal)].offset =
      offsetof(Attribute, normal);
  this->attributes[ecast(MeshAttribute::UV)].format = VK_FORMAT_R32G32_SFLOAT;
  this->attributes[ecast(MeshAttribute::UV)].offset = offsetof(Attribute, uv);
  position_stride = sizeof(vec3);
  attribute_stride = sizeof(Attribute);

  info.size = indices.size() * sizeof(uint16_t);
  info.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  ibo = device.create_buffer(info, indices.data());
  ibo_offset = 0;
  index_type = VK_INDEX_TYPE_UINT16;
  count = indices.size();
  topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;

  bake();
}

void SphereMesh::on_device_destroyed(const DeviceCreatedEvent&) {
  reset();
}

CubeMesh::CubeMesh() {
  static_aabb = AABB(vec3(-1.0f), vec3(1.0f));
  material = StockMaterials::get().get_checkerboard();
  EVENT_MANAGER_REGISTER_LATCH(CubeMesh, on_device_created, on_device_destroyed,
                               DeviceCreatedEvent);
}

void CubeMesh::on_device_created(const DeviceCreatedEvent& created) {
  auto& device = created.get_device();

  static const int8_t N = -128;
  static const int8_t P = +127;

  static const int8_t positions[] = {
      // Near
      N,
      N,
      P,
      P,
      P,
      N,
      P,
      P,
      N,
      P,
      P,
      P,
      P,
      P,
      P,
      P,

      // Far
      P,
      N,
      N,
      P,
      N,
      N,
      N,
      P,
      P,
      P,
      N,
      P,
      N,
      P,
      N,
      P,

      // Left
      N,
      N,
      N,
      P,
      N,
      N,
      P,
      P,
      N,
      P,
      N,
      P,
      N,
      P,
      P,
      P,

      // Right
      P,
      N,
      P,
      P,
      P,
      N,
      N,
      P,
      P,
      P,
      P,
      P,
      P,
      P,
      N,
      P,

      // Top
      N,
      P,
      P,
      P,
      P,
      P,
      P,
      P,
      N,
      P,
      N,
      P,
      P,
      P,
      N,
      P,

      // Bottom
      N,
      N,
      N,
      P,
      P,
      N,
      N,
      P,
      N,
      N,
      P,
      P,
      P,
      N,
      P,
      P,
  };

  static const int8_t attr[] = {
      // Near
      0,
      0,
      P,
      0,
      P,
      0,
      0,
      0,
      0,
      P,
      0,
      0,
      P,
      0,
      P,
      0,
      0,
      0,
      P,
      P,
      0,
      0,
      P,
      0,
      P,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      P,
      0,
      P,
      0,
      0,
      0,
      P,
      0,

      // Far
      0,
      0,
      N,
      0,
      N,
      0,
      0,
      0,
      0,
      P,
      0,
      0,
      N,
      0,
      N,
      0,
      0,
      0,
      P,
      P,
      0,
      0,
      N,
      0,
      N,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      N,
      0,
      N,
      0,
      0,
      0,
      P,
      0,

      // Left
      N,
      0,
      0,
      0,
      0,
      0,
      P,
      0,
      0,
      P,
      N,
      0,
      0,
      0,
      0,
      0,
      P,
      0,
      P,
      P,
      N,
      0,
      0,
      0,
      0,
      0,
      P,
      0,
      0,
      0,
      N,
      0,
      0,
      0,
      0,
      0,
      P,
      0,
      P,
      0,

      // Right
      P,
      0,
      0,
      0,
      0,
      0,
      N,
      0,
      0,
      P,
      P,
      0,
      0,
      0,
      0,
      0,
      N,
      0,
      P,
      P,
      P,
      0,
      0,
      0,
      0,
      0,
      N,
      0,
      0,
      0,
      P,
      0,
      0,
      0,
      0,
      0,
      N,
      0,
      P,
      0,

      // Top
      0,
      P,
      0,
      0,
      P,
      0,
      0,
      0,
      0,
      P,
      0,
      P,
      0,
      0,
      P,
      0,
      0,
      0,
      P,
      P,
      0,
      P,
      0,
      0,
      P,
      0,
      0,
      0,
      0,
      0,
      0,
      P,
      0,
      0,
      P,
      0,
      0,
      0,
      P,
      0,

      // Bottom
      0,
      N,
      0,
      0,
      P,
      0,
      0,
      0,
      0,
      P,
      0,
      N,
      0,
      0,
      P,
      0,
      0,
      0,
      P,
      P,
      0,
      N,
      0,
      0,
      P,
      0,
      0,
      0,
      0,
      0,
      0,
      N,
      0,
      0,
      P,
      0,
      0,
      0,
      P,
      0,
  };

  BufferCreateInfo vbo_info = {};
  vbo_info.domain = BufferDomain::Device;
  vbo_info.size = sizeof(positions);
  vbo_info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  vbo_position = device.create_buffer(vbo_info, positions);
  position_stride = 4;

  attributes[ecast(MeshAttribute::Position)].offset = 0;
  attributes[ecast(MeshAttribute::Position)].format = VK_FORMAT_R8G8B8A8_SNORM;

  attributes[ecast(MeshAttribute::Normal)].offset = 0;
  attributes[ecast(MeshAttribute::Normal)].format = VK_FORMAT_R8G8B8A8_SNORM;
  attributes[ecast(MeshAttribute::Tangent)].offset = 4;
  attributes[ecast(MeshAttribute::Tangent)].format = VK_FORMAT_R8G8B8A8_SNORM;
  attributes[ecast(MeshAttribute::UV)].offset = 8;
  attributes[ecast(MeshAttribute::UV)].format = VK_FORMAT_R8G8_SNORM;
  attribute_stride = 10;

  vbo_info.size = sizeof(attr);
  vbo_attributes = device.create_buffer(vbo_info, attr);

  static const uint16_t indices[] = {
      0,  1,  2,  3,  2,  1,  4,  5,  6,  7,  6,  5,  8,  9,  10, 11, 10, 9,
      12, 13, 14, 15, 14, 13, 16, 17, 18, 19, 18, 17, 20, 21, 22, 23, 22, 21,
  };
  BufferCreateInfo ibo_info = {};
  ibo_info.size = sizeof(indices);
  ibo_info.domain = BufferDomain::Device;
  ibo_info.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  ibo = device.create_buffer(ibo_info, indices);

  vertex_offset = 0;
  ibo_offset = 0;
  count = 36;
  bake();
}

void CubeMesh::on_device_destroyed(const DeviceCreatedEvent&) {
  reset();
}

SkyCylinder::SkyCylinder(std::string bg_path) : bg_path(move(bg_path)) {
  EVENT_MANAGER_REGISTER_LATCH(SkyCylinder, on_device_created,
                               on_device_destroyed, DeviceCreatedEvent);
}

struct SkyCylinderRenderInfo {
  Program* program;
  const ImageView* view;
  const Sampler* sampler;
  vec3 color;
  float scale;

  const Buffer* vbo;
  const Buffer* ibo;
  unsigned count;
};

struct CylinderVertex {
  vec3 pos;
  vec2 uv;
};

static void skycylinder_render(CommandBuffer& cmd,
                               const RenderQueueData* infos,
                               unsigned instances) {
  for (unsigned i = 0; i < instances; i++) {
    auto* info =
        static_cast<const SkyCylinderRenderInfo*>(infos[i].render_info);

    cmd.set_program(*info->program);
    cmd.set_texture(2, 0, *info->view, *info->sampler);

    vec4 color_scale(info->color, info->scale);
    cmd.push_constants(&color_scale, 0, sizeof(color_scale));

    auto vp = cmd.get_viewport();
    vp.minDepth = 1.0f;
    vp.maxDepth = 1.0f;
    cmd.set_viewport(vp);

    cmd.set_vertex_attrib(0, 0, VK_FORMAT_R32G32B32_SFLOAT,
                          offsetof(CylinderVertex, pos));
    cmd.set_vertex_attrib(1, 0, VK_FORMAT_R32G32_SFLOAT,
                          offsetof(CylinderVertex, uv));
    cmd.set_vertex_binding(0, *info->vbo, 0, sizeof(CylinderVertex));
    cmd.set_index_buffer(*info->ibo, 0, VK_INDEX_TYPE_UINT16);
    cmd.set_primitive_restart(true);
    cmd.set_primitive_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP);
    cmd.draw_indexed(info->count);
  }
}

void SkyCylinder::on_device_created(const DeviceCreatedEvent& created) {
  auto& device = created.get_device();
  texture = nullptr;
  if (!bg_path.empty())
    texture = device.get_texture_manager().request_texture(bg_path);

  std::vector<CylinderVertex> v;
  std::vector<uint16_t> indices;
  for (unsigned i = 0; i < 33; i++) {
    float x = cos(2.0f * pi<float>() * i / 32.0f);
    float z = sin(2.0f * pi<float>() * i / 32.0f);
    v.push_back({vec3(x, +1.0f, z), vec2(i / 32.0f, 0.0f)});
    v.push_back({vec3(x, -1.0f, z), vec2(i / 32.0f, 1.0f)});
  }

  for (unsigned i = 0; i < 33; i++) {
    indices.push_back(2 * i + 0);
    indices.push_back(2 * i + 1);
  }

  indices.push_back(0xffff);

  unsigned ring_offset = v.size();
  v.push_back({vec3(0.0f, 1.0f, 0.0f), vec2(0.5f, 0.0f)});
  v.push_back({vec3(0.0f, -1.0f, 0.0f), vec2(0.5f, 1.0f)});

  for (unsigned i = 0; i < 32; i++) {
    indices.push_back(ring_offset);
    indices.push_back(2 * i);
    indices.push_back(2 * (i + 1));
    indices.push_back(0xffff);
  }

  for (unsigned i = 0; i < 32; i++) {
    indices.push_back(ring_offset + 1);
    indices.push_back(2 * (i + 1) + 1);
    indices.push_back(2 * i + 1);
    indices.push_back(0xffff);
  }

  BufferCreateInfo info = {};
  info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  info.size = v.size() * sizeof(CylinderVertex);
  info.domain = BufferDomain::Device;
  vbo = device.create_buffer(info, v.data());

  info.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  info.size = indices.size() * sizeof(uint16_t);
  ibo = device.create_buffer(info, indices.data());

  count = indices.size();

  Vulkan::SamplerCreateInfo sampler_info = {};
  sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sampler_info.maxAnisotropy = 1.0f;
  sampler_info.magFilter = VK_FILTER_LINEAR;
  sampler_info.minFilter = VK_FILTER_LINEAR;
  sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  sampler_info.maxLod = VK_LOD_CLAMP_NONE;
  sampler = device.create_sampler(sampler_info);
}

void SkyCylinder::on_device_destroyed(const DeviceCreatedEvent&) {
  texture = nullptr;
  vbo.reset();
  ibo.reset();
  sampler.reset();
}

void SkyCylinder::get_render_info(const RenderContext&,
                                  const CachedSpatialTransformComponent*,
                                  RenderQueue& queue) const {
  SkyCylinderRenderInfo info;

  info.view = &texture->get_image()->get_view();

  Hasher h;
  h.pointer(info.view);

  auto instance_key = h.get();
  auto sorting_key =
      RenderInfo::get_background_sort_key(Queue::OpaqueEmissive, 0, 0);
  info.sampler = sampler.get();
  info.color = color;
  info.scale = scale;

  info.ibo = ibo.get();
  info.vbo = vbo.get();
  info.count = count;

  auto* cylinder_info = queue.push<SkyCylinderRenderInfo>(
      Queue::OpaqueEmissive, instance_key, sorting_key, skycylinder_render,
      nullptr);

  if (cylinder_info) {
    info.program = queue.get_shader_suites()[ecast(RenderableType::SkyCylinder)]
                       .get_program(DrawPipeline::Opaque, 0, 0)
                       .get();
    *cylinder_info = info;
  }
}

Skybox::Skybox(std::string bg_path, bool latlon)
    : bg_path(move(bg_path)), is_latlon(latlon) {
  EVENT_MANAGER_REGISTER_LATCH(Skybox, on_device_created, on_device_destroyed,
                               DeviceCreatedEvent);
}

struct SkyboxRenderInfo {
  Program* program;
  const ImageView* view;
  const Sampler* sampler;
  vec3 color;
};

static void skybox_render(CommandBuffer& cmd,
                          const RenderQueueData* infos,
                          unsigned instances) {
  for (unsigned i = 0; i < instances; i++) {
    auto* info = static_cast<const SkyboxRenderInfo*>(infos[i].render_info);

    cmd.set_program(*info->program);

    if (info->view)
      cmd.set_texture(2, 0, *info->view, *info->sampler);

    cmd.push_constants(&info->color, 0, sizeof(info->color));

    CommandBufferUtil::set_quad_vertex_state(cmd);
    cmd.set_cull_mode(VK_CULL_MODE_NONE);
    cmd.set_primitive_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP);
    cmd.draw(4);
  }
}

void Skybox::update_irradiance() {
  if (device && !irradiance_path.empty() && !irradiance_texture) {
    auto& texture_manager = device->get_texture_manager();
    auto cube_path = bg_path + ".cube";
    irradiance_texture =
        texture_manager.register_deferred_texture(irradiance_path);
    texture_manager.register_texture_update_notification(
        is_latlon ? cube_path : bg_path, [this](Vulkan::Texture& tex) {
          irradiance_texture->replace_image(convert_cube_to_ibl_diffuse(
              *device, tex.get_image()->get_view()));
        });
  }
}

void Skybox::update_reflection() {
  if (device && !reflection_path.empty() && !reflection_texture) {
    auto& texture_manager = device->get_texture_manager();
    auto cube_path = bg_path + ".cube";
    reflection_texture =
        texture_manager.register_deferred_texture(reflection_path);
    texture_manager.register_texture_update_notification(
        is_latlon ? cube_path : bg_path, [this](Vulkan::Texture& tex) {
          reflection_texture->replace_image(convert_cube_to_ibl_specular(
              *device, tex.get_image()->get_view()));
        });
  }
}

void Skybox::enable_irradiance(const std::string& path) {
  irradiance_path = path;
  update_irradiance();
}

void Skybox::enable_reflection(const std::string& path) {
  reflection_path = path;
  update_reflection();
}

void Skybox::get_render_info(const RenderContext& context,
                             const CachedSpatialTransformComponent*,
                             RenderQueue& queue) const {
  SkyboxRenderInfo info;

  if (texture)
    info.view = &texture->get_image()->get_view();
  else
    info.view = nullptr;

  Hasher h;
  h.pointer(info.view);

  auto instance_key = h.get();
  auto sorting_key =
      RenderInfo::get_background_sort_key(Queue::OpaqueEmissive, 0, 0);

  info.sampler =
      &context.get_device().get_stock_sampler(StockSampler::LinearClamp);
  info.color = color;

  auto* skydome_info = queue.push<SkyboxRenderInfo>(
      Queue::OpaqueEmissive, instance_key, sorting_key, skybox_render, nullptr);

  if (skydome_info) {
    auto flags = texture ? MATERIAL_EMISSIVE_BIT : 0;
    info.program = queue.get_shader_suites()[ecast(RenderableType::Skybox)]
                       .get_program(DrawPipeline::Opaque, 0, flags)
                       .get();
    *skydome_info = info;
  }
}

void Skybox::on_device_created(const Vulkan::DeviceCreatedEvent& created) {
  texture = nullptr;
  device = &created.get_device();

  if (!bg_path.empty()) {
    if (is_latlon) {
      auto& texture_manager = created.get_device().get_texture_manager();
      texture_manager.request_texture(bg_path);

      auto cube_path = bg_path + ".cube";
      texture = texture_manager.register_deferred_texture(cube_path);

      texture_manager.register_texture_update_notification(
          bg_path, [this](Vulkan::Texture& tex) {
            texture->replace_image(
                convert_equirect_to_cube(*device, tex.get_image()->get_view()));
          });
    } else
      texture =
          created.get_device().get_texture_manager().request_texture(bg_path);

    update_irradiance();
    update_reflection();
  }
}

void Skybox::on_device_destroyed(const DeviceCreatedEvent&) {
  device = nullptr;
  texture = nullptr;
  irradiance_texture = nullptr;
  reflection_texture = nullptr;
}

struct TexturePlaneInfo {
  Vulkan::Program* program;
  const Vulkan::ImageView* reflection;
  const Vulkan::ImageView* refraction;
  const Vulkan::ImageView* normal;

  struct Push {
    vec4 normal;
    vec4 tangent;
    vec4 bitangent;
    vec4 position;
    vec4 dPdx;
    vec4 dPdy;
    vec4 offset_scale;
    vec4 base_emissive;
  };
  Push push;
};

static void texture_plane_render(CommandBuffer& cmd,
                                 const RenderQueueData* infos,
                                 unsigned instances) {
  for (unsigned i = 0; i < instances; i++) {
    auto& info = *static_cast<const TexturePlaneInfo*>(infos[i].render_info);
    cmd.set_program(*info.program);
    if (info.reflection)
      cmd.set_texture(2, 0, *info.reflection,
                      Vulkan::StockSampler::TrilinearClamp);
    if (info.refraction)
      cmd.set_texture(2, 1, *info.refraction,
                      Vulkan::StockSampler::TrilinearClamp);
    cmd.set_texture(2, 2, *info.normal, Vulkan::StockSampler::TrilinearWrap);
    CommandBufferUtil::set_quad_vertex_state(cmd);
    cmd.set_primitive_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP);
    cmd.set_cull_mode(VK_CULL_MODE_NONE);
    cmd.push_constants(&info.push, 0, sizeof(info.push));
    cmd.draw(4);
  }
}

TexturePlane::TexturePlane(const std::string& normal) : normal_path(normal) {
  EVENT_MANAGER_REGISTER_LATCH(TexturePlane, on_device_created,
                               on_device_destroyed, DeviceCreatedEvent);
  EVENT_MANAGER_REGISTER(TexturePlane, on_frame_time, FrameTickEvent);
}

bool TexturePlane::on_frame_time(const FrameTickEvent& tick) {
  elapsed = tick.get_elapsed_time();
  return true;
}

void TexturePlane::on_device_created(const DeviceCreatedEvent& created) {
  normalmap =
      created.get_device().get_texture_manager().request_texture(normal_path);
}

void TexturePlane::on_device_destroyed(const DeviceCreatedEvent&) {
  normalmap = nullptr;
}

void TexturePlane::setup_render_pass_resources(RenderGraph& graph) {
  reflection = nullptr;
  refraction = nullptr;

  if (need_reflection)
    reflection = &graph.get_physical_texture_resource(
        graph.get_texture_resource(reflection_name).get_physical_index());
  if (need_refraction)
    refraction = &graph.get_physical_texture_resource(
        graph.get_texture_resource(refraction_name).get_physical_index());
}

void TexturePlane::setup_render_pass_dependencies(RenderGraph&,
                                                  RenderPass& target) {
  if (need_reflection)
    target.add_texture_input(reflection_name);
  if (need_refraction)
    target.add_texture_input(refraction_name);
}

void TexturePlane::set_scene(Scene* scene) {
  this->scene = scene;
}

void TexturePlane::render_main_pass(Vulkan::CommandBuffer& cmd,
                                    const mat4& proj,
                                    const mat4& view) {
  LightingParameters lighting = *base_context->get_lighting_parameters();
  lighting.shadow_near = nullptr;

  context.set_lighting_parameters(&lighting);
  context.set_camera(proj, view);

  visible.clear();
  scene->gather_visible_opaque_renderables(context.get_visibility_frustum(),
                                           visible);
  scene->gather_visible_transparent_renderables(
      context.get_visibility_frustum(), visible);
  scene->gather_background_renderables(visible);
  renderer->set_mesh_renderer_options_from_lighting(lighting);
  renderer->begin();
  renderer->push_renderables(context, visible);
  renderer->flush(cmd, context);
}

void TexturePlane::set_plane(const vec3& position,
                             const vec3& normal,
                             const vec3& up,
                             float extent_up,
                             float extent_across) {
  this->position = position;
  this->normal = normal;
  this->up = up;
  rad_up = extent_up;
  rad_x = extent_across;

  dpdx = normalize(cross(normal, up)) * extent_across;
  dpdy = normalize(up) * -extent_up;
}

void TexturePlane::set_zfar(float zfar) {
  this->zfar = zfar;
}

void TexturePlane::add_render_pass(RenderGraph& graph, Type type) {
  auto& device = graph.get_device();

  AttachmentInfo color, depth, reflection_blur;
  color.format = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
  depth.format = device.get_default_depth_format();

  color.size_x = scale_x;
  color.size_y = scale_y;
  depth.size_x = scale_x;
  depth.size_y = scale_y;

  reflection_blur.size_x = 0.5f * scale_x;
  reflection_blur.size_y = 0.5f * scale_y;
  reflection_blur.levels = 0;

  auto& name = type == Reflection ? reflection_name : refraction_name;

  auto& lighting =
      graph.add_pass(name + "-lighting", VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT);
  lighting.add_color_output(name + "-HDR", color);
  lighting.set_depth_stencil_output(name + "-depth", depth);

  lighting.set_get_clear_depth_stencil(
      [](VkClearDepthStencilValue* value) -> bool {
        if (value) {
          value->depth = 1.0f;
          value->stencil = 0;
        }
        return true;
      });

  lighting.set_get_clear_color([](unsigned, VkClearColorValue* value) -> bool {
    if (value)
      memset(value, 0, sizeof(*value));
    return true;
  });

  lighting.set_need_render_pass([this]() -> bool {
    // No point in rendering reflection/refraction if we cannot even see it :)
    vec3 c0 = position + dpdx + dpdy;
    vec3 c1 = position - dpdx - dpdy;
    AABB aabb(min(c0, c1), max(c0, c1));
    if (!base_context->get_visibility_frustum().intersects(aabb))
      return false;

    // Only render if we are above the plane.
    float plane_test =
        dot(base_context->get_render_parameters().camera_position - position,
            normal);
    return plane_test > 0.0f;
  });

  lighting.set_build_render_pass([this, type](Vulkan::CommandBuffer& cmd) {
    if (type == Reflection) {
      mat4 proj, view;
      float z_near;
      compute_plane_reflection(
          proj, view, base_context->get_render_parameters().camera_position,
          position, normal, up, rad_up, rad_x, z_near, zfar);
      renderer->set_mesh_renderer_options(Renderer::ENVIRONMENT_ENABLE_BIT |
                                          Renderer::SHADOW_ENABLE_BIT);

      if (zfar > z_near)
        render_main_pass(cmd, proj, view);
    } else if (type == Refraction) {
      mat4 proj, view;
      float z_near;
      compute_plane_refraction(
          proj, view, base_context->get_render_parameters().camera_position,
          position, normal, up, rad_up, rad_x, z_near, zfar);
      renderer->set_mesh_renderer_options(Renderer::ENVIRONMENT_ENABLE_BIT |
                                          Renderer::SHADOW_ENABLE_BIT |
                                          Renderer::REFRACTION_ENABLE_BIT);

      if (zfar > z_near)
        render_main_pass(cmd, proj, view);
    }
  });

  lighting.add_texture_input("shadow-main");

  auto& reflection_blur_pass =
      graph.add_pass(name, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT);
  reflection_blur_pass.add_texture_input(name + "-HDR");
  reflection_blur_pass.add_color_output(name, reflection_blur);
  reflection_blur_pass.set_build_render_pass(
      [this, &reflection_blur_pass](Vulkan::CommandBuffer& cmd) {
        reflection_blur_pass.set_texture_inputs(
            cmd, 0, 0, Vulkan::StockSampler::LinearClamp);
        CommandBufferUtil::draw_quad(cmd, "builtin://shaders/quad.vert",
                                     "builtin://shaders/blur.frag",
                                     {{"METHOD", 6}});
      });
}

#if 0
void TexturePlane::apply_water_depth_tint(Vulkan::CommandBuffer &cmd)
{
	auto &device = cmd.get_device();
	cmd.set_quad_state();
	cmd.set_input_attachments(0, 1);
	cmd.set_blend_enable(true);
	cmd.set_blend_op(VK_BLEND_OP_ADD);
	CommandBufferUtil::set_quad_vertex_state(cmd);
	cmd.set_depth_test(true, false);
	cmd.set_depth_compare(VK_COMPARE_OP_GREATER);

	struct Tint
	{
		mat4 inv_view_proj;
		vec3 falloff;
	} tint;

	tint.inv_view_proj = context.get_render_parameters().inv_view_projection;
	tint.falloff = vec3(1.0f / 1.5f, 1.0f / 2.5f, 1.0f / 5.0f);
	cmd.push_constants(&tint, 0, sizeof(tint));

	cmd.set_blend_factors(VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_SRC_COLOR, VK_BLEND_FACTOR_ZERO);
	auto *program = device.get_shader_manager().register_graphics("builtin://shaders/water_tint.vert",
	                                                              "builtin://shaders/water_tint.frag");
	auto variant = program->register_variant({});
	cmd.set_program(*program->get_program(variant));
	cmd.draw(4);
}
#endif

void TexturePlane::add_render_passes(RenderGraph& graph) {
  if (need_reflection)
    add_render_pass(graph, Reflection);
  if (need_refraction)
    add_render_pass(graph, Refraction);
}

RendererType TexturePlane::get_renderer_type() {
  return RendererType::GeneralForward;
}

void TexturePlane::set_base_renderer(Renderer* renderer) {
  this->renderer = renderer;
}

void TexturePlane::set_base_render_context(const RenderContext* context) {
  base_context = context;
}

void TexturePlane::get_render_info(const RenderContext& context,
                                   const CachedSpatialTransformComponent*,
                                   RenderQueue& queue) const {
  TexturePlaneInfo info;
  info.reflection = reflection;
  info.refraction = refraction;
  info.normal = &normalmap->get_image()->get_view();
  info.push.normal = vec4(normalize(normal), 0.0f);
  info.push.position = vec4(position, 0.0f);
  info.push.dPdx = vec4(dpdx, 0.0f);
  info.push.dPdy = vec4(dpdy, 0.0f);
  info.push.tangent = vec4(normalize(dpdx), 0.0f);
  info.push.bitangent = vec4(normalize(dpdy), 0.0f);
  info.push.offset_scale = vec4(vec2(0.03 * elapsed), vec2(2.0f));
  info.push.base_emissive = vec4(base_emissive, 0.0f);

  Hasher h;
  if (info.reflection)
    h.u64(info.reflection->get_cookie());
  else
    h.u32(0);

  if (info.refraction)
    h.u64(info.refraction->get_cookie());
  else
    h.u32(0);

  h.u64(info.normal->get_cookie());
  auto instance_key = h.get();
  auto sorting_key = RenderInfo::get_sort_key(context, Queue::OpaqueEmissive,
                                              h.get(), h.get(), position);
  auto* plane_info =
      queue.push<TexturePlaneInfo>(Queue::OpaqueEmissive, instance_key,
                                   sorting_key, texture_plane_render, nullptr);

  if (plane_info) {
    unsigned mat_mask = MATERIAL_EMISSIVE_BIT;
    mat_mask |= info.refraction ? MATERIAL_EMISSIVE_REFRACTION_BIT : 0;
    mat_mask |= info.reflection ? MATERIAL_EMISSIVE_REFLECTION_BIT : 0;
    info.program =
        queue.get_shader_suites()[ecast(RenderableType::TexturePlane)]
            .get_program(DrawPipeline::Opaque, 0, mat_mask)
            .get();
    *plane_info = info;
  }
}

}  // namespace Granite
