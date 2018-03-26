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
#include <string>
#include "animation_system.hpp"
#include "gltf.hpp"
#include "scene.hpp"

namespace Granite {
class SceneLoader {
 public:
  SceneLoader();
  void load_scene(const std::string& path);

  Scene& get_scene() { return *scene; }

  std::unique_ptr<AnimationSystem> consume_animation_system();

 private:
  struct SubsceneData {
    std::unique_ptr<GLTF::Parser> parser;
    std::vector<AbstractRenderableHandle> meshes;
  };
  std::unordered_map<std::string, SubsceneData> subscenes;

  std::unique_ptr<Scene> scene;
  std::unique_ptr<AnimationSystem> animation_system;
  void parse_scene_format(const std::string& path, const std::string& json);
  void parse_gltf(const std::string& path);

  Scene::NodeHandle build_tree_for_subscene(const SubsceneData& subscene);
  void load_animation(const std::string& path, Importer::Animation& animation);
};
}  // namespace Granite
