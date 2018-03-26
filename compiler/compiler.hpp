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

#include <stdint.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Granite {
enum class Stage {
  Vertex,
  TessControl,
  TessEvaluation,
  Geometry,
  Fragment,
  Compute
};

class GLSLCompiler {
 public:
  void set_stage(Stage stage) { this->stage = stage; }

  void set_source(std::string source, std::string path) {
    this->source = std::move(source);
    source_path = std::move(path);
  }

  void set_source_from_file(const std::string& path);
  bool preprocess();

  std::vector<uint32_t> compile(
      const std::vector<std::pair<std::string, int>>* defines = nullptr);

  const std::unordered_set<std::string>& get_dependencies() const {
    return dependencies;
  }

  const std::string& get_error_message() const { return error_message; }

 private:
  std::string source;
  std::string source_path;
  Stage stage = Stage::Compute;

  std::unordered_set<std::string> dependencies;
  std::string preprocessed_source;
  std::string error_message;

  static Stage stage_from_path(const std::string& path);
  bool parse_variants(const std::string& source, const std::string& path);
};
}  // namespace Granite
