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

#include "camera.hpp"
#include "glm/gtx/matrix_decompose.hpp"
#include "input.hpp"
#include "scene.hpp"
#include "transforms.hpp"
#include "vulkan_events.hpp"

using namespace Vulkan;

namespace Granite {
void Camera::set_depth_range(float znear, float zfar) {
  this->znear = znear;
  this->zfar = zfar;
}

void Camera::set_aspect(float aspect) {
  this->aspect = aspect;
}

void Camera::set_fovy(float fovy) {
  this->fovy = fovy;
}

mat4 Camera::get_view() const {
  return mat4_cast(rotation) * glm::translate(-position);
}

void Camera::set_position(const vec3& pos) {
  position = pos;
}

void Camera::set_rotation(const quat& rot) {
  rotation = rot;
}

void Camera::look_at(const vec3& eye, const vec3& at, const vec3& up) {
  position = eye;
  rotation = ::Granite::look_at(at - eye, up);
}

mat4 Camera::get_projection() const {
  return projection(fovy, aspect, znear * transform_z_scale,
                    zfar * transform_z_scale);
}

vec3 Camera::get_position() const {
  return position;
}

vec3 Camera::get_front() const {
  static const vec3 z(0.0f, 0.0f, -1.0f);
  return conjugate(rotation) * z;
}

vec3 Camera::get_right() const {
  static const vec3 right(1.0f, 0.0f, 0.0f);
  return conjugate(rotation) * right;
}

vec3 Camera::get_up() const {
  static const vec3 up(0.0f, 1.0f, 0.0f);
  return conjugate(rotation) * up;
}

void Camera::set_transform(const mat4& m) {
  vec3 skew;
  vec4 perspective;
  vec3 s, t;
  quat r;
  glm::decompose(m, s, r, t, skew, perspective);

  position = t;
  rotation = conjugate(r);
  transform_z_scale = s.x;
}

FPSCamera::FPSCamera() {
  EVENT_MANAGER_REGISTER(FPSCamera, on_mouse_move, MouseMoveEvent);
  EVENT_MANAGER_REGISTER(FPSCamera, on_orientation, OrientationEvent);
  EVENT_MANAGER_REGISTER(FPSCamera, on_touch_down, TouchDownEvent);
  EVENT_MANAGER_REGISTER(FPSCamera, on_touch_up, TouchUpEvent);
  EVENT_MANAGER_REGISTER(FPSCamera, on_input_state, InputStateEvent);
  EVENT_MANAGER_REGISTER(FPSCamera, on_joypad_state, JoypadStateEvent);
  EVENT_MANAGER_REGISTER_LATCH(FPSCamera, on_swapchain, on_swapchain,
                               SwapchainParameterEvent);
}

bool FPSCamera::on_touch_down(const TouchDownEvent&) {
  pointer_count++;
  return true;
}

bool FPSCamera::on_touch_up(const TouchUpEvent&) {
  assert(pointer_count > 0);
  pointer_count--;
  return true;
}

void FPSCamera::on_swapchain(const SwapchainParameterEvent& state) {
  set_aspect(state.get_aspect_ratio());
}

bool FPSCamera::on_input_state(const InputStateEvent& state) {
  position +=
      3.0f * get_front() * float(pointer_count) * float(state.get_delta_time());

  if (state.get_key_pressed(Key::W))
    position += 3.0f * get_front() * float(state.get_delta_time());
  else if (state.get_key_pressed(Key::S))
    position -= 3.0f * get_front() * float(state.get_delta_time());
  if (state.get_key_pressed(Key::D))
    position += 3.0f * get_right() * float(state.get_delta_time());
  else if (state.get_key_pressed(Key::A))
    position -= 3.0f * get_right() * float(state.get_delta_time());
  return true;
}

bool FPSCamera::on_joypad_state(const JoypadStateEvent& state) {
  auto& p0 = state.get_state(0);

  position += 3.0f * get_front() * -p0.get_axis(JoypadAxis::LeftY) *
              float(state.get_delta_time());
  position += 3.0f * get_right() * p0.get_axis(JoypadAxis::LeftX) *
              float(state.get_delta_time());

  float dx = 2.0f * p0.get_axis(JoypadAxis::RightX) * state.get_delta_time();
  float dy = 1.0f * p0.get_axis(JoypadAxis::RightY) * state.get_delta_time();

  quat pitch = angleAxis(dy, vec3(1.0f, 0.0f, 0.0f));
  quat yaw = angleAxis(dx, vec3(0.0f, 1.0f, 0.0f));
  rotation = normalize(pitch * rotation * yaw);

  return true;
}

bool FPSCamera::on_mouse_move(const MouseMoveEvent& m) {
  if (!m.get_mouse_button_pressed(MouseButton::Right))
    return true;

  auto dx = float(m.get_delta_x());
  auto dy = float(m.get_delta_y());
  quat pitch = angleAxis(dy * 0.02f, vec3(1.0f, 0.0f, 0.0f));
  quat yaw = angleAxis(dx * 0.02f, vec3(0.0f, 1.0f, 0.0f));
  rotation = normalize(pitch * rotation * yaw);

  return true;
}

bool FPSCamera::on_orientation(const OrientationEvent& o) {
  rotation = conjugate(o.get_rotation());
  return true;
}
}  // namespace Granite
