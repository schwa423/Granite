#version 450
precision mediump float;

layout(location = 0) out mediump vec3 Emissive;
layout(location = 0) in highp vec2 vUV;
layout(set = 2, binding = 0) uniform mediump sampler2D uCylinder;

layout(push_constant, std430) uniform Registers
{
    vec3 color;
    float xz_scale;
} registers;

void main()
{
    Emissive = texture(uCylinder, vUV).rgb * registers.color;
}