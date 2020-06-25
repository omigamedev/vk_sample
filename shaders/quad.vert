#version 450

layout (binding = 0) uniform ubo_t { 
    mat4 model;
    mat4 view;
    mat4 proj; 
} ubo;

layout (location = 0) in vec3 v_pos;
layout (location = 1) in vec2 v_uvs;

layout (location = 0) out vec2 f_uvs;

void main()
{
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(v_pos, 1);
    f_uvs = v_uvs;
}
