#version 450

layout (binding = 0) uniform ubo_t { 
    mat4 model;
    mat4 view;
    mat4 proj; 
} ubo;

layout (location = 0) in vec3 v_pos;

void main()
{
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(v_pos, 1);
}
