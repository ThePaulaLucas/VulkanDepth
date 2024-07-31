#version 450

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec4 fragPos;

layout(location = 0) out vec4 outColor;

void main() {
    float depth = fragPos.z / fragPos.w; // Clip space depth
    //outColor = vec4(depth, depth, depth, 1.0);
    outColor = vec4(fragColor.x, fragColor.y, fragColor.z, 1.0);
}