#version 450

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec4 fragPos;

layout(location = 0) out vec4 outColor;

const float near_plane = 0.1;
const float far_plane = 10.0;

vec3 colorNear = vec3(0.0, 0.0, 1.0);
vec3 colorFar = vec3(1.0, 0.0, 0.0); 

void main() {
    //float depth = fragPos.z / fragPos.w; // Clip space depth
    //outColor = vec4(depth, depth, depth, 1.0);
    // Accessing the depth value
    float depth = gl_FragCoord.z;

    // Alternativa: Normalização adaptativa (opcional)
    //float adjustedDepth = (linearDepth - minDepth) / (maxDepth - minDepth);

    float linearDepth = (2.0 * near_plane) / (far_plane + near_plane - depth * (far_plane - near_plane));
    float expDepth = 1.0 - exp(-5.0 * linearDepth);

    float contrastDepth = clamp(expDepth * 5.0, 0.0, 1.0);

    outColor = vec4(vec3(contrastDepth), 1.0);

    // Alternatively, you can output the original fragment color
    //outColor = vec4(fragColor.x, fragColor.y, fragColor.z, 1.0);
}