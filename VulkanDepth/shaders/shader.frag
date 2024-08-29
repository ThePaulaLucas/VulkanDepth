#version 450

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec4 fragPos;

layout(location = 0) out vec4 outColor;

void main() {
    //float depth = fragPos.z / fragPos.w; // Clip space depth
    //outColor = vec4(depth, depth, depth, 1.0);

    // Accessing the depth value
    float depth = gl_FragCoord.z;

    // Aplicar uma curva de gamma para ajustar o contraste
    float gamma = 0.2;  // Valores menores que 1.0 aumentam o contraste
    float adjustedDepth = pow(depth, gamma);

    // Mapeia o valor ajustado para tons de cinza
    //outColor = vec4(adjustedDepth, adjustedDepth, adjustedDepth, 1.0);

    // Alternatively, you can output the original fragment color
    outColor = vec4(fragColor.x, fragColor.y, fragColor.z, 1.0);
}