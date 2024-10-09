C:/VulkanSDK/1.3.280.0/Bin/glslc.exe shader.vert -o vert.spv
C:/VulkanSDK/1.3.280.0/Bin/glslc.exe shader.frag -o frag.spv
C:/VulkanSDK/1.3.280.0/Bin/glslc.exe compute_shader.comp -o compute_shader.spv --target-env=vulkan1.2
pause

::C:/VulkanSDK/1.3.280.0/Bin/glslc.exe -DENABLE_DEBUG_PRINTF shader.vert -o vert.spv
::C:/VulkanSDK/1.3.280.0/Bin/glslc.exe -DENABLE_DEBUG_PRINTF shader.frag -o frag.spv
::C:/VulkanSDK/1.3.280.0/Bin/glslc.exe -DENABLE_DEBUG_PRINTF compute_shader.comp -o compute_shader.spv --target-env=vulkan1.2