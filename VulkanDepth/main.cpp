#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cmath>
#include <cstring>
#include <string>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <array>
#include <optional>
#include <set>
#include <thread>
#include <mutex>

#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#endif

const uint32_t WIDTH = 1600;
const uint32_t HEIGHT = 1200;

const int MAX_FRAMES_IN_FLIGHT = 1;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    std::optional<uint32_t> computeFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value() && computeFamily.has_value();
    }
};
struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

        return attributeDescriptions;
    }
};

struct Camera {
    glm::mat3 K;  // Intrinsic matrix
    glm::mat3 R;  // Rotation matrix
    glm::vec3 T;  // Translation vector
};

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

int N_SURFACES;
int CURRENT_INDEX_SURFACE = 0;

// Função para executar o script MATLAB e capturar a saída
std::string execMATLAB(const char* cmd) {
    std::cout << "Running matlab lofting.. \n";
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

std::pair<std::vector<Vertex>, std::vector<Camera>> parseCurvesAndCameras(const std::string& matlabOutput) {
    std::vector<Vertex> curvas;
    std::vector<Camera> cameras;
    Camera currentCamera;
    std::istringstream iss(matlabOutput);
    std::string line;
    float x, y, z;
    int countPoints = 0;
    int countCameras = 0;

    // Identificador de seção
    enum Section { NONE, CURVES, CAMERA_INTRINSICS, CAMERA_ROTATION, CAMERA_TRANSLATION, NUMBER_OF_SURFACES } section = NONE;

    int row = 0;

    std::cout << "Starting parsing curves and cameras...\n";

    while (std::getline(iss, line)) {
        std::istringstream pointStream(line);

        if (line.find("Curva") != std::string::npos) {
            section = CURVES;
            continue;
        }
        if (line.find("Camera Intrinsics (K)") != std::string::npos) {
            section = CAMERA_INTRINSICS;
            row = 0;
            continue;
        }
        if (line.find("Camera Rotation (R)") != std::string::npos) {
            section = CAMERA_ROTATION;
            row = 0;
            continue;
        }
        if (line.find("Camera Translation (T)") != std::string::npos) {
            section = CAMERA_TRANSLATION;
            continue;
        }
        if (line.find("Number of surfaces") != std::string::npos) {
            section = NUMBER_OF_SURFACES;
            continue;
        }

        // Processar a seção atual com base no identificador 'section'
        switch (section) {
        case CURVES:
            // Parse dos pontos da curva (x, y, z)
            if (pointStream >> x >> y >> z) {
                glm::vec3 color = glm::vec3(1.0f, 0.5f, 0.0f); // Cor padrão
                curvas.push_back({ {x, y, z}, color, {0.0f, 0.0f} });
                countPoints++;
            }
            break;

        case CAMERA_INTRINSICS:
            // Parse da matriz intrínseca K (3x3)
            if (pointStream >> x >> y >> z) {
                currentCamera.K[0][row] = x;
                currentCamera.K[1][row] = y;
                currentCamera.K[2][row] = z;
                row++;
                if (row == 3) section = NONE;  // Terminou de ler a matriz K
            }
            break;

        case CAMERA_ROTATION:
            // Parse da matriz de rotação R (3x3)
            if (pointStream >> x >> y >> z) {
                currentCamera.R[0][row] = x;
                currentCamera.R[1][row] = y;
                currentCamera.R[2][row] = z;
                row++;
                if (row == 3) section = NONE;  // Terminou de ler a matriz R
            }
            break;

        case CAMERA_TRANSLATION:
            // Parse do vetor de translação T (1x3)
            if (pointStream >> x >> y >> z) {
                currentCamera.T = glm::vec3(x, y, z);
                cameras.push_back(currentCamera);
                countCameras++;
                section = NONE;  // Terminou de ler a translação
            }
            break;

        case NUMBER_OF_SURFACES:
            if (pointStream >> x) {
                N_SURFACES = x;
                std::cout << "Total number of surface: " << N_SURFACES << "\n";
                section = NONE;
            }
            break;

        default:
            break;
        }
    }

    std::cout << "Parsing ends. Points from curves: " << countPoints << " - Cameras: " << countCameras << "\n";

    return { curvas, cameras };
}

std::string matlabCommandCurveAndCamera = "matlab -batch \"cd('C:\\Users\\lucas\\Desktop\\LOFTING_ZICHANG\\Surface_by_Lofting_From_3D_Curves-main\\Surface_by_Lofting_From_3D_Curves-main'); occlusion_consistency_check;\""; // occlusion_consistency_check; run_loft
std::string matlabOutputCurveAndCamera = execMATLAB(matlabCommandCurveAndCamera.c_str());

std::pair<std::vector<Vertex>, std::vector<Camera>> result = parseCurvesAndCameras(matlabOutputCurveAndCamera);

//Buffer that stores curve points
std::vector<Vertex> pointVertices = result.first;
std::vector<Camera> cameras = result.second;

std::vector<Vertex> model_transformed_pointVertices;
std::vector<Vertex> clip_transformed_pointVertices;

std::vector<std::pair<std::vector<Vertex>, std::vector<uint16_t>>> parseAllSurfaces(const std::string& matlabOutput) {
    std::vector<std::pair<std::vector<Vertex>, std::vector<uint16_t>>> surfaces;
    std::istringstream iss(matlabOutput);
    std::string line;
    float x, y, z;
    int idx1, idx2, idx3;

    glm::vec3 color = glm::vec3(0.0f, 1.0f, 0.0f);  // Cor padrão para superfícies

    std::vector<Vertex> currentVertices;
    std::vector<uint16_t> currentIndices;

    bool parsingPoints = false;
    bool parsingTriangles = false;
    bool isFirstSurface = true;  // Flag para identificar a primeira superfície

    while (std::getline(iss, line)) {
        std::istringstream pointStream(line);

        // Detectar o início de uma nova superfície
        if (line.find("NOVA SUPERFICIE") != std::string::npos) {
            // Para a primeira superfície, não há necessidade de salvar ainda
            if (!isFirstSurface) {
                // Armazenar a superfície anterior antes de começar uma nova
                if (!currentVertices.empty() || !currentIndices.empty()) {
                    surfaces.push_back({ currentVertices, currentIndices });
                    currentVertices.clear();
                    currentIndices.clear();
                }
            }
            isFirstSurface = false;  // Após a primeira ocorrência, considerar outras superfícies
            continue; // Ignorar a linha "NOVA SUPERFICIE"
        }

        // Detectar início dos pontos de superfície
        if (line.find("Points:") != std::string::npos) {
            parsingPoints = true;
            parsingTriangles = false;
            continue;
        }

        // Detectar início dos triângulos de superfície
        if (line.find("Triangles:") != std::string::npos) {
            parsingPoints = false;
            parsingTriangles = true;
            continue;
        }

        // Processar pontos de superfície
        if (parsingPoints) {
            if (pointStream >> x >> y >> z) {
                currentVertices.push_back({ {x, y, z}, color, {0.0f, 0.0f} });
            }
        }

        // Processar triângulos de superfície
        if (parsingTriangles) {
            if (pointStream >> idx1 >> idx2 >> idx3) {
                currentIndices.push_back(static_cast<uint16_t>(idx1));
                currentIndices.push_back(static_cast<uint16_t>(idx2));
                currentIndices.push_back(static_cast<uint16_t>(idx3));
            }
        }
    }

    // Adicionar a última superfície processada, incluindo a primeira
    if (!currentVertices.empty() || !currentIndices.empty()) {
        surfaces.push_back({ currentVertices, currentIndices });
    }

    return surfaces;
}

std::string matlabCommandSurfaceIndex() {
    std::string matlabCommandSurfaceIndex = "matlab -batch \"cd('C:\\Users\\lucas\\Desktop\\LOFTING_ZICHANG\\Surface_by_Lofting_From_3D_Curves-main\\Surface_by_Lofting_From_3D_Curves-main'); read_surfaces;\""; // occlusion_consistency_check; run_loft
    return matlabCommandSurfaceIndex;
}
std::string matlabOutputSurface = execMATLAB(matlabCommandSurfaceIndex().c_str());
auto resultSurf = parseAllSurfaces(matlabOutputSurface);

//Buffer that stores surfaces vertex
std::vector<Vertex> vertices = resultSurf[CURRENT_INDEX_SURFACE].first;
std::vector<uint16_t> indices = resultSurf[CURRENT_INDEX_SURFACE].second;

// Inicializa o arquivo CSV para os dados de performance
std::string performanceLogFile = "performance_log.csv";
void initializePerformanceLog(const std::string& filename) {
    std::ofstream file(filename, std::ios::out);

    if (file.is_open()) {
        // Escreve o cabeçalho com uma coluna extra para o método
        file << "SurfaceIndex,Method,ExecutionTime,numberPoints\n";
        file.close();
    }
    else {
        std::cerr << "Erro ao criar o arquivo de log de performance!\n";
    }
}

void logPerformanceData(const std::string& filename, int surfaceIndex, const std::string& method, double executionTime, int num_points) {
    std::ofstream file(filename, std::ios::out | std::ios::app);

    if (file.is_open()) {
        file << surfaceIndex << "," << method << "," << executionTime << "," << num_points << "\n";
        file.close();
    }
    else {
        std::cerr << "Erro ao escrever no arquivo de log de performance!\n";
    }
}

void printMatrix(const glm::mat3& matrix, const std::string& name) {
    std::cout << name << ":\n";
    for (int i = 0; i < 3; i++) {
        std::cout << matrix[i][0] << " " << matrix[i][1] << " " << matrix[i][2] << std::endl;
    }
    std::cout << std::endl;
}

void printMatrix(const glm::mat4& matrix, const std::string& name) {
    std::cout << name << ":\n";
    for (int i = 0; i < 4; i++) {
        std::cout << matrix[i][0] << " " << matrix[i][1] << " " << matrix[i][2] << " " << matrix[i][3] << std::endl;
    }
    std::cout << std::endl;
}

void printVector(const glm::vec3& vector, const std::string& name) {
    std::cout << name << ": " << vector.x << " " << vector.y << " " << vector.z << "\n\n";
}

bool isPointOccludedByTriangle(
    const glm::vec3& point, const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
    const glm::vec3& ndc_point, const glm::vec3& tr_v0, const glm::vec3& tr_v1, const glm::vec3& tr_v2) {

    // Compute triangle normal in world space
    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;
    glm::vec3 triangleNormal = glm::normalize(glm::cross(edge1, edge2));

    if (glm::length(triangleNormal) < 1e-6f) {
        return false;
    }

    // Check if point is in front of or behind the triangle plane
    float distance = glm::dot(triangleNormal, point - v0);

    if (distance > 0) {
        return false;
    }

    // Check if the point is within the triangle in NDC space
    auto IsPointInTriangle = [](const glm::vec3& pt, const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2) {
        glm::vec3 u = v1 - v0;
        glm::vec3 v = v2 - v0;
        glm::vec3 w = pt - v0;

        float uu = glm::dot(u, u);
        float uv = glm::dot(u, v);
        float vv = glm::dot(v, v);
        float wu = glm::dot(w, u);
        float wv = glm::dot(w, v);

        float denominator = uv * uv - uu * vv;

        if (std::abs(denominator) < 1e-6f) {
            return false;
        }

        float s = (uv * wv - vv * wu) / denominator;
        float t = (uv * wu - uu * wv) / denominator;

        return (s >= 0.0f) && (t >= 0.0f) && (s + t <= 1.0f);
    };

    return IsPointInTriangle(ndc_point, tr_v0, tr_v1, tr_v2);
}

class DepthApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window;

    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;

    VkQueue graphicsQueue;
    VkQueue presentQueue;
    VkQueue computeQueue;

    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;
    std::vector<VkFramebuffer> swapChainFramebuffers;

    VkRenderPass renderPass;
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorSetLayout computeDescriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipelineLayout computePipelineLayout;
    VkPipeline graphicsTrianglePipeline;
    VkPipeline graphicsPointPipeline;
    VkPipeline computePipeline;

    VkCommandPool commandPool;
    VkCommandPool computeCommandPool;

    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;
    VkFormat depthFormat;
    VkSampler depthSampler;

    VkImage textureImage;
    VkDeviceMemory textureImageMemory;
    VkImageView textureImageView;
    VkSampler textureSampler;

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    bool depthBufferCopied = false;
    VkBuffer pointVertexBuffer;
    VkDeviceMemory pointVertexBufferMemory;
    VkBuffer resultBuffer;
    VkDeviceMemory resultBufferMemory;
    VkBuffer vec3Buffer;
    VkDeviceMemory vec3BufferMemory;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;

    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;
    std::vector<VkDescriptorSet> computeDescriptorSets;

    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkCommandBuffer> computeCommandBuffers;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    uint32_t currentFrame = 0;

    bool framebufferResized = false;

    //Used to create the output file of ninary representing the occluded points
    int file_index = 0;

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<DepthApplication*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    void initVulkan() {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createComputeDescriptorSetLayout();
        createGraphicsTrianglePipeline();
        createGraphicsPointPipeline();
        createComputePipeline();
        createCommandPool();
        createDepthResources();
        createDepthSampler();
        createFramebuffers();
        createTextureImage();
        createTextureImageView();
        createTextureSampler();
        createVertexBuffer();
        createIndexBuffer();
        createPointVertexBuffer();
        createOccludedResultBuffer();
        createVec3Buffer();
        createStagingBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createComputeDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
    }

    void mainLoop() {
        initializePerformanceLog(performanceLogFile);

        std::pair<std::vector<Vertex>, std::vector<Vertex>> transformed_points = transformPoint(pointVertices);
        model_transformed_pointVertices = transformed_points.first;
        clip_transformed_pointVertices = transformed_points.second;

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();

            //Take another patch of surface and update the buffers
            if (CURRENT_INDEX_SURFACE < N_SURFACES) {
                testSceneOccludedDirectAlgorithm();
                file_index++;

                CURRENT_INDEX_SURFACE++;

                //Buffer that stores surfaces vertex
                vertices.clear();
                indices.clear();

                vertices = resultSurf[CURRENT_INDEX_SURFACE].first;
                indices = resultSurf[CURRENT_INDEX_SURFACE].second;

                updateVertexBuffer();
                updateIndexBuffer();
            }
        }

        vkDeviceWaitIdle(device);
    }

    void cleanupSwapChain() {
        vkDestroyImageView(device, depthImageView, nullptr);
        vkDestroyImage(device, depthImage, nullptr);
        vkFreeMemory(device, depthImageMemory, nullptr);

        for (auto framebuffer : swapChainFramebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        for (auto imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }

        vkDestroySwapchainKHR(device, swapChain, nullptr);
    }

    void cleanup() {
        cleanupSwapChain();

        vkDestroyPipeline(device, graphicsTrianglePipeline, nullptr);
        vkDestroyPipeline(device, graphicsPointPipeline, nullptr);
        vkDestroyPipeline(device, computePipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyPipelineLayout(device, computePipelineLayout, nullptr);

        vkDestroyRenderPass(device, renderPass, nullptr);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroyBuffer(device, uniformBuffers[i], nullptr);
            vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
        }

        vkDestroyDescriptorPool(device, descriptorPool, nullptr);

        vkDestroySampler(device, textureSampler, nullptr);
        vkDestroyImageView(device, textureImageView, nullptr);

        vkDestroyImage(device, textureImage, nullptr);
        vkFreeMemory(device, textureImageMemory, nullptr);

        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

        //Buffers
        vkDestroyBuffer(device, indexBuffer, nullptr);
        vkFreeMemory(device, indexBufferMemory, nullptr);

        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);

        vkDestroyBuffer(device, pointVertexBuffer, nullptr);
        vkFreeMemory(device, pointVertexBufferMemory, nullptr);

        vkDestroyBuffer(device, resultBuffer, nullptr);
        vkFreeMemory(device, resultBufferMemory, nullptr);

        // Liberar o sampler e a imagem de profundidade
        vkDestroySampler(device, depthSampler, nullptr);
        vkDestroyImageView(device, depthImageView, nullptr);
        vkDestroyImage(device, depthImage, nullptr);
        vkFreeMemory(device, depthImageMemory, nullptr);

        cleanupStagingBuffer();

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }

        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyCommandPool(device, computeCommandPool, nullptr);

        vkDestroyDevice(device, nullptr);

        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);

        glfwTerminate();
    }

    void recreateSwapChain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(device);

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createDepthResources();
        createFramebuffers();
    }

    void createInstance() {
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("Validation layers requested, but not available!");
        }

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Vulkan Application";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_2;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        auto extensions = getRequiredExtensions();
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME); // Add debug utils extension
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        VkValidationFeaturesEXT validationFeatures{};
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            populateDebugMessengerCreateInfo(debugCreateInfo);

            // Enable the debug printf extension
            VkValidationFeatureEnableEXT enables[] = { VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT };
            validationFeatures.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
            validationFeatures.enabledValidationFeatureCount = 1;
            validationFeatures.pEnabledValidationFeatures = enables;

            // Chain the validation features to the pNext of debugCreateInfo
            debugCreateInfo.pNext = &validationFeatures;
            createInfo.pNext = &debugCreateInfo;
        }
        else {
            createInfo.enabledLayerCount = 0;
            createInfo.pNext = nullptr;
        }

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create instance!");
        }
        setupDebugMessenger();
    }

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |  // Added this line
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
    }

    void setupDebugMessenger() {
        if (!enableValidationLayers) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }

    void createSurface() {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        std::cout << "Found " << deviceCount << " GPU(s) with Vulkan support.\n";

        for (const auto& device : devices) {
            VkPhysicalDeviceProperties deviceProperties;
            vkGetPhysicalDeviceProperties(device, &deviceProperties);
            std::cout << "GPU Name: " << deviceProperties.deviceName << "\n";

            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    void createLogicalDevice() {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkPhysicalDeviceFeatures deviceFeatures{};
        deviceFeatures.samplerAnisotropy = VK_TRUE;

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();

        createInfo.pEnabledFeatures = &deviceFeatures;

        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else {
            createInfo.enabledLayerCount = 0;
        }

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }

        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
        vkGetDeviceQueue(device, indices.computeFamily.value(), 0, &computeQueue);
    }

    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;

        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        if (indices.graphicsFamily != indices.presentFamily) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }

        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size());

        for (uint32_t i = 0; i < swapChainImages.size(); i++) {
            swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);
        }
    }

    //Render Pass
    void createRenderPass() {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = findDepthFormat();
        depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; //verify
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL; //VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; //VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthAttachmentRef{};
        depthAttachmentRef.attachment = 1;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        subpass.pDepthStencilAttachment = &depthAttachmentRef;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        std::array<VkAttachmentDescription, 2> attachments = { colorAttachment, depthAttachment };
        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    //Create Descriptor Set Layout
    void createDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding uboLayoutBinding{};
        uboLayoutBinding.binding = 0;
        uboLayoutBinding.descriptorCount = 1;
        uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboLayoutBinding.pImmutableSamplers = nullptr;
        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutBinding samplerLayoutBinding{};
        samplerLayoutBinding.binding = 1;
        samplerLayoutBinding.descriptorCount = 1;
        samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerLayoutBinding.pImmutableSamplers = nullptr;
        samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        std::array<VkDescriptorSetLayoutBinding, 2> bindings = { uboLayoutBinding, samplerLayoutBinding };
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor set layout!");
        }
    }

    void createComputeDescriptorSetLayout() {
        std::array<VkDescriptorSetLayoutBinding, 4> bindings{};

        // Binding 0: Uniform Buffer Object (UBO)
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;  // Corrigido para COMPUTE
        bindings[0].pImmutableSamplers = nullptr;

        // Binding 1: Buffer de Pontos
        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;  // Corrigido para COMPUTE
        bindings[1].pImmutableSamplers = nullptr;

        // Binding 2: Depth Buffer (imagem 2D)
        bindings[2].binding = 2;
        bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;  // Corrigido para COMPUTE
        bindings[2].pImmutableSamplers = nullptr;

        // Binding 3: Buffer de Resultados de Oclusão
        bindings[3].binding = 3;
        bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[3].descriptorCount = 1;
        bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;  // Corrigido para COMPUTE
        bindings[3].pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &computeDescriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor set layout for compute shader!");
        }
    }

    void updateComputeDescriptorSet(VkDescriptorSet descriptorSet, VkBuffer uniformBuffer, VkBuffer pointsBuffer, VkImageView depthImageView, VkSampler depthSampler, VkBuffer resultsBuffer) {
        std::array<VkWriteDescriptorSet, 4> descriptorWrites{};

        // Binding 0: Uniform Buffer Object (UBO)
        VkDescriptorBufferInfo uboBufferInfo{};
        uboBufferInfo.buffer = uniformBuffer;
        uboBufferInfo.offset = 0;
        uboBufferInfo.range = sizeof(UniformBufferObject);

        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = computeDescriptorSets[currentFrame];;
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &uboBufferInfo;

        // Binding 1: Buffer de Pontos
        VkDescriptorBufferInfo pointsBufferInfo{};
        pointsBufferInfo.buffer = vec3Buffer;
        pointsBufferInfo.offset = 0;
        pointsBufferInfo.range = VK_WHOLE_SIZE;

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = computeDescriptorSets[currentFrame];;
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pBufferInfo = &pointsBufferInfo;

        // Binding 2: Depth Buffer como imagem
        VkDescriptorImageInfo depthImageInfo{};
        depthImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        depthImageInfo.imageView = depthImageView;
        depthImageInfo.sampler = depthSampler;

        descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[2].dstSet = computeDescriptorSets[currentFrame];;
        descriptorWrites[2].dstBinding = 2;
        descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[2].descriptorCount = 1;
        descriptorWrites[2].pImageInfo = &depthImageInfo;

        // Binding 3: Buffer de Resultados de Oclusão
        VkDescriptorBufferInfo resultsBufferInfo{};
        resultsBufferInfo.buffer = resultsBuffer;
        resultsBufferInfo.offset = 0;
        resultsBufferInfo.range = VK_WHOLE_SIZE;

        descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[3].dstSet = computeDescriptorSets[currentFrame];;
        descriptorWrites[3].dstBinding = 3;
        descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[3].descriptorCount = 1;
        descriptorWrites[3].pBufferInfo = &resultsBufferInfo;

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }

    //Create Pipelines
    void createGraphicsTrianglePipeline() {
        auto vertShaderCode = readFile("shaders/vert.spv");
        auto fragShaderCode = readFile("shaders/frag.spv");

        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; //VK_PRIMITIVE_TOPOLOGY_POINT_LIST // VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; 
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_NONE; // VK_CULL_MODE_NONE VK_CULL_MODE_BACK_BIT
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.stencilTestEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsTrianglePipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }

    void createGraphicsPointPipeline() {
        auto vertShaderCode = readFile("shaders/vert.spv");
        auto fragShaderCode = readFile("shaders/frag.spv");

        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST; //VK_PRIMITIVE_TOPOLOGY_POINT_LIST // VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; 
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_NONE; // VK_CULL_MODE_NONE VK_CULL_MODE_BACK_BIT
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_FALSE; //VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.stencilTestEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPointPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }

    void createComputePipeline() {
        // 1. Carregar o shader de computação
        auto computeShaderCode = readFile("shaders/compute_shader.spv");
        VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

        // 2. Definir o estágio do shader de computação
        VkPipelineShaderStageCreateInfo shaderStageInfo{};
        shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shaderStageInfo.module = computeShaderModule;
        shaderStageInfo.pName = "main";  // Nome da função de entrada no shader

        // 3. Criar o layout do pipeline de computação usando o mesmo descriptor set layout
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;  // Número de descritores que você usará
        pipelineLayoutInfo.pSetLayouts = &computeDescriptorSetLayout;  // O layout criado anteriormente

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &computePipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline layout!");
        }

        // 4. Criar o compute pipeline
        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage = shaderStageInfo;
        pipelineInfo.layout = computePipelineLayout;  // Use o novo layout criado para o compute shader

        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline!");
        }

        // 5. Destruir o módulo de shader após criar o pipeline
        vkDestroyShaderModule(device, computeShaderModule, nullptr);
    }

    void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            std::array<VkImageView, 2> attachments = {
                swapChainImageViews[i],
                depthImageView
            };

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }

    void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT | VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics command pool!");
        }

        VkCommandPoolCreateInfo computePoolInfo{};
        computePoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        computePoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT | VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        computePoolInfo.queueFamilyIndex = queueFamilyIndices.computeFamily.value();

        if (vkCreateCommandPool(device, &computePoolInfo, nullptr, &computeCommandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics command pool!");
        }
    }

    //Depth
    void createStagingBuffer() {
        VkDeviceSize bufferSize = swapChainExtent.width * swapChainExtent.height * sizeof(float);

        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = bufferSize;
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &stagingBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create staging buffer!");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, stagingBuffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &stagingBufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate staging buffer memory!");
        }

        vkBindBufferMemory(device, stagingBuffer, stagingBufferMemory, 0);
    }

    void cleanupStagingBuffer() {
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createDepthResources() {
        VkFormat depthFormat = findDepthFormat();
        createImage(swapChainExtent.width, swapChainExtent.height, depthFormat,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            depthImage, depthImageMemory);
        depthImageView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);

        transitionImageLayout(depthImage, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    }

    void createDepthSampler() {
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;  // Filtro de magnificação
        samplerInfo.minFilter = VK_FILTER_LINEAR;  // Filtro de minificação
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.anisotropyEnable = VK_FALSE;
        samplerInfo.maxAnisotropy = 1.0f;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;

        if (vkCreateSampler(device, &samplerInfo, nullptr, &depthSampler) != VK_SUCCESS) {
            throw std::runtime_error("failed to create depth sampler!");
        }
    }

    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
        for (VkFormat format : candidates) {
            VkFormatProperties props;
            vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

            if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
                return format;
            }
            else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }

        throw std::runtime_error("failed to find supported format!");
    }

    VkFormat findDepthFormat() {
        return findSupportedFormat(
            { VK_FORMAT_D32_SFLOAT },
            //{ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
            VK_IMAGE_TILING_OPTIMAL,
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
        );
    }

    bool hasStencilComponent(VkFormat format) {
        return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
    }

    void copyDepthBufferToStagingBuffer() {
        // Create a fence to ensure GPU operations are complete before proceeding
        VkFence copyFence;
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

        if (vkCreateFence(device, &fenceInfo, nullptr, &copyFence) != VK_SUCCESS) {
            throw std::runtime_error("failed to create fence!");
        }

        // Allocate a temporary command buffer for the copy operation
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        // Begin recording commands
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        // Transition the depth image layout to TRANSFER_SRC_OPTIMAL for copying
        transitionImageLayoutCB(commandBuffer, depthImage, VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

        // Copy the depth image to the staging buffer
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = { swapChainExtent.width, swapChainExtent.height, 1 };

        vkCmdCopyImageToBuffer(commandBuffer, depthImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stagingBuffer, 1, &region);

        // Transition the depth image layout back to DEPTH_STENCIL_ATTACHMENT_OPTIMAL for the next frame
        transitionImageLayoutCB(commandBuffer, depthImage, VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        // End recording of the command buffer
        vkEndCommandBuffer(commandBuffer);

        // Submit the command buffer and wait for it to finish
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, copyFence);
        vkWaitForFences(device, 1, &copyFence, VK_TRUE, UINT64_MAX);

        // Cleanup
        vkDestroyFence(device, copyFence, nullptr);
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    float* mapStagingBuffer() {
        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, VK_WHOLE_SIZE, 0, &data);
        return static_cast<float*>(data); // Return a pointer to the mapped memory
    }

    void unmapStagingBuffer() {
        vkUnmapMemory(device, stagingBufferMemory);
    }

    float getDepthValueAtCoord(float* depthValues, int x, int y) {
        // Ensure x and y are within bounds
        if (x >= 0 && x < swapChainExtent.width && y >= 0 && y < swapChainExtent.height) {
            int index = y * swapChainExtent.width + x;
            return depthValues[index];
        }
        else {
            return -1.0f; // Return an error value or handle this appropriately
        }
    }

    void copyImageToBuffer(VkImage image, VkBuffer buffer, uint32_t width, uint32_t height, VkFormat format) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkImageSubresourceLayers subresource = {};
        subresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT; //VK_IMAGE_ASPECT_COLOR_BIT; // VK_IMAGE_ASPECT_DEPTH_BIT;
        subresource.mipLevel = 0;
        subresource.baseArrayLayer = 0;
        subresource.layerCount = 1;

        VkBufferImageCopy region = {};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource = subresource;
        region.imageOffset = { 0, 0, 0 };
        region.imageExtent = { width, height, 1 };

        vkCmdCopyImageToBuffer(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &region);

        endSingleTimeCommands(commandBuffer);
    }

    //Texture
    void createTextureImage() {
        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load("textures/texture.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        VkDeviceSize imageSize = texWidth * texHeight * 4;

        if (!pixels) {
            throw std::runtime_error("failed to load texture image!");
        }

        VkBuffer _stagingBuffer;
        VkDeviceMemory _stagingBufferMemory;
        createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, _stagingBuffer, _stagingBufferMemory);

        void* data;
        vkMapMemory(device, _stagingBufferMemory, 0, imageSize, 0, &data);
        memcpy(data, pixels, static_cast<size_t>(imageSize));
        vkUnmapMemory(device, _stagingBufferMemory);

        stbi_image_free(pixels);

        createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

        transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        copyBufferToImage(_stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
        transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        vkDestroyBuffer(device, _stagingBuffer, nullptr);
        vkFreeMemory(device, _stagingBufferMemory, nullptr);
    }

    void createTextureImageView() {
        textureImageView = createImageView(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);
    }

    void createTextureSampler() {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.anisotropyEnable = VK_TRUE;
        samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

        if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
            throw std::runtime_error("failed to create texture sampler!");
        }
    }

    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = aspectFlags;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        VkImageView imageView;
        if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image view!");
        }

        return imageView;
    }

    void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = tiling;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image!");
        }

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate image memory!");
        }

        vkBindImageMemory(device, image, imageMemory, 0);
    }

    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;

        // Handle depth and stencil aspect masks based on format
        if (format == VK_FORMAT_D32_SFLOAT || format == VK_FORMAT_D16_UNORM || format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT) {
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            if (format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT) {
                barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
            }
        }
        else {
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        }

        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        VkPipelineStageFlags sourceStage;
        VkPipelineStageFlags destinationStage;

        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
            // Transitioning from undefined, no previous access
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
            // Preparing for transfer from depth/stencil attachment
            barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

            sourceStage = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
            // After transfer, transitioning back to depth/stencil attachment
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            // Initial transition to be ready for transfer as destination
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            // Ready for shader read after transfer
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else {
            std::cerr << "Unsupported layout transition from " << oldLayout << " to " << newLayout << std::endl;
            throw std::invalid_argument("Unsupported layout transition!");
        }

        vkCmdPipelineBarrier(
            commandBuffer,
            sourceStage, destinationStage,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );

        endSingleTimeCommands(commandBuffer);
    }

    void transitionImageLayoutCB(VkCommandBuffer commandBuffer, VkImage image, VkFormat format,
        VkImageLayout oldLayout, VkImageLayout newLayout) {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;

        if (format == VK_FORMAT_D32_SFLOAT || format == VK_FORMAT_D32_SFLOAT_S8_UINT) {
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            if (format == VK_FORMAT_D32_SFLOAT_S8_UINT) {
                barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
            }
        }
        else {
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        }

        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        VkPipelineStageFlags sourceStage;
        VkPipelineStageFlags destinationStage;

        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

            sourceStage = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;

        }
        else if (oldLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            destinationStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

        }
        else if (oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            sourceStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;

        }
        else {
            throw std::invalid_argument("Unsupported layout transition");
        }

        vkCmdPipelineBarrier(
            commandBuffer,
            sourceStage, destinationStage,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );
    }

    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = { 0, 0, 0 };
        region.imageExtent = {
            width,
            height,
            1
        };

        vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        endSingleTimeCommands(commandBuffer);
    }

    //Create Buffers
    void createVertexBuffer() {
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        VkBuffer _stagingBuffer;
        VkDeviceMemory _stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, _stagingBuffer, _stagingBufferMemory);

        void* data;
        vkMapMemory(device, _stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, vertices.data(), (size_t)bufferSize);
        vkUnmapMemory(device, _stagingBufferMemory);

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

        copyBuffer(_stagingBuffer, vertexBuffer, bufferSize);

        vkDestroyBuffer(device, _stagingBuffer, nullptr);
        vkFreeMemory(device, _stagingBufferMemory, nullptr);
    }

    void createIndexBuffer() {
        VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        VkBuffer _stagingBuffer;
        VkDeviceMemory _stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, _stagingBuffer, _stagingBufferMemory);

        void* data;
        vkMapMemory(device, _stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, indices.data(), (size_t)bufferSize);
        vkUnmapMemory(device, _stagingBufferMemory);

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

        copyBuffer(_stagingBuffer, indexBuffer, bufferSize);

        vkDestroyBuffer(device, _stagingBuffer, nullptr);
        vkFreeMemory(device, _stagingBufferMemory, nullptr);
    }

    void createPointVertexBuffer() {
        VkDeviceSize bufferSize = sizeof(pointVertices[0]) * pointVertices.size();

        VkBuffer _stagingBuffer;
        VkDeviceMemory _stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, _stagingBuffer, _stagingBufferMemory);

        void* data;
        vkMapMemory(device, _stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, pointVertices.data(), (size_t)bufferSize);
        vkUnmapMemory(device, _stagingBufferMemory);

        VkBufferUsageFlags usageFlags = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

        createBuffer(bufferSize, usageFlags,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            pointVertexBuffer, pointVertexBufferMemory);

        copyBuffer(_stagingBuffer, pointVertexBuffer, bufferSize);

        vkDestroyBuffer(device, _stagingBuffer, nullptr);
        vkFreeMemory(device, _stagingBufferMemory, nullptr);
    }

    void createOccludedResultBuffer() {

        VkDeviceSize bufferSize = sizeof(int) * pointVertices.size();  // Tamanho do buffer baseado no número de pontos

        createBuffer(bufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,  // Para usar no compute shader
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,  // Mapeável para a CPU
            resultBuffer,
            resultBufferMemory);
    }

    void createVec3Buffer() {
        VkDeviceSize bufferSize = sizeof(glm::vec4) * pointVertices.size();

        // Create a staging buffer for the vec3 data
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuffer,
            stagingBufferMemory);

        // Map the staging buffer memory and copy the vec3 data
        void* data;
        VkResult result = vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        if (result != VK_SUCCESS) {
            std::cerr << "Failed to map staging buffer memory. Error code: " << result << "\n";
            return;
        }

        // Copy vec3 data from pointVertices
        std::vector<glm::vec4> vec3Data;
        vec3Data.reserve(pointVertices.size());
        for (const auto& vertex : pointVertices) {
            vec3Data.push_back(glm::vec4(vertex.pos.x, vertex.pos.y, vertex.pos.z, 0.0f));
        }
        memcpy(data, vec3Data.data(), static_cast<size_t>(bufferSize));

        vkUnmapMemory(device, stagingBufferMemory);

        // Create the actual buffer with device-local memory for the vec3 data
        createBuffer(bufferSize,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            vec3Buffer,
            vec3BufferMemory);

        // Copy data from the staging buffer to the vec3 buffer
        copyBuffer(stagingBuffer, vec3Buffer, bufferSize);

        // Clean up the staging buffer
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    //Update Buffers
    void updateVertexBuffer() {
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        // Criar staging buffer
        VkBuffer _stagingBuffer;
        VkDeviceMemory _stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            _stagingBuffer, _stagingBufferMemory);

        // Mapear a memória e copiar os novos dados de vértices para o staging buffer
        void* data;
        vkMapMemory(device, _stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, vertices.data(), (size_t)bufferSize);
        vkUnmapMemory(device, _stagingBufferMemory);

        // Copiar os dados do staging buffer para o buffer de vértices na GPU
        copyBuffer(_stagingBuffer, vertexBuffer, bufferSize);

        // Destruir o staging buffer e liberar sua memória
        vkDestroyBuffer(device, _stagingBuffer, nullptr);
        vkFreeMemory(device, _stagingBufferMemory, nullptr);
    }

    void updateIndexBuffer() {
        VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        // Criar staging buffer
        VkBuffer _stagingBuffer;
        VkDeviceMemory _stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            _stagingBuffer, _stagingBufferMemory);

        // Mapear a memória e copiar os novos dados de índices para o staging buffer
        void* data;
        vkMapMemory(device, _stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, indices.data(), (size_t)bufferSize);
        vkUnmapMemory(device, _stagingBufferMemory);

        // Copiar os dados do staging buffer para o buffer de índices na GPU
        copyBuffer(_stagingBuffer, indexBuffer, bufferSize);

        // Destruir o staging buffer e liberar sua memória
        vkDestroyBuffer(device, _stagingBuffer, nullptr);
        vkFreeMemory(device, _stagingBufferMemory, nullptr);
    }

    void createUniformBuffers() {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);

        uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);

            vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
        }
    }

    //Descriptor Sets
    void createDescriptorPool() {
        constexpr uint32_t EXTRA_SETS = 10;  // Additional sets for safety
        std::array<VkDescriptorPoolSize, 3> poolSizes{};

        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = MAX_FRAMES_IN_FLIGHT * 2;  // Para gráficos e computação

        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[1].descriptorCount = 2 * MAX_FRAMES_IN_FLIGHT * 2;  // Para gráficos e computação

        poolSizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[2].descriptorCount = MAX_FRAMES_IN_FLIGHT * 2;  // Para gráficos e computação

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = (MAX_FRAMES_IN_FLIGHT * 2) + EXTRA_SETS; // Ajuste para comportar ambos os pipelines

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor pool!");
        }
    }

    void createDescriptorSets() {
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts = layouts.data();

        descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView = textureImageView;
            imageInfo.sampler = textureSampler;

            std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = descriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &bufferInfo;

            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = descriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pImageInfo = &imageInfo;

            vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
    }

    void createComputeDescriptorSets() {
        // Criação de N descriptor sets, um para cada frame em flight
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, computeDescriptorSetLayout);

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;  // Certifique-se de ter criado um pool de descritores
        allocInfo.descriptorSetCount = static_cast<uint32_t>(layouts.size());
        allocInfo.pSetLayouts = layouts.data();

        computeDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        if (vkAllocateDescriptorSets(device, &allocInfo, computeDescriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate compute descriptor sets!");
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            VkDescriptorBufferInfo uboBufferInfo{};
            uboBufferInfo.buffer = uniformBuffers[i];
            uboBufferInfo.offset = 0;
            uboBufferInfo.range = sizeof(UniformBufferObject);

            VkDescriptorBufferInfo vec3BufferInfo{};
            vec3BufferInfo.buffer = vec3Buffer;
            vec3BufferInfo.offset = 0;
            vec3BufferInfo.range = VK_WHOLE_SIZE;

            VkDescriptorImageInfo depthImageInfo{};
            depthImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            depthImageInfo.imageView = depthImageView;
            depthImageInfo.sampler = depthSampler;

            VkDescriptorBufferInfo resultsBufferInfo{};
            resultsBufferInfo.buffer = resultBuffer;
            resultsBufferInfo.offset = 0;
            resultsBufferInfo.range = VK_WHOLE_SIZE;

            std::array<VkWriteDescriptorSet, 4> descriptorWrites{};

            // Binding 0: Uniform Buffer (UBO)
            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = computeDescriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &uboBufferInfo;

            // Binding 1: Buffer de Pontos
            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = computeDescriptorSets[i];
            descriptorWrites[1].dstBinding = 1;  // Update to the binding for vec3Buffer
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pBufferInfo = &vec3BufferInfo;

            // Binding 2: Depth Buffer
            descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[2].dstSet = computeDescriptorSets[i];
            descriptorWrites[2].dstBinding = 2;
            descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[2].descriptorCount = 1;
            descriptorWrites[2].pImageInfo = &depthImageInfo;

            descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[3].dstSet = computeDescriptorSets[i];
            descriptorWrites[3].dstBinding = 3;
            descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[3].descriptorCount = 1;
            descriptorWrites[3].pBufferInfo = &resultsBufferInfo;

            vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create buffer!");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);
        /*std::cout << "Memory Requirements: Size=" << memRequirements.size
            << ", Alignment=" << memRequirements.alignment
            << ", Memory Type Bits=" << memRequirements.memoryTypeBits << std::endl;*/

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;


        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate buffer memory!");
        }

        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    VkCommandBuffer beginSingleTimeCommands() {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        return commandBuffer;
    }

    void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = 0;
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        endSingleTimeCommands(commandBuffer);
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createCommandBuffers() {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }

        computeCommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        VkCommandBufferAllocateInfo computeAllocInfo{};
        computeAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        computeAllocInfo.commandPool = computeCommandPool;  // Certifique-se de que tem um command pool para compute
        computeAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        computeAllocInfo.commandBufferCount = static_cast<uint32_t>(computeCommandBuffers.size());

        if (vkAllocateCommandBuffers(device, &computeAllocInfo, computeCommandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate compute command buffers!");
        }
    }

    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swapChainExtent;

        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
        clearValues[1].depthStencil = { 1.0f, 0 };

        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsTrianglePipeline);

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)swapChainExtent.width;
        viewport.height = (float)swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = swapChainExtent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        // ---- Renderizar Triângulos ----
        VkBuffer vertexBuffers[] = { vertexBuffer };
        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

        vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT16);

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

        // Desenhar triângulos
        vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

        // ---- Renderizar Pontos ----
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPointPipeline);
        VkBuffer pointBuffers[] = { pointVertexBuffer };
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, pointBuffers, offsets);

        // Desenhar pontos (usando vkCmdDraw)
        vkCmdDraw(commandBuffer, static_cast<uint32_t>(pointVertices.size()), 1, 0, 0);

        vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    void createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create synchronization objects for a frame!");
            }
        }
    }

    //Uniform Buffer
    void updateUniformBuffer(uint32_t currentImage) {

        UniformBufferObject ubo{};
        ubo = initUbo(ubo);

        void* data;
        vkMapMemory(device, uniformBuffersMemory[currentFrame], 0, sizeof(ubo), 0, &data);

        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
        vkUnmapMemory(device, uniformBuffersMemory[currentFrame]);
    }

    UniformBufferObject initUbo(UniformBufferObject ubo) {
        ubo.model = glm::mat4(1.0f);
        //ubo.model[1][1] *= -1;

        //std::vector<std::vector<float>> K = {
        //    { 2, 0, 400 },   // Fator de escala em x (focal length), centro da câmera em x = 400 (meio de uma janela 800x600)
        //    { 0, 3, 300 },   // Fator de escala em y (focal length), centro da câmera em y = 300
        //    { 0, 0, 1 }
        //};

        //std::vector<std::vector<float>> R = {
        //    { 1,  0,  0 },
        //    { 0, -1,  0 },  // Inversão no eixo Y
        //    { 0,  0,  1 }
        //};

        //std::vector<float> t = { 0.0f, 0.0f, 5.0f };  // Translação da câmera

        //Camera from matlab
        //std::vector<std::vector<float>> K = {
        //    {2917.9f, 0.0f,    800.f},   // Fator de escala em x (focal length), centro da câmera em x = 400 (meio de uma janela 800x600)
        //    { 0.0f,   2917.9f, 600.f},   // Fator de escala em y (focal length), centro da câmera em y = 300
        //    { 0.0f,   0.0f,    1.0f }
        //};

        //std::vector<std::vector<float>> R = {
        //    { 0.3136,  0.0239, -0.9492},
        //    { -0.2881, 0.9550, -0.0711 },  // Inversão no eixo Y
        //    { 0.9048,  0.2958 , 0.3064 }
        //};

        //R espelhada
        /*std::vector<std::vector<float>> R = {
            { -0.9492,  0.0239, 0.3136},
            { -0.0711, 0.9550, -0.2881 },
            { 0.3064,  0.2958 , 0.9048 }
        };*/

        //std::vector<float> t = { -5.1092f, -2.2337f, 7.5941f };  // Translação da câmera
        //std::vector<float> t = { -5.912186f, 0.008942f, - 7.335489f }; //C_t = -R' * t

        Camera& firstCamera = cameras[0];

        // Imprime as matrizes e o vetor de translação da primeira câmera
        /*printMatrix(firstCamera.K, "Primeira Camera Intrinsics (K matlab)");
        printMatrix(firstCamera.R, "Primeira Camera Rotation (R matlab)");
        printVector(firstCamera.T, "Primeira Camera Translation (T matlab)");*/

        // Construindo a matriz de rotação a partir de R (transposta necessária para Vulkan)
        glm::mat3 R = glm::mat3(
            glm::vec3(firstCamera.R[0][0], firstCamera.R[0][1], firstCamera.R[0][2]),
            glm::vec3(firstCamera.R[1][0], firstCamera.R[1][1], firstCamera.R[1][2]),
            glm::vec3(firstCamera.R[2][0], firstCamera.R[2][1], firstCamera.R[2][2])
        );

        /*R = glm::mat4(1.0f);
        R[1][1] *= -1;
        R[0][0] *= -1;*/

        // Corrigindo a translação usando a matriz de rotação transposta
        /*glm::vec3 t = -glm::transpose(R) * firstCamera.T;
        printVector(t, "C_t (matlab)");*/

        //t = glm::vec3(firstCamera.T[0], firstCamera.T[1], firstCamera.T[2]);

        // Matriz intrínseca da câmera (K)
        glm::mat3 K = glm::mat3(
            glm::vec3(firstCamera.K[0][0], firstCamera.K[0][1], firstCamera.K[0][2]),
            glm::vec3(firstCamera.K[1][0], firstCamera.K[1][1], firstCamera.K[1][2]),
            glm::vec3(firstCamera.K[2][0], firstCamera.K[2][1], firstCamera.K[2][2])
        );

        /*printMatrix(K, "Primeira Camera Intrinsics (K Vulkan)");
        printMatrix(R, "Primeira Camera Rotation (R Vulkan)");*/

        // Montando a matriz de visão (R e t)
        /*ubo.view = glm::mat4(
            glm::vec4(-R[0][0], -R[0][1], -R[0][2], 0.0f),
            glm::vec4(-R[1][0], -R[1][1], -R[1][2], 0.0f),
            glm::vec4(R[2][0], R[2][1], R[2][2], 0.0f),
            glm::vec4(t[0], t[1], t[2], 1.0f)
        );*/

        glm::vec3 C_t = -glm::transpose(R) * firstCamera.T;
        glm::vec3 direction = glm::normalize(glm::vec3(firstCamera.R[0][2], firstCamera.R[1][2], firstCamera.R[2][2]));
        glm::vec3 up = glm::normalize(-glm::vec3(firstCamera.R[0][1], firstCamera.R[1][1], firstCamera.R[2][1]));
        ubo.view = glm::lookAt(C_t, C_t + direction, up);

        // Parâmetros de clipping e dimensão da janela
        static float zNear = 0.1f;
        static float zFar = 1000.0f;
        static float width = WIDTH;
        static float height = HEIGHT;

        /*static float width = static_cast<float>(swapChainExtent.width);
        static float height = static_cast<float>(swapChainExtent.height);*/

        //static float width = static_cast<float>(swapChainExtent.width);
        //static float height = static_cast<float>(swapChainExtent.height);

        //std::cout << height << " " << width << "\n";

        // Matriz de projeção
        ubo.proj = glm::mat4(
            glm::vec4(2 * K[0][0] / width, 0.0f, 0.0f, 0.0f),
            glm::vec4(0.0f, -2 * K[1][1] / height, 0.0f, 0.0f),
            glm::vec4(2 * K[2][0] / width - 1, 2 * K[2][1] / height - 1, zFar / (zNear - zFar), -1.0f),
            glm::vec4(0.0f, 0.0f, zFar * zNear / (zNear - zFar), 0.0f)
        );

        //printMatrix(ubo.view, "View Vulkan");
        //printMatrix(ubo.proj, "Proj Vulkan");

        return ubo;
    }

    //Direct Algorithm
    void testSceneOccludedDirectAlgorithm() {
        int count = 0;
        UniformBufferObject ubo{};
        ubo = initUbo(ubo);

        std::ofstream outputFile("output_occlusion/occlusion_output_direct_" + std::to_string(file_index) + ".txt");
        if (!outputFile.is_open()) {
            std::cerr << "Não foi possível abrir o arquivo para escrita!\n";
            return;
        }

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<Vertex> transformed_vertices;

        for (size_t i = 0; i < vertices.size(); i++) {
            glm::vec4 pt = glm::vec4(vertices[i].pos, 1.0f);

            glm::vec4 transformed_pt = ubo.proj * ubo.view * ubo.model * pt;
            transformed_pt = transformed_pt / transformed_pt.w;

            float x = (transformed_pt.x * 0.5f + 0.5f) * swapChainExtent.width;
            float y = (transformed_pt.y * 0.5f + 0.5f) * swapChainExtent.height;

            glm::vec3 screen_pos = glm::vec3(x, y, transformed_pt.z);
            glm::vec3 color = glm::vec3(1.0f, 0.5f, 0.0f);

            transformed_vertices.push_back({ screen_pos, color, {0.0f, 0.0f} });
        }

        for (size_t i = 0; i < pointVertices.size(); ++i) {
            glm::vec3 model_point = pointVertices[i].pos;
            glm::vec3 ndc_point = clip_transformed_pointVertices[i].pos;
            bool isOccluded = false;

            for (size_t j = 0; j < indices.size(); j += 3) {
                int id0 = indices[j];
                int id1 = indices[j + 1];
                int id2 = indices[j + 2];

                glm::vec3 v0 = vertices[id0].pos;
                glm::vec3 v1 = vertices[id1].pos;
                glm::vec3 v2 = vertices[id2].pos;

                glm::vec3 tr_v0 = transformed_vertices[id0].pos;
                glm::vec3 tr_v1 = transformed_vertices[id1].pos;
                glm::vec3 tr_v2 = transformed_vertices[id2].pos;

                count++;
                if (isPointOccludedByTriangle(model_point, v0, v1, v2, ndc_point, tr_v0, tr_v1, tr_v2)) {
                    isOccluded = true;
                    break;
                }

                /*if (isOccluded) {
                    std::cout << "Point " << i << " is occluded by triangle (" << id0 << ", " << id1 << ", " << id2 << ")." << "\n";
                }
                else {
                    std::cout << "Point " << i << " is NOT occluded by triangle (" << id0 << ", " << id1 << ", " << id2 << ")." << "\n";
                }*/
            }
            outputFile << (isOccluded ? 1 : 0) << "\n";
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Execution time Direct Algorithm: " << elapsed.count() << " seconds, with " << count << " tests." << "\n\n";
        
        logPerformanceData(performanceLogFile, CURRENT_INDEX_SURFACE, "Direct Method", elapsed.count(), pointVertices.size());

        outputFile.close();
    }

    //Draw
    void drawFrame() {
        // Wait for the previous frame to finish
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        // Acquire the next image from the swapchain
        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapChain();
            return;
        }
        else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        // Update the uniform buffer for the current frame
        updateUniformBuffer(currentFrame);

        // Reset the fence for the current frame
        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        // Reset and record the command buffer for rendering
        vkResetCommandBuffer(commandBuffers[currentFrame], 0);
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

        // Submit the command buffer to the graphics queue
        submitGraphicsCommandBuffer();

        // Wait for the graphics queue to finish rendering
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        // Copy the depth buffer to the staging buffer
        copyDepthBufferToStagingBuffer();

        // Run the compute shader to process occlusion and time it
        auto computeStart = std::chrono::high_resolution_clock::now();
        executeComputeShader();
        auto computeEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> computeElapsed = computeEnd - computeStart;
        std::cout << "Execution time Z-Buffer Algorithm in compute shader: " << computeElapsed.count() << " seconds.\n";
        logPerformanceData(performanceLogFile, CURRENT_INDEX_SURFACE, "Compute Shader Z-Buffer", computeElapsed.count(), pointVertices.size());


        // Print the occlusion results to a file from the compute shader
        printOcclusionResultsFile(clip_transformed_pointVertices.size());

        // Present the final image
        presentImage(imageIndex);

        // Optionally: Test depth occlusion (if needed)
        testPointDepth();

        // Update the current frame
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void submitGraphicsCommandBuffer() {
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

        VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }
    }

    void executeComputeShader() {
        // Update the descriptor set for the compute shader
        updateComputeDescriptorSet(
            computeDescriptorSets[currentFrame],
            uniformBuffers[currentFrame],  // Uniform buffer
            vec3Buffer,                   // The buffer for the points
            depthImageView,               // The depth image view
            depthSampler,                 // The sampler for the depth image
            resultBuffer                  // The buffer for occlusion results
        );

        // Separate command buffer for compute shader
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        vkResetCommandBuffer(computeCommandBuffers[currentFrame], 0);
        vkBeginCommandBuffer(computeCommandBuffers[currentFrame], &beginInfo);

        // Transition depth buffer for compute shader usage
        transitionImageLayoutCB(computeCommandBuffers[currentFrame], depthImage, VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        // Bind the compute pipeline and descriptor set
        vkCmdBindPipeline(computeCommandBuffers[currentFrame], VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
        vkCmdBindDescriptorSets(computeCommandBuffers[currentFrame], VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSets[currentFrame], 0, nullptr);

        // Dispatch the compute shader
        size_t numPoints = pointVertices.size();
        int numWorkGroupsX = (numPoints + 63) / 64;  // Assuming a local size of 64
        vkCmdDispatch(computeCommandBuffers[currentFrame], numWorkGroupsX, 1, 1);

        // Transition depth buffer back to its original layout
        transitionImageLayoutCB(computeCommandBuffers[currentFrame], depthImage, VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        // End recording the command buffer
        vkEndCommandBuffer(computeCommandBuffers[currentFrame]);

        // Submit the compute command buffer
        VkSubmitInfo computeSubmitInfo{};
        computeSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        computeSubmitInfo.commandBufferCount = 1;
        computeSubmitInfo.pCommandBuffers = &computeCommandBuffers[currentFrame];

        if (vkQueueSubmit(computeQueue, 1, &computeSubmitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit compute command buffer!");
        }

        // Wait for the compute shader to complete
        vkQueueWaitIdle(computeQueue);
    }

    void presentImage(uint32_t imageIndex) {
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;

        // Use the semaphore that signals when rendering is finished
        VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
        presentInfo.pWaitSemaphores = signalSemaphores;

        VkSwapchainKHR swapChains[] = { swapChain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;

        VkResult result = vkQueuePresentKHR(presentQueue, &presentInfo);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        }
        else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present swap chain image!");
        }
    }

    int* mapResultBuffer() {
        void* data;
        vkMapMemory(device, resultBufferMemory, 0, VK_WHOLE_SIZE, 0, &data);
        return static_cast<int*>(data);  // Assuming int is used for occlusion results (0 or 1)
    }

    void unmapResultBuffer() {
        vkUnmapMemory(device, resultBufferMemory);
    }

    void printOcclusionResults(int numPoints) {
        int* results = mapResultBuffer();
        if (!results) {
            std::cerr << "Failed to map result buffer!\n";
            return;
        }

        for (int i = 0; i < numPoints; ++i) {
            std::cout << "Point " << i << ": " << results[i] << (results[i] == 1 ? "Occluded" : "Not Occluded") << "\n";
        }

        unmapResultBuffer();
    }

    void printOcclusionResultsFile(int numPoints) {
        // Map the results buffer (where the compute shader stored occlusion results)
        void* data;
        vkMapMemory(device, resultBufferMemory, 0, VK_WHOLE_SIZE, 0, &data);
        int* occlusionResults = static_cast<int*>(data);

        // Open a file to write the results
        std::ofstream outputFile("output_occlusion/occlusion_output_compute_" + std::to_string(file_index) + ".txt");
        if (!outputFile.is_open()) {
            std::cerr << "Failed to open output file!\n";
            vkUnmapMemory(device, resultBufferMemory);
            return;
        }

        for (int i = 0; i < numPoints; i++) {
            outputFile << occlusionResults[i] << "\n";
        }

        // Close the file
        outputFile.close();

        // Unmap the memory when done
        vkUnmapMemory(device, resultBufferMemory);
    }

    //Z-buffer Algorithm
    void testPointDepth() {
        // Wait for rendering to complete before proceeding =================
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
        //==================================
        UniformBufferObject _ubo{};
        _ubo = initUbo(_ubo);
        int count = 0;
        std::cout << std::fixed << std::setprecision(10);
        int isOccluded = 0;

        std::vector<int> occlusionResults(clip_transformed_pointVertices.size(), 0);

        //std::cout << currentFrame << "\n";
        std::ofstream outputFile("output_occlusion/occlusion_output_frame_" + std::to_string(file_index) + ".txt");
        if (!outputFile.is_open()) {
            std::cerr << "Não foi possível abrir o arquivo para escrita!\n";
            return;
        }

        float* depthValues = mapStagingBuffer();
        if (!depthValues) {
            std::cerr << "Failed to map the staging buffer!\n";
            return;
        }

        //WITHOUT PARALELIZATION
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < clip_transformed_pointVertices.size(); i++) {
            isOccluded = 0;

            /*if (i < 10) {
                std::cout << pointVertices[i].pos.x << " " << pointVertices[i].pos.y << " " << pointVertices[i].pos.z << "\n";
                std::cout << clip_transformed_pointVertices[i].pos.x << " " << clip_transformed_pointVertices[i].pos.y << " " << clip_transformed_pointVertices[i].pos.z << "\n";
            }*/

            // Fetch the depth value at this screen position
            float depthValue = getDepthValueAtCoord(depthValues,
                static_cast<int>(clip_transformed_pointVertices[i].pos.x),
                static_cast<int>(clip_transformed_pointVertices[i].pos.y));

            //std::cout << i << ": " << ndc.z - depthValue << "\n";
            float bias = 0.00001;
            if (clip_transformed_pointVertices[i].pos.z - (depthValue + bias) > 0.000003) {
                isOccluded = 1;
            }

            //outputFile << isOccluded <<"\n";
            occlusionResults[i] = isOccluded;
            count++;
        }

        //PARALELIZATION
        // Get available hardware concurrency (number of cores/threads)
        //int numThreads = std::thread::hardware_concurrency();
        //int numPoints = clip_transformed_pointVertices.size();
        //int chunkSize = numPoints / numThreads;

        //// Parallel execution using threads
        //auto processChunk = [&](int start, int end) {
        //    for (int i = start; i < end; ++i) {
        //        int x = static_cast<int>(clip_transformed_pointVertices[i].pos.x);
        //        int y = static_cast<int>(clip_transformed_pointVertices[i].pos.y);
        //        float depthValue = getDepthValueAtCoord(depthValues, x, y);

        //        if (clip_transformed_pointVertices[i].pos.z - depthValue > 0.00001) {
        //            occlusionResults[i] = 1;  // Occluded
        //        }
        //        else {
        //            occlusionResults[i] = 0;  // Not occluded
        //        }
        //    }
        //};

        //// Launch threads to process each chunk of the points
        //std::vector<std::thread> threads;
        //auto start = std::chrono::high_resolution_clock::now();
        //for (int i = 0; i < numThreads; ++i) {
        //    int startIdx = i * chunkSize;
        //    int endIdx = (i == numThreads - 1) ? numPoints : (i + 1) * chunkSize;  // Last thread handles the rest
        //    threads.emplace_back(processChunk, startIdx, endIdx);
        //}

        //// Join all threads (wait for completion)
        //for (auto& thread : threads) {
        //    thread.join();
        //}

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Execution time Z-Buffer Algorithm: " << elapsed.count() << " seconds for " << count << " tests.\n";
        logPerformanceData(performanceLogFile, CURRENT_INDEX_SURFACE, "Z-Buffer", elapsed.count(), pointVertices.size());

        // Write occlusion results to the output file after ensuring all threads are done
        for (const auto& result : occlusionResults) {
            outputFile << result << "\n";
        }

        unmapStagingBuffer();

        outputFile.close();
    }

    std::pair<std::vector<Vertex>, std::vector<Vertex>> transformPoint(const std::vector<Vertex> pointCurve) {
        std::vector<Vertex> model_curvas;
        std::vector<Vertex> clip_curvas;

        UniformBufferObject _ubo{};
        _ubo = initUbo(_ubo);

        for (int i = 0; i < pointCurve.size(); i++) {
            glm::vec3 vertex = pointCurve[i].pos;

            // Transform the vertex position into clip space 
            glm::vec4 vertexPos = glm::vec4(vertex, 1.0f);
            glm::vec4 model = _ubo.model * vertexPos;

            glm::vec3 color = glm::vec3(1.0f, 0.5f, 0.0f);

            glm::vec3 model_pos = glm::vec3(model.x, model.y, model.z);
            model_curvas.push_back({ model_pos, color, {0.0f, 0.0f} });

            glm::vec4 view = _ubo.view * model;
            glm::vec4 clip = _ubo.proj * view;

            // Espaço de coordenadas normalizadas (NDC)
            glm::vec3 ndc = glm::vec3(clip) / clip.w;

            // Converte coordenadas NDC para coordenadas de tela (x e y, variando de -1 a 1)
            float x = (ndc.x * 0.5f + 0.5f) * swapChainExtent.width;
            float y = (ndc.y * 0.5f + 0.5f) * swapChainExtent.height;

            glm::vec3 clip_pos = glm::vec3(x, y, ndc.z);
            clip_curvas.push_back({ clip_pos, color, {0.0f, 0.0f} });
        }

        return { model_curvas, clip_curvas };
    }

    VkShaderModule createShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }

        return shaderModule;
    }

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;
            }
        }

        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }
        else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
        SwapChainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }

    bool isDeviceSuitable(VkPhysicalDevice device) {
        QueueFamilyIndices indices = findQueueFamilies(device);

        bool extensionsSupported = checkDeviceExtensionSupport(device);

        return indices.isComplete() && extensionsSupported;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
        QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            // Verifica se esta fila suporta gráficos
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
            }

            // Verifica se esta fila suporta apresentação
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

            if (presentSupport) {
                indices.presentFamily = i;
            }

            // Verifica se esta fila suporta computação
            if (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) {
                indices.computeFamily = i;
            }

            // Verifica se encontramos todas as filas necessárias
            if (indices.isComplete()) {
                break;
            }

            i++;
        }

        return indices;
    }

    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    bool checkValidationLayerSupport() {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layerName : validationLayers) {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) {
                return false;
            }
        }

        return true;
    }

    static std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }

        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();

        return buffer;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

        return VK_FALSE;
    }
};

int main() {
    DepthApplication app;

    try {
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
