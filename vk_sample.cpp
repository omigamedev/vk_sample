#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

#include <windows.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/matrix4x4.h>

vk::PhysicalDevice physical_device;
vk::UniqueDevice device;

struct vertex_t
{
    glm::vec3 pos;
    glm::vec3 nor;
    glm::vec2 uvs;
    vertex_t() = default;
    vertex_t(glm::vec3 pos) : pos(pos), nor(0), uvs(0) {}
    vertex_t(glm::vec2 pos) : pos(glm::vec3(pos, 0)), nor(0), uvs(0) {}
    vertex_t(glm::vec3 pos, glm::vec3 nor) : pos(pos), nor(nor), uvs(0) {}
    vertex_t(glm::vec3 pos, glm::vec3 nor, glm::vec2 uvs) : pos(pos), nor(nor), uvs(uvs) {}
};

struct uniform_buffers_t
{
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
    static constexpr uint32_t mvp_size = sizeof(model) + sizeof(view) + sizeof(proj);
    uint8_t pad2[0x100 - mvp_size & ~0x100]; // alignment

    glm::vec4 col;
    uint8_t pad3[0x100 - sizeof(col) & ~0x100]; // alignment
};

struct mesh_t
{
    std::vector<vertex_t> vertices;
    std::vector<uint32_t> indices;
    glm::mat4 model;
    vk::DeviceSize off_index;
    vk::DeviceSize off_vertex;
    vk::DeviceSize off_uniform;
};


vk::UniqueShaderModule load_shader_module(const std::string& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("cannot open the file " + path);
    size_t size = file.tellg();
    file.seekg(std::ios::beg);
    std::unique_ptr<char[]> buffer = std::make_unique<char[]>(size);
    file.read(buffer.get(), size);

    vk::ShaderModuleCreateInfo module_info;
    module_info.codeSize = size;
    module_info.pCode = reinterpret_cast<uint32_t*>(buffer.get());
    vk::UniqueShaderModule m = device->createShaderModuleUnique(module_info);
    return m;
}

uint32_t find_memory(const vk::MemoryRequirements& req, vk::MemoryPropertyFlags flags)
{
    vk::PhysicalDeviceMemoryProperties mp = physical_device.getMemoryProperties();
    for (uint32_t mem_i = 0; mem_i < mp.memoryTypeCount; mem_i++)
        if ((1 << mem_i) & req.memoryTypeBits && (mp.memoryTypes[mem_i].propertyFlags & flags) == flags)
            return mem_i;
    throw std::runtime_error("find_memory failed");
}

HWND create_window()
{
    WNDCLASS wc{ 0 };
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.hInstance = GetModuleHandle(0);
    wc.lpszClassName = L"MainVulkanWindow";
    wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
    wc.lpfnWndProc = DefWindowProc;
    if (!RegisterClass(&wc))
        exit(1);
    RECT r = { 0, 0, 800, 600 };
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, false);
    return CreateWindow(wc.lpszClassName, L"Vulkan", WS_OVERLAPPEDWINDOW | WS_VISIBLE | WS_SYSMENU, 0, 0,
        r.right - r.left, r.bottom - r.top, NULL, NULL, wc.hInstance, nullptr);
}

int main()
{
    std::vector<const char*> inst_layers;
    inst_layers.push_back("VK_LAYER_KHRONOS_validation");
    std::vector<const char*> inst_extentions;
    inst_extentions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
    inst_extentions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
    vk::ApplicationInfo app_info;
    app_info.pApplicationName = "vk_sample";
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pEngineName = "custom";
    app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.apiVersion = VK_VERSION_1_2;
    vk::InstanceCreateInfo instance_info;
    instance_info.pApplicationInfo = &app_info;
    instance_info.enabledLayerCount = (uint32_t)inst_layers.size();
    instance_info.ppEnabledLayerNames = inst_layers.data();
    instance_info.enabledExtensionCount = (uint32_t)inst_extentions.size();
    instance_info.ppEnabledExtensionNames = inst_extentions.data();
    vk::UniqueInstance inst = vk::createInstanceUnique(instance_info);

    HWND hWnd = create_window();
    vk::Win32SurfaceCreateInfoKHR surface_info;
    surface_info.hinstance = GetModuleHandle(0);
    surface_info.hwnd = hWnd;
    vk::UniqueSurfaceKHR surface = inst->createWin32SurfaceKHRUnique(surface_info);

    uint32_t device_family_index = 0;
    for (auto pd : inst->enumeratePhysicalDevices())
    {
        std::vector<vk::QueueFamilyProperties> families = pd.getQueueFamilyProperties();
        for (int family_index = 0; family_index < families.size(); family_index++)
        {
            if (families[family_index].queueFlags & vk::QueueFlagBits::eGraphics && pd.getSurfaceSupportKHR(family_index, *surface))
            {
                vk::PhysicalDeviceProperties props = pd.getProperties();
                std::cout << "Device trovato: " << props.deviceName << std::endl;
                
                device_family_index = family_index;
                physical_device = pd;

                std::vector<const char*> device_layers;
                std::vector<const char*> device_extentions;
                device_extentions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

                float queue_priority[1] = { 1.f };
                vk::DeviceQueueCreateInfo queue_info;
                queue_info.queueFamilyIndex = family_index;
                queue_info.queueCount = 1;
                queue_info.pQueuePriorities = queue_priority;
                vk::DeviceCreateInfo device_info;
                device_info.queueCreateInfoCount = 1;
                device_info.pQueueCreateInfos = &queue_info;
                device_info.enabledLayerCount = (uint32_t)device_layers.size();
                device_info.ppEnabledLayerNames = device_layers.data();
                device_info.enabledExtensionCount = (uint32_t)device_extentions.size();
                device_info.ppEnabledExtensionNames = device_extentions.data();
                device = pd.createDeviceUnique(device_info);

                break;
            }
        }
        if (device)
            break;
    }

    if (!device)
    {
        std::cout << "Device not found\n";
        return EXIT_FAILURE;
    }

    vk::SurfaceCapabilitiesKHR surface_caps = physical_device.getSurfaceCapabilitiesKHR(*surface);

    vk::SwapchainCreateInfoKHR swapchain_info;
    swapchain_info.surface = *surface;
    swapchain_info.minImageCount = 2;
    swapchain_info.imageFormat = vk::Format::eB8G8R8A8Unorm;
    swapchain_info.imageColorSpace = vk::ColorSpaceKHR::eSrgbNonlinear;
    swapchain_info.imageExtent = surface_caps.currentExtent;
    swapchain_info.imageArrayLayers = 1;
    swapchain_info.imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst;
    swapchain_info.presentMode = vk::PresentModeKHR::eFifo;
    swapchain_info.clipped = true;
    vk::UniqueSwapchainKHR swapchain = device->createSwapchainKHRUnique(swapchain_info);

    vk::Queue q = device->getQueue(device_family_index, 0);

    vk::CommandPoolCreateInfo cmd_pool_info;
    cmd_pool_info.queueFamilyIndex = device_family_index;
    vk::UniqueCommandPool cmd_pool = device->createCommandPoolUnique(cmd_pool_info);
    

    // Load texture

    int tex_width = 0;
    int tex_height = 0;
    int tex_comp = 0;
    auto tex_data = std::unique_ptr<uint8_t[]>(stbi_load("donuts.jpg", &tex_width, &tex_height, &tex_comp, 4));

    vk::ImageCreateInfo tex_info;
    tex_info.imageType = vk::ImageType::e2D;
    tex_info.format = vk::Format::eR8G8B8A8Unorm;
    tex_info.extent = vk::Extent3D(tex_width, tex_height, 1);
    tex_info.mipLevels = 1;
    tex_info.arrayLayers = 1;
    tex_info.samples = vk::SampleCountFlagBits::e1;
    tex_info.tiling = vk::ImageTiling::eLinear;
    tex_info.usage = vk::ImageUsageFlagBits::eSampled;
    tex_info.initialLayout = vk::ImageLayout::ePreinitialized;
    vk::UniqueImage tex = device->createImageUnique(tex_info);
    vk::MemoryRequirements tex_mem_req = device->getImageMemoryRequirements(*tex);
    uint32_t tex_mem_idx = find_memory(tex_mem_req,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    vk::UniqueDeviceMemory tex_mem = device->allocateMemoryUnique(vk::MemoryAllocateInfo(tex_mem_req.size, tex_mem_idx));
    vk::SubresourceLayout tex_layout = device->getImageSubresourceLayout(*tex, vk::ImageSubresource(vk::ImageAspectFlagBits::eColor, 0, 0));
    device->bindImageMemory(*tex, *tex_mem, 0);
    if (void* ptr = device->mapMemory(*tex_mem, 0, VK_WHOLE_SIZE))
    {
        for (int row = 0; row < tex_height; row++)
            std::copy_n(
                tex_data.get() + row * tex_width * 4, 
                tex_width * 4, 
                reinterpret_cast<uint8_t*>(ptr) + row * tex_layout.rowPitch);
        device->unmapMemory(*tex_mem);
    }

    vk::ImageViewCreateInfo tex_view_info;
    tex_view_info.image = *tex;
    tex_view_info.viewType = vk::ImageViewType::e2D;
    tex_view_info.format = tex_info.format;
    tex_view_info.components = vk::ComponentMapping();
    tex_view_info.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    tex_view_info.subresourceRange.baseMipLevel = 0;
    tex_view_info.subresourceRange.levelCount = 1;
    tex_view_info.subresourceRange.baseArrayLayer = 0;
    tex_view_info.subresourceRange.layerCount = 1;
    vk::UniqueImageView tex_view = device->createImageViewUnique(tex_view_info);
    
    vk::SamplerCreateInfo sampler_info;
    vk::UniqueSampler sampler = device->createSamplerUnique(sampler_info);

    vk::CommandBufferAllocateInfo cmd_tex_info;
    cmd_tex_info.commandPool = *cmd_pool;
    cmd_tex_info.level = vk::CommandBufferLevel::ePrimary;
    cmd_tex_info.commandBufferCount = 1;
    auto cmd_tex = std::move(device->allocateCommandBuffersUnique(cmd_tex_info).front());
    cmd_tex->begin(vk::CommandBufferBeginInfo({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit }));
    {
        vk::ImageMemoryBarrier barrier;
        barrier.image = *tex;
        barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        barrier.srcAccessMask = {};
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
        barrier.oldLayout = vk::ImageLayout::ePreinitialized;
        barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        cmd_tex->pipelineBarrier(
            vk::PipelineStageFlagBits::eAllCommands,
            vk::PipelineStageFlagBits::eFragmentShader,
            vk::DependencyFlagBits::eByRegion,
            nullptr, nullptr, barrier);
    }
    cmd_tex->end();

    vk::CommandBuffer cmd_tex_buf = *cmd_tex;
    vk::SubmitInfo tex_submit_info;
    tex_submit_info.commandBufferCount = 1;
    tex_submit_info.pCommandBuffers = &cmd_tex_buf;
    q.submit(tex_submit_info, nullptr);
    q.waitIdle();


    // Create quad buffers
    
    std::vector<uint32_t> quad_indices{ 0, 1, 2, 0, 2, 3 };
    std::vector<vertex_t> quad_vertives{
        { glm::vec3(-1,  1, 0), glm::vec3(), glm::vec2(0, 0) },
        { glm::vec3(-1, -1, 0), glm::vec3(), glm::vec2(0, 1) },
        { glm::vec3( 1, -1, 0), glm::vec3(), glm::vec2(1, 1) },
        { glm::vec3( 1,  1, 0), glm::vec3(), glm::vec2(1, 0) },
    };
    vk::BufferCreateInfo quad_buffers_info;
    quad_buffers_info.flags = {};
    quad_buffers_info.size = quad_indices.size() * sizeof(uint32_t) + 
        quad_vertives.size() * sizeof(vertex_t) + 
        sizeof(uniform_buffers_t) + 0x100;
    quad_buffers_info.usage = vk::BufferUsageFlagBits::eIndexBuffer | 
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eUniformBuffer;
    vk::UniqueBuffer quad_buffers = device->createBufferUnique(quad_buffers_info);

    vk::MemoryRequirements quad_buffers_req = device->getBufferMemoryRequirements(*quad_buffers);
    uint32_t quad_buffers_mem_idx = find_memory(quad_buffers_req,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    vk::UniqueDeviceMemory quad_buffers_mem = device->allocateMemoryUnique(vk::MemoryAllocateInfo(quad_buffers_req.size, quad_buffers_mem_idx));
    device->bindBufferMemory(*quad_buffers, *quad_buffers_mem, 0);

    vk::DeviceSize uniform_offset = 0;
    if (void* ptr = device->mapMemory(*quad_buffers_mem, 0, VK_WHOLE_SIZE))
    {
        uint32_t* idx = reinterpret_cast<uint32_t*>(ptr);
        std::copy(quad_indices.begin(), quad_indices.end(), idx);
        vertex_t* vtx = reinterpret_cast<vertex_t*>(idx + quad_indices.size());
        std::copy(quad_vertives.begin(), quad_vertives.end(), vtx);
        void* uniform_unaligned = reinterpret_cast<void*>(vtx + quad_vertives.size());
        std::align(0x100, sizeof(uniform_buffers_t), uniform_unaligned, quad_buffers_req.size);
        uniform_buffers_t* uniform = reinterpret_cast<uniform_buffers_t*>(uniform_unaligned);
        uniform_offset = reinterpret_cast<uint8_t*>(uniform_unaligned) - reinterpret_cast<uint8_t*>(ptr);
        uniform->model = glm::scale(glm::vec3(0.5f));
        uniform->proj = glm::ortho(-1.f, 1.f, -1.f, 1.f, -1.f, 1.f);
        uniform->view = glm::identity<glm::mat4>();
        device->unmapMemory(*quad_buffers_mem);
    }

    // Pipeline layout

    std::vector<vk::DescriptorSetLayoutBinding> descrset_layout_bindings{
        vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex),
        vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment),
        //vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment),
    };
    vk::DescriptorSetLayoutCreateInfo descrset_layout_info;
    descrset_layout_info.bindingCount = (uint32_t)descrset_layout_bindings.size();
    descrset_layout_info.pBindings = descrset_layout_bindings.data();
    vk::UniqueDescriptorSetLayout descrset_layout = device->createDescriptorSetLayoutUnique(descrset_layout_info);

    vk::PipelineLayoutCreateInfo pipeline_layout_info;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &descrset_layout.get();
    pipeline_layout_info.pushConstantRangeCount = 0;
    vk::UniquePipelineLayout pipeline_layout = device->createPipelineLayoutUnique(pipeline_layout_info);


    // Create depth buffer

    vk::ImageCreateInfo depth_info;
    depth_info.imageType = vk::ImageType::e2D;
    depth_info.format = vk::Format::eD16Unorm;
    depth_info.extent = vk::Extent3D(surface_caps.currentExtent, 1);
    depth_info.mipLevels = 1;
    depth_info.arrayLayers = 1;
    depth_info.samples = vk::SampleCountFlagBits::e1;
    depth_info.tiling = vk::ImageTiling::eOptimal; // check support
    depth_info.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eInputAttachment;
    depth_info.initialLayout = vk::ImageLayout::eUndefined;
    vk::UniqueImage depth = device->createImageUnique(depth_info);
    vk::MemoryRequirements depth_mem_req = device->getImageMemoryRequirements(*depth);
    uint32_t depth_mem_idx = find_memory(depth_mem_req, vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk::UniqueDeviceMemory depth_mem = device->allocateMemoryUnique({ depth_mem_req.size, depth_mem_idx });
    device->bindImageMemory(*depth, *depth_mem, 0);
    vk::ImageViewCreateInfo depth_view_info;
    depth_view_info.image = *depth;
    depth_view_info.viewType = vk::ImageViewType::e2D;
    depth_view_info.format = depth_info.format;
    depth_view_info.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;
    depth_view_info.subresourceRange.baseArrayLayer = 0;
    depth_view_info.subresourceRange.baseMipLevel = 0;
    depth_view_info.subresourceRange.layerCount = 1;
    depth_view_info.subresourceRange.levelCount = 1;
    vk::UniqueImageView depth_view = device->createImageViewUnique(depth_view_info);


    // Create Renderpass

    std::array<vk::AttachmentDescription, 2> renderpass_attachments;
    // color buffer
    renderpass_attachments[0].format = swapchain_info.imageFormat;
    renderpass_attachments[0].samples = vk::SampleCountFlagBits::e1;
    renderpass_attachments[0].loadOp = vk::AttachmentLoadOp::eClear;
    renderpass_attachments[0].storeOp = vk::AttachmentStoreOp::eStore;
    renderpass_attachments[0].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    renderpass_attachments[0].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    renderpass_attachments[0].initialLayout = vk::ImageLayout::eColorAttachmentOptimal;
    renderpass_attachments[0].finalLayout = vk::ImageLayout::ePresentSrcKHR;
    // depth buffer
    renderpass_attachments[1].format = depth_info.format;
    renderpass_attachments[1].samples = vk::SampleCountFlagBits::e1;
    renderpass_attachments[1].loadOp = vk::AttachmentLoadOp::eClear;
    renderpass_attachments[1].storeOp = vk::AttachmentStoreOp::eDontCare;
    renderpass_attachments[1].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    renderpass_attachments[1].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    renderpass_attachments[1].initialLayout = vk::ImageLayout::eUndefined;
    renderpass_attachments[1].finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

    std::array<vk::SubpassDescription, 1> renderpass_subpasses;

    vk::AttachmentReference renderpass_references_first;
    renderpass_references_first.attachment = 0;
    renderpass_references_first.layout = vk::ImageLayout::eColorAttachmentOptimal;
    vk::AttachmentReference renderpass_references_first_depth;
    renderpass_references_first_depth.attachment = 1; // depth
    renderpass_references_first_depth.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

    renderpass_subpasses[0].pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    renderpass_subpasses[0].colorAttachmentCount = 1;
    renderpass_subpasses[0].pColorAttachments = &renderpass_references_first;
    renderpass_subpasses[0].pDepthStencilAttachment = &renderpass_references_first_depth;

    vk::RenderPassCreateInfo renderpass_info;
    renderpass_info.attachmentCount = (uint32_t)renderpass_attachments.size();
    renderpass_info.pAttachments = renderpass_attachments.data();
    renderpass_info.subpassCount = (uint32_t)renderpass_subpasses.size();
    renderpass_info.pSubpasses = renderpass_subpasses.data();
    renderpass_info.dependencyCount = 0;
    renderpass_info.pDependencies = nullptr;
    vk::UniqueRenderPass renderpass = device->createRenderPassUnique(renderpass_info);


    // Create pipeline

    vk::UniqueShaderModule module_triangle_vert = load_shader_module("shaders/quad.vert.spv");
    vk::UniqueShaderModule module_triangle_frag = load_shader_module("shaders/quad.frag.spv");
    std::vector<vk::PipelineShaderStageCreateInfo> pipeline_stages{
        vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex, *module_triangle_vert, "main"),
        vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment, *module_triangle_frag, "main"),
    };

    std::vector<vk::VertexInputBindingDescription> pipeline_input_bindings{
        vk::VertexInputBindingDescription(0, sizeof(vertex_t), vk::VertexInputRate::eVertex),
    };
    std::vector<vk::VertexInputAttributeDescription> pipeline_input_attributes{
        vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(vertex_t, pos)),
        vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32Sfloat, offsetof(vertex_t, uvs)),
    };
    vk::PipelineVertexInputStateCreateInfo pipeline_input;
    pipeline_input.vertexBindingDescriptionCount = (uint32_t)pipeline_input_bindings.size();
    pipeline_input.pVertexBindingDescriptions = pipeline_input_bindings.data();
    pipeline_input.vertexAttributeDescriptionCount = (uint32_t)pipeline_input_attributes.size();
    pipeline_input.pVertexAttributeDescriptions = pipeline_input_attributes.data();

    vk::PipelineInputAssemblyStateCreateInfo pipeline_assembly;
    pipeline_assembly.topology = vk::PrimitiveTopology::eTriangleList;
    pipeline_assembly.primitiveRestartEnable = false;

    vk::Viewport pipeline_viewport_vp{ 0, 0, (float)surface_caps.currentExtent.width, (float)surface_caps.currentExtent.height, 0.f, 1.f };
    vk::Rect2D pipeline_viewport_scissor{ {}, surface_caps.currentExtent };
    vk::PipelineViewportStateCreateInfo pipeline_viewport;
    pipeline_viewport.viewportCount = 1;
    pipeline_viewport.pViewports = &pipeline_viewport_vp;
    pipeline_viewport.scissorCount = 1;
    pipeline_viewport.pScissors = &pipeline_viewport_scissor;

    vk::PipelineRasterizationStateCreateInfo pipeline_raster;
    pipeline_raster.depthClampEnable = false;
    pipeline_raster.rasterizerDiscardEnable = false;
    pipeline_raster.polygonMode = vk::PolygonMode::eFill;
    pipeline_raster.cullMode = vk::CullModeFlagBits::eBack;
    pipeline_raster.frontFace = vk::FrontFace::eClockwise;
    pipeline_raster.depthBiasEnable = false;
    pipeline_raster.lineWidth = 1.f;

    vk::PipelineMultisampleStateCreateInfo pipeline_multisample;
    pipeline_multisample.rasterizationSamples = vk::SampleCountFlagBits::e1;
    pipeline_multisample.sampleShadingEnable = false;

    vk::PipelineDepthStencilStateCreateInfo pipeline_depth;
    pipeline_depth.depthTestEnable = true;
    pipeline_depth.depthWriteEnable = true;
    pipeline_depth.depthCompareOp = vk::CompareOp::eLess;
    pipeline_depth.depthBoundsTestEnable = false;
    pipeline_depth.stencilTestEnable = false;

    vk::ColorComponentFlags color_mask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    std::array<vk::PipelineColorBlendAttachmentState, 1> pipeline_blend_attachments;
    pipeline_blend_attachments[0].blendEnable = false;
    pipeline_blend_attachments[0].colorWriteMask = color_mask;
    vk::PipelineColorBlendStateCreateInfo pipeline_blend;
    pipeline_blend.logicOpEnable = false;
    pipeline_blend.attachmentCount = (uint32_t)pipeline_blend_attachments.size();
    pipeline_blend.pAttachments = pipeline_blend_attachments.data();

    vk::PipelineDynamicStateCreateInfo pipeline_dynamic;
    pipeline_dynamic.dynamicStateCount = 0;

    vk::GraphicsPipelineCreateInfo pipeline_info;
    pipeline_info.stageCount = (uint32_t)pipeline_stages.size();
    pipeline_info.pStages = pipeline_stages.data();
    pipeline_info.pVertexInputState = &pipeline_input;
    pipeline_info.pInputAssemblyState = &pipeline_assembly;
    pipeline_info.pTessellationState = nullptr;
    pipeline_info.pViewportState = &pipeline_viewport;
    pipeline_info.pRasterizationState = &pipeline_raster;
    pipeline_info.pMultisampleState = &pipeline_multisample;
    pipeline_info.pDepthStencilState = &pipeline_depth;
    pipeline_info.pColorBlendState = &pipeline_blend;
    pipeline_info.pDynamicState = &pipeline_dynamic;
    pipeline_info.layout = *pipeline_layout;
    pipeline_info.renderPass = *renderpass;
    pipeline_info.subpass = 0;
    vk::UniquePipeline pipeline = device->createGraphicsPipelineUnique(nullptr, pipeline_info).value;

    // Load scene
    vk::DeviceSize scene_tot_vertex_size = 0;
    std::vector<mesh_t> meshes;
    Assimp::Importer imp;
    const aiScene* scene = imp.ReadFile("data\\buildings.fbx", aiProcessPreset_TargetRealtime_Fast);
    for (UINT mesh_index = 0; mesh_index < scene->mNumMeshes; mesh_index++)
    {
        mesh_t& m = meshes.emplace_back();
        aiMesh* mesh = scene->mMeshes[mesh_index];
        m.vertices = std::vector<vertex_t>(mesh->mNumVertices);
        for (UINT vertex_index = 0; vertex_index < mesh->mNumVertices; vertex_index++)
        {
            m.vertices[vertex_index].pos.x = mesh->mVertices[vertex_index].x;
            m.vertices[vertex_index].pos.y = mesh->mVertices[vertex_index].y;
            m.vertices[vertex_index].pos.z = mesh->mVertices[vertex_index].z;
            m.vertices[vertex_index].nor.x = -mesh->mNormals[vertex_index].x;
            m.vertices[vertex_index].nor.y = -mesh->mNormals[vertex_index].y;
            m.vertices[vertex_index].nor.z = -mesh->mNormals[vertex_index].z;
            m.vertices[vertex_index].uvs.x = mesh->mTextureCoords[0][vertex_index].x;
            m.vertices[vertex_index].uvs.y = mesh->mTextureCoords[0][vertex_index].y;
        }
        m.indices = std::vector<uint32_t>(mesh->mNumFaces * 3);
        for (UINT face_index = 0; face_index < mesh->mNumFaces; face_index++)
        {
            aiFace& face = mesh->mFaces[face_index];
            for (UINT index = 0; index < face.mNumIndices; index++)
                m.indices[face_index * 3 + index] = face.mIndices[index];
        }
        scene_tot_vertex_size += mesh->mNumVertices * sizeof(vertex_t);
        scene_tot_vertex_size += mesh->mNumFaces * 3 * sizeof(uint32_t);
    }
    aiNode* root = scene->mRootNode;
    std::vector<glm::mat4> matrices(root->mNumChildren);
    for (UINT child_index = 0; child_index < root->mNumChildren; child_index++)
    {
        aiNode* child = root->mChildren[child_index];
        if (child->mNumMeshes == 1)
        {
            UINT mesh_index = child->mMeshes[0];
            matrices[mesh_index] = glm::transpose(glm::make_mat4(reinterpret_cast<float*>(&child->mTransformation)));
        }
    }

    scene_tot_vertex_size = ((scene_tot_vertex_size - 1) & (~0xFF)) + 0x100;

    vk::BufferCreateInfo scene_buffer_info;
    scene_buffer_info.size = scene_tot_vertex_size + meshes.size() * sizeof(uniform_buffers_t);
    scene_buffer_info.usage = vk::BufferUsageFlagBits::eVertexBuffer | 
        vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eUniformBuffer;
    vk::UniqueBuffer scene_buffer = device->createBufferUnique(scene_buffer_info);

    vk::MemoryRequirements scene_buffer_req = device->getBufferMemoryRequirements(*scene_buffer);
    uint32_t scene_buffer_mem_idx = find_memory(scene_buffer_req,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    vk::UniqueDeviceMemory scene_buffer_mem = device->allocateMemoryUnique(vk::MemoryAllocateInfo(scene_buffer_req.size, scene_buffer_mem_idx));
    device->bindBufferMemory(*scene_buffer, *scene_buffer_mem, 0);

    for (UINT mesh_index = 0; mesh_index < meshes.size(); mesh_index++)
    {
        meshes[mesh_index].model = matrices[mesh_index];
    }

    if (void* ptr = device->mapMemory(*scene_buffer_mem, 0, VK_WHOLE_SIZE))
    {
        uint32_t* idx_2 = reinterpret_cast<uint32_t*>(ptr);
        uint32_t* idx = reinterpret_cast<uint32_t*>(ptr);
        for (auto& m : meshes)
        {
            std::copy(m.indices.begin(), m.indices.end(), idx);
            m.off_index = reinterpret_cast<uint8_t*>(idx) - reinterpret_cast<uint8_t*>(ptr);
            idx += m.indices.size();
        }
        vertex_t* vtx = reinterpret_cast<vertex_t*>(idx);
        for (auto& m : meshes)
        {
            std::copy(m.vertices.begin(), m.vertices.end(), vtx);
            m.off_vertex = reinterpret_cast<uint8_t*>(vtx) - reinterpret_cast<uint8_t*>(ptr);
            vtx += m.vertices.size();
        }
        uniform_buffers_t* uniform = reinterpret_cast<uniform_buffers_t*>(reinterpret_cast<uint8_t*>(ptr) + scene_tot_vertex_size);
        for (auto& m : meshes)
        {
            m.off_uniform = reinterpret_cast<uint8_t*>(uniform) - reinterpret_cast<uint8_t*>(ptr);
            uniform->model = glm::scale(glm::vec3(0.01f)) * m.model;
            uniform->proj = glm::perspective(glm::radians(85.f), 
                (float)surface_caps.currentExtent.width / (float)surface_caps.currentExtent.height, 1.0f, 1000.f);
            uniform->view = glm::lookAt(glm::vec3(0, 3, 30), glm::vec3(0), glm::vec3(0, -1, 0));
            uniform++;
        }
        device->unmapMemory(*scene_buffer_mem);
    }

    std::vector<vk::DescriptorPoolSize> descrpool_sizes{
        vk::DescriptorPoolSize{ vk::DescriptorType::eUniformBuffer, (uint32_t)(1 + meshes.size()) },
        vk::DescriptorPoolSize{ vk::DescriptorType::eSampledImage, (uint32_t)(1 + meshes.size()) },
    };
    vk::UniqueDescriptorPool descrpool = device->createDescriptorPoolUnique({ {}, 2 + (uint32_t)meshes.size() * 2,
        (uint32_t)descrpool_sizes.size(), descrpool_sizes.data() });

    std::vector<vk::DescriptorSetLayout> scene_descr_sets_layouts(meshes.size(), *descrset_layout);
    std::vector<vk::UniqueDescriptorSet> scene_descr_sets = device->allocateDescriptorSetsUnique({ *descrpool,
        (uint32_t)scene_descr_sets_layouts.size(), scene_descr_sets_layouts.data() });

    for (UINT mesh_index = 0; mesh_index < meshes.size(); mesh_index++)
    {
        // model, view, proj
        vk::DescriptorBufferInfo descr_sets_buffer;
        descr_sets_buffer.buffer = *scene_buffer;
        descr_sets_buffer.offset = meshes[mesh_index].off_uniform;
        descr_sets_buffer.range = uniform_buffers_t::mvp_size;

        vk::DescriptorImageInfo descr_sets_tex;
        descr_sets_tex.sampler = *sampler;
        descr_sets_tex.imageView = *tex_view;
        descr_sets_tex.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

        std::vector<vk::WriteDescriptorSet> descr_sets_write{
            vk::WriteDescriptorSet(*scene_descr_sets[mesh_index], 0, 0, 1,
                vk::DescriptorType::eUniformBuffer, nullptr, &descr_sets_buffer, nullptr),
            vk::WriteDescriptorSet(*scene_descr_sets[mesh_index], 1, 0, 1,
                vk::DescriptorType::eCombinedImageSampler, &descr_sets_tex, nullptr, nullptr),
        };

        device->updateDescriptorSets(descr_sets_write, nullptr);
    }

    // Create descriptorsets

    std::vector<vk::DescriptorSetLayout> descr_sets_layouts(1, *descrset_layout);
    std::vector<vk::UniqueDescriptorSet> descr_sets = device->allocateDescriptorSetsUnique({ *descrpool,
        (uint32_t)descr_sets_layouts.size(), descr_sets_layouts.data() });
    
    // model, view, proj
    vk::DescriptorBufferInfo descr_sets_buffer;
    descr_sets_buffer.buffer = *quad_buffers;
    descr_sets_buffer.offset = uniform_offset;
    descr_sets_buffer.range = uniform_buffers_t::mvp_size;
    
    vk::DescriptorImageInfo descr_sets_tex;
    descr_sets_tex.sampler = *sampler;
    descr_sets_tex.imageView = *tex_view;
    descr_sets_tex.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    std::vector<vk::WriteDescriptorSet> descr_sets_write{
        vk::WriteDescriptorSet(*descr_sets[0], 0, 0, 1,
            vk::DescriptorType::eUniformBuffer, nullptr, &descr_sets_buffer, nullptr),
        vk::WriteDescriptorSet(*descr_sets[0], 1, 0, 1,
            vk::DescriptorType::eCombinedImageSampler, &descr_sets_tex, nullptr, nullptr),
    };

    device->updateDescriptorSets(descr_sets_write, nullptr);

    // Record drawing command

    std::vector<vk::Image> swapchain_images = device->getSwapchainImagesKHR(*swapchain);

    vk::CommandBufferAllocateInfo cmd_info;
    cmd_info.commandPool = *cmd_pool;
    cmd_info.level = vk::CommandBufferLevel::ePrimary;
    cmd_info.commandBufferCount = (uint32_t)swapchain_images.size();
    std::vector<vk::UniqueCommandBuffer> cmds = device->allocateCommandBuffersUnique(cmd_info);
    std::vector<vk::CommandBuffer> cmds_naked(cmds.size());
    std::vector<vk::UniqueFramebuffer> framebuffers(swapchain_images.size());
    std::vector<vk::UniqueImageView> swapchain_views(swapchain_images.size());

    for (int image_index = 0; image_index < swapchain_images.size(); image_index++)
    {
        cmds[image_index]->begin(vk::CommandBufferBeginInfo({}));
        {
            /*
            vk::ImageMemoryBarrier barrier;
            barrier.image = swapchain_images[image_index];
            barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount = 1;

            barrier.srcAccessMask = {};
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.oldLayout = vk::ImageLayout::eUndefined;
            barrier.newLayout = vk::ImageLayout::eTransferDstOptimal;
            cmds[image_index]->pipelineBarrier(
                vk::PipelineStageFlagBits::eAllCommands,
                vk::PipelineStageFlagBits::eTransfer,
                vk::DependencyFlagBits::eByRegion,
                nullptr, nullptr, barrier);

            std::array<float, 4> color_red = { 1.f, 0.f, 0.f, 1.f };
            cmds[image_index]->clearColorImage(
                swapchain_images[image_index],
                vk::ImageLayout::eTransferDstOptimal,
                vk::ClearColorValue(color_red),
                barrier.subresourceRange);

            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eMemoryRead;
            barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
            barrier.newLayout = vk::ImageLayout::ePresentSrcKHR;
            cmds[image_index]->pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eBottomOfPipe,
                vk::DependencyFlagBits::eByRegion,
                nullptr, nullptr, barrier);
            */
            vk::ImageMemoryBarrier barrier;
            barrier.image = swapchain_images[image_index];
            barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount = 1;
            barrier.srcAccessMask = {};
            barrier.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
            barrier.oldLayout = vk::ImageLayout::eUndefined;
            barrier.newLayout = vk::ImageLayout::eColorAttachmentOptimal;
            cmds[image_index]->pipelineBarrier(
                vk::PipelineStageFlagBits::eAllCommands,
                vk::PipelineStageFlagBits::eColorAttachmentOutput,
                vk::DependencyFlagBits::eByRegion,
                nullptr, nullptr, barrier);

            vk::ImageViewCreateInfo view_info;
            view_info.image = swapchain_images[image_index];
            view_info.viewType = vk::ImageViewType::e2D;
            view_info.format = swapchain_info.imageFormat;
            view_info.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
            view_info.subresourceRange.baseArrayLayer = 0;
            view_info.subresourceRange.baseMipLevel = 0;
            view_info.subresourceRange.layerCount = 1;
            view_info.subresourceRange.levelCount = 1;
            swapchain_views[image_index] = device->createImageViewUnique(view_info);
            std::vector<vk::ImageView> framebuffer_attachments{
                *swapchain_views[image_index],
                *depth_view,
            };
            vk::FramebufferCreateInfo framebuffer_info;
            framebuffer_info.renderPass = *renderpass;
            framebuffer_info.attachmentCount = (uint32_t)framebuffer_attachments.size();
            framebuffer_info.pAttachments = framebuffer_attachments.data();
            framebuffer_info.width = surface_caps.currentExtent.width;
            framebuffer_info.height = surface_caps.currentExtent.height;
            framebuffer_info.layers = 1;
            framebuffers[image_index] = device->createFramebufferUnique(framebuffer_info);

            std::array<float, 4> color_red = { 1.f, 0.f, 0.f, 1.f };
            std::array<vk::ClearValue, 2> clear_values{
                vk::ClearColorValue(color_red),
                vk::ClearDepthStencilValue(1.f, 0),
            };
            vk::RenderPassBeginInfo renderpass_begin_info;
            renderpass_begin_info.renderPass = *renderpass;
            renderpass_begin_info.framebuffer = *framebuffers[image_index];
            renderpass_begin_info.renderArea.extent = surface_caps.currentExtent;
            renderpass_begin_info.renderArea.offset = vk::Offset2D{};
            renderpass_begin_info.clearValueCount = (uint32_t)clear_values.size();
            renderpass_begin_info.pClearValues = clear_values.data();
            cmds[image_index]->beginRenderPass(renderpass_begin_info, vk::SubpassContents::eInline);
            {
                cmds[image_index]->bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);
                
                //cmds[image_index]->bindVertexBuffers(0, *quad_buffers, { quad_indices.size() * sizeof(uint32_t) });
                //cmds[image_index]->bindIndexBuffer(*quad_buffers, 0, vk::IndexType::eUint32);
                //cmds[image_index]->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipeline_layout, 0, *descr_sets[0], {});
                //cmds[image_index]->drawIndexed((uint32_t)quad_indices.size(), 1, 0, 0, 0);

                for (int mesh_index = 0; mesh_index < meshes.size(); mesh_index++)
                {
                    cmds[image_index]->bindVertexBuffers(0, *scene_buffer, { meshes[mesh_index].off_vertex });
                    cmds[image_index]->bindIndexBuffer(*scene_buffer, meshes[mesh_index].off_index, vk::IndexType::eUint32);
                    cmds[image_index]->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipeline_layout, 0, *scene_descr_sets[mesh_index], {});
                    cmds[image_index]->drawIndexed((uint32_t)meshes[mesh_index].indices.size(), 1, 0, 0, 0);
                }
            }
            cmds[image_index]->endRenderPass();
        }
        cmds[image_index]->end();
        cmds_naked[image_index] = *cmds[image_index];
    }

    vk::UniqueFence fence = device->createFenceUnique({});
    auto start = std::chrono::high_resolution_clock::now();
    while (true)
    {
        MSG msg;
        if (PeekMessage(&msg, hWnd, 0, 0, PM_REMOVE) > 0)
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration_cast<std::chrono::duration<float>>(now - start).count();
        start = now;

        auto next_image = device->acquireNextImageKHR(*swapchain, UINT64_MAX, nullptr, *fence);

        if (next_image.result == vk::Result::eSuccess)
        {
            if (void* ptr = device->mapMemory(*scene_buffer_mem, scene_tot_vertex_size, meshes.size() * sizeof(uniform_buffers_t)))
            {
                static float angle = 0.f;
                angle += 10 * dt;
                glm::mat4 view = glm::lookAt(glm::vec3(glm::cos(glm::radians(angle)), 1, glm::sin(glm::radians(angle))) * 2.f, glm::vec3(0), glm::vec3(0, -1, 0));
                glm::mat4 proj = glm::perspective(glm::radians(85.f),
                    (float)surface_caps.currentExtent.width / (float)surface_caps.currentExtent.height, 0.01f, 100.f);
                uniform_buffers_t* uniform = reinterpret_cast<uniform_buffers_t*>(reinterpret_cast<uint8_t*>(ptr));
                for (auto& m : meshes)
                {
                    uniform->model = glm::scale(glm::vec3(0.01f)) * m.model;
                    uniform->proj = proj;
                    uniform->view = view;
                    uniform++;
                }
                device->unmapMemory(*scene_buffer_mem);
            }

            vk::SubmitInfo submit_info;
            submit_info.commandBufferCount = (uint32_t)cmds_naked.size();
            submit_info.pCommandBuffers = cmds_naked.data();
            q.submit(submit_info, nullptr);
            q.waitIdle();
            
            vk::Result present_result;
            vk::SwapchainKHR swapchains[] = { *swapchain };
            vk::PresentInfoKHR present_info;
            present_info.swapchainCount = 1;
            present_info.pSwapchains = swapchains;
            present_info.pImageIndices = &next_image.value;
            present_info.pResults = &present_result;
            q.presentKHR(present_info);
            device->waitForFences(*fence, true, UINT64_MAX);
            device->resetFences(*fence);
        }

    }

    
    return EXIT_SUCCESS;
}
