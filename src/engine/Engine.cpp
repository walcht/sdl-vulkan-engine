#include "Engine.hpp"
#include "Utils.hpp"
#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>
#include <map>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include <stb/stb_image.h>

namespace vkengine {
VkEngine::VkEngine(const std::string &title,
                   const std::string &app_identifier) {
  m_Title = title;
  m_AppIdentifier = app_identifier;

  init_window();
  init_vkinstance();
  init_surface();
  init_physical_device();
  init_logical_device();
  init_swap_chain();
  init_swap_image_views();
  init_descriptor_set_layouts();
  init_graphics_pipeline();
  create_command_pool();
  create_texture_image("textures/test.png");
  create_texture_image_view();
  create_texture_sampler();
  create_index_buffer();
  create_vertex_buffer();
  create_uniform_buffers();
  create_descriptor_pool();
  create_descriptor_sets();
  create_command_buffers();
  create_sync_objects();
}

VkEngine::~VkEngine() {}

void VkEngine::init_window() {
  /* optional: set initial app metadata */
  SDL_SetAppMetadata(m_Title.c_str(), "1.0", m_AppIdentifier.c_str());

  /* initialize the video subsystem */
  if (!SDL_Init(SDL_INIT_VIDEO)) {
    throw std::runtime_error(
        std::format("SDL_Init failed. Reason: {}", SDL_GetError()));
  }

  if ((m_Window = SDL_CreateWindow(m_Title.c_str(), 640, 480,
                                   SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE)) ==
      nullptr) {
    throw std::runtime_error(
        std::format("SDL_CreateWindow failed. Reason: {}", SDL_GetError()));
  }
}

void VkEngine::init_vkinstance() {
  /* get required Vulkan extensions by SDL3 - these are REQUIRED extensions, so
   * we fail if any of them is not available */
  uint32_t sdl_extensions_count = 0;
  const char *const *sdl_extensions =
      SDL_Vulkan_GetInstanceExtensions(&sdl_extensions_count);
  std::vector<const char *> required_exts(sdl_extensions_count);
  required_exts.assign(sdl_extensions, sdl_extensions + sdl_extensions_count);

  vk::raii::Context m_Context;

  /* check if required extensions are supported by the provided Vulkan */
  auto available_props = m_Context.enumerateInstanceExtensionProperties();
  for (auto const &required_ext : required_exts) {
    if (std::ranges::none_of(available_props, [=](auto const &extension_prop) {
          return strcmp(required_ext, extension_prop.extensionName) == 0;
        })) {
      throw std::runtime_error(
          "Some of the SDL3 required Vulkan extensions are not available");
    }
  }

  vk::ApplicationInfo app_info{
      .pApplicationName = m_Title.c_str(),
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .engineVersion = VK_MAKE_VERSION(1, 0, 0),
      .apiVersion = vk::ApiVersion14,
  };

  /* validation layers setup */
#ifndef NDEBUG
  /* add other validation layers here - this is only used for Debug builds */
  std::vector<char const *> val_layers{"VK_LAYER_KHRONOS_validation"};

  auto available_layers = m_Context.enumerateInstanceLayerProperties();
  /* if any_of the required validation layers does NOT exist in the vector of
   * availalbe layers => exit with failure */
  if (std::ranges::any_of(val_layers, [&](auto const &val_layer) {
        return std::ranges::none_of(available_layers, [&](auto const &l) {
          return strcmp(l.layerName, val_layer) == 0;
        });
      })) {
    throw std::runtime_error(
        "one or more of the required validation layers are not available");
  }
#endif

  vk::InstanceCreateInfo create_info{
      .pApplicationInfo = &app_info,
#ifndef NDEBUG
      .enabledLayerCount = static_cast<uint32_t>(val_layers.size()),
      .ppEnabledLayerNames = val_layers.data(),
#endif
      .enabledExtensionCount = static_cast<uint32_t>(required_exts.size()),
      .ppEnabledExtensionNames = required_exts.data(),
  };
  m_VkInstance = vk::raii::Instance(m_Context, create_info);
};

void VkEngine::init_surface() {
  VkSurfaceKHR _tmpVkSurface; /* because passing &m_VkSurface does not work */
  if (!SDL_Vulkan_CreateSurface(m_Window, *m_VkInstance, nullptr,
                                &_tmpVkSurface)) {
    throw std::runtime_error(std::format(
        "failed to created SDL Vulkan surface. Reason: {}", SDL_GetError()));
  }
  m_VkSurface = vk::raii::SurfaceKHR(m_VkInstance, _tmpVkSurface, nullptr);
}

void VkEngine::init_physical_device() {
  /* get available physical devices with Vulkan support */
  auto devices = m_VkInstance.enumeratePhysicalDevices();
  if (devices.empty()) {
    throw std::runtime_error("no devices available with Vulkan support");
  }

  /* the idea here is to assign a score to each available device and pick the
   * one with the highest score - for that we need an ordered map with support
   * for identical keys (because scores can be identical) */
  struct PhysicalDeviceCandidate {
    const vk::raii::PhysicalDevice *pPhysicalDevice;
    uint32_t graphicsQueueFamilyIdx;
    uint32_t presentQueueFamilyIdx;
  };
  std::multimap<uint32_t, PhysicalDeviceCandidate> candidates;
  for (const auto &device : devices) {
    auto deviceProps = device.getProperties();
    auto deviceFeats = device.getFeatures();
    auto deviceExts = device.enumerateDeviceExtensionProperties();
    uint32_t score{0};

    /* support for Vulkan >= 1.4 is crucial */
    if (!(deviceProps.apiVersion >= VK_API_VERSION_1_4)) {
      continue;
    }

    /* geometry shader support is crucial */
    if (!deviceFeats.geometryShader) {
      continue;
    }

    /* anisotropy sampling support is crucial */
    if (!deviceFeats.samplerAnisotropy) {
      continue;
    }

    /* check if the required graphics queue families are available */
    auto queue_family_props = device.getQueueFamilyProperties();
    uint32_t graphics_queue_family_idx;
    try {
      /* get the index of the first queue that supports graphics cmds */
      graphics_queue_family_idx =
          utils::find_first_graphics_familty_queue_idx(queue_family_props);
    } catch (const std::runtime_error &e) {
      continue;
    }

    /* check if the required presentation queue family is the same as the
     * just found graphics queue family => this results in better performance */
    uint32_t present_queue_family_idx;
    if (device.getSurfaceSupportKHR(graphics_queue_family_idx, *m_VkSurface)) {
      present_queue_family_idx = graphics_queue_family_idx;
    }

    /* try to find another graphics + present family queue */
    if (present_queue_family_idx == queue_family_props.size()) {
      /* search for a queue family that supports presentation */
      for (size_t i{0}; i < queue_family_props.size(); ++i) {
        if (((queue_family_props[i].queueFlags &
              vk::QueueFlagBits::eGraphics) !=
             static_cast<vk::QueueFlags>(0)) &&
            device.getSurfaceSupportKHR(static_cast<uint32_t>(i),
                                        *m_VkSurface)) {
          present_queue_family_idx = graphics_queue_family_idx =
              static_cast<uint32_t>(i);
          break;
        }
      }
    }

    /* this means that we couldn't find a family queue that supports both
     * graphics and presentations cmds => look for a present family queue */
    if (present_queue_family_idx == queue_family_props.size()) {
      score -= 1000; /* because present != graphics queue => bad performance */
      for (size_t i{0}; i < queue_family_props.size(); ++i) {
        if (device.getSurfaceSupportKHR(static_cast<uint32_t>(i),
                                        *m_VkSurface)) {
          present_queue_family_idx = static_cast<uint32_t>(i);
          break;
        }
      }
    }

    /* this means that we couldn't find any family queue that supports window
     * presentation cmds => probably a compute-only device */
    if (present_queue_family_idx == queue_family_props.size()) {
      continue;
    }

    /* check if required extensions are available for this device */
    bool extsSatisfied = true;
    for (auto requiredExt : m_RequiredDevExts) {
      if (std::ranges::find_if(deviceExts, [=](const auto &deviceExt) {
            return strcmp(deviceExt.extensionName, requiredExt) == 0;
          }) == deviceExts.end()) {
        extsSatisfied = false;
        break;
      }
    }
    if (!extsSatisfied) {
      continue;
    }

    /* check if required features are supported for this device */
    auto features = device.template getFeatures2<
        vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features,
        vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
    bool supports_required_features =
        features.template get<vk::PhysicalDeviceVulkan11Features>()
            .shaderDrawParameters &&
        features.template get<vk::PhysicalDeviceVulkan13Features>()
            .synchronization2 &&
        features.template get<vk::PhysicalDeviceVulkan13Features>()
            .dynamicRendering &&
        features
            .template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>()
            .extendedDynamicState;
    if (!supports_required_features) {
      continue;
    }

    /* discrete GPUs are usually just better */
    if (deviceProps.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
      score += 1000;
    }

    /* add to the score the max allowed 2D image dimension - the higher the
     * better - add other desired properties here */
    score += deviceProps.limits.maxImageDimension2D;

    candidates.insert(std::make_pair(
        score, PhysicalDeviceCandidate{
                   .pPhysicalDevice = &device,
                   .graphicsQueueFamilyIdx = graphics_queue_family_idx,
                   .presentQueueFamilyIdx = present_queue_family_idx,
               }));
  }

  /* pick the highest scoring candidate */
  if (!candidates.empty()) {
    auto best = candidates.rbegin()->second;
    m_PhysicalDevice = *(best.pPhysicalDevice);
    m_GraphicsQueueFamilyIdx = best.graphicsQueueFamilyIdx;
    m_PresentQueueFamilyIdx = best.presentQueueFamilyIdx;
    return;
  }

  /* no physical device was found that supported the required features :'( */
  throw std::runtime_error("failed to find a suitable physical device!");
}

void VkEngine::init_logical_device() {
  /* graphics queue priority (only 1 graphics queue is needed) */
  float graphics_queue_priority{0.0f};

  /* describe the queues that will be requested to be created */
  std::vector<vk::DeviceQueueCreateInfo> device_queue_create_infos{{
      .queueFamilyIndex = m_GraphicsQueueFamilyIdx,
      .queueCount = 1,
      .pQueuePriorities = &graphics_queue_priority,
  }};

  // TODO:  do we need to also add another DeviceQueueCreateInfo in case
  //        graphicsQueueFamilyIdx != presentQueueFamilyIdx ?

  /* StuctureChain simply chains provided structures using their pNext fields */
  vk::StructureChain<vk::PhysicalDeviceFeatures2,
                     vk::PhysicalDeviceVulkan11Features,
                     vk::PhysicalDeviceVulkan13Features,
                     vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>
      feature_chain{
          {.features = {.samplerAnisotropy = true}},
          {.shaderDrawParameters = true},
          {.synchronization2 = true, .dynamicRendering = true},
          {.extendedDynamicState = true},
      };

  /* required device extensions (e.g., for rendering to a window, etc.) */
  std::vector<const char *> device_extensions = {
      vk::KHRSwapchainExtensionName, /* for presenting rendered images to
                                        windows. Some devices may only provide
                                        compute operations and this may not be
                                        present hence why it is a feature */
      vk::KHRSpirv14ExtensionName,
      vk::KHRSynchronization2ExtensionName,
      vk::KHRCreateRenderpass2ExtensionName,
  };

  vk::DeviceCreateInfo device_create_info{
      .pNext = &feature_chain.get<vk::PhysicalDeviceFeatures2>(),
      .queueCreateInfoCount =
          static_cast<uint32_t>(device_queue_create_infos.size()),
      .pQueueCreateInfos = device_queue_create_infos.data(),
      .enabledExtensionCount = static_cast<uint32_t>(device_extensions.size()),
      .ppEnabledExtensionNames = device_extensions.data(),
  };

  m_Device = vk::raii::Device(m_PhysicalDevice, device_create_info);
  m_GraphicsQueue = m_Device.getQueue(m_GraphicsQueueFamilyIdx, 0);
  // TODO: this will probably fail if presentQueueFamily != graphicsQueueFamily
  m_PresentQueue = m_Device.getQueue(m_PresentQueueFamilyIdx, 0);
}

void VkEngine::init_swap_chain() {
  /* make sure swapchain is uninitialized */
  if (m_SwapChain.getDevice() != nullptr || !m_SwapChainImgs.empty() ||
      !m_SwapChainImgViews.empty()) {
    throw std::runtime_error(
        "init_swap_chain should be called on an uninitialized swapchain");
  }

  vk::SurfaceCapabilitiesKHR surface_capabilities =
      m_PhysicalDevice.getSurfaceCapabilitiesKHR(*m_VkSurface);

  auto [surface_format, colorspace] = choose_swap_surface_format(
      m_PhysicalDevice.getSurfaceFormatsKHR(*m_VkSurface));

  vk::Extent2D extent = get_swap_extent(surface_capabilities);
  vk::PresentModeKHR present_mode = choose_swap_present_mode(
      m_PhysicalDevice.getSurfacePresentModesKHR(*m_VkSurface));

  vk::SwapchainCreateInfoKHR swap_chain_create_info{
      .surface = *m_VkSurface,
      .minImageCount = choose_swap_min_img_count(surface_capabilities),
      .imageFormat = surface_format,
      .imageColorSpace = colorspace,
      .imageExtent = extent,
      .imageArrayLayers = 1, /* non-stereo apps have to set this to 1 */
      .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
      .preTransform = surface_capabilities.currentTransform,
      .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
      .presentMode = present_mode,
      .clipped = vk::True,
      .oldSwapchain = VK_NULL_HANDLE, /* TODO: handle window resizing */
  };

  /* in case present queue != graphics queue, we have to specify how the
   * swapchain images will be accessed across queues and which queues will
   * access it - to keep it simple, just set it to concurrent if they are
   * different */
  std::vector<uint32_t> queue_family_indices{m_GraphicsQueueFamilyIdx,
                                             m_PresentQueueFamilyIdx};
  if (m_GraphicsQueueFamilyIdx != m_PresentQueueFamilyIdx) {
    swap_chain_create_info.imageSharingMode = vk::SharingMode::eConcurrent;
    swap_chain_create_info.queueFamilyIndexCount =
        static_cast<uint32_t>(queue_family_indices.size());
    swap_chain_create_info.pQueueFamilyIndices = queue_family_indices.data();
  } else {
    swap_chain_create_info.imageSharingMode = vk::SharingMode::eExclusive;
    swap_chain_create_info.queueFamilyIndexCount = 0;
    swap_chain_create_info.pQueueFamilyIndices = nullptr;
  }

  m_SwapChain = m_Device.createSwapchainKHR(swap_chain_create_info);
  m_SwapChainImgs = m_SwapChain.getImages();
  m_SwapChainImgFormat = surface_format;
  m_SwapChainExtent = extent;
}

void VkEngine::cleanup_swap_chain() {
  m_SwapChainImgs.clear();
  m_SwapChainImgViews.clear();

  /* or simply assign a nullptr to it (which invokes the move assignment and
   * clears up the swapchain resources) */
  m_SwapChain.clear();
}

void VkEngine::recreate_swap_chain() {
  /* wait until GPU is not doing anymore work */
  m_Device.waitIdle();

  cleanup_swap_chain();

  init_swap_chain();
  init_swap_image_views();
}

vk::SurfaceFormatKHR VkEngine::choose_swap_surface_format(
    const std::vector<vk::SurfaceFormatKHR> &surface_formats) const {
  if (surface_formats.empty()) {
    std::runtime_error("no format to choose from");
  }
  for (const vk::SurfaceFormatKHR &f : surface_formats) {
    if (f.format == vk::Format::eB8G8R8A8Srgb &&
        f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
      return f;
    }
  }
  SDL_LogWarn(SDL_LOG_CATEGORY_RENDER,
              "[warn] non-linear SRGB framebuffers are not available");
  /* as a fallback - just return the first supported format */
  return surface_formats[0];
}

vk::PresentModeKHR VkEngine::choose_swap_present_mode(
    const std::vector<vk::PresentModeKHR> &present_modes) const {
  for (const vk::PresentModeKHR &p : present_modes) {
    if (p == vk::PresentModeKHR::eMailbox) {
      return p;
    }
  }
  SDL_LogWarn(SDL_LOG_CATEGORY_RENDER,
              "[warn] mailbox (tripple buffering) presentation mode is not "
              "avaialble");
  /* return the guranteed Fifo (double buffering) presentation mode */
  return vk::PresentModeKHR::eFifo;
}

vk::Extent2D VkEngine::get_swap_extent(
    const vk::SurfaceCapabilitiesKHR &capabilities) const {
  if (capabilities.currentExtent.width !=
      std::numeric_limits<uint32_t>::max()) {
    return capabilities.currentExtent;
  }
  /* because of high DPI displays, the previously set resolution/extent may
   * not match with the framebuffer extent in PIXELS (Vulkan uses pixels unit).
   * We have to find to query the window size in PIXELS from the window wrapper
   * library (here it's SDL) */
  int width, height;
  SDL_GetWindowSizeInPixels(m_Window, &width, &height);
  return {
      std::clamp<uint32_t>(width, 0, 100),
      std::clamp<uint32_t>(height, 0, 100),
  };
}

uint32_t VkEngine::choose_swap_min_img_count(
    vk::SurfaceCapabilitiesKHR surface_capabilities) const {
  /* how many images to have in the swapchain? we can get the min number of
   * images the device supports for the swapchain. To avoid waiting for the
   * driver to make a swapchain image available, we do + 1 */
  uint32_t min_img_count = std::max(3u, surface_capabilities.minImageCount);
  if ((0 < surface_capabilities.maxImageCount) &&
      (surface_capabilities.maxImageCount < min_img_count)) {
    SDL_LogWarn(SDL_LOG_CATEGORY_RENDER,
                "[warn] could not set desirable swapchain image count "
                "(minImageCount + 1)");
    min_img_count = surface_capabilities.maxImageCount;
  }
  return min_img_count;
}

void VkEngine::init_swap_image_views() {
  if (m_SwapChain.getDevice() == nullptr || m_SwapChainImgs.empty()) {
    throw std::runtime_error(
        "cannot create swapchain image views out of uninitialized swapchain");
  }

  /* swapchain image views should be empty */
  if (!m_SwapChainImgViews.empty()) {
    throw std::runtime_error(
        "init_swap_image_views should be called on an uninisialized swapchain");
  }

  for (size_t i{0}; i < m_SwapChainImgs.size(); ++i) {
    m_SwapChainImgViews.push_back(
        create_image_view(m_SwapChainImgs[i], m_SwapChainImgFormat));
  }
}

void VkEngine::init_descriptor_set_layouts() {
  vk::DescriptorSetLayoutBinding mvp_layout_binding{
      .binding = 0,
      .descriptorType = vk::DescriptorType::eUniformBuffer,
      .descriptorCount = 1,
      .stageFlags = vk::ShaderStageFlagBits::eVertex,
      .pImmutableSamplers = nullptr,
  };
  vk::DescriptorSetLayoutCreateInfo layout_info{
      .bindingCount = 1,
      .pBindings = &mvp_layout_binding,
  };
  m_DesriptorSetLayout = vk::raii::DescriptorSetLayout(m_Device, layout_info);
}

void VkEngine::init_graphics_pipeline() {
  /* a shader module is a thin wrapper around shader code which may contain
   * multiple entry points */
  vk::raii::ShaderModule base_shader_module = create_shader_module(
      utils::read_binary(utils::get_spirv_shaders_path() / "base.spv"));

  /* specify the vertex shader entry point in the shader module */
  vk::PipelineShaderStageCreateInfo vert_shader_stage_info{
      .stage = vk::ShaderStageFlagBits::eVertex,
      .module = base_shader_module,
      .pName = "vert_main",
  };

  /* specify the fragment shader entry point in the shader module */
  vk::PipelineShaderStageCreateInfo frag_shader_stage_info{
      .stage = vk::ShaderStageFlagBits::eFragment,
      .module = base_shader_module,
      .pName = "frag_main",
  };

  vk::PipelineShaderStageCreateInfo shader_stages[] = {vert_shader_stage_info,
                                                       frag_shader_stage_info};

  auto vertex_binding_desc = Vertex::get_binding_description();
  auto vertex_attributes_desc = Vertex::get_attribute_descriptions();
  vk::PipelineVertexInputStateCreateInfo vertex_input_state_info{
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &vertex_binding_desc,
      .vertexAttributeDescriptionCount = vertex_attributes_desc.size(),
      .pVertexAttributeDescriptions = vertex_attributes_desc.data()};

  vk::PipelineInputAssemblyStateCreateInfo input_assembly_create_info{
      .topology = vk::PrimitiveTopology::eTriangleList,
  };

  std::vector<vk::DynamicState> dynamic_states = {
      vk::DynamicState::eViewport,
      vk::DynamicState::eScissor,
  };
  vk::PipelineDynamicStateCreateInfo dynamic_state_create_info{
      .dynamicStateCount = static_cast<uint32_t>(dynamic_states.size()),
      .pDynamicStates = dynamic_states.data(),
  };

  vk::PipelineViewportStateCreateInfo viewport_state{
      .viewportCount = 1,
      /* .pViewports = &viewport, omitted because of dynamic state */
      .scissorCount = 1,
      /* .pScissors  = &scissor, omitted because of dynamic state */
  };

  /* configure the rasterizer stage */
  vk::PipelineRasterizationStateCreateInfo rasterizer{
      .depthClampEnable = vk::False,
      .rasterizerDiscardEnable = vk::False,
      .polygonMode = vk::PolygonMode::eFill,
      .cullMode = vk::CullModeFlagBits::eBack,
      .frontFace = vk::FrontFace::eCounterClockwise,
      .depthBiasEnable = vk::False,
      .depthBiasSlopeFactor = 1.0f,
      .lineWidth = 1.0f,
  };

  /* configure multisampling */
  vk::PipelineMultisampleStateCreateInfo multisampling{
      .rasterizationSamples = vk::SampleCountFlagBits::e1,
      .sampleShadingEnable = vk::False,
  };

  /* configure shader-output-to-color-attachment color blending */
  vk::PipelineColorBlendAttachmentState color_blending_attachment_state{
      .blendEnable = vk::False, /* pass frag shader color output as it is */
      .colorWriteMask =
          vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
  };

  /* I still don't understand this */
  vk::PipelineColorBlendStateCreateInfo color_blending{
      .logicOpEnable = vk::False,
      .attachmentCount = 1,
      .pAttachments = &color_blending_attachment_state,
  };

  /* pipeline layout (for shader uniforms, push constants, etc.) */
  vk::PipelineLayoutCreateInfo pipeline_layout_create_info{
      .setLayoutCount = 1,
      .pSetLayouts = &*m_DesriptorSetLayout,
      .pushConstantRangeCount = 0,
  };

  m_PipelineLayout =
      vk::raii::PipelineLayout(m_Device, pipeline_layout_create_info);

  /* avoid creating render pass objects and use dynamic rendering (Vk >= 1.3) */
  vk::PipelineRenderingCreateInfo pipeline_rendering_create_info{
      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &m_SwapChainImgFormat,
  };

  /* graphics pipeline creation struct - finally :-) */
  vk::GraphicsPipelineCreateInfo pipeline_info{
      .pNext = &pipeline_rendering_create_info,
      .stageCount = 2,
      .pStages = shader_stages,
      .pVertexInputState = &vertex_input_state_info,
      .pInputAssemblyState = &input_assembly_create_info,
      .pViewportState = &viewport_state,
      .pRasterizationState = &rasterizer,
      .pMultisampleState = &multisampling,
      .pColorBlendState = &color_blending,
      .pDynamicState = &dynamic_state_create_info,
      .layout = m_PipelineLayout,
      .renderPass = nullptr,
  };

  /* finally - create the god damn pipeline object :-) */
  m_Pipeline = vk::raii::Pipeline(m_Device, nullptr, pipeline_info);
}

void VkEngine::create_command_pool() {
  /* each cmd buffer is submitted on a particular device queue and each cmd pool
   * can only allocate buffers for a single type of queue hence why we pass the
   * graphics queue index */
  vk::CommandPoolCreateInfo cmd_pool_create_info{
      .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
      .queueFamilyIndex = m_GraphicsQueueFamilyIdx,
  };

  m_GraphicsCmdPool = vk::raii::CommandPool(m_Device, cmd_pool_create_info);
}

void VkEngine::create_image(uint32_t width, uint32_t height, vk::Format format,
                            vk::ImageTiling tiling,
                            vk::ImageUsageFlags usage_flags,
                            vk::MemoryPropertyFlags mem_property_flags,
                            vk::raii::Image &img,
                            vk::raii::DeviceMemory &img_memory) {
  vk::ImageCreateInfo img_info{
      .imageType = vk::ImageType::e2D,
      .format = format,
      .extent =
          {
              .width = width,
              .height = height,
              .depth = 1,
          },
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = vk::SampleCountFlagBits::e1, /* no multisampling */
      .tiling = tiling,
      .usage = usage_flags,
      .sharingMode = vk::SharingMode::eExclusive,
  };

  img = vk::raii::Image(m_Device, img_info);

  vk::MemoryRequirements mem_reqs = img.getMemoryRequirements();
  vk::MemoryAllocateInfo alloc_info{
      .allocationSize = mem_reqs.size,
      .memoryTypeIndex = find_memory_type(
          m_PhysicalDevice, mem_reqs.memoryTypeBits, mem_property_flags)};
  img_memory = vk::raii::DeviceMemory(m_Device, alloc_info);
  img.bindMemory(img_memory, 0);
}

void VkEngine::create_texture_image(std::string_view filename) {
  int width, height, nbr_channels;
  stbi_uc *img_data = stbi_load(filename.data(), &width, &height, &nbr_channels,
                                STBI_rgb_alpha /* RGBA */);
  if (!img_data)
    throw std::runtime_error("failed to load texture image");

  vk::DeviceSize img_size = width * height * 4;

  vk::raii::Buffer img_staging_buff{nullptr};
  vk::raii::DeviceMemory img_memory{nullptr};
  create_buffer(m_PhysicalDevice, m_Device, img_size,
                vk::BufferUsageFlagBits::eTransferSrc,
                vk::MemoryPropertyFlagBits::eHostCoherent, img_staging_buff,
                img_memory);

  /* copy loaded image data into the staging buffer */
  void *dst_data_ptr = img_memory.mapMemory(0, img_size);
  std::memcpy(dst_data_ptr, img_data, img_size);
  img_memory.unmapMemory();

  /* TODO: potential resource leak if code above throws an exception */
  stbi_image_free(img_data);

  create_image(
      width, height, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal,
      vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
      vk::MemoryPropertyFlagBits::eDeviceLocal, m_TextureImg,
      m_TextureImgMemory);

  /* now transition the image to a layout that is optimal for it to be written
   * to as the destination for a transfer operation */
  transition_image_layout(m_TextureImg, vk::ImageLayout::eUndefined,
                          vk::ImageLayout::eTransferDstOptimal);

  copy_buffer_to_image(img_staging_buff, m_TextureImg,
                       static_cast<uint32_t>(width),
                       static_cast<uint32_t>(height));

  /* then transition it again to a layout that is optimal for fragment shader
   * reading operations */
  transition_image_layout(m_TextureImg, vk::ImageLayout::eTransferDstOptimal,
                          vk::ImageLayout::eShaderReadOnlyOptimal);
}

vk::raii::ImageView VkEngine::create_image_view(vk::Image img,
                                                vk::Format format) {
  vk::ImageViewCreateInfo view_create_info{
      .image = img,
      .viewType = vk::ImageViewType::e2D,
      .format = format,
      .components =
          {
              .r = vk::ComponentSwizzle::eIdentity,
              .g = vk::ComponentSwizzle::eIdentity,
              .b = vk::ComponentSwizzle::eIdentity,
              .a = vk::ComponentSwizzle::eIdentity,
          },
      .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                           .baseMipLevel = 0,
                           .levelCount = 1,
                           .baseArrayLayer = 0,
                           .layerCount = 1},
  };
  return vk::raii::ImageView(m_Device, view_create_info);
}

void VkEngine::create_texture_image_view() {
  m_TextureImgView = create_image_view(m_TextureImg, vk::Format::eR8G8B8A8Srgb);
}

void VkEngine::create_texture_sampler() {
  vk::PhysicalDeviceProperties props = m_PhysicalDevice.getProperties();
  vk::SamplerCreateInfo create_info{
      .magFilter = vk::Filter::eLinear,
      .minFilter = vk::Filter::eLinear,
      .mipmapMode = vk::SamplerMipmapMode::eLinear,
      .addressModeU = vk::SamplerAddressMode::eRepeat,
      .addressModeV = vk::SamplerAddressMode::eRepeat,
      .addressModeW = vk::SamplerAddressMode::eRepeat,
      .mipLodBias = 0.0f,
      .anisotropyEnable = vk::True,
      /* maximum quality. TODO: make this configurable */
      .maxAnisotropy = props.limits.maxSamplerAnisotropy,
      .compareEnable = vk::False,
      .compareOp = vk::CompareOp::eAlways,
  };
  m_TextureSampler = vk::raii::Sampler(m_Device, create_info);
}

uint32_t
VkEngine::find_memory_type(vk::raii::PhysicalDevice const &physical_device,
                           uint32_t mem_type_bitmask,
                           vk::MemoryPropertyFlags properties) {
  vk::PhysicalDeviceMemoryProperties mem_props{
      physical_device.getMemoryProperties()};
  for (uint32_t i{0}; i < mem_props.memoryTypeCount; ++i) {
    if (VkEngine::is_bit_set(mem_type_bitmask, i) &&
        is_all_bits_set(mem_props.memoryTypes[i].propertyFlags, properties)) {
      return i;
    }
  }
  throw std::runtime_error("failed to find required memory type");
}

void VkEngine::copy_buffer(vk::raii::Buffer &src, vk::raii::Buffer &dst,
                           vk::DeviceSize size) {

  vk::raii::CommandBuffer cb{begin_single_time_cmds()};

  {
    cb.copyBuffer(src, dst, vk::BufferCopy(0, 0, size));
  }

  end_single_time_cmds(cb);
}

void VkEngine::create_vertex_buffer() {
  vk::DeviceSize buff_size =
      sizeof(TEST_RECTANGLE_VERTICES[0]) * TEST_RECTANGLE_VERTICES.size();

  /* Create the host-visible (temporary?) staging buffer and its backing memory.
   * Notice in the usage flags that will be used as the source for a memory
   * transfer operation which is from this staging buffer (source) to the actual
   * vertex buffer that is in device-memory (or at least device-visible) */
  vk::raii::Buffer staging_buff{nullptr};
  vk::raii::DeviceMemory staging_memory{nullptr};
  create_buffer(m_PhysicalDevice, m_Device, buff_size,
                vk::BufferUsageFlagBits::eTransferSrc,
                vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent,
                staging_buff, staging_memory);
  /* Copy data from CPU memory (RAM) to the just-created staging buffer memory
   * (which is probably on RAM - idk for sure) */
  {
    /* Before doing so, we need to map the memory to host (i.e., CPU) accessible
     * memory */
    void *staging_data_ptr = staging_memory.mapMemory(0, vk::WholeSize);
    std::memcpy(staging_data_ptr, TEST_RECTANGLE_VERTICES.data(), buff_size);

    /* don't forget to unmap this - keep this mapped only incase you are
     * potentially at least a copy each frame to this data */
    staging_memory.unmapMemory();
  }

  /* Now we create the device-visible (more performant) buffer and its backing
   * memory. Notice in the flags that this will be used as a vertex buffer AND
   * as a transfer destination for some memory transfer operation which is
   * from the previously created staging buffer to this in-GPU-memory buffer. */
  create_buffer(m_PhysicalDevice, m_Device, buff_size,
                vk::BufferUsageFlagBits::eVertexBuffer |
                    vk::BufferUsageFlagBits::eTransferDst,
                vk::MemoryPropertyFlagBits::eDeviceLocal, m_VertexBuff,
                m_VertexBuffMemory);

  copy_buffer(staging_buff, m_VertexBuff, buff_size);
}

void VkEngine::create_index_buffer() {
  vk::DeviceSize buff_size =
      sizeof(TEST_RECTANGLE_INDICES[0]) * TEST_RECTANGLE_INDICES.size();

  vk::raii::Buffer staging_buff{nullptr};
  vk::raii::DeviceMemory staging_memory{nullptr};
  create_buffer(m_PhysicalDevice, m_Device, buff_size,
                vk::BufferUsageFlagBits::eTransferSrc,
                vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent,
                staging_buff, staging_memory);
  {
    void *staging_data_ptr = staging_memory.mapMemory(0, vk::WholeSize);
    std::memcpy(staging_data_ptr, TEST_RECTANGLE_INDICES.data(), buff_size);
    staging_memory.unmapMemory();
  }

  create_buffer(m_PhysicalDevice, m_Device, buff_size,
                vk::BufferUsageFlagBits::eIndexBuffer |
                    vk::BufferUsageFlagBits::eTransferDst,
                vk::MemoryPropertyFlagBits::eDeviceLocal, m_IndexBuff,
                m_IndexBuffMemory);

  copy_buffer(staging_buff, m_IndexBuff, buff_size);
}

void VkEngine::create_uniform_buffers() {
  m_UniformBuffers.clear();
  m_UniformBufferMemories.clear();
  m_UniformBufferMaps.clear();

  for (size_t i{0}; i < MAX_NBR_FRAMES_IN_FLIGHT; ++i) {
    vk::DeviceSize buff_size{sizeof(MVP)};
    vk::raii::Buffer buff({});
    vk::raii::DeviceMemory buff_mem({});
    create_buffer(m_PhysicalDevice, m_Device, buff_size,
                  vk::BufferUsageFlagBits::eUniformBuffer,
                  vk::MemoryPropertyFlagBits::eHostVisible |
                      vk::MemoryPropertyFlagBits::eHostCoherent,
                  buff, buff_mem);
    m_UniformBuffers.emplace_back(std::move(buff));
    m_UniformBufferMemories.emplace_back(std::move(buff_mem));
    /* keep the memories persistently mapped (i.e., use persistent mapping
     * technique) since we are expected to update the memory every frame */
    m_UniformBufferMaps.emplace_back(
        m_UniformBufferMemories[i].mapMemory(0, buff_size));
  }
}

void VkEngine::update_mvp(uint32_t curr_frame) {
  static auto start_time{std::chrono::high_resolution_clock::now()};

  auto current_time{std::chrono::high_resolution_clock::now()};
  auto delta{std::chrono::duration<float, std::chrono::seconds::period>(
                 current_time - start_time)
                 .count()};

  MVP mvp{
      .model = glm::rotate(glm::mat4(1.0f), delta * glm::radians(90.0f),
                           glm::vec3(0.0f, 0.0f, 1.0f)),
      .view =
          glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                      glm::vec3(0.0f, 0.0f, 1.0f)),
      .proj = glm::perspective(glm::radians(45.0f),
                               static_cast<float>(m_SwapChainExtent.width) /
                                   static_cast<float>(m_SwapChainExtent.height),
                               0.1f, 10.f),
  };

  /* flip Y coordinates because glm is designed for OpenGL and not Vulkan */
  mvp.proj[1][1] *= -1;

  /* then copy the mvp struct to the actual uniform buffer memory */
  std::memcpy(m_UniformBufferMaps[curr_frame], &mvp, sizeof(mvp));
}

void VkEngine::create_descriptor_pool() {
  vk::DescriptorPoolSize pool_size{
      .type = vk::DescriptorType::eUniformBuffer,
      .descriptorCount = MAX_NBR_FRAMES_IN_FLIGHT,
  };

  vk::DescriptorPoolCreateInfo pool_create_info{
      .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
      .maxSets = MAX_NBR_FRAMES_IN_FLIGHT, /* a set foreach in-flight frame */
      .poolSizeCount = 1,
      .pPoolSizes = &pool_size,
  };

  m_DesriptorPool = vk::raii::DescriptorPool(m_Device, pool_create_info);
}

void VkEngine::create_descriptor_sets() {
  std::vector<vk::DescriptorSetLayout> layouts(MAX_NBR_FRAMES_IN_FLIGHT,
                                               *m_DesriptorSetLayout);
  vk::DescriptorSetAllocateInfo alloc_info{
      .descriptorPool = m_DesriptorPool,
      .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
      .pSetLayouts = layouts.data(),
  };

  m_DescriptorSets = m_Device.allocateDescriptorSets(alloc_info);

  for (size_t i{0}; i < MAX_NBR_FRAMES_IN_FLIGHT; ++i) {
    vk::DescriptorBufferInfo buffer_info{
        .buffer = m_UniformBuffers[i],
        .offset = 0,
        .range = sizeof(MVP),
    };

    vk::WriteDescriptorSet descriptor_write{
        .dstSet = m_DescriptorSets[i],
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eUniformBuffer,
        .pBufferInfo = &buffer_info,
    };

    m_Device.updateDescriptorSets(descriptor_write, {});
  }
}

vk::raii::CommandBuffer VkEngine::begin_single_time_cmds() {
  vk::CommandBufferAllocateInfo alloc_info{
      .commandPool = m_GraphicsCmdPool,
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount = 1,
  };

  vk::raii::CommandBuffer cb{
      std::move(m_Device.allocateCommandBuffers(alloc_info).front())};

  /* The eOneTimeSubmit tells the driver that this cb is only used once. This
   * might potentially result in optimizations. */
  vk::CommandBufferBeginInfo begin_info{
      .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
  };
  cb.begin(begin_info);

  return cb;
}

void VkEngine::end_single_time_cmds(vk::raii::CommandBuffer &cb) {
  /* end recording of commands */
  cb.end();

  /* submit the cb synchronously (i.e., wait for it to finish) */
  m_GraphicsQueue.submit(
      vk::SubmitInfo{
          .commandBufferCount = 1,
          .pCommandBuffers = &*cb,
      },
      nullptr);

  /* We could have used a fence here which will allow us to schedule multiple
   * transfer cmds simultaneously and wait on all of them. */
  m_GraphicsQueue.waitIdle();
}

void VkEngine::transition_image_layout(vk::raii::Image const &img,
                                       vk::ImageLayout from_layout,
                                       vk::ImageLayout to_layout) {
  vk::raii::CommandBuffer cb{begin_single_time_cmds()};
  {
    vk::ImageMemoryBarrier barrier{
        .oldLayout = from_layout,
        .newLayout = to_layout,
        .image = img,
        .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                             .baseMipLevel = 0,
                             .levelCount = 1,
                             .baseArrayLayer = 0,
                             .layerCount = 1},
    };

    vk::PipelineStageFlags src_stage;
    vk::PipelineStageFlags dst_stage;

    /* undefined -> anything: "anything" does not have to wait on anything */
    if (from_layout == vk::ImageLayout::eUndefined /* don't care layout */ &&
        to_layout == vk::ImageLayout::eTransferDstOptimal) {
      barrier.srcAccessMask = {};
      barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
      src_stage = vk::PipelineStageFlagBits::eTopOfPipe;
      dst_stage = vk::PipelineStageFlagBits::eTransfer;
    }
    /* transfer dst -> shader read: shader reads in the fragment shader SHOULD
     * wait on transfer writes */
    else if (from_layout == vk::ImageLayout::eTransferDstOptimal &&
             to_layout == vk::ImageLayout::eShaderReadOnlyOptimal) {
      barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
      barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
      src_stage = vk::PipelineStageFlagBits::eTransfer;
      dst_stage = vk::PipelineStageFlagBits::eFragmentShader;
    } else {
      throw std::invalid_argument("unsupported image layout transition");
    }
    cb.pipelineBarrier(src_stage, dst_stage, {}, {}, nullptr, barrier);
  }
  end_single_time_cmds(cb);
}

void VkEngine::copy_buffer_to_image(vk::raii::Buffer const &buff,
                                    vk::raii::Image const &img, uint32_t width,
                                    uint32_t height) {
  vk::raii::CommandBuffer cb{begin_single_time_cmds()};
  {
    vk::BufferImageCopy region{
        .bufferOffset = 0,
        .bufferRowLength = 0,   /* 0 means tightly-packed pixels */
        .bufferImageHeight = 0, /* 0 means tightly-packed pixels */
        /* to which subresource of the image we want to copy the data to */
        .imageSubresource =
            {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        /* at which offset of the image we want to copy the data to */
        .imageOffset =
            {
                .x = 0,
                .y = 0,
                .z = 0,
            },
        /* which subregions/extent of the image we want to copy the data to */
        .imageExtent =
            {
                .width = width,
                .height = height,
                .depth = 1,
            },
    };
    /* we assume that the image's layout is set to transfer dst optimal */
    cb.copyBufferToImage(buff, img, vk::ImageLayout::eTransferDstOptimal,
                         {region});
  }
  end_single_time_cmds(cb);
}

void VkEngine::create_command_buffers() {
  vk::CommandBufferAllocateInfo cmd_buffer_allocate_info{
      .commandPool = m_GraphicsCmdPool,
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount = MAX_NBR_FRAMES_IN_FLIGHT,
  };
  m_GraphicsCmdBuffers =
      std::move(vk::raii::CommandBuffers(m_Device, cmd_buffer_allocate_info));
}

void VkEngine::record_command_buffer(uint32_t curr_frame_idx,
                                     uint32_t swapchain_image_idx) const {
  vk::raii::CommandBuffer const &cb = m_GraphicsCmdBuffers[curr_frame_idx];
  cb.begin({});

  /* First, transition the swap chain image into an a layout suitable for color
   * attachment operations (i.e., shader writing to it, etc.). */
  {
    /* To do so, we need to set up a pipeline barrier - ... a what? You have to
     * understand something crucial about command buffers in Vulkan - there is
     * no gurantee whatsoever about the order in which they are going to be
     * executed. This means that if you record commands A->B->C the driver may
     * or may not execute these cmds in that submission order.
     *
     * It is your job, as a Vulkan application developer, to explecitely set
     * synchronization mechanisms to ensure a particular executaion order
     * whithin a subset of the recorded cmds.
     *
     * Now back to image layout transition, it is crucial for the layout
     * transition to occur BEFORE any cmds that will write to that image (think
     * of draw cmds).
     * */
    vk::ImageMemoryBarrier2 pipeline_barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
        .srcAccessMask = {},
        .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = m_SwapChainImgs[swapchain_image_idx],
        .subresourceRange =
            {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
    };

    vk::DependencyInfo dependency_info{
        .dependencyFlags = {},
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &pipeline_barrier,
    };
    cb.pipelineBarrier2(dependency_info);
  }

  /* Then, setup the rendering info; which area to render to, which image view
   * to use for rendering, which clear color, which operations to perform before
   * and after rendering, etc. */
  auto clear_color = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 0.0f};
  vk::RenderingAttachmentInfo attachment_info{
      .imageView = m_SwapChainImgViews[swapchain_image_idx],
      .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
      .loadOp = vk::AttachmentLoadOp::eClear,
      .storeOp = vk::AttachmentStoreOp::eStore,
      .clearValue = clear_color,
  };
  vk::RenderingInfo rendering_info{
      .renderArea =
          {
              .offset = {0, 0},
              .extent = m_SwapChainExtent,
          },
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &attachment_info,
  };

  { /* BEGIN RENDERING */

    /* then begin rendering (oof finally) */
    cb.beginRendering(rendering_info);

    /* then bind the graphics pipeline */
    cb.bindPipeline(vk::PipelineBindPoint::eGraphics, m_Pipeline);

    /* remember that we have set the viewport and its scissor as dynamic
     * pipeline variables hence why we need to specify them before issuing any
     * 'vkCmd' calls */
    cb.setViewport(0,
                   vk::Viewport{
                       .x = 0.0f,
                       .y = 0.0f,
                       .width = static_cast<float>(m_SwapChainExtent.width),
                       .height = static_cast<float>(m_SwapChainExtent.height),
                       .minDepth = 0.0f,
                       .maxDepth = 1.0f,
                   });
    cb.setScissor(0, vk::Rect2D{
                         .offset =
                             vk::Offset2D{
                                 .x = 0,
                                 .y = 0,
                             },
                         .extent = m_SwapChainExtent,
                     });

    /* now we can finally use the drawing 'vkCmd' calls which will actually draw
     * stuff on the screen */

    { /* BEGIN DRAWING CMDS */

      cb.bindVertexBuffers(
          0, *m_VertexBuff,
          {0} /* offsets are NOT in bytes rather in set stride unit */);
      cb.bindIndexBuffer(*m_IndexBuff, 0, vk::IndexType::eUint16);
      cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_PipelineLayout,
                            0, *m_DescriptorSets[curr_frame_idx], nullptr);
      cb.drawIndexed(TEST_RECTANGLE_INDICES.size(), 1, 0, 0, 0);

    } /* END DRAWING CMDS*/

    cb.endRendering();

  } /* END RENDERING */

  /* after rendering, transition the swapchain image back to a layout that is
   * optimal for presentation */
  {
    vk::ImageMemoryBarrier2 pipeline_barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe,
        .dstAccessMask = {},
        .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .newLayout = vk::ImageLayout::ePresentSrcKHR,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = m_SwapChainImgs[swapchain_image_idx],
        .subresourceRange =
            {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
    };

    vk::DependencyInfo dependency_info{
        .dependencyFlags = {},
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &pipeline_barrier,
    };
    cb.pipelineBarrier2(dependency_info);
  }

  cb.end();
}

void VkEngine::create_sync_objects() {
  m_RenderFinishedSems.clear();
  m_PresentCompleteSems.clear();
  m_DrawFences.clear();

  /* For each swapchain image, we need to create seperate semaphores
   * (semaphores sync operations within a queue).
   *
   * Why not MAX_NBR_FRAMES_IN_FLIGHT semaphores? You can think of the following
   * scenario where we have 2 in-flight frames (MAX_NBR_FRAMES_IN_FLIGHT = 2)
   * and 3 swapchain images that can be acquired. The third call to
   * acquireNextImage may re-use some semaphores that are potentially still
   * needed by another swapchain image. */
  for (int i = 0; i < m_SwapChainImgs.size(); ++i) {
    m_RenderFinishedSems.emplace_back(m_Device, vk::SemaphoreCreateInfo());
    m_PresentCompleteSems.emplace_back(m_Device, vk::SemaphoreCreateInfo());
  }

  /* For each in-flight frame, we need to create seperate fences
   * (fences sync operations between the host and the device) */
  for (int i = 0; i < MAX_NBR_FRAMES_IN_FLIGHT; ++i) {
    m_DrawFences.emplace_back(
        m_Device,
        vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
  }
}

[[nodiscard]] vk::raii::ShaderModule
VkEngine::create_shader_module(const std::vector<char> &code) const {
  vk::ShaderModuleCreateInfo shader_module_create_info{
      .codeSize = code.size() * sizeof(char),
      .pCode = reinterpret_cast<const uint32_t *>(code.data()),
  };

  vk::raii::ShaderModule shader_module(m_Device, shader_module_create_info);
  return shader_module;
}

void VkEngine::draw_frame() {

  /* now wait for the previously submitted rendering cmds to finish (i.e., wait
   * for draw fence to be signaled) */
  while (vk::Result::eTimeout ==
         m_Device.waitForFences(*m_DrawFences[m_CurrFrameIdx], vk::True,
                                UINT64_MAX))
    ;

  /* Acquire an available swap chain image to render to (of course - we have to
   * make sure that this is NOT an image that is currently being rendered to).
   *
   * But why signal a semaphore you may ask - well, as per Vulkan specification:
   * "The presentation engine may not have finished reading from the image at
   * the time it is acquired, so the application must use semaphore and/or fence
   * to ensure that the image layout and contents are not modified until the
   * presentation engine reads have completed"
   * */
  auto [result, img_idx] = m_SwapChain.acquireNextImage(
      UINT64_MAX, *m_PresentCompleteSems[m_CurrSemphIdx], nullptr);

  /* acquireNextImage may return in its Result that the swapchain is no longer
   * compatible and it may need to be recreated (e.g., window resize) */
  if (result == vk::Result::eErrorOutOfDateKHR) {
    recreate_swap_chain();
    return;
  }

  if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
    throw std::runtime_error("failed to acquire swapchain image");
  }

  /* update MVP projection matrix */
  update_mvp(m_CurrFrameIdx);

  /* then record the rendering cmds into the current command buffer */
  m_GraphicsCmdBuffers[m_CurrFrameIdx].reset();
  record_command_buffer(m_CurrFrameIdx, img_idx);

  /* Then submit the previously recorded command buffer to the graphics queue.
   *
   * We wait for the presentation semaphore to be signaled before starting the
   * execution of these cmds (i.e., before we start rendering). We have to wait
   * for the presentation semaphore to be signaled because the presentation
   * engine may still be reading from that acquired image (read comment above
   * on acquireNextImage).
   *
   * We signal the render-finished semaphore because we want to wait on it
   * before presenting the image (that is, we introduced a sync that states
   * "rendering to the image has to be finished before presenting it". Read
   * comment under on presentKHR).
   * */
  vk::PipelineStageFlags wait_dst_stage_mask{
      vk::PipelineStageFlagBits::eColorAttachmentOutput};
  vk::SubmitInfo submit_info{
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &(*m_PresentCompleteSems[m_CurrSemphIdx]),
      .pWaitDstStageMask = &wait_dst_stage_mask,
      .commandBufferCount = 1,
      .pCommandBuffers = &(*m_GraphicsCmdBuffers[m_CurrFrameIdx]),
      .signalSemaphoreCount = 1,
      .pSignalSemaphores = &(*m_RenderFinishedSems[img_idx]),
  };
  /* reset the drawing-finished fence (if we don't do this, the fence will
   * remain in a signaled state). To make your code more future-proof, make sure
   * to reset the fence exactly before submiting the work that will signal it */
  m_Device.resetFences(*m_DrawFences[m_CurrFrameIdx]);
  m_GraphicsQueue.submit(submit_info, *m_DrawFences[m_CurrFrameIdx]);

  /* After making sure that rendering has finished, we have to present the just
   * rendered frame.
   *
   * We wait on the render-finished semaphore to be signaled before we start
   * presenting the image (otherwise we may start presenting an unfinished -
   * not completely rendered - image. Or we may get an error. Just try it out).
   * */
  vk::PresentInfoKHR present_info{
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &(*m_RenderFinishedSems[img_idx]),
      .swapchainCount = 1,
      .pSwapchains = &(*m_SwapChain),
      .pImageIndices = &img_idx,
  };
  result = m_GraphicsQueue.presentKHR(present_info);

  if (result == vk::Result::eErrorOutOfDateKHR ||
      result == vk::Result::eSuboptimalKHR || is_window_resized) {
    is_window_resized = false;
    recreate_swap_chain();
  } else if (result != vk::Result::eSuccess) {
    throw std::runtime_error("failed to present swapchain image");
  }

  m_CurrFrameIdx = (m_CurrFrameIdx + 1) % MAX_NBR_FRAMES_IN_FLIGHT;
  m_CurrSemphIdx = (m_CurrSemphIdx + 1) % m_SwapChainImgs.size();
}

void VkEngine::create_buffer(vk::raii::PhysicalDevice const &physical_device,
                             vk::raii::Device const &device,
                             vk::DeviceSize size,
                             vk::BufferUsageFlags usage_flags,
                             vk::MemoryPropertyFlags mem_props,
                             vk::raii::Buffer &buffer,
                             vk::raii::DeviceMemory &dev_mem) {
  /* a vertex buffer is backed by a buffer (duh?). Create the buffer with
   * required size and set usage flags accordingly */
  vk::BufferCreateInfo buffer_info{
      .size = size,
      .usage = usage_flags,
      /* eExclusive because the buffer is NOT shared between multiple queues
       * (i.e., here it is used exclusively whithin the graphics queue) */
      .sharingMode = vk::SharingMode::eExclusive,

  };
  buffer = vk::raii::Buffer(device, buffer_info);

  /* The buffer was created but is still NOT backed by any memory (i.e., the
   * acutal memory that will hold the data has not been yet allocated).
   *
   * To create the backing memory, we need to first fetch its requirements from
   * the VkBuffer object we have just created (and other requirements - see
   * below) */
  vk::MemoryRequirements mem_requirements{buffer.getMemoryRequirements()};
  uint32_t mem_type_idx = find_memory_type(
      physical_device, mem_requirements.memoryTypeBits, mem_props);
  vk::MemoryAllocateInfo mem_alloc_info{
      /* could be != buffer_info.size */
      .allocationSize = mem_requirements.size,
      .memoryTypeIndex = mem_type_idx,
  };
  dev_mem = vk::raii::DeviceMemory(device, mem_alloc_info);

  /* The memory is now successfully created but is still not bound to the
   * VkBuffer object we have created earlier (i.e., we haven't done anything
   * yet in regards to backing the VkBuffer with memory). The memoryOffset is 0
   * because this memory is specifically allocated for this vertex buffer (i.e.,
   * if we wanted to back multiple vertex buffers with a single VkDeviceMemory
   * then we would need to supply offsets for the VkBuffer objects) */
  buffer.bindMemory(dev_mem, 0);
}

} // namespace vkengine
