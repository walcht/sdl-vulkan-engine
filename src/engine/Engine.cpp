#include "Engine.hpp"
#include "Utils.hpp"
#include "vulkan/vulkan.hpp"
#include <map>

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
  init_graphics_pipeline();
  create_command_pool();
  create_command_buffer();
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
                                   SDL_WINDOW_VULKAN)) == nullptr) {
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
          {},
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
  if (m_SwapChainImgs.empty()) {
    throw std::runtime_error(
        "cannot create swapchain image views out of empty images");
  }

  vk::ImageViewCreateInfo imageViewCreateInfo{
      .viewType = vk::ImageViewType::e2D,
      .format = m_SwapChainImgFormat,
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

  for (const auto &img : m_SwapChainImgs) {
    imageViewCreateInfo.image = img;
    m_SwapChainImgViews.emplace_back(m_Device, imageViewCreateInfo);
  }
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

  /* keep the vertex input state to default because we are hardcoding the
   * vertices */
  vk::PipelineVertexInputStateCreateInfo vertex_input_state_info;

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
      .frontFace = vk::FrontFace::eClockwise,
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
      .setLayoutCount = 0,
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

void VkEngine::create_command_buffer() {
  vk::CommandBufferAllocateInfo cmd_buffer_allocate_info{
      .commandPool = m_GraphicsCmdPool,
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount = 1,
  };

  /* transfer ownership of the create cmd buffer to the class cmd buffer handle.
   * Remember that it makes no sense for cmd buffers to have copy assignment */
  m_GraphicsCmdBuffer = std::move(
      vk::raii::CommandBuffers(m_Device, cmd_buffer_allocate_info).front());
}

void VkEngine::record_command_buffer(uint32_t swapchain_image_idx) const {
  m_GraphicsCmdBuffer.begin({});

  /* First, transition the swap chain image into an a layout suitable for color
   * attachment operations (i.e., shader writing to it, etc.). */
  transition_imaga_layout(m_SwapChainImgs[swapchain_image_idx],
                          vk::ImageLayout::eUndefined,
                          vk::ImageLayout::eColorAttachmentOptimal, {},
                          vk::AccessFlagBits2::eColorAttachmentWrite,
                          vk::PipelineStageFlagBits2::eTopOfPipe,
                          vk::PipelineStageFlagBits2::eColorAttachmentOutput);

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
    m_GraphicsCmdBuffer.beginRendering(rendering_info);

    /* then bind the graphics pipeline */
    m_GraphicsCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
                                     m_Pipeline);

    /* remember that we have set the viewport and its scissor as dynamic
     * pipeline variables hence why we need to specify them before issuing any
     * 'vkCmd' calls */
    m_GraphicsCmdBuffer.setViewport(
        0, vk::Viewport{
               .x = 0.0f,
               .y = 0.0f,
               .width = static_cast<float>(m_SwapChainExtent.width),
               .height = static_cast<float>(m_SwapChainExtent.height),
               .minDepth = 0.0f,
               .maxDepth = 1.0f,
           });
    m_GraphicsCmdBuffer.setScissor(0, vk::Rect2D{
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

      m_GraphicsCmdBuffer.draw(3, 1, 0, 0);

    } /* END DRAWING CMDS*/

    m_GraphicsCmdBuffer.endRendering();

  } /* END RENDERING */

  /* after rendering, transition the swapchain image back to a layout that is
   * optimal for presentation */
  transition_imaga_layout(m_SwapChainImgs[swapchain_image_idx],
                          vk::ImageLayout::eColorAttachmentOptimal,
                          vk::ImageLayout::ePresentSrcKHR,
                          vk::AccessFlagBits2::eColorAttachmentWrite, {},
                          vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                          vk::PipelineStageFlagBits2::eBottomOfPipe);

  m_GraphicsCmdBuffer.end();
}

void VkEngine::transition_imaga_layout(
    vk::Image image, vk::ImageLayout old_layout, vk::ImageLayout new_layout,
    vk::AccessFlags2 src_access_mask, vk::AccessFlags2 dst_access_mask,
    vk::PipelineStageFlags2 src_stage_mask,
    vk::PipelineStageFlags2 dst_stage_mask) const {
  /* To do so, we need to set up a pipeline barrier - ... a what? You have to
   * understand something crucial about command buffers in Vulkan - there is no
   * gurantee whatsoever about the order in which they are going to be executed.
   * This means that if you record commands A->B->C the driver may or may not
   * execute these cmds in that submission order.
   *
   * It is your job, as a Vulkan application developer, to explecitely set
   * synchronization mechanisms to ensure a particular executaion order whithin
   * a subset of the recorded cmds.
   *
   * Now back to image layout transition, it is crucial for the layout
   * transition to occur BEFORE any cmds that will write to that image (think of
   * draw cmds).
   * */
  vk::ImageMemoryBarrier2 pipeline_barrier{
      .srcStageMask = src_stage_mask,
      .srcAccessMask = src_access_mask,
      .dstStageMask = dst_stage_mask,
      .dstAccessMask = dst_access_mask,
      .oldLayout = old_layout,
      .newLayout = new_layout,
      .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
      .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
      .image = image,
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

  m_GraphicsCmdBuffer.pipelineBarrier2(dependency_info);
}

void VkEngine::create_sync_objects() {
  m_PresentCompleteSem =
      vk::raii::Semaphore(m_Device, vk::SemaphoreCreateInfo{});
  m_RenderFinishedSem =
      vk::raii::Semaphore(m_Device, vk::SemaphoreCreateInfo{});
  m_DrawFence =
      vk::raii::Fence(m_Device, {
                                    .flags = vk::FenceCreateFlagBits::eSignaled,
                                });
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

void VkEngine::draw_frame() const {

  /* block the host (us - CPU) until all operations on the graphics queue are
   * done. */
  m_GraphicsQueue.waitIdle();

  /* acquire an available swap chain image to render to (of course - we have to
   * make sure that this is NOT an image that is being rendered to).
   *
   * But why signal a semaphore you may askw - well, as per Vulkan
   * specification:
   * "The presentation engine may not have finished reading from the image at
   * the time it is acquired, so the application must use semaphore and/or fence
   * to ensure that the image layout and contents are not modified until the
   * presentation engine reads have completed"
   * */
  auto [result, img_idx] =
      m_SwapChain.acquireNextImage(UINT64_MAX, *m_PresentCompleteSem, nullptr);

  /* reset the drawing-finished fence */
  m_Device.resetFences(*m_DrawFence);

  /* then record the rendering cmds */
  record_command_buffer(img_idx);

  /* then submit the previously recorded command buffer to the graphics queue.
   * We wait for the presentation semaphore to be signaled before starting the
   * execution of these cmds (i.e., before we start rendering) */
  vk::PipelineStageFlags wait_dst_stage_mask{
      vk::PipelineStageFlagBits::eColorAttachmentOutput};
  vk::SubmitInfo submit_info{
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &(*m_PresentCompleteSem),
      .pWaitDstStageMask = &wait_dst_stage_mask,
      .commandBufferCount = 1,
      .pCommandBuffers = &(*m_GraphicsCmdBuffer),
      .signalSemaphoreCount = 1,
      .pSignalSemaphores = &(*m_RenderFinishedSem),
  };
  m_GraphicsQueue.submit(submit_info, *m_DrawFence);

  /* now wait for the previously submitted rendering cmds to finish (i.e., wait
   * for draw fence to be signaled) */
  while (vk::Result::eTimeout ==
         m_Device.waitForFences(*m_DrawFence, vk::True, UINT64_MAX))
    ;

  /* after making sure that rendering has finished, we have to present the just
   * rendered frame */
  vk::PresentInfoKHR present_info{
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &(*m_RenderFinishedSem),
      .swapchainCount = 1,
      .pSwapchains = &(*m_SwapChain),
      .pImageIndices = &img_idx,
  };
  result = m_GraphicsQueue.presentKHR(present_info);
  switch (result) {
  case vk::Result::eSuccess:
    break;
  case vk::Result::eSuboptimalKHR:
    break;
  default:
    break;
  }
}

} // namespace vkengine
