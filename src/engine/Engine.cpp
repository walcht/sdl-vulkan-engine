#include "Engine.hpp"
#include <map>

void VkEngine::pickPhysicalDevice() {
  /* get available physical devices with Vulkan support */
  auto devices = m_VkInstance.enumeratePhysicalDevices();
  if (devices.empty()) {
    throw std::runtime_error("no devices available with Vulkan support");
  }

  /* the idea here is to assign a score to each available device and pick the
   * one with the highest score - for that we need an ordered map with support
   * for identical keys (because scores can be identical) */
  std::multimap<uint32_t, const vk::raii::PhysicalDevice *> candidates;
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
    auto queueFamilies = device.getQueueFamilyProperties();
    const auto qfpIter =
        std::ranges::find_if(queueFamilies, [](const auto &qfp) {
          return (qfp.queueFlags & vk::QueueFlagBits::eGraphics) !=
                 static_cast<vk::QueueFlags>(0);
        });
    if (qfpIter == queueFamilies.end()) {
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

    /* discrete GPUs are usually just better */
    if (deviceProps.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
      score += 1000;
    }

    /* add to the score the max allowed 2D image dimension - the higher the
     * better - add other desired properties here */
    score += deviceProps.limits.maxImageDimension2D;

    candidates.insert(std::make_pair(score, &device));
  }

  /* pick the highest scoring candidate */
  if (!candidates.empty()) {
    m_PhysicalDevice = *(candidates.rbegin()->second);
  }

  throw std::runtime_error("failed to find a suitable physical device!");
}

void VkEngine::createVkInstance(const vk::ApplicationInfo &appInfo,
                                const std::vector<const char *> &requiredExts,
                                const std::vector<const char *> &valLayers) {
  vk::raii::Context m_Context;
  /* check if required extensions are supported by the provided Vulkan */
  auto available_props = m_Context.enumerateInstanceExtensionProperties();
  for (auto const &requiredExt : requiredExts) {
    if (std::ranges::none_of(available_props, [=](auto const &extension_prop) {
          return strcmp(requiredExt, extension_prop.extensionName) == 0;
        })) {
      throw std::runtime_error(
          "Some of the SDL3 required Vulkan extensions are not available");
    }
  }

  /* validation layers setup */
#ifndef NDEBUG
  auto available_layers = m_Context.enumerateInstanceLayerProperties();
  /* if any_of the required validation layers does NOT exist in the vector of
   * availalbe layers => exit with failure */
  if (std::ranges::any_of(valLayers, [&](auto const &val_layer) {
        return std::ranges::none_of(available_layers, [&](auto const &l) {
          return strcmp(l.layerName, val_layer) == 0;
        });
      })) {
    throw std::runtime_error(
        "one or more of the required validation layers are not available");
  }
#endif

  vk::InstanceCreateInfo create_info{
      .pApplicationInfo = &appInfo,
#ifndef NDEBUG
      .enabledLayerCount = static_cast<uint32_t>(valLayers.size()),
      .ppEnabledLayerNames = valLayers.data(),
#endif
      .enabledExtensionCount = static_cast<uint32_t>(requiredExts.size()),
      .ppEnabledExtensionNames = requiredExts.data(),
  };
  m_VkInstance = vk::raii::Instance(m_Context, create_info);
};
