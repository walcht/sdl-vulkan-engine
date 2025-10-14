#pragma once
#include "Engine.hpp"
#include <vulkan/vulkan_raii.hpp>

/* Vulkan Engine wrapper - this holds all necessary Vulkan utilities for
 * rendering or compute purposes */
class VkEngine {
public:
  VkEngine() = delete;

  /* engine takes ownership of the Vulkan instance */
  VkEngine(const vk::ApplicationInfo &appInfo,
           const std::vector<const char *> &requiredExts,
           const std::vector<const char *> &valLayers) {
    createVkInstance(appInfo, requiredExts, valLayers);
    pickPhysicalDevice();
  }

  ~VkEngine();

private:
  /* nullptr because constructors are deleted */
  vk::raii::Instance m_VkInstance = nullptr;
  vk::raii::PhysicalDevice m_PhysicalDevice = nullptr;

  const std::vector<const char *> m_RequiredDevExts = {
      vk::KHRSwapchainExtensionName,
      vk::KHRSpirv14ExtensionName,
      vk::KHRSynchronization2ExtensionName,
      vk::KHRCreateRenderpass2ExtensionName,
  };

  /* */
  void pickPhysicalDevice();

  /* */
  void createVkInstance(const vk::ApplicationInfo &appInfo,
                        const std::vector<const char *> &requiredExts,
                        const std::vector<const char *> &valLayers);
};
