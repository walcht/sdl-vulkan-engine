#pragma once

/* make VulkanHpp function calls accept structure parameters directly (easier
 * to map to the original Vulkan C API) */
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS 1
#include "Engine.hpp"
#include "SDL3/SDL_video.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan_raii.hpp>

namespace vkengine {

/* Vulkan Engine wrapper - this holds all necessary Vulkan utilities for
 * rendering or compute purposes */
class VkEngine {
public:
  VkEngine() = delete;
  void draw_frame();

  /* engine takes ownership of the Vulkan instance */
  VkEngine(const std::string &title, const std::string &app_identifier);
  ~VkEngine();

public:
  static constexpr int MAX_NBR_FRAMES_IN_FLIGHT = 2;

private:
  std::string m_Title;
  std::string m_AppIdentifier;
  SDL_Window *m_Window = nullptr;

  /* nullptr because constructors are deleted */
  vk::raii::Instance m_VkInstance = nullptr;
  vk::raii::PhysicalDevice m_PhysicalDevice = nullptr;
  vk::raii::Device m_Device = nullptr;
  vk::raii::Queue m_GraphicsQueue = nullptr;
  vk::raii::Queue m_PresentQueue = nullptr;
  vk::raii::SurfaceKHR m_VkSurface = nullptr;
  vk::raii::SwapchainKHR m_SwapChain = nullptr;
  vk::raii::PipelineLayout m_PipelineLayout = nullptr;
  vk::raii::Pipeline m_Pipeline = nullptr;
  vk::raii::CommandPool m_GraphicsCmdPool = nullptr;
  std::vector<vk::raii::CommandBuffer> m_GraphicsCmdBuffers;
  std::vector<vk::raii::Semaphore> m_PresentCompleteSems;
  std::vector<vk::raii::Semaphore> m_RenderFinishedSems;
  std::vector<vk::raii::Fence> m_DrawFences;

  /* swapchain stuff */
  std::vector<vk::Image> m_SwapChainImgs;
  std::vector<vk::raii::ImageView> m_SwapChainImgViews;
  vk::Format m_SwapChainImgFormat{vk::Format::eUndefined};
  vk::Extent2D m_SwapChainExtent;

  uint32_t m_GraphicsQueueFamilyIdx;
  uint32_t m_PresentQueueFamilyIdx;

  const std::vector<const char *> m_RequiredDevExts = {
      vk::KHRSwapchainExtensionName,
      vk::KHRSpirv14ExtensionName,
      vk::KHRSynchronization2ExtensionName,
  };

  /* index of current in-flight frame [0, MAX_NBR_FRAMES_IN_FLIGHT[ */
  uint32_t m_CurrFrameIdx = 0;

  /* index of current semaphore [0, swapChainImgs.size()[ */
  uint32_t m_CurrSemphIdx = 0;

private:
  void init_window();

  void init_surface();

  /* */
  void init_physical_device();

  /* */
  void init_vkinstance();

  void init_logical_device();

  void init_swap_chain();

  /* choose the most suitable format for the swapchain from a vector of
   * supported formats */
  vk::SurfaceFormatKHR choose_swap_surface_format(
      const std::vector<vk::SurfaceFormatKHR> &surface_formats) const;

  vk::PresentModeKHR choose_swap_present_mode(
      const std::vector<vk::PresentModeKHR> &present_modes) const;

  vk::Extent2D
  get_swap_extent(const vk::SurfaceCapabilitiesKHR &capabilities) const;

  uint32_t choose_swap_min_img_count(
      vk::SurfaceCapabilitiesKHR surface_capabilities) const;

  void init_swap_image_views();

  void init_graphics_pipeline();

  void create_command_pool();

  void create_command_buffers();

  void record_command_buffer(const vk::raii::CommandBuffer &cb,
                             uint32_t swapchain_image_idx) const;

  void create_sync_objects();

  void transition_imaga_layout(const vk::raii::CommandBuffer &cb,
                               vk::Image image, vk::ImageLayout old_layout,
                               vk::ImageLayout new_layout,
                               vk::AccessFlags2 src_access_mask,
                               vk::AccessFlags2 dst_access_mask,
                               vk::PipelineStageFlags2 src_stage_mask,
                               vk::PipelineStageFlags2 dst_stage_mask) const;

  [[nodiscard]] vk::raii::ShaderModule
  create_shader_module(const std::vector<char> &code) const;
};

} // namespace vkengine
