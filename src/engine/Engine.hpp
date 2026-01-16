#pragma once

/* Vulkan strcut alignments - does NOT work for nested structs! */
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES

/* make VulkanHpp function calls accept structure parameters directly (easier
 * to map to the original Vulkan C API) */
#include <concepts>
#include <cstddef>
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS 1
#include "SDL3/SDL_video.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3/SDL_vulkan.h>
#include <glm/glm.hpp>
#include <vulkan/vulkan_raii.hpp>

namespace vkengine {

struct Vertex {
  glm::vec2 pos;
  glm::vec3 color;

  static vk::VertexInputBindingDescription get_binding_description() {
    return {/* index of binding */
            .binding = 0,
            /* nbr bytes from one entry to next */
            .stride = sizeof(Vertex),
            /* instanced vs per-Vertex rendering (TODO) */
            .inputRate = vk::VertexInputRate::eVertex};
  }

  static std::array<vk::VertexInputAttributeDescription, 2>
  get_attribute_descriptions() {
    return {
        vk::VertexInputAttributeDescription{.location = 0,
                                            .binding = 0,
                                            .format = vk::Format::eR32G32Sfloat,
                                            .offset = offsetof(Vertex, pos)},
        vk::VertexInputAttributeDescription{.location = 1,
                                            .binding = 0,
                                            .format =
                                                vk::Format::eR32G32B32Sfloat,
                                            .offset = offsetof(Vertex, color)},
    };
  }
};

struct MVP {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
};

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

  /* This should be set externally by whatever window manager library is used
   * (e.g., SDL3). This is used because it is not guranteed that the GPU driver
   * returns vk::Result::eErrorOutOfDateKHR upon acquireNextImage call */
  bool is_window_resized = false;

private:
  void init_window();

  void init_surface();

  /* */
  void init_physical_device();

  /* */
  void init_vkinstance();

  void init_logical_device();

  void init_swap_chain();

  void cleanup_swap_chain();

  void recreate_swap_chain();

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

  void init_descriptor_set_layouts();

  void init_graphics_pipeline();

  void create_command_pool();

  void create_vertex_buffer();

  void create_index_buffer();

  void create_uniform_buffers();

  void create_command_buffers();

  void record_command_buffer(uint32_t curr_frame_idx,
                             uint32_t swapchain_image_idx) const;

  void create_sync_objects();

  void transition_image_layout(vk::CommandBuffer cb, vk::Image image,
                               vk::ImageLayout old_layout,
                               vk::ImageLayout new_layout,
                               vk::AccessFlags2 src_access_mask,
                               vk::AccessFlags2 dst_access_mask,
                               vk::PipelineStageFlags2 src_stage_mask,
                               vk::PipelineStageFlags2 dst_stage_mask) const;

  [[nodiscard]] vk::raii::ShaderModule
  create_shader_module(const std::vector<char> &code) const;

  /* copies data from source buffer to destination buffer on the device (i.e.,
   * data copy operation is performed using a vkCmdCopyBuffer call).
   *
   * A command buffer is spawned, the vkCmdCopyBuffer call is recorded, the
   * execution of the command buffer is waited upon. */
  void copy_buffer(vk::raii::Buffer &src, vk::raii::Buffer &dst,
                   vk::DeviceSize size);

  void update_mvp(uint32_t curr_frame);

  void create_descriptor_pool();

  void create_descriptor_sets();

  /****************************** STATIC UTILS ********************************/

  /* Returns true if the corresponding bit index is set in the provided
   * bitfield. */
  template <std::unsigned_integral T, std::unsigned_integral P>
  static inline bool is_bit_set(T bitfield, P bit_idx) {
    return bitfield & (1 << bit_idx);
  }

  template <typename T, typename P>
  static inline bool is_all_bits_set(T bitfield, P bitmask) {
    return (bitfield & bitmask) == bitmask;
  }

  static void create_buffer(vk::raii::PhysicalDevice const &physical_device,
                            vk::raii::Device const &device, vk::DeviceSize size,
                            vk::BufferUsageFlags usage_flags,
                            vk::MemoryPropertyFlags mem_props,
                            vk::raii::Buffer &buffer,
                            vk::raii::DeviceMemory &dev_mem);

  static uint32_t
  find_memory_type(vk::raii::PhysicalDevice const &physical_device,
                   uint32_t mem_type_filter,
                   vk::MemoryPropertyFlags properties);

  /****************************************************************************/

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
  vk::raii::PipelineLayout m_PipelineLayout = nullptr;
  vk::raii::Pipeline m_Pipeline = nullptr;
  vk::raii::CommandPool m_GraphicsCmdPool = nullptr;
  std::vector<vk::raii::CommandBuffer> m_GraphicsCmdBuffers;

  /*************************** VERTEX BUFFER SHIT *****************************/

  vk::raii::Buffer m_VertexBuff{nullptr};
  vk::raii::DeviceMemory m_VertexBuffMemory{nullptr};
  vk::raii::Buffer m_IndexBuff{nullptr};
  vk::raii::DeviceMemory m_IndexBuffMemory{nullptr};

  /****************************************************************************/

  /***************************** SYNC PRIMITIVES ******************************/

  std::vector<vk::raii::Semaphore> m_PresentCompleteSems;
  std::vector<vk::raii::Semaphore> m_RenderFinishedSems;
  std::vector<vk::raii::Fence> m_DrawFences;

  /****************************************************************************/

  /***************************** DESCRIPTOR SETS ******************************/

  vk::raii::DescriptorPool m_DesriptorPool{nullptr};
  vk::raii::DescriptorSetLayout m_DesriptorSetLayout{nullptr};
  std::vector<vk::raii::DescriptorSet> m_DescriptorSets;
  std::vector<vk::raii::Buffer> m_UniformBuffers;
  std::vector<vk::raii::DeviceMemory> m_UniformBufferMemories;
  std::vector<void *> m_UniformBufferMaps;

  /****************************************************************************/

  /***************************** SWAPCHAIN STUFF ******************************/

  vk::raii::SwapchainKHR m_SwapChain = nullptr;
  /* this is NOT a vk::raii::Image vector because the vk::raii::SwapchainKHR is
   * the object owning the swapchain images and we are just copying their
   * handles here */
  std::vector<vk::Image> m_SwapChainImgs;
  std::vector<vk::raii::ImageView> m_SwapChainImgViews;
  vk::Format m_SwapChainImgFormat{vk::Format::eUndefined};
  vk::Extent2D m_SwapChainExtent;

  /****************************************************************************/

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

  /********************* CONST DATA FOR TESTING PURPOSES **********************/

  /* Vertex triangle data (for testing purposes) */
  const std::vector<Vertex> TEST_RECTANGLE_VERTICES{
      {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
      {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
      {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
      {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}},
  };

  const std::vector<uint16_t> TEST_RECTANGLE_INDICES{
      0, 1, 2, /* triangle 0 */
      2, 3, 0, /* triangle 1 */
  };

  /****************************************************************************/
};

} // namespace vkengine
