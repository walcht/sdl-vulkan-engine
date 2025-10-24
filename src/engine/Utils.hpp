#pragma once

#include <cstdint>
#include <vector>
#include <vulkan/vulkan_raii.hpp>
#include <filesystem>

namespace vkengine
{

namespace utils
{

uint32_t find_first_graphics_familty_queue_idx(
    const std::vector<vk::QueueFamilyProperties> &queue_family_props);

std::vector<char> read_binary(const std::string &filepath);

std::filesystem::path get_spirv_shaders_path();

}        // namespace utils

}        // namespace vkengine
