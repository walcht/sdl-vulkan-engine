#include "Utils.hpp"
#include <cstdint>
#include <vector>
#include <vulkan/vulkan_raii.hpp>
#include <fstream>

namespace fs = std::filesystem;

namespace vkengine
{

namespace utils
{
uint32_t find_first_graphics_familty_queue_idx(
    const std::vector<vk::QueueFamilyProperties> &queue_family_props)
{
	auto itr = std::ranges::find_if(queue_family_props, [](auto &qfp) {
		return (qfp.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlags>(0);
	});
	if (itr == queue_family_props.end())
	{
		throw std::runtime_error("failed to find a queue family supporting graphics cmds");
	}
	return static_cast<uint32_t>(std::distance(queue_family_props.begin(), itr));
}

std::vector<char> read_binary(const std::string &filepath)
{
	std::ifstream file(filepath, std::ios::ate | std::ios::binary);
	if (!file.is_open())
	{
		throw std::runtime_error(std::format("failed to open file {}", filepath));
	}

	std::vector<char> bytearray(file.tellg());
	file.read(bytearray.data(), static_cast<std::streamsize>(bytearray.size()));
	file.close();
	return bytearray;
}

fs::path get_spirv_shaders_path()
{
	/* ../../../shaders/ */
	return fs::current_path().parent_path().parent_path().parent_path() / "shaders";
}

}        // namespace utils

}        // namespace vkengine
