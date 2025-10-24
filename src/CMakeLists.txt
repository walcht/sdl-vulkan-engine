add_library(VkEngine STATIC)
add_library(VkEngine::VkEngine ALIAS VkEngine)

################################################################################
## VULKAN SETUP
################################################################################
find_package(Vulkan REQUIRED)

# initially - I was going to use C++20 modules but due to poor clangd support
# for them, I decided to stick to the old header includes

# add_library(VkCppModule)
# add_library(Vulkan::cppm ALIAS VkCppModule)
# 
# target_compile_definitions(VkCppModule
#   PUBLIC
#   VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
#   VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
# )
# 
# target_include_directories(VkCppModule PRIVATE "${Vulkan_INCLUDE_DIR}")
# target_link_libraries(VkCppModule PUBLIC Vulkan::Vulkan)
# set_target_properties(VkCppModule PROPERTIES CXX_STANDARD 20)
# target_sources(VkCppModule
#   PUBLIC
#   FILE_SET cxx_modules TYPE CXX_MODULES
#   BASE_DIRS
#   "${Vulkan_INCLUDE_DIR}"
#   FILES
#   "${Vulkan_INCLUDE_DIR}/vulkan/vulkan.cppm"
# )

################################################################################
## ENGINE SOURCE FILES (*.cpp)
################################################################################
target_sources(VkEngine
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/engine/Engine.cpp
    ${CMAKE_CURRENT_LIST_DIR}/engine/Utils.cpp
)


################################################################################
## ENGINE HEADER FILES (*.hpp)
################################################################################
target_include_directories(VkEngine
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/engine/
)


################################################################################
## LINK WITH 3rd PATIE LIBRARIES
################################################################################
target_link_libraries(VkEngine
  PUBLIC
    Vulkan::Vulkan
)


################################################################################
## INSTALLATION
################################################################################
install(TARGETS VkEngine DESTINATION .)
