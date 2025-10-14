cmake_minimum_required(VERSION 3.11)

# instruct CMake that what follows is the description of a distinct software
# project of a given name
project(SDLVulkanEngine LANGUAGES CXX)

# we are using C++20
set(CMAKE_CXX_STANDARD 20 CACHE STRING "CMake CXX Standard")

# C++20 modules support is only available for new compilers. See:
# https://cmake.org/cmake/help/latest/manual/cmake-cxxmodules.7.html
# if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
#   if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "15")
#     message(
#       FATAL_ERROR
#       "GCC version >= 15 is required for C++20 modules support"
#     )
#   endif()
# elseif(CMAKE_CXX_COMPILER_ID STREQUAL "CLANG")
#   if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "18.1.2")
#     message(
#       FATAL_ERROR
#       "CLANG version >= 18.1.2 is required for C++20 modules support"
#     )
#   endif()
# elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
#   if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "14.36")
#     message(
#       FATAL_ERROR
#       "MSVC version >= 14.36 is required for C++20 modules support"
#     )
#   endif()
# endif()

# enable C++ module dependency scanning
# set(CMAKE_CXX_SCAN_FOR_MODULES ON)

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
## FETCH DEPENDENCIES
################################################################################
include(FetchContent)
# SDL3 core - for creating a Vulkan context
FetchContent_Declare(
  SDL3
  GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
  GIT_TAG a8589a84226a6202831a3d49ff4edda4acab9acd # v3.2.24
)
FetchContent_MakeAvailable(SDL3)

# GLM - graphics-related mathematics library
#CPMAddPackage("gh:g-truc/glm@1.0.1")

# gtest - for unit testing
#CPMAddPackage("gh:google/googletest@1.17.0")

# cgltf - gltf loader
# CPMAddPackage("gh:jkuhlmann/cgltf@1.15")

# imgui - imediate mode GUI
# CPMAddPackage("gh:ocornut/imgui@1.92.3")

# build the engine code as a shared library separately - this has the benefit of
# not recompiling non-changing engine code and having some potential language
# bindings in a different language (e.g., lua - to make development easier)

################################################################################
## EXECUTABLE BUILD INSTRUCTIONS
################################################################################

# our final target is an executalbe (not a library)
add_executable(${PROJECT_NAME} main.cpp)

# add our VkEngine
add_subdirectory(src/)

# link with SDL3 shared libraries, Vulkan, etc.
target_link_libraries(
  ${PROJECT_NAME} PRIVATE
    SDL3::SDL3
    Vulkan::Vulkan
    VkEngine::VkEngine
    # Vulkan::cppm  # Vulkan module - see VULKAN SETUP above
)
