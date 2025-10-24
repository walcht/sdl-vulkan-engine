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
## Shader Compilation Function
################################################################################
# this is quite frankly the ugliest, pos function declaration syntax I have
# ever seen in my entire life ...
set(SLANGC_EXECUTABLE "slangc")
function (add_slang_shader_target TARGET)
  cmake_parse_arguments("SHADER" "" "" "SOURCES" ${ARGN})
  set(SHADERS_DIR ${CMAKE_SOURCE_DIR}/shaders)
  add_custom_command(
    OUTPUT ${SHADERS_DIR}/slang.spv
    COMMAND ${SLANGC_EXECUTABLE} ${SHADER_SOURCES} -target spirv
      -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name
      -entry vert_main -entry frag_main -o slang.spv
    WORKING_DIRECTORY ${SHADERS_DIR}
    DEPENDS ${SHADERS_DIR} ${SHADER_SOURCES}
    COMMENT "Compiling Slang Shaders"
    VERBATIM
  )
  add_custom_target(${TARGET} DEPENDS ${SHADERS_DIR}/slang.spv)
endfunction()

################################################################################
## EXECUTABLE BUILD INSTRUCTIONS
################################################################################

# our final target is an executalbe (not a library)
add_executable(${PROJECT_NAME} main.cpp)

# add our VkEngine
add_subdirectory(src/)

# compile the shaders into SPIRV
add_slang_shader_target(SPIRV_SHADERS
  SOURCES
    ${CMAKE_SOURCE_DIR}/shaders/base.slang
  )

# make sure that shaders are compiled/built before the project
add_dependencies(${PROJECT_NAME} SPIRV_SHADERS)

# link with SDL3 shared libraries, Vulkan, etc.
target_link_libraries(
  ${PROJECT_NAME} PRIVATE
    SDL3::SDL3
    VkEngine::VkEngine
    # Vulkan::cppm  # Vulkan module - see VULKAN SETUP above
)
