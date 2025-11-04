cmake_minimum_required(VERSION 3.2)

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
## Shader Compilation Function
################################################################################
# compile shaders into build/shaders/ subfolder
set(SHADERS_IN_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(SHADERS_OUT_DIR ${CMAKE_BINARY_DIR}/shaders)
set(SLANGC_EXECUTABLE "slangc")

# ugly cmake stuff ...
function (add_slang_shader_target TARGET)
  cmake_parse_arguments("SHADER" "" "" "SOURCES" ${ARGN})
  cmake_path(GET SHADER_SOURCES STEM SHADER_SOURCES_FILENAME)

  set(SHADER_OUTPUT_PATH ${SHADERS_OUT_DIR}/${SHADER_SOURCES_FILENAME}.spv)
  message("SHADER_OUTPUT_PATH=${SHADER_OUTPUT_PATH}")

  add_custom_command(
    OUTPUT ${SHADER_OUTPUT_PATH}
    COMMAND ${SLANGC_EXECUTABLE} ${SHADER_SOURCES} -target spirv
      -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name
      -entry vert_main -entry frag_main -o ${SHADER_OUTPUT_PATH}
    WORKING_DIRECTORY ${SHADERS_IN_DIR}
    DEPENDS ${SHADERS_IN_DIR} ${SHADER_SOURCES}
    COMMENT "Compiling Slang shaders"
    VERBATIM
  )
  add_custom_target(${TARGET} DEPENDS ${SHADER_OUTPUT_PATH})
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
    VkEngine::VkEngine
)
