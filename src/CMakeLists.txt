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
## FETCH DEPENDENCIES
################################################################################
include(FetchContent)
# SDL3 core - for creating a Vulkan context
fetchcontent_declare(
  SDL3
  GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
  GIT_TAG a8589a84226a6202831a3d49ff4edda4acab9acd # v3.2.24
)
fetchcontent_makeavailable(SDL3)


# fetch stb_image.h in case it is not already available at external/stb/
# this is a bit ugly since we have to fetch the entire repo for a single file
# but this is the only method that works (that I know of)
set(STB_DIR ${CMAKE_SOURCE_DIR}/external/stb)
if (NOT EXISTS ${STB_DIR}/stb_image.h)
  message("fetching stb git repository for stb_image.h dependency ...")
  if (NOT EXISTS ${STB_DIR})
    file(MAKE_DIRECTORY ${STB_DIR})
    file(DOWNLOAD https://github.com/nothings/stb/archive/refs/heads/master.zip
      ${STB_DIR}/stb.zip SHOW_PROGRESS)
    file(ARCHIVE_EXTRACT INPUT ${STB_DIR}/stb.zip DESTINATION ${STB_DIR})
    file(COPY ${STB_DIR}/stb-master/stb_image.h DESTINATION ${STB_DIR})
    file(REMOVE_RECURSE ${STB_DIR}/stb-master ${STB_DIR}/stb.zip)
  endif()
  message("stb_image.h available at ${STB_DIR}/external/stb/stb_image.h")
endif()

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
    ${CMAKE_SOURCE_DIR}/external/
)


################################################################################
## LINK WITH 3rd PATIE LIBRARIES
################################################################################
target_link_libraries(VkEngine
  PUBLIC
    Vulkan::Vulkan
    # Vulkan::cppm  # Vulkan module - see VULKAN SETUP above
    SDL3::SDL3
)


################################################################################
## INSTALLATION
################################################################################
install(TARGETS VkEngine DESTINATION .)
