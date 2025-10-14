add_library(VkEngine STATIC)
add_library(VkEngine::VkEngine ALIAS VkEngine)

# add source files here - essentially any .cpp in any relative subdirectory
# should be added
target_sources(VkEngine
  PRIVATE
    engine/Engine.cpp
)


target_include_directories(VkEngine
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/engine/
)

install(TARGETS VkEngine DESTINATION .)
