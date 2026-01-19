#pragma once

#include "Engine.hpp"
namespace vkengine {
class GameObject {
public:
  void set_vertex_buffer_data();
  void set_vertex_buffer_data(std::vector<Vertex> &&data) {
    m_vertex_data = std::move(data);
  }

private:
  std::vector<Vertex> m_vertex_data;
};
} // namespace vkengine
