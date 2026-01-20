//////////////////////////////////////////
// ATTRIBUTES
//////////////////////////////////////////
struct VertexInput {
  float3 pos;
  float3 color;
  float2 uv;
}

struct VertexOutput {
  float4 pos : SV_Position;
  float3 color;
  float2 uv;
};

//////////////////////////////////////////
// DESCRIPTOR RESOURCES
//////////////////////////////////////////
struct UniformBuffer {
  float4x4  model;
  float4x4  view;
  float4x4  proj;
};
ConstantBuffer<UniformBuffer> mvp;

Sampler2D texture;

//////////////////////////////////////////
// PUSH CONSTANTS
//////////////////////////////////////////

[shader("vertex")]
VertexOutput vert_main(VertexInput i) {
  VertexOutput o;
  o.pos = mul(mvp.proj, mul(mvp.view, mul(mvp.model, float4(i.pos, 1.0))));
  o.color = i.color;
  o.uv = i.uv;
  return o;
}

[shader("fragment")]
float4 frag_main(VertexOutput vert_in) : SV_Target {
  return texture.Sample(vert_in.uv);
}
