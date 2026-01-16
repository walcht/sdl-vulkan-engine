//////////////////////////////////////////
// ATTRIBUTES
//////////////////////////////////////////
struct VertexInput {
  float2 pos;
  float3 color;
}

struct VertexOutput {
  float4 pos : SV_Position;
  float3 color;
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

//////////////////////////////////////////
// PUSH CONSTANTS
//////////////////////////////////////////

[shader("vertex")]
VertexOutput vert_main(VertexInput i) {
  VertexOutput o;
  o.pos = mul(mvp.proj, mul(mvp.view, mul(mvp.model, float4(i.pos, 0.0, 1.0))));
  o.color = i.color;
  return o;
}

[shader("fragment")]
float4 frag_main(VertexOutput vert_in) : SV_Target {
  float3 color = vert_in.color;
  return float4(color, 1.0);
}
