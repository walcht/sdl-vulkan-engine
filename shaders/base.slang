/*
static float2 positions[3] = float2[](
  float2(0.0, -0.5),
  float2(0.5, 0.5),
  float2(-0.5, 0.5)
);

static float3 colors[3] = float3[](
  float3(1.0, 0.0, 0.0),
  float3(0.0, 1.0, 0.0),
  float3(0.0, 0.0, 1.0)
);
*/

struct VertexInput {
  float2 position;
  float3 color;
}

struct VertexOutput {
  float3 color;
  float4 sv_position : SV_Position;
};

[shader("vertex")]
VertexOutput vert_main(VertexInput i) {
  VertexOutput o;
  o.sv_position = float4(i.position, 0.0, 1.0);
  o.color = i.color;
  return o;
}

[shader("fragment")]
float4 frag_main(VertexOutput vert_in) : SV_Target {
  float3 color = vert_in.color;
  return float4(color, 1.0);
}
