// Per-vertex data from the vertex shader.
struct GeometryShaderInput
{
	min16float4 pos     : SV_POSITION;
	min16float3 normal   : NORMAL0;
	min16float3 pos_ws   : POSITIONT;
	uint        instId  : TEXCOORD0;  // SV_InstanceID % 2
};

// Per-vertex data passed to the rasterizer.
struct GeometryShaderOutput
{
    min16float4 pos     : SV_POSITION;
	min16float3 normal   : NORMAL0;
	min16float3 pos_ws   : POSITIONT;
    uint        rtvId   : SV_RenderTargetArrayIndex;
};

// This geometry shader is a pass-through that leaves the geometry unmodified 
// and sets the render target array index.
[maxvertexcount(3)]
void main(triangle GeometryShaderInput input[3], inout TriangleStream<GeometryShaderOutput> outStream)
{
    GeometryShaderOutput output;
    [unroll(3)]
    for (int i = 0; i < 3; ++i)
    {
        output.pos   = input[i].pos;
		output.pos_ws = input[i].pos_ws;
        output.normal = input[i].normal;
        output.rtvId = input[i].instId;
        outStream.Append(output);
    }
}
