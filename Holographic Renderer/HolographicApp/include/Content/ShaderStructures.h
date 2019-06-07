#pragma once

namespace HolographicApp
{
    // Constant buffer used to send sampling parameters to the vertex shader.
    struct SamplingCameraConstantBuffer {
        DirectX::XMFLOAT4X4 transformation;
        DirectX::XMFLOAT4 parameters;
        DirectX::XMFLOAT4 resolution;
        DirectX::XMFLOAT4 rendering_parameters;
    };

    // Assert that the constant buffer remains 16-byte aligned (best practice).
    static_assert((sizeof(SamplingCameraConstantBuffer) % (sizeof(float) * 4)) == 0, "Model constant buffer size must be 16-byte aligned (16 bytes is the length of four floats).");

    // Used to send per-vertex data to the vertex shader.
    struct VertexPosition {
        DirectX::XMFLOAT3 pos;
    };

    // Constant buffer used to send hologram position transform to the shader pipeline.
    struct ModelConstantBuffer
    {
        DirectX::XMFLOAT4X4 model;
    };

    // Assert that the constant buffer remains 16-byte aligned (best practice).
    static_assert((sizeof(ModelConstantBuffer) % (sizeof(float) * 4)) == 0, "Model constant buffer size must be 16-byte aligned (16 bytes is the length of four floats).");

	// Constant buffer used to send hologram position transform to the shader pipeline.
	struct FibConstantBuffer {
		DirectX::XMFLOAT4 Origin_R;
		DirectX::XMFLOAT4 CameraPos[2];
		DirectX::XMFLOAT4 Resolution; //M_fib, M_used, N_fib
		DirectX::XMFLOAT4X4 normal_mat;
	};

	// Assert that the constant buffer remains 16-byte aligned (best practice).
	static_assert((sizeof(FibConstantBuffer) % (sizeof(float) * 4)) == 0, "Model constant buffer size must be 16-byte aligned (16 bytes is the length of four floats).");


    // Used to send per-vertex data to the vertex shader.
    struct VertexPositionColor
    {
        DirectX::XMFLOAT3 pos;
        DirectX::XMFLOAT3 color;
    };
}