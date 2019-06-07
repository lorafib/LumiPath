#pragma once
#include "..\Common\DeviceResources.h"
#include "..\Common\StepTimer.h"
#include "ShaderStructures.h"

namespace HolographicApp
{
    // This sample renderer instantiates a basic rendering pipeline.
    class FibRenderer
    {
    public:
        FibRenderer(const std::shared_ptr<DX::DeviceResources>& deviceResources);
        void CreateDeviceDependentResources();
        void ReleaseDeviceDependentResources();
        void Update(const DX::StepTimer& timer);
        void Render();

        // Repositions the sample hologram.
        void PositionHologram(Windows::UI::Input::Spatial::SpatialPointerPose^ pointerPose);

        // Property accessors.
        void SetPosition(Windows::Foundation::Numerics::float3 pos) { m_position = pos;  }
        Windows::Foundation::Numerics::float3 GetPosition()         { return m_position; }

    private:
        // Cached pointer to device resources.
        std::shared_ptr<DX::DeviceResources>            m_deviceResources;

        // Direct3D resources for cube geometry.
        Microsoft::WRL::ComPtr<ID3D11InputLayout>       m_inputLayout;
        Microsoft::WRL::ComPtr<ID3D11Buffer>            m_vertexBuffer;
        Microsoft::WRL::ComPtr<ID3D11Buffer>            m_indexBuffer;
        Microsoft::WRL::ComPtr<ID3D11VertexShader>      m_vertexShader;
        Microsoft::WRL::ComPtr<ID3D11GeometryShader>    m_geometryShader;
        Microsoft::WRL::ComPtr<ID3D11PixelShader>       m_pixelShader;
		Microsoft::WRL::ComPtr<ID3D11Buffer>            m_modelConstantBuffer;
		Microsoft::WRL::ComPtr<ID3D11Buffer>            m_fibConstantBuffer;
		//TEx
		Microsoft::WRL::ComPtr<ID3D11ShaderResourceView>    m_textureRV0;
		Microsoft::WRL::ComPtr<ID3D11ShaderResourceView>    m_textureRV1;

		// Rasterizer states, for different rendering modes.
		Microsoft::WRL::ComPtr<ID3D11RasterizerState>   m_defaultRasterizerState;
        // System resources for cube geometry.
        ModelConstantBuffer                             m_modelConstantBufferData;
		FibConstantBuffer								m_fibConstantBufferData;
        uint32                                          m_indexCount = 0;

        // Variables used with the rendering loop.
        bool                                            m_loadingComplete = false;
        float                                           m_degreesPerSecond = 5.f;
        Windows::Foundation::Numerics::float3           m_position = { 0.f, 0.f, 0.f };
		float											m_radius = 0.33;
        // If the current D3D Device supports VPRT, we can avoid usi5ng a geometry
        // shader just to set the render target array index.s
        bool                                            m_usingVprtShaders = false;
    
	// Textures
		inline void loadTexture(std::string filename = "assets/hypersphere_0_2.ppm");
		inline void loadTextureParted();

		unsigned char *m_texture = nullptr;
		unsigned char *m_texture_part0 = nullptr;
		unsigned char *m_texture_part1 = nullptr; 

		const uint32_t MAX_TEXTURE_DIM_SIZE = 16384;
		//  m_width_fib matches the size of the origin point set for the mapping
		//, m_width_used is the number of origin that are actually used. 
		//  m_width_used should be equal to the texture size!
		uint32_t m_width_fib, m_width_used;
		//  m_height_fib matches the size of the direction point set for the mapping
		// m_height_part0 is MAX_TEXTURE_DIM_SIZE, m_height_part0 is rest 
		uint32_t m_height_fib, m_height_part0, m_height_part1;
		//old params temporarily used during texture loading
		uint32_t m_maximum, m_width, m_height;
		float m_sampling_width = 1268, m_sampling_height = 720;
    
	};
}
