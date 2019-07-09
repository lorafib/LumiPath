
#include "pch.h"
#include "FibRenderer.h"
#include "Common\DirectXHelper.h"
#include <fstream>
#include <sstream>


#define M_PI 3.1415926535897932384626433832795028841971693993751058209749445923

using namespace HolographicApp;
using namespace Concurrency;
using namespace DirectX;
using namespace Windows::Foundation::Numerics;
using namespace Windows::UI::Input::Spatial;

// Loads vertex and pixel shaders from files and instantiates the cube geometry.
FibRenderer::FibRenderer(const std::shared_ptr<DX::DeviceResources>& deviceResources) :
    m_deviceResources(deviceResources)
{
    CreateDeviceDependentResources();
}

// This function uses a SpatialPointerPose to position the world-locked hologram
// two meters in front of the user's heading.
void FibRenderer::PositionHologram(SpatialPointerPose^ pointerPose)
{
    if (pointerPose != nullptr)
    {
        // Get the gaze direction relative to the given coordinate system.
        const float3 headPosition    = pointerPose->Head->Position;
        const float3 headDirection   = pointerPose->Head->ForwardDirection;

        // The hologram is positioned two meters along the user's gaze direction.
        constexpr float distanceFromUser    = 0.75f; // meters
        const float3 gazeAtTwoMeters        = headPosition + (distanceFromUser * headDirection);

        // This will be used as the translation component of the hologram's
        // model transform.
        SetPosition(gazeAtTwoMeters);
    }
}

// Called once per frame. Rotates the cube, and calculates and sets the model matrix
// relative to the position transform indicated by hologramPositionTransform.
void FibRenderer::Update(const DX::StepTimer& timer)
{
    // Rotate the cube.
    // Convert degrees to radians, then convert seconds to rotation angle.
    const float    radiansPerSecond = XMConvertToRadians(m_degreesPerSecond);
    const double   totalRotation    = /*timer.GetTotalSeconds() **/ radiansPerSecond;
    const float    radians          = static_cast<float>(fmod(totalRotation, XM_2PI));
	const XMMATRIX modelRotation = XMMatrixRotationY(-radians);// XMMatrixMultiply(XMMatrixRotationX(XMConvertToRadians(M_PI)), XMMatrixRotationY(-radians));
	const XMMATRIX modelScale = XMMatrixScaling(m_radius, m_radius, m_radius);

    // Position the cube.
    const XMMATRIX modelTranslation = XMMatrixTranslationFromVector(XMLoadFloat3(&m_position));

    // Multiply to get the transform matrix.
    // Note that this transform does not enforce a particular coordinate system. The calling
    // class is responsible for rendering this content in a consistent manner.
    //const XMMATRIX modelTransform   = XMMatrixMultiply(modelRotation, modelTranslation);
    XMMATRIX modelTransform   = XMMatrixMultiply(modelScale, modelRotation);
    modelTransform   = XMMatrixMultiply(modelTransform, modelTranslation);

    // The view and projection matrices are provided by the system; they are associated
    // with holographic cameras, and updated on a per-camera basis.
    // Here, we provide the model transform for the sample hologram. The model transform
    // matrix is transposed to prepare it for the shader.
    XMStoreFloat4x4(&m_modelConstantBufferData.model, XMMatrixTranspose(modelTransform));

	//set params fr fibconstant buffer to do inverse mapping
	XMStoreFloat4(&m_fibConstantBufferData.Origin_R, XMVectorSet((FLOAT)m_position.x, (FLOAT)m_position.y, (FLOAT)m_position.z, (FLOAT)m_radius));
	XMStoreFloat4(&m_fibConstantBufferData.Resolution, XMVectorSet((FLOAT)m_width_fib, (FLOAT)m_width_used, (FLOAT)m_height_fib, 1.0f));
	XMStoreFloat4x4(&m_fibConstantBufferData.normal_mat, XMMatrixTranspose(XMMatrixInverse(nullptr, modelTransform)));
	//XMFLOAT4 pos[2] = 
	//XMStoreFloat4(&m_fibConstantBufferData.CameraPos, XMVectorSet((FLOAT)m_width_fib, (FLOAT)m_width_used, (FLOAT)m_height_fib, !renderTexture ? 1.f : -1.f));



    // Loading is asynchronous. Resources must be created before they can be updated.
    if (!m_loadingComplete)
    {
        return;
    }

    // Use the D3D device context to update Direct3D device-based resources.
    const auto context = m_deviceResources->GetD3DDeviceContext();

    // Update the model transform buffer for the hologram.
    context->UpdateSubresource(
        m_modelConstantBuffer.Get(),
        0,
        nullptr,
        &m_modelConstantBufferData,
        0,
        0
        );

	// Update the fib params buffer for the hologram.
	context->UpdateSubresource(
		m_fibConstantBuffer.Get(),
		0,
		nullptr,
		&m_fibConstantBufferData,
		0,
		0
	);
}

// Renders one frame using the vertex and pixel shaders.
// On devices that do not support the D3D11_FEATURE_D3D11_OPTIONS3::
// VPAndRTArrayIndexFromAnyShaderFeedingRasterizer optional feature,
// a pass-through geometry shader is also used to set the render 
// target array index.
void FibRenderer::Render()
{
    // Loading is asynchronous. Resources must be created before drawing can occur.
    if (!m_loadingComplete)
    {
        return;
    }

    const auto context = m_deviceResources->GetD3DDeviceContext();

    // Each vertex is one instance of the VertexPositionColor struct.
    const UINT stride = sizeof(VertexPositionColor);
    const UINT offset = 0;
    context->IASetVertexBuffers(
        0,
        1,
        m_vertexBuffer.GetAddressOf(),
        &stride,
        &offset
        );
    context->IASetIndexBuffer(
        m_indexBuffer.Get(),
        DXGI_FORMAT_R16_UINT, // Each index is one 16-bit unsigned integer (short).
        0
        );
    context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    context->IASetInputLayout(m_inputLayout.Get());

    // Attach the vertex shader.
    context->VSSetShader(
        m_vertexShader.Get(),
        nullptr,
        0
        );
    // Apply the model constant buffer to the pixel shader.
    context->VSSetConstantBuffers(
        0,
        1,
        m_modelConstantBuffer.GetAddressOf()
        );

    if (!m_usingVprtShaders)
    {
        // On devices that do not support the D3D11_FEATURE_D3D11_OPTIONS3::
        // VPAndRTArrayIndexFromAnyShaderFeedingRasterizer optional feature,
        // a pass-through geometry shader is used to set the render target 
        // array index.
        context->GSSetShader(
            m_geometryShader.Get(),
            nullptr,
            0
            );
    }

    // Attach the pixel shader.
    context->PSSetShader(
        m_pixelShader.Get(),
        nullptr,
        0
        );

	// Apply the model constant buffer to the vertex shader.
	context->PSSetConstantBuffers(
		0,
		1,
		m_fibConstantBuffer.GetAddressOf()
	);

	// Bind Texture
	context->PSSetShaderResources(0, 1, m_textureRV0.GetAddressOf());
	context->PSSetShaderResources(1, 1, m_textureRV1.GetAddressOf());

    // Draw the objects.
    context->DrawIndexedInstanced(
        m_indexCount,   // Index count per instance.
        2,              // Instance count.
        0,              // Start index location.
        0,              // Base vertex location.
        0               // Start instance location.
        );
}



void FibRenderer::CreateDeviceDependentResources()
{
    m_usingVprtShaders = m_deviceResources->GetDeviceSupportsVprt();

    // On devices that do support the D3D11_FEATURE_D3D11_OPTIONS3::
    // VPAndRTArrayIndexFromAnyShaderFeedingRasterizer optional feature
    // we can avoid using a pass-through geometry shader to set the render
    // target array index, thus avoiding any overhead that would be 
    // incurred by setting the geometry shader stage.
    std::wstring vertexShaderFileName = m_usingVprtShaders ? L"ms-appx:///VprtCubeSpinnerVertexShader.cso" : L"ms-appx:///CubeSpinnerVertexShader.cso";

    // Load shaders asynchronously.
    task<std::vector<byte>> loadVSTask = DX::ReadDataAsync(vertexShaderFileName);
    task<std::vector<byte>> loadPSTask = DX::ReadDataAsync(L"ms-appx:///CubeSpinnerPixelShader.cso");

    task<std::vector<byte>> loadGSTask;
    if (!m_usingVprtShaders)
    {
        // Load the pass-through geometry shader.
        loadGSTask = DX::ReadDataAsync(L"ms-appx:///CubeSpinnerGeometryShader.cso");
    }

    // After the vertex shader file is loaded, create the shader and input layout.
    task<void> createVSTask = loadVSTask.then([this] (const std::vector<byte>& fileData)
    {
        DX::ThrowIfFailed(
            m_deviceResources->GetD3DDevice()->CreateVertexShader(
                fileData.data(),
                fileData.size(),
                nullptr,
                &m_vertexShader
                )
            );

        constexpr std::array<D3D11_INPUT_ELEMENT_DESC, 2> vertexDesc =
        {{
            { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0,  0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
            { "COLOR",    0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        }};

        DX::ThrowIfFailed(
            m_deviceResources->GetD3DDevice()->CreateInputLayout(
                vertexDesc.data(),
                vertexDesc.size(),
                fileData.data(),
                fileData.size(),
                &m_inputLayout
                )
            );
    });

    // After the pixel shader file is loaded, create the shader and constant buffer.
    task<void> createPSTask = loadPSTask.then([this] (const std::vector<byte>& fileData)
    {
        DX::ThrowIfFailed(
            m_deviceResources->GetD3DDevice()->CreatePixelShader(
                fileData.data(),
                fileData.size(),
                nullptr,
                &m_pixelShader
                )
            );

        const CD3D11_BUFFER_DESC constantBufferDesc(sizeof(ModelConstantBuffer), D3D11_BIND_CONSTANT_BUFFER);
        DX::ThrowIfFailed(
            m_deviceResources->GetD3DDevice()->CreateBuffer(
                &constantBufferDesc,
                nullptr,
                &m_modelConstantBuffer
                )
            );

		const CD3D11_BUFFER_DESC fibconstantBufferDesc(sizeof(FibConstantBuffer), D3D11_BIND_CONSTANT_BUFFER);
		DX::ThrowIfFailed(
			m_deviceResources->GetD3DDevice()->CreateBuffer(
				&fibconstantBufferDesc,
				nullptr,
				&m_fibConstantBuffer
			)
		);
    });

    task<void> createGSTask;
    if (!m_usingVprtShaders)
    {
        // After the pass-through geometry shader file is loaded, create the shader.
        createGSTask = loadGSTask.then([this] (const std::vector<byte>& fileData)
        {
            DX::ThrowIfFailed(
                m_deviceResources->GetD3DDevice()->CreateGeometryShader(
                    fileData.data(),
                    fileData.size(),
                    nullptr,
                    &m_geometryShader
                    )
                );
        });
    }

    // Once all shaders are loaded, create the mesh.
    task<void> shaderTaskGroup = m_usingVprtShaders ? (createPSTask && createVSTask) : (createPSTask && createVSTask && createGSTask);
    task<void> createCubeTask  = shaderTaskGroup.then([this] ()
    {
		/////////////////////////////////
		// and now witch a sphere
		const uint32_t tessU = 70;
		const uint32_t tessV = 35;
		const float maxTheta = M_PI;

		std::vector<VertexPositionColor> sphereVertices = std::vector<VertexPositionColor>();
		const uint32_t num_vertices = (tessU + 1) * tessV;
		sphereVertices.reserve(num_vertices);

		std::vector<unsigned short> indices;
		const uint32_t num_indices = 6 * tessU * (tessV - 1);
		indices.reserve(num_indices);

		float phi_step = 2.0f * M_PI / (float)tessU;
		float theta_step = maxTheta / (float)(tessV - 1);

		// Latitudinal rings.
		// Starting at the south pole going upwards on the y-axis.
		for (int latitude = 0; latitude < tessV; latitude++) // theta angle
		{
			float theta = (float)latitude * theta_step;
			float sinTheta = sinf(theta);
			float cosTheta = cosf(theta);

			float texv = (float)latitude / (float)(tessV - 1); // Range [0.0f, 1.0f]

															   // Generate vertices along the latitudinal rings.
															   // On each latitude there are tessU + 1 vertices.
															   // The last one and the first one are on identical positions, but have different texture coordinates!
															   // DAR FIXME Note that each second triangle connected to the two poles has zero area!
			for (int longitude = 0; longitude <= tessU; longitude++) // phi angle
			{
				float phi = (float)longitude * phi_step;
				float sinPhi = sinf(phi);
				float cosPhi = cosf(phi);

				float texu = (float)longitude / (float)tessU; // Range [0.0f, 1.0f]

															  // Unit sphere coordinates are the normals.
				XMFLOAT3 normal = XMFLOAT3(cosPhi * sinTheta,
					-cosTheta,                 // -y to start at the south pole.
					-sinPhi * sinTheta);
				VertexPositionColor vposcol;

				vposcol.pos = XMFLOAT3(normal.x * m_radius, normal.y * m_radius, normal.z * m_radius);
				vposcol.color = normal;
				//attrib.tangent = optix::make_float3(-sinPhi, 0.0f, -cosPhi);
				//attrib.texcoord = optix::make_float3(texu, texv, 0.0f);

				sphereVertices.push_back(vposcol);
			}
		}

		D3D11_SUBRESOURCE_DATA vertexBufferData = { 0 };
		vertexBufferData.pSysMem = sphereVertices.data();
		vertexBufferData.SysMemPitch = 0;
		vertexBufferData.SysMemSlicePitch = 0;
		const CD3D11_BUFFER_DESC vertexBufferDesc(sizeof(VertexPositionColor) * sphereVertices.size(), D3D11_BIND_VERTEX_BUFFER);
		DX::ThrowIfFailed(
			m_deviceResources->GetD3DDevice()->CreateBuffer(
				&vertexBufferDesc,
				&vertexBufferData,
				&m_vertexBuffer
			)
		);

		// We have generated tessU + 1 vertices per latitude.
		const uint32_t columns = tessU + 1;

		// Calculate indices.
		for (int latitude = 0; latitude < tessV - 1; latitude++) {
			for (int longitude = 0; longitude < tessU; longitude++) {
				indices.push_back(latitude      * columns + longitude);  // lower left
				indices.push_back(latitude      * columns + longitude + 1);  // lower right
				indices.push_back((latitude + 1) * columns + longitude + 1);  // upper right 

				indices.push_back((latitude + 1) * columns + longitude + 1);  // upper right 
				indices.push_back((latitude + 1) * columns + longitude);  // upper left
				indices.push_back(latitude      * columns + longitude);  // lower left
			}
		}

		m_indexCount = indices.size();

		D3D11_SUBRESOURCE_DATA indexBufferData = { 0 };
		indexBufferData.pSysMem = indices.data();
		indexBufferData.SysMemPitch = 0;
		indexBufferData.SysMemSlicePitch = 0;
		CD3D11_BUFFER_DESC indexBufferDesc(sizeof(unsigned short) * indices.size(), D3D11_BIND_INDEX_BUFFER);
		DX::ThrowIfFailed(
			m_deviceResources->GetD3DDevice()->CreateBuffer(
				&indexBufferDesc,
				&indexBufferData,
				&m_indexBuffer
			)
		);
		//////////////////////////////////////////////////////////////////////////////////////
		// Load texture
		loadTextureParted();
		//push part 0 to gpu
		CD3D11_TEXTURE2D_DESC textureDesc0(DXGI_FORMAT_R8G8B8A8_UNORM, m_width_used, m_height_part0, 1, 1);
		D3D11_SUBRESOURCE_DATA textureData0;
		ZeroMemory(&textureData0, sizeof(textureData0));
		textureData0.pSysMem = (void*)m_texture_part0;
		textureData0.SysMemPitch = m_width_used * 4 * sizeof(unsigned char);
		textureData0.SysMemSlicePitch = m_width_used * m_height_part0 * 4 * sizeof(unsigned char);
		Microsoft::WRL::ComPtr<ID3D11Texture2D> testTexture2D0;
		DX::ThrowIfFailed(
			m_deviceResources->GetD3DDevice()->CreateTexture2D(
				&textureDesc0,
				&textureData0,
				&testTexture2D0
			)
		);
		DX::ThrowIfFailed(
			m_deviceResources->GetD3DDevice()->CreateShaderResourceView(
				testTexture2D0.Get(),
				nullptr,
				&m_textureRV0
			)
		);
		testTexture2D0.Detach();

		//push part1 to gpu
		CD3D11_TEXTURE2D_DESC textureDesc1(DXGI_FORMAT_R8G8B8A8_UNORM, m_width_used, m_height_part1, 1, 1);
		D3D11_SUBRESOURCE_DATA textureData1;
		ZeroMemory(&textureData1, sizeof(textureData1));
		textureData1.pSysMem = (void*)m_texture_part1;
		textureData1.SysMemPitch = m_width_used * 4 * sizeof(unsigned char);
		textureData1.SysMemSlicePitch = m_width_used * m_height_part1 * 4 * sizeof(unsigned char);
		Microsoft::WRL::ComPtr<ID3D11Texture2D> testTexture2D1;
		DX::ThrowIfFailed(
			m_deviceResources->GetD3DDevice()->CreateTexture2D(
				&textureDesc1,
				&textureData1,
				&testTexture2D1
			)
		);
		DX::ThrowIfFailed(
			m_deviceResources->GetD3DDevice()->CreateShaderResourceView(
				testTexture2D1.Get(),
				nullptr,
				&m_textureRV1
			)
		);
		testTexture2D1.Detach();

		/////////////////////////////////////////////////////////////////////////////////////////////////
  //      // Load mesh vertices. Each vertex has a position and a color.
  //      // Note that the cube size has changed from the default DirectX app
  //      // template. Windows Holographic is scaled in meters, so to draw the
  //      // cube at a comfortable size we made the cube width 0.2 m (20 cm).
  //      static const std::array<VertexPositionColor, 8> cubeVertices =
  //      {{
  //          { XMFLOAT3(-0.1f, -0.1f, -0.1f), XMFLOAT3(0.0f, 0.0f, 0.0f) },
  //          { XMFLOAT3(-0.1f, -0.1f,  0.1f), XMFLOAT3(0.0f, 0.0f, 1.0f) },
  //          { XMFLOAT3(-0.1f,  0.1f, -0.1f), XMFLOAT3(0.0f, 1.0f, 0.0f) },
  //          { XMFLOAT3(-0.1f,  0.1f,  0.1f), XMFLOAT3(0.0f, 1.0f, 1.0f) },
  //          { XMFLOAT3( 0.1f, -0.1f, -0.1f), XMFLOAT3(1.0f, 0.0f, 0.0f) },
  //          { XMFLOAT3( 0.1f, -0.1f,  0.1f), XMFLOAT3(1.0f, 0.0f, 1.0f) },
  //          { XMFLOAT3( 0.1f,  0.1f, -0.1f), XMFLOAT3(1.0f, 1.0f, 0.0f) },
  //          { XMFLOAT3( 0.1f,  0.1f,  0.1f), XMFLOAT3(1.0f, 1.0f, 1.0f) },
  //      }};

  //      D3D11_SUBRESOURCE_DATA vertexBufferData = {0};
  //      vertexBufferData.pSysMem                = cubeVertices.data();
  //      vertexBufferData.SysMemPitch            = 0;
  //      vertexBufferData.SysMemSlicePitch       = 0;
  //      const CD3D11_BUFFER_DESC vertexBufferDesc(sizeof(VertexPositionColor) * cubeVertices.size(), D3D11_BIND_VERTEX_BUFFER);
  //      DX::ThrowIfFailed(
  //          m_deviceResources->GetD3DDevice()->CreateBuffer(
  //              &vertexBufferDesc,
  //              &vertexBufferData,
  //              &m_vertexBuffer
  //              )
  //          );

  //      // Load mesh indices. Each trio of indices represents
  //      // a triangle to be rendered on the screen.
  //      // For example: 2,1,0 means that the vertices with indexes
  //      // 2, 1, and 0 from the vertex buffer compose the
  //      // first triangle of this mesh.
  //      // Note that the winding order is clockwise by default.
  //      constexpr std::array<unsigned short, 36> cubeIndices =
  //      {{
  //          2,1,0, // -x
  //          2,3,1,

  //          6,4,5, // +x
  //          6,5,7,

  //          0,1,5, // -y
  //          0,5,4,

  //          2,6,7, // +y
  //          2,7,3,

  //          0,4,6, // -z
  //          0,6,2,

  //          1,3,7, // +z
  //          1,7,5,
  //      }};

  //      m_indexCount = cubeIndices.size();

  //      D3D11_SUBRESOURCE_DATA indexBufferData  = {0};
  //      indexBufferData.pSysMem                 = cubeIndices.data();
  //      indexBufferData.SysMemPitch             = 0;
  //      indexBufferData.SysMemSlicePitch        = 0;
  //      CD3D11_BUFFER_DESC indexBufferDesc(sizeof(unsigned short) * cubeIndices.size(), D3D11_BIND_INDEX_BUFFER);
  //      DX::ThrowIfFailed(
  //      m_deviceResources->GetD3DDevice()->CreateBuffer(
  //              &indexBufferDesc,
  //              &indexBufferData,
  //              &m_indexBuffer
  //              )
  //          );
    });

    // Once the cube is loaded, the object is ready to be rendered.
    createCubeTask.then([this] ()
    {
		// Create a default rasterizer state descriptor.
		D3D11_RASTERIZER_DESC rasterizerDesc = CD3D11_RASTERIZER_DESC(D3D11_DEFAULT);

		// Create the default rasterizer state.
		m_deviceResources->GetD3DDevice()->CreateRasterizerState(&rasterizerDesc, m_defaultRasterizerState.GetAddressOf());

		// Change settings for wireframe rasterization.
		//rasterizerDesc.AntialiasedLineEnable = true;
		rasterizerDesc.CullMode = D3D11_CULL_FRONT;						//XXX triangles seem to face inwards!!!
		//rasterizerDesc.FillMode = D3D11_FILL_WIREFRAME;

		// Create a wireframe rasterizer state.
		m_deviceResources->GetD3DDevice()->CreateRasterizerState(&rasterizerDesc, m_defaultRasterizerState.GetAddressOf());
		m_deviceResources->GetD3DDeviceContext()->RSSetState(m_defaultRasterizerState.Get());

        m_loadingComplete = true;
    });
}

void FibRenderer::ReleaseDeviceDependentResources()
{
    m_loadingComplete  = false;
    m_usingVprtShaders = false;
    m_vertexShader.Reset();
    m_inputLayout.Reset();
    m_pixelShader.Reset();
    m_geometryShader.Reset();
    m_modelConstantBuffer.Reset();
	m_fibConstantBuffer.Reset();
    m_vertexBuffer.Reset();
    m_indexBuffer.Reset();
	m_textureRV0.Reset();
	m_textureRV1.Reset();

	m_defaultRasterizerState.Reset();
}


void FibRenderer::loadTexture(std::string filename) {
	std::ifstream ifs;
	std::string s2;
	std::string header;

	ifs.open(filename, std::ios::binary);
	//ifs.open("assets/output_test.ppm", std::ios::binary);
	if (ifs.fail()) {
		OutputDebugString(L"Cannot load test texture!\n");
		exit(-1);
	}
	ifs >> header;
	if (strcmp(header.c_str(), "P6") != 0) throw("Can't read input file");//to header einai to p6
	ifs >> m_width >> m_height >> m_maximum;
	ifs.get();
	if (m_texture) delete[] m_texture;
	try {
		OutputDebugString((L"Current Memory Limit: " + std::to_wstring((double)Windows::System::MemoryManager::AppMemoryUsageLimit / 1024. / 1024.) + L"MB\n").c_str());
		OutputDebugString((L"Current Memory Usage: " + std::to_wstring((double)Windows::System::MemoryManager::AppMemoryUsage / 1024. / 1024.) + L"MB\n").c_str());
		m_texture = new unsigned char[4 * m_width*m_height];
	}
	catch (std::exception& e) {
		std::wstring wc(strlen(e.what()), L'#');
		size_t out_size;
		mbstowcs_s(&out_size, &wc[0], strlen(e.what()) + 1, e.what(), strlen(e.what()));
		OutputDebugString((L"Cannot allocate memory: " + wc + L"\n").c_str());
		OutputDebugString((L"Current Memory Limit: " + std::to_wstring((double)Windows::System::MemoryManager::AppMemoryUsageLimit / 1024. / 1024.) + L"MB\n").c_str());
		OutputDebugString((L"Current Memory Usage: " + std::to_wstring((double)Windows::System::MemoryManager::AppMemoryUsage / 1024. / 1024.) + L"MB\n").c_str());
		exit(-1);
	}
	int count = 0;
	unsigned char io_texture[3];
	for (unsigned int j = 0; j < (unsigned int)m_height; ++j) {
		for (unsigned int i = 0; i < (unsigned int)m_width; ++i) {
			unsigned int idx = j * m_width + i;
			ifs.read((char *)io_texture, 3);
			m_texture[idx * 4u] = io_texture[0];
			m_texture[idx * 4u + 1u] = io_texture[1];
			m_texture[idx * 4u + 2u] = io_texture[2];
			m_texture[idx * 4u + 3u] = 128;
		}
	}
	ifs.close();
	OutputDebugString(L"Texture is loaded\n");
}

void FibRenderer::loadTextureParted() {
	OutputDebugString(L"Loading the  texture.. part 0...\n");

	loadTexture("assets/hypersphere_0_2.ppm");
	m_width_used = m_width;
	m_height_part0 = m_height;

	//sanity checks
	if (m_height != 16384)
		OutputDebugString(L"Texture is has wrong height! Check again!\n");
	m_texture_part0 = m_texture;
	//reset m_texture that content isn't deleted
	m_texture = nullptr;


	OutputDebugString(L"Loading the  texture.. part 1...\n");
	loadTexture("assets/hypersphere_1_2.ppm");
	//more sanity checks
	if (m_width_used != m_width)
		OutputDebugString(L"Texture is has wrong height! Check again!\n");
	m_height_part1 = m_height;
	m_height_fib = m_height_part0 + m_height_part1;
	m_texture_part1 = m_texture;

	//TODO not hardcode!!!!
	m_width_fib = 12288;
	OutputDebugString(L"WARNING: m_width_fib is currently hardcoded to 12288!!!!!\n");


	OutputDebugString(L"Texture is loaded\n");
}