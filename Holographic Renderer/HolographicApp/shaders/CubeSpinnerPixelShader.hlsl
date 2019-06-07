//--------------------------------------------------------------------------------------
Texture2D Texture0 : register(t0);
Texture2D Texture1 : register(t1);
//--------------------------------------------------------------------------------------// Per-pixel color data passed through the pixel shader.
cbuffer FibConstantBuffer : register(b0) {
	float4 Origin_R;
	float4 CameraPos[2];
	float4 Resolution; //M_fib, M_used, N_fib
	float4x4 normal_mat;
}
//--------------------------------------------------------------------------------------
cbuffer ViewProjectionConstantBuffer : register(b1) {
	float4x4 view[2];
	float4x4 projection[2];
	float4x4 inverse_view[2];
	float4x4 inverse_projection[2];
};
//--------------------------------------------------------------------------------------
struct PixelShaderInput
{
	min16float4 pos     : SV_POSITION;
	min16float3 normal   : NORMAL0;
	min16float3 pos_ws   : POSITIONT;
	uint        rtvId   : SV_RenderTargetArrayIndex;
};

#define M_PI 3.1415926535897932384626433832795028841971693993751058209749445923
#define M_2PI 6.2831853071795864769252867665590057683943387987502116419498891846
#define MAX_TEXTURE_DIM_SIZE 16384.0
static const float _root5 = sqrt(5.f);
#define SQRT_5 _root5
static const float _Phi = (_root5 + 1.f) * 0.5f;
#define GOLDEN_RATIO_PHI _Phi
static const float _iPhi = _Phi - 1.f;
#define INV_PHI _iPhi

static const float M_fib = Resolution[0]; // 12288.0f; // Resolution[0];
static const float M_used = Resolution[1]; // 0.5f* M_fib;
static const float N_fib = Resolution[2]; // 24576.0f;
static const float _stepsize = sqrt(4.f * M_PI / M_fib); //TODO
static const float _resolution = M_fib; //TODO
										//float fractOfProduct(const float a, const float b) { return fma(a, b, -trunc(a * b)) /*a*b - floorf(a*b)*/; }
#define fractOfProduct(A,B) mad((A),(B),-floor((A)*(B)))

float2 _getPhiAndZ(unsigned int i, float num_fib) {
	float z0 = 1.f - 1.f / num_fib;
	return float2(M_2PI * frac(float(i)*_iPhi), z0 - 2.f * float(i) / num_fib);
}

float3 convertToCartesian(float2 p) {
	return float3(sin(p[0]) * cos(p[1]), cos(p[0]), sin(p[0])*sin(p[1]));
}

float3 convertToCartesian(const float theta, const float phi) {
	return float3(sin(theta) * cos(phi), cos(theta), sin(phi)*sin(theta));
}

float2 getPolar(in unsigned int n, in float numFibPoints) {
	float2 polar = _getPhiAndZ(n, numFibPoints);
	float theta = acos(polar.y);
	polar.y = polar.x;
	polar.x = theta;
	return polar;
}

float3 getCartesian(in unsigned int n, in float numFibPoints) {
	float2 polar = getPolar(n, numFibPoints);
	return convertToCartesian(polar.x, polar.y);
}

float inverseMapping(float3 p, float num_fib) {
	float phi = min(atan2(p[2], p[0]), M_PI);
	float cos_theta = p[1];
	float z0 = 1.f - 1.f / num_fib;
	// Get grid region by eq. (5): log_(Phi^2) (sqrt(5)*n*pi*(1-z^2))
	float k_dash = log(_root5 * num_fib * M_PI *  (1.f - cos_theta * cos_theta)) / log(_Phi + 1.0f); // _Phi^2 = _Phi + 1
	float k = max(2.f, floor(k_dash));
	// Compute fib number using the approxation by Phi^k/sqrt(5)
	float fib_k = pow(_Phi, k) / _root5;
	float fib_0 = round(fib_k);
	float fib_1 = round(fib_k * _Phi);
	// Compute matrix B by eq. (8) and (13)
	float2x2 B;
	B[0][0] = M_2PI * (frac((fib_0 + 1.f)*_iPhi) - _iPhi);
	B[0][1] = M_2PI * (frac((fib_1 + 1.f)*_iPhi) - _iPhi);
	B[1][0] = -2.f * fib_0 / num_fib;
	B[1][1] = -2.f * fib_1 / num_fib;
	float2x2 inv_B;
	float det = (B[0][0] * B[1][1] - B[0][1] * B[1][0]);
	inv_B[0][0] = B[1][1] / det;
	inv_B[0][1] = -B[0][1] / det;
	inv_B[1][0] = -B[1][0] / det;
	inv_B[1][1] = B[0][0] / det;
	// Get local grid point by eq. (10)
	float2 c = float2(inv_B[0][0] * phi + inv_B[0][1] * (cos_theta - z0), inv_B[1][0] * phi + inv_B[1][1] * (cos_theta - z0));
	unsigned int nearest = 0;
	float distance = 10000000.f;
	for (unsigned int j = 0; j < 4; ++j) {
		float2 corner = floor(c) + float2(j / 2, j % 2);
		float temp_idx = dot(float2(fib_0, fib_1), corner);
		unsigned int temp_cell = temp_idx < 0.f ? 0 : temp_idx >= num_fib ? num_fib - 1 : temp_idx;
		float2 phi_z = _getPhiAndZ(temp_cell, num_fib);
		float3 temp_p = convertToCartesian(float2(acos(phi_z[1]), phi_z[0]));
		float d = dot(temp_p - p, temp_p - p);
		if (d < distance) {
			distance = d;
			nearest = temp_cell;
		}
	}
	return nearest;
}

float4 loadColor(in uint x, in uint y) {
	float4 color = float4(0.0, 0.0, 0.0, 0.0);
	if (x >= M_used) {
		return color;
	}
	if (y < MAX_TEXTURE_DIM_SIZE) {
		color = Texture0.Load(int3(x, MAX_TEXTURE_DIM_SIZE - y - 1, 0));
	} else {
		const uint h_part1 = N_fib - MAX_TEXTURE_DIM_SIZE;
		y -= MAX_TEXTURE_DIM_SIZE;
		color = Texture1.Load(int3(x, h_part1 - y - 1, 0));
	}
	return color;
}

float4 smoothSF_second_2x2(in float3 p, in unsigned int numFibPointsN, unsigned int first_fibIndex) {
	const float numFibPointsNf = float(numFibPointsN);
	float phi = min(atan2(p.z, p.x), M_PI);
	float cos_theta = p.y;
	float z0 = 1.f - 1.f / numFibPointsNf;
	// Get grid region by eq. (5): log_(Phi^2) (sqrt(5)*n*pi*(1-z^2))
	//float k_dash = logf(SQRT_5 * numFibPoints * M_PIf *  optix::clamp((1.f - cos_theta * cos_theta), 0.f, 1.f))
	//	/ logf(GOLDEN_RATIO_PHI + 1.0f); // _Phi^2 = _Phi + 1
	float k_dash = log(SQRT_5 * numFibPointsNf * M_PI *  (1.f - cos_theta * cos_theta)) / log(GOLDEN_RATIO_PHI + 1.0f); // _Phi^2 = _Phi + 1

	float k = max(2.f, floor(k_dash));

	//TODO check is new
	float filterWidth = sqrt(SQRT_5)*sqrt(4.0f * M_PI / (SQRT_5*numFibPointsNf));

	float4 a = float4(0.f, 0.f, 0.f, 0.f);

	const float color_eps = 1.0 / 20.f;
	if (k <= 4) {
		float weightSum = 0;

		float offset = cos_theta < 0.f ? (numFibPointsNf - 16.f) : 0;
		for (float i = 0; i < 16; ++i) {
			float3 q = getCartesian(i + offset, numFibPointsNf);

			float delta = length(q - p);
			float weight = smoothstep(filterWidth, 0, delta);

			//early out
			if (weight < color_eps)
				continue;
			a += max(loadColor(first_fibIndex, int(i + offset)) * weight, float4(0.f, 0.f, 0.f, 0.f));

			//a += optix::fmaxf(optix::rtTex2DFetch<float4>(texID, int(first_fibIndex), int(i + offset)) * weight, make_float4(0.f));
			//a += make_float4(getRndColor(i + offset, numFibPoints), 0) * weight;
			weightSum += weight;
		}
		a /= weightSum;
		return a;
	} else {

		// Compute fib number using the approxation by Phi^k/sqrt(5)
		float fib_k = pow(GOLDEN_RATIO_PHI, k) / SQRT_5;
		float fib_0 = round(fib_k);
		float fib_1 = round(fib_k * GOLDEN_RATIO_PHI);
		// Compute matrix B by eq. (8) and (13)
		float B[4];
		B[0] = M_2PI * (fractOfProduct(fib_0 + 1.f, INV_PHI) - INV_PHI);
		B[1] = -2.f * fib_0 / numFibPointsNf;
		B[2] = M_2PI * (fractOfProduct(fib_1 + 1.f, INV_PHI) - INV_PHI);
		B[3] = -2.f * fib_1 / numFibPointsNf;
		float inv_B[4];
		float det = (B[0] * B[3] - B[1] * B[2]);
		inv_B[0] = B[3] / det;
		inv_B[1] = -B[2] / det;
		inv_B[2] = -B[1] / det;
		inv_B[3] = B[0] / det;
		// Get local grid point by eq. (10)
		float2 c = float2(inv_B[0] * phi + inv_B[1] * (cos_theta - z0),
			inv_B[2] * phi + inv_B[3] * (cos_theta - z0)); //TODO vectorize

		c.x = round(c.x); //TODO check
		c.y = round(c.y); //TODO check

						  //TODO also try this but adapt filtersize!!!!
						  /*const unsigned int NUM_CORNERS = 4;
						  float2 corners[NUM_CORNERS];
						  corners[0] = c;
						  corners[1] = c + make_float2(0.f, 1.f);
						  corners[2] = c + make_float2(1.f, 1.f);
						  corners[3] = c + make_float2(1.f, 0.f);*/

		const unsigned int NUM_CORNERS = 5;
		float2 corners[NUM_CORNERS];
		corners[0] = c;
		corners[1] = c + float2(-1.0f, 0.0f);
		corners[2] = c + float2(1.f, 0.f);
		corners[3] = c + float2(0.f, -1.f);
		corners[4] = c + float2(0.f, 1.f);

		unsigned int temp_cell[NUM_CORNERS];
		float weightSum = 0.f;
		for (unsigned int j = 0; j < NUM_CORNERS; ++j) {
			float temp_idx = dot(float2(fib_0, fib_1), corners[j]);
			if (temp_idx >= numFibPointsNf) continue;//should be equivalent to: if (abs(cosTheta) > 1) continue; TODO check!!
			if (temp_idx < 0.f) continue;
			temp_cell[j] = (unsigned int)(temp_idx);
			float3 q = getCartesian(temp_cell[j], numFibPointsNf);

			float delta = length(q - p);
			float weight = smoothstep(filterWidth, 0.f, delta);

			//early out
			if (weight < color_eps)
				continue;
			a += max(loadColor(first_fibIndex, temp_cell[j]) * weight, float4(0.f, 0.f, 0.f, 0.f)); //TODO y 
			weightSum += weight;
		}
		a = a / weightSum;
	}
	return a;
}

float4 smoothSF_first_2x2(in float3 p_first, in float3 p_second, in unsigned int numFibPointsM, in unsigned int numFibPointsN) {
	const float numFibPointsMf = float(numFibPointsM);
	float phi = min(atan2(p_first.z, p_first.x), M_PI);
	float cos_theta = p_first.y;
	float z0 = 1.f - 1.f / numFibPointsMf;
	// Get grid region by eq. (5): log_(Phi^2) (sqrt(5)*n*pi*(1-z^2))
	//float k_dash = logf(SQRT_5 * numFibPoints * M_PIf *  optix::clamp((1.f - cos_theta * cos_theta), 0.f, 1.f))
	//	/ logf(GOLDEN_RATIO_PHI + 1.0f); // _Phi^2 = _Phi + 1
	float k_dash = log(SQRT_5 * numFibPointsMf * M_PI *  (1.f - cos_theta * cos_theta)) / log(GOLDEN_RATIO_PHI + 1.0f); // _Phi^2 = _Phi + 1

	float k = max(2.f, floor(k_dash));

	//TODO check is new
	float filterWidth = sqrt(SQRT_5)*sqrt(4.0f * M_PI / (SQRT_5*numFibPointsMf));

	float4 a = float4(0.f, 0.f, 0.f, 0.f);
	const float color_eps = 1.f / 100.f;
	if (k <= 4) {
		float weightSum = 0;

		float offset = cos_theta < 0.f ? (numFibPointsMf - 16.f) : 0;
		for (float i = 0; i < 16; ++i) {
			float3 q = getCartesian(i + offset, numFibPointsMf);

			float delta = length(q - p_first);
			float weight = smoothstep(filterWidth, 0, delta);
			if (weight < color_eps)
				continue;
			float4 smooth_second_col = smoothSF_second_2x2(p_second, numFibPointsN, i + offset);
			a += smooth_second_col * weight;
			weightSum += weight;
		}
		a /= weightSum;
		return a;
	} else {

		// Compute fib number using the approxation by Phi^k/sqrt(5)
		float fib_k = pow(GOLDEN_RATIO_PHI, k) / SQRT_5;
		float fib_0 = round(fib_k);
		float fib_1 = round(fib_k * GOLDEN_RATIO_PHI);
		// Compute matrix B by eq. (8) and (13)
		float B[4];
		B[0] = M_2PI * (fractOfProduct(fib_0 + 1.f, INV_PHI) - INV_PHI);
		B[1] = -2.f * fib_0 / numFibPointsMf;
		B[2] = M_2PI * (fractOfProduct(fib_1 + 1.f, INV_PHI) - INV_PHI);
		B[3] = -2.f * fib_1 / numFibPointsMf;
		float inv_B[4];
		float det = (B[0] * B[3] - B[1] * B[2]);
		inv_B[0] = B[3] / det;
		inv_B[1] = -B[2] / det;
		inv_B[2] = -B[1] / det;
		inv_B[3] = B[0] / det;
		// Get local grid point by eq. (10)
		float2 c = float2(inv_B[0] * phi + inv_B[1] * (cos_theta - z0),
			inv_B[2] * phi + inv_B[3] * (cos_theta - z0)); //TODO vectorize

		c.x = round(c.x); //TODO check
		c.y = round(c.y); //TODO check


		const unsigned int NUM_CORNERS = 5;
		float2 corners[NUM_CORNERS];
		corners[0] = c;
		corners[1] = c + float2(-1.0f, 0.0f);
		corners[2] = c + float2(1.f, 0.f);
		corners[3] = c + float2(0.f, -1.f);
		corners[4] = c + float2(0.f, 1.f);

		unsigned int temp_cell[NUM_CORNERS];
		float weightSum = 0.f;
		float4 smooth_second_col;
		for (unsigned int j = 0; j < NUM_CORNERS; ++j) {
			float temp_idx = dot(float2(fib_0, fib_1), corners[j]);
			if (temp_idx >= numFibPointsMf) continue;//should be equivalent to: if (abs(cosTheta) > 1) continue; TODO check!!
			if (temp_idx < 0.f) continue;
			temp_cell[j] = (unsigned int)(temp_idx);
			float3 q = getCartesian(temp_cell[j], numFibPointsMf);

			float delta = length(q - p_first);
			float weight = smoothstep(filterWidth, 0.f, delta);
			//early out
			if (weight < color_eps)
				continue;
			smooth_second_col = smoothSF_second_2x2(p_second, numFibPointsN, temp_cell[j]);
			a += smooth_second_col * weight;
			//	a += weight * float4(1.f, 0.f, 0.f, 0.f);
			weightSum += weight;
		}
		a = a / weightSum;
	}
	return a;
}


// The pixel shader passes through the color data. The color data from 
// is interpolated and assigned to a pixel at the rasterization step.
min16float4 main(PixelShaderInput input) : SV_TARGET
{
	int idx = input.rtvId % 2;
	float4 camera_pos = mul(float4(0, 0, 0, 1), inverse_view[idx]);
	camera_pos = camera_pos / camera_pos.w;

	float3 normal = normalize(input.normal);
	float3 origin = Origin_R.xyz; //world space
	float r = Origin_R.w;

	float3 first_hitpoint = normal; // in local unit sphere coordinates

	//find second hitpoint
	float3 first_hitpoint_ws = input.pos_ws;
	float3 ray_dir = normalize(first_hitpoint_ws - camera_pos.xyz);
	float t = dot(origin - first_hitpoint_ws, ray_dir);

	float3 second_hitpoint_ws = first_hitpoint_ws + 2*t * ray_dir;
	float3 second_hitpoint = normalize(mul(float4(second_hitpoint_ws, 1), (normal_mat)).xyz);
	
	float idx0 = inverseMapping(first_hitpoint, M_fib);
	float idx1 = inverseMapping(second_hitpoint, N_fib);

	const float discard_eps = 0.04;
	//lonly sample if fib point is actually used
	if (idx0 < M_used) {
		float idx1 = inverseMapping(second_hitpoint, N_fib);
		if (false) {//nearest neighbor
			int x = idx0;
			int y = idx1;
			float4 color = loadColor(x, y);
			if (dot(color.xyz, color.xyz)< discard_eps)
				discard;
			return color;
		} else {
			float4 color = smoothSF_first_2x2(first_hitpoint, second_hitpoint, M_fib, N_fib);
			if (dot(color.xyz, color.xyz)< discard_eps)
				discard;
			return color;
		}
	} else {
		discard;
		return float4(0,0,0,0);
	}
}
