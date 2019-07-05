#pragma once

#ifndef SHADER_COMMON_H
#define SHADER_COMMON_H

#ifndef INFINITYf
#define INFINITYf 100000.f
#endif
#ifndef SQRT_5
#define SQRT_5 2.2360679774997896964091736687312762354406183596115257
#endif
#ifndef M_2PIf
#define M_2PIf 6.2831853071795864769252867665590057683943387987502116419498891846f
#endif
#define GOLDEN_RATIO_PHI ((SQRT_5 + 1.0) * 0.5)
#define INV_PHI (GOLDEN_RATIO_PHI -1.0)// property that Phi - 1 = Phi^(-1)
#ifndef EPS
#define EPS 0.000001f
#endif

__device__ float fractOfProduct(const float a, const float b) { return fmaf(a, b, -truncf(a * b)) /*a*b - floorf(a*b)*/; }

__device__ float3 convertToCartesian(const float theta, const float phi) { 
	return make_float3(sinf(theta) * cosf(phi), cosf(theta), sinf(phi)*sinf(theta)); }

__device__ float2 getPhiAndZ(unsigned int n, float numFibPoints) {
	// phi, z by eq. (2): phi = 2*pi*[i / Phi], z = 1 - (2*i + 1) / n
	float z0 = 1.f - 1.f / numFibPoints;
	return make_float2(M_2PIf * fractOfProduct((float)n, INV_PHI), z0 - 2.f * float(n) / numFibPoints);
}

__device__ float2 getPolar(unsigned int n, float numFibPoints) {
	float2 polar = getPhiAndZ(n, numFibPoints);
	float theta = acosf(polar.y);
	polar.y = polar.x;
	polar.x = theta;
	return polar;
}

__device__ float3 getCartesian(unsigned int n, float numFibPoints) {
	float2 polar = getPolar(n, numFibPoints);
	return convertToCartesian(polar.x, polar.y);
}

////////////////////////////////////////////////////////////////////
//sampling

__device__ float sphere_area(float r) {
	return 4.0*M_PI*r*r;
}

__device__ float disk_area(float r) {
	return M_PIf*r*r;
}

__device__ float radius_of_disk(float area) {
	float radius = area / M_PIf;
	radius = sqrtf(radius);
	return radius;

}

__device__ float getTangentDiskRadius(const float numFibPoints) {
	//const stuff
	const float fib_sphere_radius = 1.0f;
	const float fib_sphere_area = sphere_area(fib_sphere_radius);
	const float area_per_point = fib_sphere_area / numFibPoints;
	const float surface_radius = radius_of_disk(area_per_point);

	const float surface_disk_origin_ratio = sqrtf(fib_sphere_radius*fib_sphere_radius - surface_radius*surface_radius);
	//float3 surface_disk_origin = (unit_origin*fib_sphere_radius)*(surface_disk_origin_ratio / fib_sphere_radius);
	//path3 surface_disk_circle = lookAt(surface_disk_origin, O, (0, 1, 0))*path3(circle((0, 0), surface_radius));

	const float tangent_disk_radius = surface_radius*(fib_sphere_radius / surface_disk_origin_ratio);
	//path3 tangent_disk_circle = lookAt(origin, O, (0, 1, 0))*path3(circle((0, 0), tangent_disk_radius));
	//const stuff end
	return tangent_disk_radius;
}

//following PBRTv 1.0 p. 653
__device__ float3 uniform_disk_sample(const float3& unit_origin, const float2& sample, const float numFibPoints) {
	float sampling_disk_phi = 2 * M_PIf*sample.x;

	float weight = sqrtf(sample.y); 
	float tangent_disk_radius = getTangentDiskRadius(numFibPoints);
	float sampling_disk_r = tangent_disk_radius *weight;
	float3 random_cartesian = make_float3(sampling_disk_r*cos(sampling_disk_phi),
		sampling_disk_r*sin(sampling_disk_phi),
		0.0f);

	float3 up = make_float3(0.0f, 1.0f, 0.0f);

	float3 f = optix::normalize(-unit_origin);
	float3 s = optix::normalize(optix::cross(f, up));
	float3 u = optix::normalize(optix::cross(s, f));

	random_cartesian = random_cartesian.x * s + random_cartesian.y * u + random_cartesian.z * f;
	random_cartesian = random_cartesian.x * make_float3(s.x, u.x, f.x)
		+ random_cartesian.y * make_float3(s.y, u.y, f.y)
		+ random_cartesian.z * make_float3(s.z, u.z, f.z);
	random_cartesian = unit_origin + random_cartesian;

	//sanity check
	//float3 thing = optix::cross(unit_origin, up);
	//float3 tangent = optix::normalize(optix::cross(thing, unit_origin));
	//float3 sphere_circle_point1 = (unit_origin + tangent*tangent_disk_radius);
	//if (optix::length(random_cartesian) - optix::length(sphere_circle_point1) > 0.001 ||
	//	optix::length(random_cartesian) < 0.999f)
	//	return make_float3(2.0, 2.0, 2.0);
	//else
	//sanity check end

	return optix::normalize(random_cartesian);
}


#endif // SHADER_COMMON_H

