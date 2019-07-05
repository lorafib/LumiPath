
#include "shader_common.h"
///
// in: 
// pixel: pixel coordinate in [0^2, screen-1]
// screen: width, height in pixels of outputbuffer
// sample: the sub-pixel location in [0,1]^2 (uniform sample)
// out:
// origin: origin of ray
// direction: direction of ray (normalized) 

//without jittering of direction samples on disk (sample is ignored)
RT_CALLABLE_PROGRAM void lens_shader_fib_hypersphere(const float2 pixel, const float2 screen, const float2 sample,
	float3& origin, float3& direction) { //return values
	
	///y is index for origin
	float3 origin_vec = getCartesian(pixel.x, sysNumOfFibPointsForRendering_M);

	const float3 directionToOrigin = optix::normalize(origin_vec);
	origin = (sysCameraCenterOfInterest + directionToOrigin*sysCameraDistance);
	
	///x is index for direction
	float3 direction_vec = getCartesian(pixel.y, sysNumOfFibPointsForRendering_N);

	const float3 directionToDirection = optix::normalize(direction_vec);
	
	float3 dirPoint = (sysCameraCenterOfInterest + directionToDirection*sysCameraDistance);
	direction = (-origin+dirPoint);

	float len_dir = optix::length(direction);
	if (len_dir <= EPS)
		// sampling from me to myself -> BAD 
		// --> tangential rays have to be considered as invalid during inverse mapping, too. 
		//alternatively, set color to background color or to color of geometry at origins postion 
		direction = make_float3(0.f,1.f,0.f); 
	else {
		direction = direction / len_dir;
	}

}


//with jittering of direction samples on disk
RT_CALLABLE_PROGRAM void lens_shader_fib_hypersphere_super_sampled(const float2 pixel, const float2 screen, const float2 sample,
	float3& origin, float3& direction) { //return values

	///x is index for origins
	float3 origin_vec = getCartesian(pixel.x, sysNumOfFibPointsForRendering_M);

	const float3 directionToOrigin = optix::normalize(origin_vec);
	const float3 origin_unjittered = (sysCameraCenterOfInterest + sysCameraDistance*directionToOrigin);
	const float3 cartesian_on_unitsphere = uniform_disk_sample(directionToOrigin, sample, float(sysNumOfFibPointsForRendering_M));

	//sanity check
	//if (optix::length(cartesian_on_unitsphere) >= 1.5) {
	//	direction = make_float3(42.f,42.f,42.f);
	//	origin = make_float3(42.f, 42.f, 42.f);
	//	return;
	//}
	//sanity check end

	origin = sysCameraCenterOfInterest + sysCameraDistance*cartesian_on_unitsphere;

	///y is index for directions
	float3 direction_vec = getCartesian(pixel.y, sysNumOfFibPointsForRendering_N);
	const float3 directionToDirection = optix::normalize(direction_vec);

	float3 dirPoint = (sysCameraCenterOfInterest + directionToDirection*sysCameraDistance);
	direction = (-origin_unjittered + dirPoint);

	float len_dir = optix::length(direction);
	if (len_dir <= EPS)
		// sampling from me to myself -> BAD 
		// --> tangential rays have to be considered as invalid during inverse mapping, too. 
		//alternatively, set color to background color or to color of geometry at origins postion 
		direction = make_float3(0.f, 1.f, 0.f); // sampling from me to myself -> BAD 
	else {
		direction = direction / len_dir;
	}
}