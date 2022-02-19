#pragma once

/* Application-wide includes */
#include <iostream>
#include <string>
#include <vector>

// These are options for COMPUTE_DEVICE.
#define TARGET_NONE 0                   // Don't use OpenCL
#define TARGET_GPU  1                   // OpenCL on GPU
#define TARGET_CPU  2                   // OpenCL on CPU

///////////////////////////////////////////////////////////////////////////////
// Parameters:
///////////////////////////////////////////////////////////////////////////////

// COMPUTE_DEVICE options:
//  TARGET_NONE = Sequential - don't use OpenCL
//  TARGET_GPU  = OpenCL on GPU
//  TARGET_CPU  = OpenCL on CPU
#define COMPUTE_DEVICE TARGET_GPU


///////////////////////////////////////////////////////////////////////////////
// DEFINITIONS & MACROS
///////////////////////////////////////////////////////////////////////////////

struct Filter
{
    const size_t size;      // size of the mask (height or width)
    const float divisor;    // the mask is divided by this
    const float* mask;      // the actual filter mask

    Filter(size_t size, float divisor, const float *mask)
        : size(size), divisor(divisor), mask(mask) {}
};
typedef struct Filter Filter;

#if COMPUTE_DEVICE == TARGET_CPU
# define TARGET_DEVICE_TYPE CL_DEVICE_TYPE_CPU
#else
# define TARGET_DEVICE_TYPE CL_DEVICE_TYPE_GPU
#endif

#if COMPUTE_DEVICE != TARGET_NONE
# define USE_OCL
#endif
