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
// DEFINITIONS & MACROS
///////////////////////////////////////////////////////////////////////////////

struct filter_s
{
    const size_t size;      // size of the mask (height or width)
    const float divisor;    // the mask is divided by this
    const float *mask;      // the actual filter mask
};
typedef struct filter_s filter_t;

#if COMPUTE_DEVICE == TARGET_CPU
# define TARGET_DEVICE_TYPE CL_DEVICE_TYPE_CPU
#else
# define TARGET_DEVICE_TYPE CL_DEVICE_TYPE_GPU
#endif

#if COMPUTE_DEVICE != TARGET_NONE
# define USE_OCL
#endif
