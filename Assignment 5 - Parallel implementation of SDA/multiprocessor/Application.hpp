#pragma once

/* Application-wide includes */
#include <iostream>
#include <string>
#include <vector>
#include <inttypes.h>

/* These are options for COMPUTE_DEVICE. */
#define TARGET_NONE     0       // Don't use OpenCL
#define TARGET_GPU      1       // OpenCL on GPU
#define TARGET_CPU      2       // OpenCL on CPU
#define TARGET_PTHREAD  3       // Pthreads on CPU

///////////////////////////////////////////////////////////////////////////////
// Parameters:
///////////////////////////////////////////////////////////////////////////////

/** 
 * COMPUTE_DEVICE options:
 * TARGET_NONE     = Sequential - don't use OpenCL
 * TARGET_GPU      = OpenCL on GPU
 * TARGET_CPU      = OpenCL on CPU
 * TARGET_PTHREAD  = Threaded on CPU
 */
#define COMPUTE_DEVICE TARGET_PTHREAD

/**
 * Number of parallel threads. Used only COMPUTE_DEVICE == TARGET_PTHREAD.
 * For maximum efficiency, the number of CPU cores is suggested as
 * the number of threads.
 */
#define NUM_THREADS 8

///////////////////////////////////////////////////////////////////////////////
// DEFINITIONS & MACROS
///////////////////////////////////////////////////////////////////////////////

#if COMPUTE_DEVICE == TARGET_CPU
# define TARGET_DEVICE_TYPE CL_DEVICE_TYPE_CPU
#else
# define TARGET_DEVICE_TYPE CL_DEVICE_TYPE_GPU
#endif

#if COMPUTE_DEVICE == TARGET_GPU || COMPUTE_DEVICE == TARGET_CPU
# define USE_OCL
#elif COMPUTE_DEVICE == TARGET_PTHREAD
# define USE_THREADS
#endif
