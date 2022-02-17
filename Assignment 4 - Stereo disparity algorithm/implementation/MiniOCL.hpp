#pragma once

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS // required for using old APIs

#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>

#include "Application.hpp"

/* Struct that contains a description of an otput image. */
struct image_buf_s
{
	cl_mem buffer;
	void *data;
	size_t origin[3];
	size_t region[3];
};
typedef struct image_buf_s image_buf_t;

/**
 * A simple wrapper class for accessing OpenCL.
 **/
class MiniOCL
{
	cl_device_type device_type; 		// OpenCL target device type
	image_buf_t outImg; 				// the output image struct (only single one)

    // OpenCL objects
    cl_platform_id platform;            // OpenCL platform
    cl_device_id device_id;             // device ID
    cl_context context;                 // context
    cl_command_queue queue;             // command queue
    cl_program program;                 // program
    cl_kernel kernel;                   // kernel
    cl_event kernelEvent;				// kernel event (profiling)

public:
	MiniOCL();
	~MiniOCL();

	// OpenCL workflow
	bool initialize(cl_device_type device_type);
	bool buildKernel(const char *fileName, const char *kernelName);
	bool executeKernel(size_t globalWidth, size_t globalHeight, size_t localWidth, size_t localHeight);

	// values
	bool setValue(cl_uint argIndex, void *value, size_t size);
	// buffers
	bool setInputBuffer(cl_uint argIndex, void *data, size_t size);
	bool setOutputBuffer(cl_uint argIndex, void *data, size_t size);
	// image buffers
	bool setInputImageBuffer(cl_uint argIndex, void *data, size_t width, size_t height);
	bool setOutputImageBuffer(cl_uint argIndex, void *data, size_t width, size_t height);

	bool displayDeviceInfo(cl_device_id device_id = NULL);
	double getExecutionTime();

private:
	bool readOutput();

};
