#pragma once

// #define CL_USE_DEPRECATED_OPENCL_2_0_APIS // required for using old APIs

#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>

#include "Application.hpp"

/* Struct that contains a description of an output image. */
typedef struct
{
	cl_mem buffer;
	void *data;
	size_t origin[3];
	size_t region[3];
} image_buf_t;

/* Struct that contains a description of an output buffer. */
typedef struct
{
	cl_mem buffer;
	void *data;
	size_t size;
} buf_t;

/**
 * A simple wrapper class for accessing OpenCL.
 */
class MiniOCL
{
	const char* kernelFileName;
	cl_device_type device_type; 		// OpenCL target device type
	// FIXME: Could be a union type between image_buf_t and buf_t?
	bool outputIsImage;					// whether to use outImg over outBuf
	image_buf_t outImg;					// the output image struct (only single one, exclusive with outBuf)
	buf_t outBuf;						// the output buffer struct (only single one, exclusive with outImg)

    // OpenCL objects
    cl_platform_id platform;            // OpenCL platform
    cl_device_id device_id;             // device ID
    cl_context context;                 // context
    cl_command_queue queue;             // command queue
    cl_program program;                 // program
    cl_kernel kernel;                   // kernel
    cl_event kernelEvent;				// kernel event (profiling)

public:
	MiniOCL(const char* kernelFileName);
	~MiniOCL();

	// OpenCL workflow
	bool initialize(cl_device_type device_type);
	bool buildKernel(const char *kernelName);
	bool executeKernel(size_t globalWidth, size_t globalHeight, size_t localWidth, size_t localHeight);

	// values
	bool setValue(cl_uint argIndex, void *value, size_t size);
	// buffers
	bool setInputBuffer(cl_uint argIndex, void *data, size_t size);
	bool setOutputBuffer(cl_uint argIndex, void *data, size_t size);
	// image buffers
	bool setInputImageBuffer(cl_uint argIndex, void *data, size_t width, size_t height, bool singleChannel);
	bool setOutputImageBuffer(cl_uint argIndex, void *data, size_t width, size_t height, bool singleChannel);

	bool displayDeviceInfo(cl_device_id device_id = NULL);
	double getExecutionTime();

private:
	bool readOutput();

};
