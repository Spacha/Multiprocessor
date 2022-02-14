#pragma once

#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

class MiniOCL
{
	unsigned int device_type;
	std::string kernelName;
	char **kernelSource;

public:
	MiniOCL(unsigned int device_type);
	bool setKernel(std::string name, char **source);
	bool setInputImage(unsigned char *data, size_t width, size_t height);
	bool setOutputImage(unsigned char *data, size_t width, size_t height);
	bool execute();
	bool displayDeviceInfo();
	double getExecutionTime();
	~MiniOCL();
};
