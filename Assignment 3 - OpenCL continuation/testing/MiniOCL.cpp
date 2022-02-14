#include <iostream>
#include "MiniOCL.hpp"

using std::cout;
using std::endl;

MiniOCL::MiniOCL(unsigned int device_type)
{
	this->device_type = device_type;
}
MiniOCL::~MiniOCL()
{
	// Destroy the object, release memory...
}

bool MiniOCL::setKernel(std::string name, char **source)
{
	cl_int err = CL_SUCCESS;

	this->kernelName = name;
	this->kernelSource = source;

	return err == CL_SUCCESS;
}

bool MiniOCL::setInputImage(unsigned char *data, size_t width, size_t height)
{
	cl_int err = CL_SUCCESS;
	// ...
	return err == CL_SUCCESS;
}
bool MiniOCL::setOutputImage(unsigned char *data, size_t width, size_t height)
{
	cl_int err = CL_SUCCESS;
	// ...
	return err == CL_SUCCESS;
}
bool MiniOCL::execute()
{
	cl_int err = CL_SUCCESS;
	// ...
	return err == CL_SUCCESS;
}
bool MiniOCL::displayDeviceInfo()
{
	cl_int err = CL_SUCCESS;
	cout << "Device information will be here..." << endl;

	return err == CL_SUCCESS;
}
double MiniOCL::getExecutionTime()
{
	// ...
	return 0.0;
}

/*
std::vector<unsigned char> originalImg;
std::vector<unsigned char> grayImg;

MiniOCL ocl;
ocl.init(TARGET_DEVICE_TYPE);
ocl.setKernel("pixel_stuff", &kernelSource);
ocl.setInputImage(static_cast<void*>(img.image.data()), img.width, img.height);
ocl.setOutputImage(static_cast<void*>(grayImg.image.data()), grayImg.width, grayImg.height);
ocl.execute();

ocl.displayDeviceInfo();

double microSeconds = ocl.getExecutionTime();

ocl.terminate();
*/
