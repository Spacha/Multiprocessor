#pragma once

#include "Application.hpp"
#include "MiniOCL.hpp"
#include "lodepng.h"

/* Struct representing a pixel (0-255) */
struct pixel_s
{
    unsigned char red;
    unsigned char green;
    unsigned char blue;
    unsigned char alpha;
};
typedef struct pixel_s pixel_t;

/* Same struct but as float (0.0f - 1.0f) */
struct pixelf_s
{
    float red;
    float green;
    float blue;
    float alpha;
};
typedef struct pixelf_s pixelf_t;

/**
 * This is my wrapper for lodepng.h that simplifies the handling of PNGs a lot.
 * The object contains image metadata and offers
 * functions for manipulating the image.
 **/
class Image
{
public:
    std::vector<unsigned char> image;   // image pixels (RGBA)
    std::string name;                   // image file name
    size_t width;                       // image width
    size_t height;                      // image height
    LodePNGColorType colorType;         // color type (see LodePNGColorType in lodepng.h)
    MiniOCL *ocl = nullptr;             // Handle to OpenCL wrapper class for parallel execution

    Image(LodePNGColorType colorType = LCT_RGBA);
    ~Image();
    void setOpenCL(MiniOCL *ocl);

    // image creation etc.
    void createEmpty(size_t width, size_t height);
    void replaceImage(std::vector<unsigned char> &newImage);
    bool load(const std::string &filename);
    bool save(const std::string &filename);

    // image manipulation
    bool convertToGrayscale();
    bool filter(const filter_t &filter);

    // helper methods
    void putPixel(unsigned int x, unsigned int y, pixel_t pixel);
    void putPixel(unsigned int x, unsigned int y, unsigned char grey);
    pixel_t getPixel(unsigned int x, unsigned int y);
    void printPixel(unsigned int x, unsigned int y);

    size_t sizeBytes();
    bool validCoordinates(unsigned int x, unsigned int y);
};
