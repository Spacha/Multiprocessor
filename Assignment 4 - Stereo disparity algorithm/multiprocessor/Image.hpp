#pragma once

#include "Application.hpp"
#include "MiniOCL.hpp"
#include "lodepng.h"

/* Struct representing a pixel (0-255) */
struct Pixel
{
    unsigned char red;
    unsigned char green;
    unsigned char blue;
    unsigned char alpha;

    Pixel(unsigned char red, unsigned char green, unsigned char blue, unsigned char alpha)
        : red(red), green(green), blue(blue), alpha(alpha) {}

    /**
     * Overloads the division operator between Pixel and a float.
     * Values of R, G and B are divided but A is not.
     */
    Pixel operator/(const float divisor)
    {
        return Pixel(
            (unsigned char)(this->red   / divisor),
            (unsigned char)(this->green / divisor),
            (unsigned char)(this->blue  / divisor),
            (unsigned char)(this->alpha)
        );
    }
};
typedef struct Pixel Pixel;

/* Same struct but as float (0.0f - 1.0f) */

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
    MiniOCL *ocl = nullptr;             // Handle to OpenCL wrapper class for parallel execution

    Image();
    ~Image();
    void setOpenCL(MiniOCL *ocl);

    // image creation etc.
    void createEmpty(size_t width, size_t height);
    void replaceImage(std::vector<unsigned char> &newImage);
    bool load(const std::string &filename);
    bool save(const std::string &filename);

    // image manipulation
    bool convertToGrayscale();
    bool filter(const Filter &filter);
    bool resize(size_t width, size_t height);
    Image *calcZNCC(Image &otherImg);
    bool crossCheck(Image &left, Image &right);
    bool occlusionFill();

    // helper methods
    void putPixel(unsigned int x, unsigned int y, Pixel pixel);
    void putPixel(unsigned int x, unsigned int y, unsigned char grey);
    Pixel getPixel(unsigned int x, unsigned int y);
    void printPixel(unsigned int x, unsigned int y);

    size_t sizeBytes();
    bool validCoordinates(unsigned int x, unsigned int y);
};
