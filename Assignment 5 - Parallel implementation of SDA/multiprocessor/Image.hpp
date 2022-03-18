#pragma once

#include <math.h>
#include <array>
#include <iomanip>          // setw
#include <pthread.h>        // Only ifdef USE_THREADS
#include "Application.hpp"
#include "Filters.hpp"
#include "MiniOCL.hpp"
#include "lodepng.h"

/* TEMP: Forward declaration. */
struct ZNCCArgs;
typedef struct ZNCCArgs ZNCCArgs;

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

    /**
     * Returns true if the pixel is a zero pixel, i.e. all channels are zero.
     */
    bool isZero()
    {
        return this->red   == 0 && this->green == 0 &&
               this->blue  == 0 && this->alpha == 0;
    }
};
typedef struct Pixel Pixel;

/**
 * This is my wrapper for lodepng.h that simplifies the handling of PNGs a lot.
 * The object contains image metadata and offers
 * functions for manipulating the image.
 */
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
    //void replaceImage(std::vector<unsigned char> &newImage);
    void replace(Image &newImage);
    bool load(const std::string &filename);
    bool save(const std::string &filename);

    // image manipulation
    bool convertToGrayscale();
    bool filter(const Filter &filter);
    bool filterMean(size_t size);
    //bool resize(size_t width, size_t height);
    bool downScale(unsigned int factor);
    bool calcZNCC(Image &otherImg, Image *disparityMap, unsigned int windowSize, unsigned int maxSearchD, bool reverse = false);
    bool crossCheck(Image &left, Image &right);
    bool occlusionFill();

    // TEMP
    void *calculateZNCC_thread(ZNCCArgs *args);

    // helper methods
    void putPixel(unsigned int x, unsigned int y, Pixel pixel);
    void putPixel(unsigned int x, unsigned int y, unsigned char grey);
    Pixel getPixel(unsigned int x, unsigned int y);
    unsigned char getGrayPixel(unsigned int x, unsigned int y);
    void printPixel(unsigned int x, unsigned int y);

    size_t sizeBytes();
    unsigned char grayAverage(unsigned int startX = 0, unsigned int startY = 0, size_t w = 0, size_t h = 0);
    bool validCoordinates(unsigned int x, unsigned int y);
};


/* TEMPORARY */
struct ZNCCArgs
{
    int tid;
    const char windowSize;
    unsigned int fromY;
    unsigned int toY;
    char dir;
    unsigned int maxSearchD;
    Image *thisImg;
    Image otherImg;
    Image *disparityMap;

    //      (int&,    const char&,           unsigned int&,      unsigned int&,    char&,                             Image*&,      , Image&,         Image*&)
    ZNCCArgs(int tid, const char windowSize, unsigned int fromY, unsigned int toY, char dir, unsigned int maxSearchD, Image *thisImg, Image otherImg, Image *disparityMap)
        : tid(tid), windowSize(windowSize), fromY(fromY), toY(toY), dir(dir), maxSearchD(maxSearchD), thisImg(thisImg), otherImg(otherImg), disparityMap(disparityMap) {}
};
