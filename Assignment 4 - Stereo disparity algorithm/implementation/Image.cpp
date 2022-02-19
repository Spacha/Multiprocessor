#include "Image.hpp"

using std::cout;
using std::endl;

/**
 * Convert the floating point representation
 * to the integer representation.
 */
pixel_t pixelf_to_pixel(pixelf_t p)
{
    return {
        (unsigned char)(255.0f * p.red),
        (unsigned char)(255.0f * p.green),
        (unsigned char)(255.0f * p.blue),
        (unsigned char)(255.0f * p.alpha)
    };
}
/**
 * Convert the integer representation
 * to the floating point representation.
 */
pixelf_t pixel_to_pixelf(pixel_t p)
{
    return {
        (float)(  p.red / 255.0f),
        (float)(p.green / 255.0f),
        (float)( p.blue / 255.0f),
        (float)(p.alpha / 255.0f)
    };
}

///////////////////////////////////////////////////////////////////////////////
// Image
///////////////////////////////////////////////////////////////////////////////

/**
 * Initializes the object and sets color type.
 * E.g. LCT_RGBA, LCT_GREY
 **/
Image::Image(LodePNGColorType colorType /* = LCT_RGBA */)
{
    this->colorType = colorType;
}

/**
 * Destructs the object.
 **/
Image::~Image()
{
    // ...
}

/**
 * In case parallel execution is to be used, an OpenCL (MiniOCL) instance is needed.
 * This method sets an already initialized MiniOCL object as a property.
 **/
void Image::setOpenCL(MiniOCL *ocl)
{
    this->ocl = ocl;
}

/**
 * Creates an empty image of given size.
 **/
void Image::createEmpty(size_t width, size_t height)
{
    this->width = width;
    this->height = height;

    this->image.clear();
    this->image.resize(this->sizeBytes(), (unsigned char)0);
    this->image.resize(this->sizeBytes(), (unsigned char)0);
}

/**
 * Replaces the current image with given image @newImage.
 * The image sizes must match exactly.
 **/
void Image::replaceImage(std::vector<unsigned char> &newImage)
{
    if (newImage.size() != this->image.size())
        throw;

    this->image = std::move(newImage);
}

/**
 * Load PNG file from disk to memory first, then decode to raw pixels in memory.
 * Returns true on success, false on fail.
 **/
bool Image::load(const std::string &filename)
{
    this->name = filename;

    unsigned err;
    std::vector<unsigned char> png;

    // load and decode
    cout << "Loading image... ";
    err = lodepng::load_file(png, filename);
    cout << "Done." << endl;

    if (err) {
        cout << "Image load error " << err << ": " << lodepng_error_text(err) << endl;
        return false;
    }

    cout << "Decoding image... ";
    err = lodepng::decode(this->image,
        (unsigned &)this->width, (unsigned &)this->height, png);
    cout << "Done." << endl;


    // the pixels are now in the vector "image", 4 bytes per pixel, ordered RGBARGBA...

    if (err) {
        cout << "Decode error " << err << ": " << lodepng_error_text(err) << endl;
        return false;
    }

    return true;
}

/**
 * Encode PNG and save it to the disk.
 * Returns true on success, false on fail.
 **/
bool Image::save(const std::string &filename)
{
    unsigned err;
    std::vector<unsigned char> png;

    cout << "Encoding image... ";
    err = lodepng::encode(png, this->image,
        (unsigned)this->width, (unsigned)this->height);
    cout << "Done." << endl;

    if (err) {
        cout << "Encode error " << err << ": "<< lodepng_error_text(err) << endl;
        return false;
    }

    cout << "Saving image... ";
    lodepng::save_file(png, filename);
    cout << "Done." << endl;

    if (err) {
        cout << "Image save error " << err << ": " << lodepng_error_text(err) << endl;
        return false;
    }

    return true;
}

///////////////////////////////////////////////////////////////////////////////
// Image manipulation
///////////////////////////////////////////////////////////////////////////////

/**
 * Converts the image to grayscale and makes the image opaque.
 * Uses the NTSC formula.
 **/
bool Image::convertToGrayscale()
{
    bool success = true;
    cout << "Transforming image to grayscale... ";

#ifdef USE_OCL /* OpenCL (GPU or CPU) */

    if (!this->ocl) {
        cout << "Cannot do parallel execution without instance of MiniOCL." << endl;
        return false;
    }

    // set up OpenCL for execution

    success = ocl->buildKernel(kernelFileName, "grayscale");

    ocl->setInputImageBuffer(
        0, static_cast<void *>(image.data()), width, height);   // image in
    ocl->setOutputImageBuffer(
        1, static_cast<void *>(image.data()), width, height);   // image out

    success = this->ocl->executeKernel(width, height, 16, 16);

#else /* No parallelization */

    for (unsigned int y = 0; y < this->height; y++)
    {
        for (unsigned int x = 0; x < this->width; x++)
        {
            pixel_t p = this->getPixel(x, y);

            // replace the pixel with a gray one
            this->putPixel(x, y, ceil(0.299*p.red + 0.587*p.green + 0.114*p.blue));
        }
    }

#endif
    // this->colorType = LCT_GREY; // update color type
    cout << "Done." << endl;
    return success;
}

/**
 * Filters the image using the given mask.
 **/
bool Image::filter(const filter_t &filter)
{
    bool success = true;
    cout << "Filtering image using a mask... ";

    // the mask size must be odd
    if (filter.size % 2 == 0) {
        cout << "Error: mask size is not odd." << endl;
        return false;
    }

#ifdef USE_OCL /* OpenCL (GPU or CPU) */

    if (!this->ocl) {
        cout << "Cannot do parallel execution without instance of MiniOCL." << endl;
        return false;
    }

    success = this->ocl->buildKernel(kernelFileName, "filter");

    ocl->setInputImageBuffer(
        0, static_cast<void *>(image.data()), width, height);               // image in
    ocl->setOutputImageBuffer(
        1, static_cast<void *>(image.data()), width, height);               // image out
    ocl->setInputBuffer(
        2, (void *)filter.mask, filter.size * filter.size * sizeof(float)); // filter mask
    ocl->setValue(
        3, (void *)&filter.size, sizeof(int));                              // filter size
    ocl->setValue(
        4, (void *)&filter.divisor, sizeof(float));                         // filter divisor

    success = ocl->executeKernel(width, height, 16, 16);

#else /* No parallelization */

    // TODO: This implementation could use different edge handling techniques.
    // std::vector<unsigned char> tempImage(4 * width * height);
    Image tempImage;
    tempImage.createEmpty(width, height);

    int d = filter.size / 2; // kernel's "edge thickness"

    for (unsigned int cy = 0; cy < height; cy++)
    {
        for (unsigned int cx = 0; cx < width; cx++)
        {
            unsigned int weight = 0;
            pixelf_t newClr = { 0.0f, 0.0f, 0.0f, 255.0f };

            // iterate over each element in the mask
            for (int y = cy - d; y <= (int)(cy + d); y++)
            {
                for (int x = cx - d; x <= (int)(cx + d); x++)
                {
                    // get pixel if within image
                    pixel_t p = validCoordinates(x, y)
                        ? this->getPixel(x, y)
                        : (pixel_t){ 0, 0, 0, 0 };

                    newClr.red   += filter.mask[weight] * (float)p.red;
                    newClr.green += filter.mask[weight] * (float)p.green;
                    newClr.blue  += filter.mask[weight] * (float)p.blue;
                    weight++;
                }
            }

            // convert back to integers (floats are used in computation)
            pixel_t pixelInt = {
                (unsigned char)(newClr.red / filter.divisor),
                (unsigned char)(newClr.green / filter.divisor),
                (unsigned char)(newClr.blue / filter.divisor),
                (unsigned char)(newClr.alpha)
            };

            // replace the pixel in the center of the mask
            tempImage.putPixel(cx, cy, pixelInt);
        }
    }

    // update the image
    this->replaceImage(tempImage.image);

#endif
    cout << "Done." << endl;
    return success;
}

bool Image::resize(size_t width, size_t height)
{
    bool success;

    // ...

    return success;

}
bool Image::calcZNCC()
{
    bool success;

    // ...

    return success;

}
bool Image::crossCheck(Image &left, Image &right)
{
    bool success;

    // ...

    return success;
}
bool Image::occlusionFill()
{
    bool success;

    // ...

    return success;
}


///////////////////////////////////////////////////////////////////////////////
// Helper methods
///////////////////////////////////////////////////////////////////////////////

/**
 * Puts given RGBA (4 channel) pixel to position (x,y).
 **/
void Image::putPixel(unsigned int x, unsigned int y, pixel_t pixel)
{
    if (!validCoordinates(x, y))
        throw;

    unsigned int i = 4*(y*width + x);
    this->image[i]   = pixel.red;
    this->image[i+1] = pixel.green;
    this->image[i+2] = pixel.blue;
    this->image[i+3] = pixel.alpha;
}

/**
 * Puts given grayscale (1 channel) pixel to position (x,y).
 **/
void Image::putPixel(unsigned int x, unsigned int y, unsigned char grey)
{
    pixel_t pixel = { grey, grey, grey, 0xff};
    this->putPixel(x, y, pixel);
}

/**
 * Returns a pixel struct containing the color values
 * of each pixel in position (x,y).
 **/
pixel_t Image::getPixel(unsigned int x, unsigned int y)
{
    if (!validCoordinates(x, y))
        throw;

    // RGBA
    unsigned int i = 4*(y*width + x);
    return { image[i], image[i+1], image[i+2], image[i+3] };
}

/**
 * Print hex color value of pixel in position (x,y).
 **/
void Image::printPixel(unsigned int x, unsigned int y)
{
    pixel_t pixel = getPixel(x, y);
    printf("Pixel (%u, %u):\tRGBA: #%02x%02x%02x%02x\n", x, y, pixel.red, pixel.green, pixel.blue, pixel.alpha);
}

/**
 * Returns true if given coordinates point to
 * a pixel coordinate, otherwise false.
 **/
bool Image::validCoordinates(unsigned int x, unsigned int y)
{
    return !(x > (width-1) || y > (height-1) || x < 0 || y < 0);
}

/**
 * Returns the size of the image in bytes.
 **/
size_t Image::sizeBytes()
{
    // TODO: Can we guarantee always having 4 channels?
    return 4 * sizeof(unsigned char) * this->width * this->height;
}
