#include "Image.hpp"

using std::cout;
using std::endl;

///////////////////////////////////////////////////////////////////////////////
// Image
///////////////////////////////////////////////////////////////////////////////

/**
 * Initializes the object.
 */
Image::Image() : width(0), height(0)
{
    // ...
}

/**
 * Destructs the object and cleans up after itself.
 */
Image::~Image()
{
    // ...
}

/**
 * In case parallel execution is to be used, an OpenCL (MiniOCL) instance is needed.
 * This method sets an already initialized MiniOCL object as a property.
 * 
 * @param ocl Initiated instance of MiniOCL for parallel processing.
 */
void Image::setOpenCL(MiniOCL *ocl)
{
    this->ocl = ocl;
}

/**
 * Creates an empty image of given image. Image will
 * contain only transparent balck pixels.
 * 
 * @param width  Image width
 * @param height Image height
 */
void Image::createEmpty(size_t width, size_t height)
{
    this->width = width;
    this->height = height;

    this->image.clear();
    this->image.resize(this->sizeBytes(), (unsigned char)0);
}

/**
 * Replaces the current image with given image @newImage.
 * The image sizes must match exactly.
 * 
 * @param newImage Image that will replace the current image.
 */
void Image::replace(Image &newImage)
{
    if (newImage.width != this->width || newImage.height != this->height)
    {
        // new image is different size -> make a new image
        this->createEmpty(newImage.width, newImage.height);
    }

    this->image = std::move(newImage.image);
}

/**
 * Load PNG file from disk to memory first, then decode to raw pixels in memory.
 * Returns true on success, false on fail.
 **/

/**
 * Load PNG file from disk to memory first, then decode to raw pixels in memory.
 * Returns true on success, false on fail.
 * 
 * @param filename Name of the image file to be loaded.
 * @return         True on success, false on fail.
 */
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
 * 
 * @param filename Name of the file to be saved.
 * @return         True on success, false on fail.
 */
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
 * Uses the NTSC formula. Depending on the flags, this can use parallel
 * processing to significantly speed up.
 * 
 * @return True on success, false on fail.
 */
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

    success = ocl->buildKernel("grayscale");

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
            Pixel p = this->getPixel(x, y);

            // replace the pixel with a gray one
            this->putPixel(x, y, ceil(0.299*p.red + 0.587*p.green + 0.114*p.blue));
        }
    }

#endif
    cout << "Done." << endl;
    return success;
}

/**
 * Filters the image using the given mask.
 * 
 * @param filter Filter to be applied to the image.
 * @return       True on success, false on fail.
 */
bool Image::filter(const Filter &filter)
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

    success = this->ocl->buildKernel("filter");

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
    Image tempImage;
    tempImage.createEmpty(width, height);

    int d = filter.size / 2; // kernel's "edge thickness"

    for (unsigned int cy = 0; cy < height; cy++)
    {
        for (unsigned int cx = 0; cx < width; cx++)
        {
            unsigned int weight = 0;
            // we need more space per pixel since we first accumulate and then divide
            unsigned long int newRed = 0, newGreen = 0, newBlue = 0, newAlpha = 0;

            // iterate over each element in the mask
            for (int y = cy - d; y <= (int)(cy + d); y++)
            {
                for (int x = cx - d; x <= (int)(cx + d); x++)
                {
                    // get pixel if within image
                    Pixel p = validCoordinates(x, y)
                        ? this->getPixel(x, y)
                        : Pixel(0, 0, 0, 0);

                    newRed   += filter.mask[weight] * p.red;
                    newGreen += filter.mask[weight] * p.green;
                    newBlue  += filter.mask[weight] * p.blue;
                    newAlpha  += filter.mask[weight] * p.alpha;
                    weight++;
                }
            }

            Pixel newClr(
                (unsigned char)(newRed / filter.divisor),
                (unsigned char)(newGreen / filter.divisor),
                (unsigned char)(newBlue / filter.divisor),
                (unsigned char)(newAlpha / filter.divisor));

            // replace the pixel in the center of the mask
            tempImage.putPixel(cx, cy, newClr);
        }
    }

    cout << "Check if can divide alpha too, then REMOVE THIS COMMENT" << endl;
    // update the image
    this->replace(tempImage);

#endif
    cout << "Done." << endl;
    return success;
}

/**
 * Dynamically creates an averaging filter and applies it to the image.
 * 
 * @param size Size of the mask.
 * @return     True on success, false on fail.
 */
bool Image::filterMean(const size_t size)
{
    bool success = false;
    if (size == 0) return false;

    // fill the mask with ones
    float *mask = (float*)malloc( size * size * sizeof(float) );

    if (mask != NULL)
    {
        // populate the mask with ones
        for (unsigned int i = 0; i < (size * size); i++) { mask[i] = 1.0f; }

        const Filter filter((const size_t)size, (const float)(size * size), mask);
        success = this->filter(filter);
    }

    free(mask);
    return success;
}

/**
 * Scales the image down by given integer factor. Current implementation
 * is quite rough: just blur using an averaging filter and then drop
 * some pixels. In this case, however, it works very well.
 *
 * @todo Different factor for x and y would be easy.
 *
 * @param factor Scaling factor.
 * @return       True on success, false on fail.
 */
bool Image::downScale(unsigned int factor)
{
    cout << "Resizing image... ";

    // Since we cannot use even-sized kernels, use the
    // closest odd-numbered kernel size towards zero.
    // For example, if factor = 4 or 5 => maskSize = 5.
    const size_t maskSize = (factor % 2 == 0) ? factor + 1 : factor;

    // filtering the image first gives a better downscaling quality
    this->filterMean(maskSize);

    Image tempImage;
    tempImage.createEmpty(this->width / factor, this->height / factor);

    for (unsigned int y = 0; y < this->height; y++)
    {
        if (y % factor == 0) continue; // skip every factor'th row

        for (unsigned int x = 0; x < this->width; x++)
        {
            if (x % factor == 0) continue; // skip every factor'th column

            // copy the pixel
            tempImage.putPixel(x / factor, y / factor, this->getPixel(x, y));
        }
    }

    this->replace(tempImage);

    cout << "Done." << endl;
    return true;
}

/**
 * Calculates the disparity map of the image compared to
 * another image. The disparity map replaces the current image.
 * 
 * @param otherImg     The image to be compared against.
 * @param disparityMap Pointer to a location to store the disparity map.
 * @param reverse      Traverse the right image to right instead of left.
 * @return             True on success, false on fail.
 */
bool Image::calcZNCC(Image &otherImg, Image *disparityMap, bool reverse /* = false */)
{
    disparityMap->createEmpty(otherImg.width, otherImg.height);

    //unsigned char leftAvg = this->grayAverage();
    //unsigned char rightAvg = otherImg.grayAverage();
    const char windowSize = 15;
    const char maxSearchD = 55;

    if (windowSize % 2 == 0)
    {
        cout << "Window size must be odd." << endl;
        return false;
    }
    
    // precalculate some...
    const char halfWindow = (windowSize - 1) / 2;

    // d = -1 -> move left (default), d = +1 -> move right
    char dir = reverse ? 1 : -1;

    float progress = 0.0f;
    float progressPerRound = 1.0f / this->height;

    for (unsigned int y = halfWindow; y < (this->height - halfWindow); y++)
    {
        for (unsigned int x = halfWindow; x < (this->width - halfWindow); x++)
        {
            unsigned int leftAvg = this->grayAverage(
                x - windowSize,
                y - windowSize,
                windowSize,
                windowSize);

            unsigned char bestD = 0;        // tracks the distance with best correlation
            float maxCorrelation = 0.0f;    // tracks the best correlation (ZNCC)

            // stops at the left/right edge
            char maxD = reverse
                ? std::min((int)maxSearchD, (int)((this->width - 1 - halfWindow) - x))
                : std::min((int)maxSearchD, (int)(x - halfWindow));
                //: std::min((int)(this->width - 1 - maxSearchD), (int)(this->width - 1 - x - halfWindow));

            for (int d = 0; d <= maxD; d++)
            {
                unsigned int rightAvg = otherImg.grayAverage(
                    x - windowSize + (dir * d),
                    y - windowSize,
                    windowSize,
                    windowSize);

                /* Calculate ZNCC */

                int upperSum = 0;
                unsigned int lowerLeftSum = 0;
                unsigned int lowerRightSum = 0;

                //cout << "getting (" << x << ", " << y << ")";
                // getting(48, 64)
                /* Calculate ZNCC(x, y, d) */
                for (int wy = -halfWindow; wy <= halfWindow; wy++) // 20
                {
                    for (int wx = -halfWindow; wx <= halfWindow; wx++) // 20
                    {
                        // difference of (left/right) image pixel from the average
                        // TODO: Not necessary for each d!
                        char leftDiff  = this->getGrayPixel(x + wx, y + wy) - leftAvg;
                        char rightDiff = otherImg.getGrayPixel(x + wx + (dir * d), y + wy) - rightAvg;

                        upperSum      += leftDiff * rightDiff;
                        lowerLeftSum  += leftDiff * leftDiff;     // leftDiff ^ 2
                        lowerRightSum += rightDiff * rightDiff;   // rightDiff ^ 2
                    }
                }
                //cout << " done" << endl;

                // Finally calculate the ZNCC value
                float correlation = (float)(upperSum / (sqrt(lowerLeftSum) * sqrt(lowerRightSum)));

                // update disparity value for pixel (x,y)
                if (correlation > maxCorrelation)
                {
                    maxCorrelation = correlation;
                    bestD = d;
                }
            }

            // put the best disparity value to the disparity map
            disparityMap->putPixel(x, y, bestD);
        }
        progress += progressPerRound;
        cout << "Calculating ZNCC... " << (unsigned int)(100 * progress) << " %\r" << std::flush;
    }

    cout << "Calculating ZNCC... Done.\r" << endl;
    return true;
}

/**
 * Performs a cross-checking for two disparity maps of the same size.
 * The result is stored as an image.
 * 
 * @param left   The left-to-right disparity map.
 * @param right  The right-to-left disparity map.
 * @return       True on success, false on fail.
 */
bool Image::crossCheck(Image &left, Image &right)
{
    /*
    The LEFT disparity map is obtained by mapping the image on the left against the one
    on the right.
    The RIGHT disparity map is obtained analogously by mapping the image on the right
    against the one on the left.

    Cross checking is a process where you compare two depth maps.

    To obtain a consolidated map, the process consists in checking that the corresponding
    pixels in the left and right disparity images are consistent. This can be done by comparing
    their absolute difference to a threshold value. For our case, it is recommended to start
    with a threshold value of 8, while the best threshold value for your implementation can
    be obtained by experimentation. If the absolute difference is larger than the threshold,
    then replace the pixel value with zero. Otherwise the pixel value remains unchanged.
    This process helps in removing the probable lack of consistency between the depth maps
    due to occlusions, noise, or algorithmic limitation.

    // If the absolute difference is larger than the threshold,
    // then replace the pixel value with zero. Otherwise the pixel value remains unchanged.

    // Spacha: I assume that the "otherwise the pixel remains unchanged" means that we
    // pick one from either left or right picture? We'll pick from the left image.
    // What about averaging between the two images?
    */
    bool success;
    cout << "Performing cross-check... ";

    // the disparity maps must be exactly the same size
    if (left.width != right.width || left.height != right.height)
        return false;

    this->createEmpty(left.width, left.height);

    unsigned char threshold = 8;

    for (unsigned int y = 0; y < this->height; y++)
    {
        for (unsigned int x = 0; x < this->width; x++)
        {
            unsigned char leftPixel = left.getGrayPixel(x, y);

            if (std::abs(leftPixel - right.getGrayPixel(x, y)) > threshold)
            {
                this->putPixel(x, y, (unsigned char)0);
            }
            else
            {
                this->putPixel(x, y, leftPixel);
            }
        }
    }

    // Do some magic, will ya?
    success = true;

    cout << "Done." << endl;
    return success;
}

/**
 * Performs an occlusion filling to the image. The image is overwritten.
 * 
 * @return       True on success, false on fail.
 */
bool Image::occlusionFill()
{
    /*
    Occlusion filling is the process of eliminating the pixels that have been
    assigned to zero by the previously calculated cross-checking. In the simplest
    form it can be done by replacing each pixel with zero value with the nearest
    non-zero pixel value. However, in this exercise it is recommended that you
    experiment with other more complex postprocessing approaches that obtain the
    pixel value in different forms.

    Some ideas:
        - take the pixel from left (or right if on the left edge)
        - take the average of the surrounding pixels (interpolate)...

    FOR x in this->image:
        FOR y in this->image:
            if !this->getPixel(x, y).isZero():  // not zero, next pixel
                continue

            // filling is needed
            Pixel newPixel(0, 0, 0, 0);

            // note: this can well be empty as well, need
            // to specifically look for non-zero pixel!
            if (x > 0)
                this->putPixel(x, y, this->getPixel(x - 1, y));
            else                         // on the left edge
                this->putPixel(x, y, this->getPixel(x + 1, y));

    */
    bool success;
    cout << "Performing occlusion fill... ";

    for (unsigned int y = 0; y < this->height; y++)
    {
        for (unsigned int x = 0; x < this->width; x++)
        {
            unsigned char p = this->getGrayPixel(x, y);

            if (p > 0) continue;

            for (int x0 = x; x0 >= 0; x0--)
            {
                unsigned char p0 = this->getGrayPixel(x0, y);

                if (p0 > 0)
                {
                    // replace current pixel (that is zero) with p0
                    this->putPixel(x, y, p0);
                    break;
                }
            }
        }
    }

    success = true;

    cout << "Done." << endl;
    return success;
}


///////////////////////////////////////////////////////////////////////////////
// Helper methods
///////////////////////////////////////////////////////////////////////////////

/**
 * Puts given RGBA (4 channel) pixel to position (x,y).
 */
void Image::putPixel(unsigned int x, unsigned int y, Pixel pixel)
{
    if (!validCoordinates(x, y))
        throw;

    // unsigned int i = 4*(y*width + x);
    const __int64 i = 4 * (y * width + x);

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
    Pixel pixel(grey, grey, grey, 0xff);
    this->putPixel(x, y, pixel);
}

/**
 * Returns a pixel struct containing the color values
 * of each pixel in position (x,y).
 */
Pixel Image::getPixel(unsigned int x, unsigned int y)
{
    if (!validCoordinates(x, y))
        throw;

    // RGBA
    // unsigned int i = 4*(y*width + x);
    const __int64 i = 4 * (y * width + x);
    return Pixel(image[i], image[i + 1], image[i + 2], image[i + 3]);
}

/**
 * Returns the pixel value of the red channel in position (x,y).
 */
unsigned char Image::getGrayPixel(unsigned int x, unsigned int y)
{
    return image[4 * (y * width + x)];
}

/**
 * Print hex color value of pixel in position (x,y).
 */
void Image::printPixel(unsigned int x, unsigned int y)
{
    Pixel pixel = getPixel(x, y);
    printf("Pixel (%u, %u):\tRGBA: #%02x%02x%02x%02x\n", x, y, pixel.red, pixel.green, pixel.blue, pixel.alpha);
}

/**
 * Returns true if given coordinates point to
 * a pixel coordinate, otherwise false.
 */
bool Image::validCoordinates(unsigned int x, unsigned int y)
{
    return !(x > (width-1) || y > (height-1) || x < 0 || y < 0);
}

///////////////////////////////////////////////////////////////////////////////
// Get information
///////////////////////////////////////////////////////////////////////////////

/**
 * Calculates the average pixel value of the image.
 * Assumes the image is in grayscale.
 *
 * @param startX    The X value to start from (default = 0).
 * @param startY    The Y value to start from (default = 0).
 * @param w         The window width (default = image width).
 * @param h         The window height (default = image height).
 * @return          Average pixel value.
 */
unsigned char Image::grayAverage(unsigned int startX, unsigned int startY, size_t w, size_t h)
{
    unsigned __int64 avg = 0;

    w = (w > 0) ? w : this->width;
    h = (h > 0) ? h : this->height;

    for (unsigned int y = startY; y < h; y++)
    {
        for (unsigned int x = startX; x < w; x++)
        {
            avg += this->getGrayPixel(x, y);
        }
    }

    return (unsigned char)(avg / (w * h));
}

/**
 * Returns the size of the image in bytes.
 */
size_t Image::sizeBytes()
{
    // TODO: Can we guarantee always having 4 channels?
    return 4 * sizeof(unsigned char) * this->width * this->height;
}
