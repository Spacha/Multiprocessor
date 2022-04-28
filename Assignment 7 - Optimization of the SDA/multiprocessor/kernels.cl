// See OpenCL Programming Guide p.342.
// OpenCL kernel. Each work item takes care of one element of C.

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE   |
                               CLK_FILTER_NEAREST;

///////////////////////////////////////////////////////////////////////////////
// GRAYSCALE KERNEL
///////////////////////////////////////////////////////////////////////////////

/**
 * Converts image @in to grayscale (still RGBA) and
 * stores the result to image @out.
 **/
__kernel void grayscale(__read_only image2d_t in,
                        __write_only image2d_t out)
{
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    float4 clr = read_imagef(in, sampler, coord);

    if (coord.x >= get_image_width(in) || coord.y >= get_image_height(in))
    {
        return;
    }

    // NTCS gray conversion
    float gray = 0.299f*clr.x + 0.587f*clr.y + 0.114f*clr.z; // float4 : (xyzw)

    write_imagef(out, coord, (float4)(gray, gray, gray, clr.w));
}

///////////////////////////////////////////////////////////////////////////////
// FILTER KERNEL
///////////////////////////////////////////////////////////////////////////////

/**
 * Applies @mask to image @in and stores the result to image @out.
 * The mask size and divisor must be provided (if the
 * mask has no common divisor, set it to 1.0f).
 **/
__kernel void filter(__read_only image2d_t in,
                     __write_only image2d_t out,
                     __constant float *mask,
                     const int maskSize, const float divisor)
{
    int d = maskSize / 2; // filter "edge thickness"
    int2 center = (int2)(get_global_id(0), get_global_id(1));
    int2 topLeft = center - d;
    int2 btmRight = center + d;

    float w = get_image_width(in);
    float h = get_image_height(in);

    if (center.x >= w || center.y >= h)
    {
        return;
    }

    // Calculate the pixel value (in center) by sweeping over the mask

    float4 clr = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    int maskIdx = 0;

    for (int x = topLeft.x; x <= btmRight.x; x++)
    {
        for (int y = topLeft.y; y <= btmRight.y; y++)
        {
            clr += read_imagef( in, sampler, (int2)(x,y) ) * mask[maskIdx];
            maskIdx++;
        }
    }

    write_imagef( out, center, (float4)(clr / divisor) );
}

///////////////////////////////////////////////////////////////////////////////
// ZNCC (DISPARITY) KERNEL
///////////////////////////////////////////////////////////////////////////////

/**
 * NOTE: Assumes grayscale image.
 * Calculates ZNCC disparity between @in_this and @in_other according to
 * @windowSize, @dir and @maxSearchD parameters. The result is written to @out.
 **/
__kernel void calc_zncc(__read_only image2d_t in_this,
                        __read_only image2d_t in_other,
                        __write_only image2d_t out,
                        char windowSize,
                        char dir,
                        unsigned int maxSearchD)
{
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    float w = get_image_width(in_this);
    float h = get_image_height(in_this);
    const char halfWindow = (windowSize - 1) / 2;

    // skip the edges
    if (pos.x < halfWindow || pos.y < halfWindow ||
        pos.x >= (w - halfWindow) || pos.y >= (h - halfWindow))
    {
        return;
    }

    // Calculate left window average.
    float windowSum = 0;
    float4 clr;
    for (unsigned int y = pos.y - halfWindow; y < pos.y + halfWindow; y++) {
        for (unsigned int x = pos.x - halfWindow; x < pos.x + halfWindow; x++) {
            clr = read_imagef( in_this, sampler, (int2)(x, y) );
            windowSum += clr[0];
        }
    }
    float leftAvg = (windowSum / (windowSize*windowSize));

    unsigned char bestD = 0;        // tracks the distance with best correlation
    float maxCorrelation = 0.0f;    // tracks the best correlation (ZNCC)

    // stops at the left/right edge (OpenCL defines generic integer min and max)
    char maxD = (dir > 0)
        ? min((int)maxSearchD, (int)((w - 1 - halfWindow) - pos.x))
        : min((int)maxSearchD, (int)(pos.x - halfWindow));

    for (int d = 0; d <= maxD; d++)
    {
         // Calculate right window average.
        windowSum = 0;
        for (unsigned int y = pos.y - halfWindow; y < pos.y + halfWindow; y++) {
            for (unsigned int x = pos.x + (dir * d) - halfWindow; x < pos.x + (dir * d) + halfWindow; x++) {
                clr = read_imagef( in_other, sampler, (int2)(x, y) );
                windowSum += clr[0];
            }}
        float rightAvg = (windowSum / (windowSize*windowSize));

        /* Calculate ZNCC */

        float upperSum = 0;
        float lowerLeftSum = 0;
        float lowerRightSum = 0;

        /* Calculate ZNCC(x, y, d) */
        for (int wy = -halfWindow; wy <= halfWindow; wy++) {
            for (int wx = -halfWindow; wx <= halfWindow; wx++) {
                // difference of (left/right) image pixel from the average
                // TODO: Not necessary for each d!
                clr = read_imagef( in_this, sampler, (int2)(pos.x + wx, pos.y + wy) );
                float leftDiff = clr[0] - leftAvg;
                clr = read_imagef( in_other, sampler, (int2)(pos.x + wx + (dir * d), pos.y + wy) );
                float rightDiff = clr[0] - rightAvg;

                upperSum      += leftDiff * rightDiff;
                lowerLeftSum  += leftDiff * leftDiff;     // leftDiff ^ 2
                lowerRightSum += rightDiff * rightDiff;   // rightDiff ^ 2
            }
        }

        // Finally calculate the ZNCC value
        float correlation = (float)(upperSum / (sqrt(lowerLeftSum) * sqrt(lowerRightSum)));

        // update disparity value for pixel (x,y)
        if (correlation > maxCorrelation)
        {
            maxCorrelation = correlation;
            bestD = d;
        }

        // put the best disparity value to the disparity map
        //args->disparityMap->putPixel(x, y, bestD);
        float p = bestD / 255.0f;
        write_imagef( out, pos, (float4)(p, p, p, 1.0f) );
    }

    //float4 clr1 = read_imagef( in_this, sampler, pos );
    //float4 clr2 = read_imagef( in_other, sampler, pos );
    //float4 clr = (float4)((clr1 + clr2) / 2);
    //write_imagef( out, pos, (float4)(leftAvg, leftAvg, leftAvg, 1.0f) );
}

///////////////////////////////////////////////////////////////////////////////
// CROSS CHECK KERNEL
///////////////////////////////////////////////////////////////////////////////

/**
 * NOTE: Assumes grayscale image.
 **/
__kernel void cross_check(__read_only image2d_t in_left,
                          __read_only image2d_t in_right,
                          __write_only image2d_t out,
                          unsigned int threshold)
{
    int2 pos = (int2)(get_global_id(0), get_global_id(1));

    float4 leftPixel = read_imagef( in_left, sampler, pos );
    float4 rightPixel = read_imagef( in_right, sampler, pos );

    // If there is a sufficiently large difference between the images,
    // replace the pixel with tranparent black pixel.
    if (fabs(leftPixel[0] - rightPixel[0]) > (float)(threshold / 255.0f))
    {
        write_imagef( out, pos, (float4)(0.0f, 0.0f, 0.0f, 0.0f) );
    } else {
        write_imagef( out, pos, leftPixel );
    }
}

///////////////////////////////////////////////////////////////////////////////
// OCCLUSION FILL KERNEL
///////////////////////////////////////////////////////////////////////////////

/**
 * Performs a very simple occlusion fill. For each pixel, the nearest non-zero
 * pixel on the left is used. If none is found, write a black opaque pixel.
 * + very fast
 * - the resulting fill has 'tears' on some edges
 **/
__kernel void occlusion_fill_left(__read_only image2d_t in,
                                  __write_only image2d_t out)
{
    int2 pos = (int2)(get_global_id(0), get_global_id(1));

    float4 clr = read_imagef( in, sampler, pos );

    // just copy the pixel...
    if (clr[0] > 0)
    {
        write_imagef( out, pos, clr );
        return;
    }

    for (int x0 = pos.x; x0 >= 0; x0--)
    {
        //unsigned char p0 = this->getGrayPixel(x0, y);
        float4 clr0 = read_imagef( in, sampler, (int2)(x0, pos.y) );

        if (clr0[0] > 0)
        {
            // replace current pixel (that is zero) with clr0
            write_imagef( out, pos, clr0 );
            return;
        }
    }

    // if no pixels found on the left, replace with black pixel
    write_imagef( out, pos, (float4)(0.0f, 0.0f, 0.0f, 1.0f) );
}

/**
 * Performs more complicated occlusion fill. For each pixel, the nearest non-zero
 * pixel is searched using a spiral-like search (square). If none is found
 * (impossible), write a black opaque pixel.
 * + more accurate fill
 * - more complex -> slightly slower
 */
__kernel void occlusion_fill_nearest(__read_only image2d_t in,
                                     __write_only image2d_t out)
{
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    float w = get_image_width(in);
    float h = get_image_height(in);

    float4 clr = read_imagef( in, sampler, pos );

    // just copy the pixel...
    if (clr[0] > 0)
    {
        write_imagef( out, pos, clr );
        return;
    }

    // Spirals out in X and Y, checking each pixel for the nearest non-zero pixel.

    //for (int x0 = pos.x; x0 >= 0; x0--)
    for (int i = 1; i < w; i++)
    {
        for (char dirX = -1; dirX <= 1; dirX++)
        {
            int x0 = pos.x + dirX * i;
            if (x0 < 0 || x0 > (w - 1))
                    continue;

            for (char dirY = -1; dirY <= 1; dirY++)
            {
                int y0 = pos.y + dirY * i;
                if (y0 < 0 || y0 > (h - 1))
                    continue;

                //unsigned char p0 = this->getGrayPixel(x0, y);
                float4 clr0 = read_imagef( in, sampler, (int2)(x0, y0) );

                if (clr0[0] > 0)
                {
                    // replace current pixel (that is zero) with clr0
                    write_imagef( out, pos, clr0 );
                    return;
                }
            }
        }
    }

    // if no pixels found on the left, replace with black pixel
    write_imagef( out, pos, (float4)(0.0f, 0.0f, 0.0f, 1.0f) );
}





///////////////////////////////////////////////////////////////////////////////
// STEREO DISPARITY KERNEL
///////////////////////////////////////////////////////////////////////////////

/**
 * A single kernel to do it all:
 * 1. Grayscale images
 * 2. Calculate ZNCC
 * 3. Cross-checking
 * 4. Occlusion filling
 **/
__kernel void calc_stereo_disparity(__read_only image2d_t in_left,
                                    __read_only image2d_t in_right,
                                    __write_only image2d_t out,
                                    char windowSize,
                                    unsigned int maxSearchD)
{
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    const char halfWindow = (windowSize - 1) / 2;

#if 0
    //
    // 1. Grayscale
    //

    // NTCS gray conversion
    float gray_l = 0.299f*clr_l.x + 0.587f*clr_l.y + 0.114f*clr_l.z; // float4 : (xyzw)
    float gray_r = 0.299f*clr_r.x + 0.587f*clr_r.y + 0.114f*clr_r.z; // float4 : (xyzw)

    write_imagef( out, pos, (float4)( gray_l, gray_l, gray_l, 1.0f ) );
    write_imagef( out_xtra, pos, (float4)( gray_r, gray_r, gray_r, 1.0f ) );

    //float g = (gray_l + gray_r) / 2;
    //float4 new_clr = (float4)( g, g, g, 1.0f );
    // TODO: output a single-channel grayscale image?
    //write_imagef( out, pos, new_clr );

    // wait for all work-items to reach this point
    barrier(CLK_GLOBAL_MEM_FENCE);

    //
    // 2. Calculate ZNCC
    //

    ////////////////////////////////////////////////////////////////////////////

    float w = get_image_width(in_left);
    float h = get_image_height(in_left);

    /*
    // skip the edges
    if (pos.x < halfWindow || pos.y < halfWindow ||
        pos.x >= (w - halfWindow) || pos.y >= (h - halfWindow))
    {
        return;
    }
    */

    char dir = -1;
    //int2 pos = (int2)(get_global_id(0), get_global_id(1));

    // Calculate left window average.
    float windowSum = 0;
    float4 clr;
    for (unsigned int y = pos.y - halfWindow; y < pos.y + halfWindow; y++) {
        for (unsigned int x = pos.x - halfWindow; x < pos.x + halfWindow; x++) {
            clr = read_imagef( in_left, sampler, (int2)(x, y) );
            windowSum += clr[0];
        }
    }

    float leftAvg = (windowSum / (windowSize*windowSize));

    unsigned char bestD = 0;        // tracks the distance with best correlation
    float maxCorrelation = 0.0f;    // tracks the best correlation (ZNCC)

    // stops at the left/right edge (OpenCL defines generic integer min and max)
    char maxD = (dir > 0)
        ? min((int)maxSearchD, (int)((w - 1 - halfWindow) - pos.x))
        : min((int)maxSearchD, (int)(pos.x - halfWindow));

    for (int d = 0; d <= maxD; d++)
    {
         // Calculate right window average.
        windowSum = 0;
        for (unsigned int y = pos.y - halfWindow; y < pos.y + halfWindow; y++) {
            for (unsigned int x = pos.x + (dir * d) - halfWindow; x < pos.x + (dir * d) + halfWindow; x++) {
                clr = read_imagef( in_right, sampler, (int2)(x, y) );
                windowSum += clr[0];
            }}
        float rightAvg = (windowSum / (windowSize*windowSize));

        /* Calculate ZNCC */

        float upperSum = 0;
        float lowerLeftSum = 0;
        float lowerRightSum = 0;

        /* Calculate ZNCC(x, y, d) */
        for (int wy = -halfWindow; wy <= halfWindow; wy++) {
            for (int wx = -halfWindow; wx <= halfWindow; wx++) {
                // difference of (left/right) image pixel from the average
                // TODO: Not necessary for each d!
                clr = read_imagef( in_left, sampler, (int2)(pos.x + wx, pos.y + wy) );
                float leftDiff = clr[0] - leftAvg;
                clr = read_imagef( in_right, sampler, (int2)(pos.x + wx + (dir * d), pos.y + wy) );
                float rightDiff = clr[0] - rightAvg;

                upperSum      += leftDiff * rightDiff;
                lowerLeftSum  += leftDiff * leftDiff;     // leftDiff ^ 2
                lowerRightSum += rightDiff * rightDiff;   // rightDiff ^ 2
            }
        }

        // Finally calculate the ZNCC value
        float correlation = (float)(upperSum / (sqrt(lowerLeftSum) * sqrt(lowerRightSum)));

        // update disparity value for pixel (x,y)
        if (correlation > maxCorrelation)
        {
            maxCorrelation = correlation;
            bestD = d;
        }

        // put the best disparity value to the disparity map
        //args->disparityMap->putPixel(x, y, bestD);
        float p = bestD / 255.0f;
        write_imagef( out, pos, (float4)(p, p, p, 1.0f) );
    }
#endif

    ////////////////////////////////////////////////////////////////////////////
}
