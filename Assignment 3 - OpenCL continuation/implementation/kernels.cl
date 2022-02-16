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

    // NTCS grey conversion
    float gray = 0.299f*clr[0] + 0.587f*clr[1] + 0.114f*clr[2];

    write_imagef(out, coord, (float4)(gray, gray, gray, 1.0f));
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
