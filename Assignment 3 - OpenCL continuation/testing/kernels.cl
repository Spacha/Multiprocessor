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

	float w = get_image_width(in);
	float h = get_image_height(in);

	if (center.x >= w || center.y >= h)
	{
		return;
	}

#if 0
	int weight = 0;
	// float4 clr = (float4)(0.0f, 0.0f, 0.0f, 1.0f);
	// float clr = 0.0f;

	// iterate over each element in the mask
	for (int y = center.y + d; y < center.y + d; y++)
	{
		for (int x = center.x + d; x < center.x + d; x++)
		{
			//clr += read_imagef(in, sampler, (int2)(x,y)) * mask[weight];
			//float4 a = read_imagef(in, sampler, (int2)(x,y));
			weight++;
		}
	}
#endif

	float a = 0.0f;
	int maskIdx = 0;
	for (int dx = -2; dx <= 2; dx++)
	{
		for (int dy = -2; dy <= 2; dy++)
		{
			float4 c = read_imagef(in, sampler, (int2)(center.x+dx, center.y+dy));
			a += (c[0] * mask[maskIdx]);

			maskIdx++;
		}
	}
	//float4 c = read_imagef(in, sampler, (int2)(center.x - 1, center.y));

	float4 clr = read_imagef(in, sampler, center);
	float newc = a / divisor;
	clr[0] = newc;
	clr[1] = newc;
	clr[2] = newc;

	write_imagef(out, center, clr);
	// write_imagef(out, center, (float4)(clr * factor));
}
