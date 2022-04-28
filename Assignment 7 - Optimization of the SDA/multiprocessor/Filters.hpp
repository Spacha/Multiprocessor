#pragma once

///////////////////////////////////////////////////////////////////////////////
// FILTERS
///////////////////////////////////////////////////////////////////////////////

struct Filter
{
    const size_t size;          // size of the mask (height or width)
    const float divisor;        // the mask is divided by this
    float* mask;                // the actual filter mask

    Filter(size_t size, float divisor, float* mask)
        : size(size), divisor(divisor), mask(mask) {}
};
typedef struct Filter Filter;
