#pragma once

///////////////////////////////////////////////////////////////////////////////
// FILTERS
///////////////////////////////////////////////////////////////////////////////

const size_t maskSize = 5;

// Mean filter (5x5)
const float meanFilterMask[maskSize*maskSize] = {
     1.0f, 1.0f, 1.0f, 1.0f,  1.0f,
     1.0f, 1.0f, 1.0f, 1.0f,  1.0f,
     1.0f, 1.0f, 1.0f, 1.0f,  1.0f,
     1.0f, 1.0f, 1.0f, 1.0f,  1.0f,
     1.0f, 1.0f, 1.0f, 1.0f,  1.0f
};
const Filter g_meanFilter(maskSize, 25.0f, meanFilterMask);

// Gaussian filter (5x5)
const float gaussianFilterMask[maskSize*maskSize] = {
     1.0f,  4.0f,  7.0f,  4.0f,  1.0f,
     4.0f, 16.0f, 26.0f, 16.0f,  4.0f,
     7.0f, 26.0f, 41.0f, 26.0f,  7.0f,
     4.0f, 16.0f, 26.0f, 16.0f,  4.0f,
     1.0f,  4.0f,  7.0f,  4.0f,  1.0f
};
const Filter g_gaussianFilter(maskSize, 273.0f, gaussianFilterMask);

// Emboss filter (5x5)
const float embossFilterMask[maskSize*maskSize] = {
     -1.0f,  0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, -1.0f, 0.0f, 0.0f, 0.0f,
      0.0f,  0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,  0.0f, 0.0f, 1.0f, 0.0f,
      0.0f,  0.0f, 0.0f, 0.0f, 1.0f
};
const Filter g_embossFilter(maskSize, 1.0f, embossFilterMask);

/* A global variable containing the filter options. */
const Filter g_filters[] = { g_meanFilter, g_gaussianFilter, g_embossFilter };
