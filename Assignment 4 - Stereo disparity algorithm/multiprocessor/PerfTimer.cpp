#include "PerfTimer.hpp"

///////////////////////////////////////////////////////////////////////////////
// PerfTimer
///////////////////////////////////////////////////////////////////////////////

PerfTimer::PerfTimer()
{
    // ...
}

PerfTimer::~PerfTimer()
{
    // ...
}

/**
 * Start/reset performance counter.
 **/
void PerfTimer::reset()
{
    QueryPerformanceFrequency(&Frequency);
    QueryPerformanceCounter(&StartingTime);
}

/**
 * Get a snapshot of the delta time in microseconds. 
 **/
long long int PerfTimer::getMicroseconds()
{
    QueryPerformanceCounter(&EndingTime);
    ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
    ElapsedMicroseconds.QuadPart *= 1000000;
    ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;

    return ElapsedMicroseconds.QuadPart;
}

/**
 * A helper for printing the execution time since start.
 **/
void PerfTimer::printTime()
{
    double us = this->getMicroseconds();

    std::string unitsStr;
    double divisor;

    if (us < 1000) {            // microseconds
        unitsStr = "us";
        divisor = 1;
    } else if (us < 1000000) {  // milliseconds
        unitsStr = "ms";
        divisor = 1000;
    } else {                    // seconds
        unitsStr = "s";
        divisor = 1000000;
    }

    printf("\t=> time: %0.3f %s\n", us / divisor, unitsStr.c_str());
}
