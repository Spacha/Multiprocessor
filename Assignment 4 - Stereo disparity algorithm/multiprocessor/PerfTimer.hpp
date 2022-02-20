#pragma once

#include <windows.h>

#include "Application.hpp"

/**
 * Used to measure performance (i.e. execution times) of different operations.
 * Requires <windows.h>
 **/
class PerfTimer
{
public:
    LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
    LARGE_INTEGER Frequency;

    PerfTimer();
    ~PerfTimer();

    void reset();
    long long int getMicroseconds();

    void printTime();
};
