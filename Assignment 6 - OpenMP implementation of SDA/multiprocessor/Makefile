# cls && g++ main.cpp MiniOCL.cpp lodepng.cpp %OCL_ROOT%/lib/x86_64/opencl.lib -Wall -I %OCL_ROOT%\include -o image-filter.exe && image-filter.exe img/im0.png
#

# Assumes that there is an environment variable OCL_ROOT
# which points to the location of the OpenCL library

# Global configs
CXX       = g++
CFLAGS    = -Wall

# Libraries
EXT_LIB   = $(OCL_ROOT)/lib/x86_64/opencl.lib
EXT_INC   = $(OCL_ROOT)/include

SRC = main.cpp PerfTimer.cpp MiniOCL.cpp Image.cpp lodepng.cpp
OUT = stereo.exe

# NOTE: DONT'T ALWAYS INCLUDE EVERYTHING FOR FUN?
all: $(SRC)
	$(CXX) $(CFLAGS) $(SRC) $(EXT_LIB) -I $(EXT_INC) -o $(OUT) -fopenmp -O3
	@echo Done. Run using: "stereo.exe LEFT_IMAGE RIGHT_IMAGE [WINDOW_SIZE=9] [MAX_SEARCH_DIST=32] [CROSS_CHECK_THRESHOLD=8] [DOWNSCALE_FACTOR=4]"
