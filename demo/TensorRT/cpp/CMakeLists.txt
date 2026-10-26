cmake_minimum_required(VERSION 3.18)
project(yolox LANGUAGES CXX)

# This sample uses the TensorRT 7/8 binding API, not TensorRT 10's tensor API.
set(TENSORRT_ROOT "" CACHE PATH "TensorRT 7/8 SDK root")
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS ${TENSORRT_ROOT} PATH_SUFFIXES include REQUIRED)
find_library(TENSORRT_LIBRARY nvinfer HINTS ${TENSORRT_ROOT} PATH_SUFFIXES lib lib64 REQUIRED)

add_executable(yolox yolox.cpp)
target_compile_features(yolox PRIVATE cxx_std_14)
target_include_directories(yolox PRIVATE include ${TENSORRT_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolox PRIVATE ${TENSORRT_LIBRARY} CUDA::cudart ${OpenCV_LIBS})
if(MSVC)
    target_compile_options(yolox PRIVATE /O2 /W3)
else()
    target_compile_options(yolox PRIVATE -O2 -Wall)
endif()
