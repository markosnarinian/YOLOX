cmake_minimum_required(VERSION 3.18)
project(yolox_megengine LANGUAGES CXX)

set(MGE_INSTALL_PATH "$ENV{MGE_INSTALL_PATH}" CACHE PATH "MegEngine SDK root")
find_path(MEGENGINE_INCLUDE_DIR megbrain/serialization/serializer.h
          HINTS ${MGE_INSTALL_PATH} PATH_SUFFIXES include REQUIRED)
find_library(MEGENGINE_LIBRARY megengine
             HINTS ${MGE_INSTALL_PATH} PATH_SUFFIXES lib lib64 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
add_executable(yolox yolox.cpp)
target_compile_features(yolox PRIVATE cxx_std_14)
target_include_directories(yolox PRIVATE ${MEGENGINE_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolox PRIVATE ${MEGENGINE_LIBRARY} ${OpenCV_LIBS})
if(ANDROID)
    target_link_libraries(yolox PRIVATE log z)
endif()
