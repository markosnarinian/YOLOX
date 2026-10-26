cmake_minimum_required(VERSION 3.22.1)
project(yoloXncnn LANGUAGES CXX)

set(ncnn_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.deps/ncnn-20250916-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn")
if(NOT EXISTS "${ncnn_DIR}/ncnnConfig.cmake")
    message(FATAL_ERROR "Missing pinned ncnn SDK for ${ANDROID_ABI}. See demo/ncnn/android/README.md dependency setup.")
endif()
find_package(ncnn 20250916 EXACT REQUIRED CONFIG)

add_library(yoloXncnn SHARED yoloXncnn_jni.cpp)
target_compile_features(yoloXncnn PRIVATE cxx_std_11)

target_link_libraries(yoloXncnn
    ncnn

    jnigraphics
)
