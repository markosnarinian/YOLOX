cmake_minimum_required(VERSION 3.16)
project(yolox_openvino_demo LANGUAGES CXX)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(OpenVINO 2025.4.1 REQUIRED COMPONENTS Runtime)

add_executable(yolox_openvino yolox_openvino.cpp)
target_compile_features(yolox_openvino PRIVATE cxx_std_17)
set_target_properties(yolox_openvino PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(yolox_openvino PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(yolox_openvino PRIVATE openvino::runtime ${OpenCV_LIBS})
