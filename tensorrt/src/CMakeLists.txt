cmake_minimum_required(VERSION 3.10)

project(detr)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

set(INSTALL_INCLUDE_DIR /usr/local/include)
set(INSTALL_LIB_DIR /usr/local/lib)

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

include_directories(/usr/local/include)
link_directories(/usr/local/lib)

include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

include_directories(${PROJECT_SOURCE_DIR}/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O2 -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(plugins SHARED 
            detr.cpp 
            calibrator.cpp
            engine.cpp
            infer.cpp
            postprocess.cpp
            write_objects.cpp
            )

find_package(OpenCV)
target_include_directories(plugins PRIVATE ${OpenCV_INCLUDE_DIRECTORIES})

target_link_libraries(plugins nvinfer nvonnxparser cudart ${OpenCV_LIBS})   

find_package    (PkgConfig)
pkg_check_modules   (GLib glib-2.0)
pkg_check_modules   (GObject gobject-2.0)

include_directories(${GLib_INCLUDE_DIRS} ${GObject_INCLUDE_DIRS})
link_directories(${GLib_LIBRARY_DIRS} ${GObject_LIBRARY_DIRS})
add_definitions(${GLib_CFLAGS_OTHER} ${GObject_CFLAGS_OTHER})

add_executable(onnx_trt_convert onnx_trt_convert.cpp)
target_include_directories(onnx_trt_convert PRIVATE 
    ${OpenCV_INCLUDE_DIRECTORIES}
    )
target_link_libraries(onnx_trt_convert PRIVATE 
    nvinfer
    nvonnxparser
    cudart
    plugins
    ${OpenCV_LIBS}
    ${GLib_LIBRARIES} 
    ${GObject_LIBRARIES}
    )

add_executable(detect detect.cpp)
target_include_directories(detect PRIVATE 
    ${OpenCV_INCLUDE_DIRECTORIES}
    )
target_link_libraries(detect PRIVATE 
    nvinfer
    nvonnxparser
    cudart
    plugins
    ${OpenCV_LIBS}
    ${GLib_LIBRARIES} 
    ${GObject_LIBRARIES}
    )