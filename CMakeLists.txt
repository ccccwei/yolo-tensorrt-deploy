cmake_minimum_required(VERSION 3.24)

project(
    yolo_deploy
    VERSION 0.0.0
    DESCRIPTION "Low-latency YOLO runtime for NVIDIA GPUs"
    LANGUAGES CXX CUDA
)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(YOLO_DEPLOY_WARNINGS_AS_ERRORS "Treat project warnings as errors" OFF)

add_library(yolo_deploy_project_options INTERFACE)
if(MSVC)
    target_compile_options(
        yolo_deploy_project_options
        INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/W4;/permissive->
    )
    if(YOLO_DEPLOY_WARNINGS_AS_ERRORS)
        target_compile_options(
            yolo_deploy_project_options
            INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/WX>
        )
    endif()
else()
    target_compile_options(
        yolo_deploy_project_options
        INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wsign-conversion>
    )
    if(YOLO_DEPLOY_WARNINGS_AS_ERRORS)
        target_compile_options(
            yolo_deploy_project_options
            INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Werror>
        )
    endif()
endif()

find_package(CUDAToolkit 11.8 REQUIRED)
find_package(OpenCV 3.4 REQUIRED COMPONENTS core imgproc imgcodecs)

set(TensorRT_ROOT "" CACHE PATH "TensorRT installation prefix")
set(_TensorRT_INCLUDE_SUFFIXES
    include
    targets/${CMAKE_LIBRARY_ARCHITECTURE}/include
)
set(_TensorRT_LIBRARY_SUFFIXES
    lib
    lib64
    lib/${CMAKE_LIBRARY_ARCHITECTURE}
    targets/${CMAKE_LIBRARY_ARCHITECTURE}/lib
)

if(TensorRT_ROOT)
    file(REAL_PATH "${TensorRT_ROOT}" _TensorRT_SEARCH_ROOT EXPAND_TILDE)
    find_path(
        TensorRT_INCLUDE_DIR
        NAMES NvInfer.h NvInferVersion.h
        PATHS "${_TensorRT_SEARCH_ROOT}"
        PATH_SUFFIXES ${_TensorRT_INCLUDE_SUFFIXES}
        NO_DEFAULT_PATH
        NO_CACHE
    )
    find_library(
        TensorRT_NVINFER_LIBRARY
        NAMES nvinfer
        PATHS "${_TensorRT_SEARCH_ROOT}"
        PATH_SUFFIXES ${_TensorRT_LIBRARY_SUFFIXES}
        NO_DEFAULT_PATH
        NO_CACHE
    )
else()
    find_path(
        TensorRT_INCLUDE_DIR
        NAMES NvInfer.h NvInferVersion.h
        HINTS /usr/local/TensorRT /opt/TensorRT
        PATH_SUFFIXES ${_TensorRT_INCLUDE_SUFFIXES}
        NO_CACHE
    )
    find_library(
        TensorRT_NVINFER_LIBRARY
        NAMES nvinfer
        HINTS /usr/local/TensorRT /opt/TensorRT
        PATH_SUFFIXES ${_TensorRT_LIBRARY_SUFFIXES}
        NO_CACHE
    )
endif()

if(NOT TensorRT_INCLUDE_DIR OR NOT TensorRT_NVINFER_LIBRARY)
    message(FATAL_ERROR "TensorRT 8.6 was not found; set TensorRT_ROOT to its installation prefix")
endif()

file(
    STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h"
    _TensorRT_VERSION_LINES
    REGEX "^#define[ \t]+NV_TENSORRT_(MAJOR|MINOR|PATCH|BUILD)[ \t]+[0-9]+"
)
foreach(_part MAJOR MINOR PATCH BUILD)
    foreach(_line IN LISTS _TensorRT_VERSION_LINES)
        if(_line MATCHES "NV_TENSORRT_${_part}[ \t]+([0-9]+)")
            set(TensorRT_VERSION_${_part} "${CMAKE_MATCH_1}")
        endif()
    endforeach()
endforeach()

if(NOT DEFINED TensorRT_VERSION_MAJOR
   OR NOT DEFINED TensorRT_VERSION_MINOR
   OR NOT DEFINED TensorRT_VERSION_PATCH)
    message(FATAL_ERROR "Could not read the TensorRT version from ${TensorRT_INCLUDE_DIR}")
endif()
if(NOT TensorRT_VERSION_MAJOR EQUAL 8 OR TensorRT_VERSION_MINOR LESS 6)
    message(
        FATAL_ERROR
        "This source tree currently validates TensorRT 8.6.x only; found "
        "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}"
    )
endif()

set(
    TensorRT_VERSION
    "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}"
)
add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
set_target_properties(
    TensorRT::nvinfer
    PROPERTIES
        IMPORTED_LOCATION "${TensorRT_NVINFER_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
)

message(STATUS "yolo_deploy ${PROJECT_VERSION}")
message(STATUS "  CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "  TensorRT:     ${TensorRT_VERSION}")

add_subdirectory(src)
