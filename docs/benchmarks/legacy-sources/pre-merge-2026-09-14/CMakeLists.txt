cmake_minimum_required(VERSION 3.24)
project(yolo_pre_merge LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit 11.8 REQUIRED)
find_package(OpenCV 3.4 REQUIRED COMPONENTS core imgproc imgcodecs)
set(TensorRT_ROOT "/usr/local/TensorRT" CACHE PATH "TensorRT 8.6 installation")
find_path(TRT_INCLUDE NvInfer.h HINTS "${TensorRT_ROOT}/include" REQUIRED)
find_library(TRT_LIBRARY nvinfer HINTS "${TensorRT_ROOT}/lib" "${TensorRT_ROOT}/lib64" REQUIRED)

add_executable(yolo_baseline src/01_baseline/main.cpp)
add_executable(yolo_benchmark src/02_benchmark/main.cpp)
add_executable(yolo_cuda_preprocess
    src/03_cuda_preprocess/main.cpp src/03_cuda_preprocess/preprocess.cu)
add_executable(yolo_gpu_postprocess
    src/07_gpu_postprocess/main.cpp src/07_gpu_postprocess/preprocess.cu
    src/07_gpu_postprocess/postprocess.cu)
add_executable(yolo_pinned_memory src/04_pinned_memory/main.cpp src/04_pinned_memory/preprocess.cu)
add_executable(yolo_async_pipeline src/05_async_pipeline/main.cpp src/05_async_pipeline/preprocess.cu)
add_executable(yolo_cuda_graph src/06_cuda_graph/main.cpp src/06_cuda_graph/preprocess.cu)
add_executable(yolo_full_graph
    src/08_full_graph/main.cpp src/08_full_graph/preprocess.cu src/08_full_graph/postprocess.cu)
add_executable(yolo_graph_pipeline
    src/09_graph_pipeline/main.cpp src/09_graph_pipeline/preprocess.cu src/09_graph_pipeline/postprocess.cu)
add_executable(yolo_preprocess_maps src/10_preprocess_maps/main.cpp src/10_preprocess_maps/preprocess.cu)
add_executable(yolo_registered_input src/11_registered_input/main.cpp src/11_registered_input/preprocess.cu)
add_executable(yolo_split_streams src/12_split_streams/main.cpp src/12_split_streams/preprocess.cu)
foreach(target yolo_baseline yolo_benchmark yolo_cuda_preprocess yolo_gpu_postprocess
               yolo_pinned_memory yolo_async_pipeline yolo_cuda_graph yolo_full_graph
               yolo_graph_pipeline yolo_preprocess_maps yolo_registered_input yolo_split_streams)
    target_include_directories(${target} SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS} "${TRT_INCLUDE}")
    target_link_libraries(${target} PRIVATE CUDA::cudart "${TRT_LIBRARY}" ${OpenCV_LIBS})
endforeach()
