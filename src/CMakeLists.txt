add_executable(yolo_baseline 01_baseline/main.cpp)
add_executable(yolo_cuda_prepost
    02_cuda_prepost/main.cpp
    02_cuda_prepost/preprocess.cu
    02_cuda_prepost/postprocess.cu
)
add_executable(yolo_pinned_memory
    03_pinned_memory/main.cpp
    03_pinned_memory/preprocess.cu
)
add_executable(yolo_async_pipeline
    04_async_pipeline/main.cpp
    04_async_pipeline/preprocess.cu
)
add_executable(yolo_cuda_graph
    05_cuda_graph/main.cpp
    05_cuda_graph/preprocess.cu
)
add_executable(yolo_full_graph
    06_full_graph/main.cpp
    06_full_graph/preprocess.cu
    06_full_graph/postprocess.cu
)
add_executable(yolo_graph_pipeline
    07_graph_pipeline/main.cpp
    07_graph_pipeline/preprocess.cu
    07_graph_pipeline/postprocess.cu
)
add_executable(yolo_preprocess_maps
    08_preprocess_maps/main.cpp
    08_preprocess_maps/preprocess.cu
)
add_executable(yolo_registered_input
    09_registered_input/main.cpp
    09_registered_input/preprocess.cu
)
add_executable(yolo_split_streams
    10_split_streams/main.cpp
    10_split_streams/preprocess.cu
)

foreach(target yolo_baseline yolo_cuda_prepost yolo_pinned_memory
               yolo_async_pipeline yolo_cuda_graph
               yolo_full_graph yolo_graph_pipeline yolo_preprocess_maps
               yolo_registered_input yolo_split_streams)
    target_include_directories(${target} SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${target} PRIVATE
        yolo_deploy_project_options CUDA::cudart TensorRT::nvinfer ${OpenCV_LIBS})
    target_compile_features(${target} PRIVATE cxx_std_17)
endforeach()

if(BUILD_TESTING)
    add_executable(preprocess_test
        ../tests/preprocess_test.cpp
        02_cuda_prepost/preprocess.cu)
    target_include_directories(preprocess_test SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(preprocess_test PRIVATE
        yolo_deploy_project_options CUDA::cudart ${OpenCV_LIBS})
    target_compile_features(preprocess_test PRIVATE cxx_std_17)
    add_test(NAME cuda_preprocess_equivalence COMMAND preprocess_test)

    add_executable(pinned_memory_test
        ../tests/pinned_memory_test.cpp
        03_pinned_memory/preprocess.cu)
    target_include_directories(pinned_memory_test SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(pinned_memory_test PRIVATE
        yolo_deploy_project_options CUDA::cudart TensorRT::nvinfer ${OpenCV_LIBS})
    target_compile_features(pinned_memory_test PRIVATE cxx_std_17)
    add_test(NAME pinned_memory_equivalence COMMAND pinned_memory_test)

    add_executable(gpu_postprocess_test
        ../tests/gpu_postprocess_test.cpp
        02_cuda_prepost/postprocess.cu)
    target_link_libraries(gpu_postprocess_test PRIVATE yolo_deploy_project_options CUDA::cudart)
    target_compile_features(gpu_postprocess_test PRIVATE cxx_std_17)
    add_test(NAME gpu_postprocess_equivalence COMMAND gpu_postprocess_test)

    add_executable(preprocess_maps_test
        ../tests/preprocess_maps_test.cpp
        08_preprocess_maps/preprocess.cu)
    target_include_directories(preprocess_maps_test SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(preprocess_maps_test PRIVATE
        yolo_deploy_project_options CUDA::cudart ${OpenCV_LIBS})
    target_compile_features(preprocess_maps_test PRIVATE cxx_std_17)
    add_test(NAME preprocess_maps_equivalence COMMAND preprocess_maps_test)

    add_executable(registered_input_test
        ../tests/registered_input_test.cpp
        09_registered_input/preprocess.cu)
    target_include_directories(registered_input_test SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(registered_input_test PRIVATE
        yolo_deploy_project_options CUDA::cudart TensorRT::nvinfer ${OpenCV_LIBS})
    target_compile_features(registered_input_test PRIVATE cxx_std_17)
    add_test(NAME registered_input_equivalence COMMAND registered_input_test)

    add_executable(preprocess_maps_benchmark
        ../tests/preprocess_maps_benchmark.cpp
        08_preprocess_maps/preprocess.cu)
    target_link_libraries(preprocess_maps_benchmark PRIVATE yolo_deploy_project_options CUDA::cudart)
    target_compile_features(preprocess_maps_benchmark PRIVATE cxx_std_17)
endif()
