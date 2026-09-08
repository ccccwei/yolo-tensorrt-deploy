add_executable(yolo_deploy main.cpp)
target_include_directories(yolo_deploy SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(
    yolo_deploy
    PRIVATE
        yolo_deploy_project_options
        CUDA::cudart
        TensorRT::nvinfer
        ${OpenCV_LIBS}
)
target_compile_features(yolo_deploy PUBLIC cxx_std_17)
