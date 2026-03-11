find_package(CUDAToolkit REQUIRED)
find_package(fmt REQUIRED)
find_package(OpenCV REQUIRED)

add_library(trt_wrapper STATIC src/ModelBuilder.cpp)
target_include_directories(trt_wrapper PUBLIC include)
target_link_libraries(
    trt_wrapper
    PUBLIC nvinfer nvonnxparser CUDA::cudart fmt::fmt ${OpenCV_LIBS}
)
