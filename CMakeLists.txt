cmake_minimum_required(VERSION 3.16)
project(dep_trt LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)

add_subdirectory(trt_wrapper)
add_subdirectory(camera)
add_subdirectory(Cifar100CNN)
add_subdirectory(YoloV8n)
