add_executable(test_cifar test_cifar100.cpp)
target_link_libraries(test_cifar PRIVATE trt_wrapper)
target_include_directories(test_cifar PRIVATE include)
target_compile_definitions(
    test_cifar
    PRIVATE PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
install(TARGETS test_cifar)
