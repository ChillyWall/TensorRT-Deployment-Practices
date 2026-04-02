add_library(cifar100cnn SHARED src/Cifar100CNN.cpp)
target_include_directories(cifar100cnn PUBLIC include)
target_link_libraries(cifar100cnn PUBLIC trt_wrapper)

add_executable(test_cifar test_cifar100.cpp)
target_link_libraries(test_cifar PUBLIC cifar100cnn)
target_compile_definitions(
    test_cifar
    PRIVATE PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
install(TARGETS test_cifar)
