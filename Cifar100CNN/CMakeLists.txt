find_package(OpenMP REQUIRED)
add_library(cifar100cnn SHARED src/Cifar100CNN.cpp)
target_include_directories(cifar100cnn PUBLIC include)
target_link_libraries(cifar100cnn PUBLIC trt_wrapper OpenMP::OpenMP_CXX)

add_executable(test_cifar test_cifar100.cpp)
target_link_libraries(test_cifar PUBLIC cifar100cnn)
target_compile_definitions(
    test_cifar
    PRIVATE PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

include(GNUInstallDirs)

install(TARGETS test_cifar
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# 安装库文件到 lib
install(TARGETS cifar100cnn
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

# RPATH 设置：
set_target_properties(test_cifar PROPERTIES
    INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
)
