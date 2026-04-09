find_package(OpenMP REQUIRED)
add_library(yolov8n SHARED src/YoloV8n.cpp)
target_include_directories(yolov8n PUBLIC include)
target_link_libraries(yolov8n PUBLIC trt_wrapper OpenMP::OpenMP_CXX)
target_compile_definitions(
    yolov8n
    PRIVATE PACKAGE_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

add_executable(test_yolo test_yolo.cpp)
target_link_libraries(test_yolo PRIVATE yolov8n camera)
target_compile_definitions(
    test_yolo
    PRIVATE PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    PACKAGE_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

include(GNUInstallDirs)

install(TARGETS test_yolo
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# 安装库文件到 lib
install(TARGETS yolov8n
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

# RPATH 设置
set_target_properties(test_yolo PROPERTIES
    INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
)
