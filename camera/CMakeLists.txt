find_package(OpenCV REQUIRED)
add_library(camera STATIC
    src/camera.cpp
)
target_include_directories(camera PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(camera PUBLIC ${OpenCV_LIBS})
target_compile_definitions(camera PRIVATE PACKAGE_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
