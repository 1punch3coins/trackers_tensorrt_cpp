cmake_minimum_required(VERSION 3.10)
project(tracker_cpp)

set(PROJECT_NAME main)
set(CMAKE_CXX_STANDARD 14)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
set(CMAKE_BUILD_TYPE Release)

find_package(CUDA REQUIRED)
set(CUDA_LIBS ${CUDA_LIBRARIES} nvinfer nvonnxparser nvinfer_plugin cudnn)

find_package(OpenCV 4.0 REQUIRED)
set(OpenCV_LIBS ${OpenCV_LIBS} opencv_imgproc opencv_core opencv_highgui opencv_imgcodecs opencv_videoio)

set(EIGEN_INCLUDE_DIRS "/usr/include/eigen3")

aux_source_directory(./tracker TRACKER_SRC_FILES)
add_executable(${PROJECT_NAME}
                ${TRACKER_SRC_FILES}
                main.cpp det_structs.h)
target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${EIGEN_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBS} ${OpenCV_LIBS})
