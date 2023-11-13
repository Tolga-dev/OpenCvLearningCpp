cmake_minimum_required(VERSION 3.23)
project(OpencvLearningCpp)

########## OPENCV PART
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 23)

add_executable(OpencvLearningCpp main.cpp)

target_link_libraries(OpencvLearningCpp ${OpenCV_LIBS})
