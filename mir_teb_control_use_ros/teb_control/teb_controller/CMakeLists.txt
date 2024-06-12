cmake_minimum_required(VERSION 3.0.2)
project(teb_controller)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  obstacle_detection
  tf
)
## System dependencies are found with CMake's conventions
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake_modules)

## System dependencies are found with CMake's conventions
find_package(OpenCV REQUIRED )
find_package(Boost REQUIRED COMPONENTS system thread graph)
find_package(Eigen3 REQUIRED)
find_package(SUITESPARSE REQUIRED)
find_package(G2O REQUIRED)
find_package(Glog REQUIRED)

# Set external include directories
set(EXTERNAL_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${SUITESPARSE_INCLUDE_DIRS} ${G2O_INCLUDE_DIR})
set(EXTERNAL_LIBS ${OpenCV_LIBS} ${SUITESPARSE_LIBRARIES} ${G2O_LIBRARIES} ${Boost_LIBRARIES})

file(GLOB CPP_SOURCES ${PROJECT_SOURCE_DIR}/src/**/*.cpp)

catkin_package(
  INCLUDE_DIRS include/${PROJECT_NAME}
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp obstacle_detection tf
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${EXTERNAL_INCLUDE_DIRS}
  include
  include/${PROJECT_NAME}/robot
  launch
  config
  include/${PROJECT_NAME}
)
add_library(${PROJECT_NAME}
  ${CPP_SOURCES}
)

target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})
install(
  TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)
install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

add_executable(teb_control src/main.cpp
  ${CPP_SOURCES}
)

target_link_libraries(teb_control ${EXTERNAL_LIBS} ${catkin_LIBRARIES})
