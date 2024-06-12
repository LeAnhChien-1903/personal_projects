cmake_minimum_required(VERSION 3.0.2)
project(obstacle_detection)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)
find_package(OpenCV REQUIRED )


find_package(catkin REQUIRED COMPONENTS
  roscpp
  message_generation
  geometry_msgs
  dynamic_reconfigure
)
#add dynamic reconfigure api
#find_package(catkin REQUIRED dynamic_reconfigure)
generate_dynamic_reconfigure_options(
  cfg/obstacle_detection.cfg
)
find_package(Eigen3 REQUIRED)

add_message_files(
  FILES
  ObstacleVertices.msg
)
generate_messages(
  DEPENDENCIES
  geometry_msgs
)

catkin_package(
  INCLUDE_DIRS include/${PROJECT_NAME}
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp message_runtime dynamic_reconfigure
)
###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  include/${PROJECT_NAME}
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  launch
  config
  cfg
)
add_library(${PROJECT_NAME}
  src/lib.cpp
  src/obstacle_detection.cpp
)

# Dynamic reconfigure: make sure configure headers are built before any node using them
add_dependencies(obstacle_detection ${PROJECT_NAME}_gencfg)
# Generate messages before compiling the lib
add_dependencies(obstacle_detection ${PROJECT_NAME}_generate_messages_cpp)

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

add_executable(obstacle_detector src/main.cpp
  src/lib.cpp
  src/obstacle_detection.cpp
)

target_link_libraries(obstacle_detector ${catkin_LIBRARIES} ${OpenCV_LIBS})