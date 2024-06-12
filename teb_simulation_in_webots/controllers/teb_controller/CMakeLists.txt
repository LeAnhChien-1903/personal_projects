cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 11)



# Setup the project.
# Its name is defined to be the controller directory name.
get_filename_component(PROJECT ${CMAKE_SOURCE_DIR} NAME)
project(${PROJECT})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake_modules)
## System dependencies are found with CMake's conventions
find_package(OpenCV REQUIRED )
find_package(Boost REQUIRED COMPONENTS system thread graph)
find_package(Eigen3 REQUIRED)
find_package(SUITESPARSE REQUIRED)
find_package(G2O REQUIRED)
# Set external include directories
set(EXTERNAL_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${SUITESPARSE_INCLUDE_DIRS} ${G2O_INCLUDE_DIR})
set(EXTERNAL_LIBS ${OpenCV_LIBS} ${SUITESPARSE_LIBRARIES} ${G2O_LIBRARIES} ${Boost_LIBRARIES})

# Get C or C++ sources in the current directory (only).
file(GLOB C_SOURCES *.c)
file(GLOB CPP_SOURCES *.cpp)
set(SOURCES ${C_SOURCES} ${CPP_SOURCES})

# Set the  Webots home path (change it according to your installation method)
set(WEBOTS_HOME "/usr/local/webots")
#set(WEBOTS_HOME "/snap/webots/current/usr/share/webots")

# Link with the Webots controller library.
link_directories($ENV{WEBOTS_HOME}/lib/controller)
set (LIBRARIES m ${CMAKE_SHARED_LIBRARY_PREFIX}Controller${CMAKE_SHARED_LIBRARY_SUFFIX} ${CMAKE_SHARED_LIBRARY_PREFIX}CppController${CMAKE_SHARED_LIBRARY_SUFFIX})
include_directories(
    $ENV{WEBOTS_HOME}/include/controller/c 
    $ENV{WEBOTS_HOME}/include/controller/cpp
    ${EXTERNAL_INCLUDE_DIRS}
    include
    include/common
    include/g2o_types
    include/global_planner
    include/homotopy_class_planner
    include/obstacles
    include/robot
    include/optimization
)
add_library(teb
    # Commom libraries 
    src/common/ultis.cpp
    # Global planner
    src/global_planner/cubic_spline.cpp
    src/global_planner/rrt_star.cpp
    # Homotopy classes planner 
    src/homotopy_class_planner/timed_elastic_band.cpp
    src/homotopy_class_planner/homotopy_class_planner.cpp
    src/homotopy_class_planner/graph_search.cpp
    # Obstacles
    src/obstacles/obstacles.cpp
    # Robot
    src/robot/robot.cpp
    # TEB optimization
    src/optimization/optimal_planner.cpp
)
# Setup the target executable.
add_executable(${PROJECT} 
    ${SOURCES}
    # Commom libraries 
    src/common/ultis.cpp
    # Global planner
    src/global_planner/cubic_spline.cpp
    src/global_planner/rrt_star.cpp
    # Homotopy classes planner 
    src/homotopy_class_planner/timed_elastic_band.cpp
    src/homotopy_class_planner/homotopy_class_planner.cpp
    src/homotopy_class_planner/graph_search.cpp
    # Obstacles
    src/obstacles/obstacles.cpp
    # Robot
    src/robot/robot.cpp
    # TEB optimization
    src/optimization/optimal_planner.cpp
)
target_link_libraries(${PROJECT} ${LIBRARIES} ${EXTERNAL_LIBS})

# Copy the target executable at the right location.
add_custom_command(TARGET ${PROJECT} POST_BUILD COMMAND ${CMAKE_COMMAND} -E
  copy ${CMAKE_BINARY_DIR}/${PROJECT} ${CMAKE_SOURCE_DIR}
)