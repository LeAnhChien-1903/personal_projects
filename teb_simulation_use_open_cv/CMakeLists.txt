cmake_minimum_required(VERSION 2.8)
get_filename_component(PROJECT ${CMAKE_SOURCE_DIR} NAME)
project(${PROJECT})
set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake_modules)

## System dependencies are found with CMake's conventions
find_package(OpenCV REQUIRED )
find_package(Boost REQUIRED COMPONENTS system thread graph)
find_package(Eigen3 REQUIRED)
find_package(SUITESPARSE REQUIRED)
find_package(G2O REQUIRED)
# Set external include directories
set(EXTERNAL_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${SUITESPARSE_INCLUDE_DIRS} ${G2O_INCLUDE_DIR})
set(EXTERNAL_LIBS ${OpenCV_LIBS} ${SUITESPARSE_LIBRARIES} ${G2O_LIBRARIES} ${Boost_LIBRARIES})

# Set include directories
include_directories( 
    ${EXTERNAL_INCLUDE_DIRS}
    include
    map
    image
    video
    include/common
    include/g2o_types
    include/global_planner
    include/homotopy_class_planner
    include/obstacles
    include/robot
    include/optimization
    include/test
)
# Set libraries
add_library(${PROJECT}
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
    src/obstacles/obstacles_ver2.cpp
    # Robot
    src/robot/robot.cpp
    # TEB optimization
    src/optimization/optimal_planner.cpp
    # Testing
)

add_executable(teb src/main.cpp 
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
    src/obstacles/obstacles_ver2.cpp
    # Robot
    src/robot/robot.cpp
    # TEB optimization
    src/optimization/optimal_planner.cpp
)
target_link_libraries(teb ${EXTERNAL_LIBS})
