cmake_minimum_required(VERSION 3.0.2)
project(collision_avoidance)

find_package(catkin REQUIRED COMPONENTS
  dynamic_reconfigure
  roscpp
  rospy
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  include/${PROJECT_NAME}
)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS 
  src/train_policy.py
  src/circle_test_5.py
  src/circle_test_10.py
  src/circle_test_15.py
  src/circle_test_20.py
  src/stage_2_test.py
  src/multi_env_test.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)