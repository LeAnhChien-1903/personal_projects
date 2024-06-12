cmake_minimum_required(VERSION 3.0.2)
project(mir100_gazebo)

find_package(catkin REQUIRED COMPONENTS
    roslaunch
    ira_laser_tools
)

###################################
## catkin specific configuration ##
###################################
catkin_package()

#############
## Install ##
#############

# Mark other files for installation (e.g. launch and bag files, etc.)
install(DIRECTORY
    config
    launch
    maps
    sdf
    rviz
    worlds
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

#############
## Testing ##
#############

roslaunch_add_file_check(launch)