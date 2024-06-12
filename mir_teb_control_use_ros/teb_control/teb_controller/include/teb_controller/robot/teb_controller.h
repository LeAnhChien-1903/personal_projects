#pragma once
#ifndef TEB_CONTROLLER_H
#define TEB_CONTROLLER_H

#include <ros/ros.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/LaserScan.h>
#include <obstacle_detection/ObstacleVertices.h>
#include <std_msgs/Bool.h>
#include <geometry_msgs/Pose2D.h>
#include <nav_msgs/OccupancyGrid.h>
#include "homotopy_class_planner/graph_search.h"
#include "homotopy_class_planner/homotopy_class_planner.h"
#include "common/ultis.h"
#include <string>
#include "common/cubic_spline.h" 

using namespace teb_local_planner;

class TEBControllerWithMap
{
private:
    // Constant of environments
    double sample_time; // Time step in seconds
    // Constant of robot constraints
    double wheel_radius; // wheel radius
    double wheel_base; // distance between two wheels
    double robot_length; // length of robot
    double robot_width; // width of robot
    double robot_radius; // radius of robot
    double max_left_velocity; // The maximum left wheel velocity (m/s)
    double max_right_velocity; // The maximum right wheel velocity (m/s)
    double min_left_velocity; // The minimum left wheel velocity (m/s)
    double min_right_velocity; // The minimum right wheel velocity (m/s)
    double max_acceleration; // The maximum acceleration (m/s^2)
    double goal_tolerance; // the goal tolerance (m)
    // ROS topic
    std::string obstacle_topic;
    std::string odometry_topic;
    std::string amcl_topic;
    std::string start_topic;
    std::string goal_topic;
    std::string vel_topic;
    std::string global_planner_topic;
    std::string local_costmap_topic; 
    std::string global_costmap_topic;
    std::string local_path_topic;
    // Robot
    PoseSE2 robot_pose; // Store the robot pose [x, y, theta]
    PoseSE2 robot_goal; // Store the robot goal [x, y, theta]
    WheelVelocity cmd_vel; // The wheel velocities of robot [v_l, v_r]
    std::vector<PoseSE2> robot_visited; // The position that the robot have visited
    std::vector<PoseSE2> robot_goal_list; // The position that the robot have to visit

    // Obstacles
    ObstContainer *obstacles;

    // Planner
    TebConfig config; // Store the configuration
    PlannerInterfacePtr planner; // Store the planner interface
    ViaPointContainer* via_points; // Store the via point container for planner
    std::vector<PoseSE2> best_path; // Store the best trajectory 
    std::vector<double> best_timediff; // Store the time step of best trajectory
    std::vector<geometry_msgs::PoseStamped> transformed_plan; // Store local ref path
    int sub_goal_index; // Index of the sub goal in the global trajectory
    // Global costmap
    std::vector<uint8_t> global_costmap; // Store the global map for check feasible path
    std::vector<double> original_position;; // Origin of global costmap   
    double resolution; // resolution of the global costmap (m/cell)
    uint32_t global_costmap_width; // Width of the global costmap
    // Local costmap parameters
    int local_costmap_height; // Height of the costmap
    int local_costmap_width; // Width of the costmap
    // double local_costmap_resolution; // Resolution of the costmap
    // Global path
    std::vector<geometry_msgs::PoseStamped> global_path; // Store the global path
    // ROS node
    ros::NodeHandle node;
    // ROS publishers and subscribers
    ros::Subscriber odometry_sub, start_sub, global_path_sub, obstacle_sub, amcl_sub, goal_sub, global_costmap_sub;
    ros::Publisher velocity_pub, visualization_pub, local_path_pub, best_path_visual_pub, robot_local_pose_pub;
    // Timer
    ros::Timer timer_;
    // Start command
    bool startCommand = false;
    // Local path
    nav_msgs::Path local_path;
    nav_msgs::Path best_path_visual;
    // Init goal and robot pose
    Eigen::Vector3d init_robot_pose;
    // Transform init robot pose
    tf::TransformBroadcaster br;
    tf::Transform transform;
    // Transform matrix
    Eigen::Matrix2d global2local;
public:
    /**
     * @brief Default Construct a new TEBControl object
     */
    TEBControllerWithMap();
    /**
     * @brief Destroy the TEBControl object
     */
    ~TEBControllerWithMap(){};
    /**
     * @brief Timer callback function 
     * 
     */
    void timerCallback(const ros::TimerEvent & msg);
    /**
    * @brief Prune global plan such that already passed poses are cut off
    * 
    * The pose of the robot is transformed into the frame of the global plan by taking the most recent tf transform.
    * If no valid transformation can be found, the method returns \c false.
    * The global plan is pruned until the distance to the robot is at least \c dist_behind_robot.
    * If no pose within the specified threshold \c dist_behind_robot can be found,
    * nothing will be pruned and the method returns \c false.
    * @remarks Do not choose \c dist_behind_robot too small (not smaller the cell size of the map), otherwise nothing will be pruned.
    * @param dist_behind_robot Distance behind the robot that should be kept [meters]
    * @return \c true if the plan is pruned, \c false in case of a transform exception or if no pose cannot be found inside the threshold
    */
    bool pruneGlobalPlan(double dist_behind_robot = 1);
    /**
    * @brief  Transforms the global plan of the robot from the planner frame to the local frame (modified).
    * 
    * The method replaces transformGlobalPlan as defined in base_local_planner/goal_functions.h 
    * such that the index of the current goal pose is returned as well as 
    * the transformation between the global plan and the planning frame.
    * @param max_plan_length Specify maximum length (cumulative Euclidean distances) of the transformed plan [if <=0: disabled; the length is also bounded by the local costmap size!
    * @return \c true if the global plan is transformed, \c false otherwise
    */
    bool transformGlobalPlan(double max_plan_length);
    /**
    * @brief Estimate the orientation of a pose from the global_plan that is treated as a local goal for the local planner.
    * 
    * If the current (local) goal point is not the final one (global)
    * substitute the goal orientation by the angle of the direction vector between 
    * the local goal and the subsequent pose of the global plan. 
    * This is often helpful, if the global planner does not consider orientations. \n
    * A moving average filter is utilized to smooth the orientation.
    * @param moving_average_length number of future poses of the global plan to be taken into account
    * @return orientation (yaw-angle) estimate
    */
    double estimateLocalGoalOrientation(int moving_average_length=3);
    /**
     * @brief Visualize the result
     * 
     */
    void visualization();
    /**
     * @brief Visualize the best path
     * 
     * @return visualization_msgs::Marker 
     */
    visualization_msgs::MarkerArray bestPathVisualization();
    /**
     * @brief Visualize the visited posed and goal pose list
     * 
     * @return visualization_msgs::MarkerArray 
     */
    visualization_msgs::MarkerArray visitedPoseVisualization();
    /**
     * @brief Initialize the publishers and subscribers
     * 
     */
    void initSubAndPub();
    /**
     * @brief Get odometry data (linear_vel, angular_vel)
     * 
     * @param msg 
     */
    void odometryCallback(const nav_msgs::Odometry msg);
    /**
     * @brief Get pose data (x, y, theta)
     * 
     * @param msg 
     */
    void acmlCallback(const geometry_msgs::PoseWithCovarianceStamped msg);
    /**
     * @brief Get start command 
     * 
     */
    void startCallback(const std_msgs::Bool msg);
    /**
     * @brief Get goal pose
     * 
     */
    void goalCallback(const geometry_msgs::PoseStamped msg);
    /**
     * @brief Get the global path
     * 
     * @param msg 
     */
    void globalPathCallback(const nav_msgs::Path msg);
    /**
     * @brief Extract obstacles from lidar
     * 
     * @param msg 
     */
    void obstaclesCallback(const obstacle_detection::ObstacleVertices msg);
    /**
     * @brief Get global cost map
     * 
     * @param msg 
     */
    void globalCostMapCallback(const nav_msgs::OccupancyGrid msg);
    /**
     * @brief Check the new path is feasible
     * 
     * @return true if the new path is feasible
     * @return false else the new path is not feasible
     */
    bool isFeasiblePath();
    /**
     * @brief Check the goal reached 
     * @return true if goal reached
     */
    bool goalReached();
    /**
     * @brief Convert left and right velocity to linear and angular velocity
     * 
     * @param wheel_vel input wheel velocities
     * @return Twist 
     */
    Twist convertWheelToVelocity(WheelVelocity wheel_vel);
    /**
     * @brief Convert linear and angular velocity to left and right velocity
     * 
     * @param linear linear velocity
     * @param angular angular velocity
     * @return WheelVelocity 
     */
    WheelVelocity convertVelocityToWheelVelocity(double linear, double angular);
    /**
     * @brief Find minimum difference between two angles
     * 
     * @param angle1 
     * @param angle2 
     * @return double 
     */
    double findDifferenceOrientation(double angle1, double angle2);
    /**
     * @brief Convert meter to pixel position
     * 
     * @param x 
     * @param cellsize 
     * @return int 
     */
    int contxy2disc(double x, double cellsize);
};

#endif