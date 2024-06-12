#pragma once
#ifndef ROBOT_H
#define ROBOT_H

// Webots libraries
#include <webots/Robot.hpp>
#include <webots/Motor.hpp>
#include <webots/Lidar.hpp>
#include <webots/Display.hpp>
#include <webots/DistanceSensor.hpp>
#include <webots/PositionSensor.hpp>
#include <webots/InertialUnit.hpp>
#include <webots/GPS.hpp>
// Local libraries
#include "graph_search.h"
#include "homotopy_class_planner.h"
#include "rrt_star.h"
#include "cubic_spline.h"
using namespace teb_local_planner;
class TEBControl : public webots::Robot 
{
private:
    // Constant of environments
	int timeStep; // Time step in milliseconds 
    double dt; // Time step in seconds
    // Constant of robot constraints
    double wheel_radius; // wheel radius
    double wheel_base; // distance between two wheels
    double robot_length; // length of robot
    double robot_width; // width of robot
    double robot_radius; // radius of robot
    double observable_range; // the observable range of robot
    double max_left_velocity; // The maximum left wheel velocity (m/s)
    double max_right_velocity; // The maximum right wheel velocity (m/s)
    double min_left_velocity; // The minimum left wheel velocity (m/s)
    double min_right_velocity; // The minimum right wheel velocity (m/s)
    double max_acceleration; // The maximum acceleration (m/s^2)
    
    // Constant of lidar 
    Eigen::Vector2d extra_position; // position of the back lidar equivalent center of robot
    double min_range; // the minimum range of the lidar
    double max_range; // the maximum range of the lidar
    double min_angle; // the minimum angle of the lidar
    double max_angle; // the maximum angle of the lidar 
    double resolution; // the resolution of the lidar
    double angular_resolution; // the angular resolution of the lidar

    // Previous position for calculate the velocity of the wheel
    double prevLeftPosition; // Previous position of left wheel
    double prevRightPosition; // Previous position of right wheel
    
    // Robot
    PoseSE2 robot_pose; // Store the robot pose [x, y, theta]
    PoseSE2 robot_goal; // Store the robot goal [x, y, theta]
    WheelVelocity cmd_vel; // The wheel velocities of robot [v_l, v_r]
    std::vector<cv::Point> robot_visited; // The position that the robot have visited
    
    // Constants for visualizations
    double gain_x; // gain to add in x-coordinate
    double gain_y; // gain to add in y-coordinate
    double map_height_pixel; // Height of the map
    double map_width_pixel; // Width of the map
    cv::Mat map_empty; // Empty map for visualization
	
    // Obstacles
    std::vector<std::vector<Eigen::Vector2d>> rectangles; // position of obstacles
    ObstContainer *obstacles;

    // Planner
    TebConfig config; // Store the configuration
    PlannerInterfacePtr planner; // Store the planner interface
    ViaPointContainer via_points; // Store the via point container for planner
    std::vector<PoseSE2> best_path; // Store the best trajectory 
    
    // Data sensor
    LaserScanData laser_scan; // The laser scan data
    LaserPointCloudCluster point_cloud_clusters; // The point cloud of the laser scan data
    Eigen::Rotation2D<double> front_transform;// Rotation matrix to transform front point cloud
    Eigen::Rotation2D<double> back_transform; // Rotation matrix to transform back point cloud
    Eigen::Rotation2D<double> robot_rotation; // Rotation matrix to transform between robot and point cloud
    Eigen::Matrix2d all_transform; // Matrix to transform point cloud
	webots::Motor* left_motor, * right_motor; // Left and right motor
	webots::PositionSensor* left_sensor, * right_sensor; // Left and right position sensor
	webots::InertialUnit* iu; // Inertial unit sensor for get orientation of robot
	webots::Lidar* front_lidar, * back_lidar, *lidar; // Lidar sensors for get scanner data
	webots::DistanceSensor* ds0, * ds1, * ds2, * ds3; // Distance sensor 
	webots::GPS* gps; // GPS sensor for get position of robot
	webots::Display* display; // Display for visualization

public:
    /**
     * @brief Default Construct a new TEBControl object
     */
    TEBControl();
    /**
     * @brief Destroy the TEBControl object
     */
    ~TEBControl(){};
    /**
     * @brief Run robot
     */
    void run();
    /**
     * @brief Visualize path tracking, obstacles, robot, create teb path
     * @param map map for visualization
     */
    void visualize(cv::Mat &map);
    /**
     * @brief Get laser scan data from lidar
     * 
     */
    void getLaserScanData();
    /**
     * @brief Extract wheel velocity from best trajectory and reachable velocity
     * @return WheelVelocity 
     */
    WheelVelocity extractWheelVelocity();
    /**
     * @brief Calculate the cost with best trajectory
     * 
     * @param current_pose current pose of robot
     * @param current_vel wheel velocity of robot
     * @return double cost with best trajectory
     */
    double calculateCostWithReference(const PoseSE2 current_pose, const WheelVelocity current_vel);
    /**
     * @brief Calculate the dynamic window from current velocity
     * 
     * @return DynamicWindow 
     */
    DynamicWindow calculateDynamicWindow();
    /**
     * @brief Calculate the reachable velocity from current velocity and dynamic window
     * 
     * @return ReachableVelocity 
     */
    ReachableVelocity calculateReachableVelocity();
    /**
     * @brief Clustering the point cloud using Adaptive Threshold distance
     * 
     */
    void clusterPointCloud();
    /**
     * @brief This function is based on ¨Efficient L-Shape Fitting for Vehicle Detection Using Laser Scanners¨
     * 
     */
    void rectangleFitting();
    /**
     * @brief Check point cloud is a convex polygon 
     * 
     * @param cluster 
     * @return true if convex polygon
     * @return false if 
     */
    bool isConvexObject(LaserPointCloud cluster);
    /**
     * @brief Extract the line segment from the point cloud 
     * 
     * @param cluster the point cloud clustered
     * @return list of start points and end points of the line segment
     */
    std::vector<LineSegment> lineExtraction(LaserPointCloud cluster);
    /**
     * @brief Find intersection point by two line
     * 
     * @param a1 a factor of line 1
     * @param b1 b factor of line 1
     * @param c1 c factor of line 1
     * @param a2 a factor of line 2
     * @param b2 b factor of line 2
     * @param c2 c factor of line 2
     * @return Eigen::Vector2d point intersection
     */
    Eigen::Vector2d lineIntersection(double a1, double b1, double c1, double a2, double b2, double c2);
    /**
     * @brief Calculate the cost of rectangle related to area criterion
     * 
     * @param C1 edge of rectangle
     * @param C2 edge of rectangle
     * @return double 
     */
    double areaCriterion(const Eigen::VectorXd & C1, const Eigen::VectorXd & C2);
    /**
     * @brief Calculate the cost of rectangle related to area closeness criterion
     * 
     * @param C1 edge of rectangle
     * @param C2 edge of rectangle
     * @param d0 minimum distance threshold 
     * @return double 
     */
    double closenessCriterion(const Eigen::VectorXd& C1, const Eigen::VectorXd &C2, const double & d0);
    /**
     * @brief Calculate the cost of rectangle related to area variance criterion
     * 
     * @param C1 edge of rectangle
     * @param C2 edge of rectangle 
     * @return double 
     */
    double varianceCriterion(const Eigen::VectorXd & C1, const Eigen::VectorXd & C2);
    /**
     * @brief Set the Robot Velocity object
     * 
     * @param wheel_vel the wheel velocities
     */
    void setRobotVelocity(WheelVelocity wheel_vel);
    // Initializes the motor and position sensor
	void initializeMotorAndPositionSensor();
    /**
     * @brief Initialize all sensors of robot
     */
	void initializeSensor();
    /**
     * @brief Check the goal reached 
     * @return true if goal reached
     */
    bool goalReached();
    /**
     * @brief Find minimum of difference between two angles
     * 
     * @param angle1 (radians)
     * @param angle2 (radians)
     * @return double (radians)
     */
    double findDifferenceOrientation(double angle1, double angle2);
    /**
     * @brief Convert left and right velocity to linear and angular velocity
     * 
     * @param wheel_vel input wheel velocities
     * @return Twist 
     */
    Twist convertWheelToVelocity(WheelVelocity wheel_vel);
    /**
     * @brief Get current robot pose [x, y, theta]
     * 
     */
    void getRobotPose();
    /**
     * @brief Get current robot velocity [v_l, v_r]
     * 
     */
    void getRobotVelocity();
    /**
     * @brief Extract the vertex of robot at point
     * @param pose the pose of the robot
     */
    std::vector<cv::Point> extractRobotVisual(PoseSE2 pose);
};


#endif
