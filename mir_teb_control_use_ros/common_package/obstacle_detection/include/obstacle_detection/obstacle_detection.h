#ifndef OBSTACLE_DETECTION_NO_MAP_H
#define OBSTACLE_DETECTION_NO_MAP_H

#include <ros/ros.h>
#include <tf/tf.h>
#include <sensor_msgs/LaserScan.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/PoseArray.h>
#include <std_msgs/Bool.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <dynamic_reconfigure/server.h>
#include <obstacle_detection/ObstacleDetectionReconfigureConfig.h>
#include <obstacle_detection/ObstacleVertices.h>

#include "lib.h"

class ObstacleDetectorWithMap
{
private:
    ros::NodeHandle node;
    std::string scan_topic, amcl_topic, start_topic, local_costmap_topic;
    std::string obstacle_topic;
    //Constants
    double sample_time; // the sample time
    double lidar_x, lidar_y, lidar_z; // the position of the lidar relative to the base_link
    double lidar_roll, lidar_pitch, lidar_yaw; // the rotation of the lidar relative to the base_link
    // Robot
    double robot_length; // the robot length
    double robot_width; // the robot width
    double robot_radius; // the robot radius
    Eigen::Vector3d robot_pose; // Store the robot pose [x, y, theta]
    
    // Obstacles
    std::vector<std::vector<Eigen::Vector2d>> obstacles; // position of obstacles
    std::vector<std::vector<Eigen::Vector2d>> prev_obstacles; // position of previous obstacles
    // Local map
    double map_height; // height of the local map (m)
    double map_width; // width of the local map (m)
    double resolution; // resolution of the local map (m/cell)
    double inflation_radius; // radius of the inflation obstacle (radius)
    double min_distance_between_points; // minimum distance between two vertices in different polygon obstacles
    Eigen::Vector2d center; // center of the local
    cv::Mat local_map; // Store the local map
    cv::Mat map; // Store the local map
    // Data sensor
    LaserScanData laser_scan; // The laser scan data
    // ROS publishers and subscribers
    ros::Subscriber laser_scan_sub, start_sub, amcl_sub;
    ros::Publisher visualization_pub, obstacle_pub, local_costmap_pub;
    // Timer
    ros::Timer timer_;
    // Dynamic reconfiguration
    dynamic_reconfigure::Server<obstacle_detection::ObstacleDetectionReconfigureConfig> dynamic_config_server;
    // Start command
    bool startCommand = false;
public:
    /**
     * @brief Construct a new Obstacle Detector object
     * 
     */
    ObstacleDetectorWithMap(){};
    /**
     * @brief Destroy the Obstacle Detector object
     * 
     */
    ~ObstacleDetectorWithMap(){};
    /**
     * @brief Initialize the obstacle detector
     * 
     * @return true 
     * @return false 
     */
    bool initialize();
    /**
     * @brief Visualize the obstacles
     * 
     * @return visualization_msgs::MarkerArray 
     */
    visualization_msgs::MarkerArray obstacleVisualization();
    /**
     * @brief Get pose data (x, y, theta)
     * 
     * @param msg 
     */
    void acmlCallback(const geometry_msgs::PoseWithCovarianceStamped msg);
    /**
     * @brief Get odometry data
     * 
     * @param msg 
     */
    void odometryCallback(const nav_msgs::Odometry msg);
    /**
     * @brief get laser scan data
     * 
     * @param msg 
     */
    void laserScanCallback(const sensor_msgs::LaserScan msg);
    /**
     * @brief get local cost map
     * 
     * @param local_costmap 
     */
    void localCostmapCallback(const nav_msgs::OccupancyGrid local_costmap);
    /**
     * @brief Get start command 
     * 
     */
    void startCallback(const std_msgs::Bool msg);
    /**
     * @brief Dynamic configuration callback function
     * 
     * @param config 
     */
    void dynamicConfigurationCallback(const obstacle_detection::ObstacleDetectionReconfigureConfig config, uint32_t level);
    /**
     * @brief Timer callback function 
     * 
     */
    void timerCallback(const ros::TimerEvent & event);
    /**
     * @brief Create a Costmap From Laser Scan data
     * 
     */
    void createCostmapFromLaserScan();
    /**
     * @brief Extract polygon obstacle from the cost map
     * 
     */
    void extractObstacleFromLocalCostmap();
    /**
     * @brief Extract equivalent obstacle from the obstacle 
     * 
     */
    void extractEquivalentObstacles();
    /**
     * @brief Check if two polygon are in the same obstacle
     * 
     * @param polygon1 
     * @param polygon2 
     * @return true 
     * @return false 
     */
    bool checkTwoEquivalentPolygon(std::vector<Eigen::Vector2d> polygon1, std::vector<Eigen::Vector2d> polygon2);
    /**
     * @brief Association the obstacle
     * 
     */
    void associateObstacle();
    /**
     * @brief 
     * 
     * @param map 
     * @param origin_x 
     * @param origin_y 
     * @param resolution 
     * @param map_height_pixel 
     */
    void visualizeObstacleOpenCV(cv::Mat &map, double origin_x, double origin_y, double resolution, double map_height_pixel);
    /**
     * @brief Calculate the area of the convex
     * 
     * @param convex 
     * @return double 
     */
    double calculateAreaOfConvex(std::vector<Eigen::Vector2d>  convex);
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
     * @brief Find minimum of difference between two angles
     * 
     * @param angle1 (radians)
     * @param angle2 (radians)
     * @return double (radians)
     */
    double findDifference(double angle1, double angle2);
    
};

#endif
