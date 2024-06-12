#ifndef ULTIS_H
#define ULTIS_H

// Eigen libraries
#include <Eigen/Dense>
#include <Eigen/Core>
// Boost libraries
#include <boost/utility.hpp>
#include <boost/type_traits.hpp>
// OpenCV libraries
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
// C++ libraries
#include <cmath>
#include <vector>
#include <random> 
#include <iostream>
#include <cassert>
#include <complex>
#include <list>
// Constants
#define SMALL_NUM 0.00000001
// Constants for rectangle fit
const double extra_distance = 0.05;
const int step_of_theta = 25;
// Line extractor constants
const int MINIMUM_POINTS_CHECK = 2;
const int MINIMUM_INDEX = 2;
const double MAXIMUM_GAP_DISTANCE = 0.5;
const double IN_LINE_THRESHOLD = 0.25;

// Map convert constants
const int meter_to_pixel = 40; // 1m ~ 40 pixels
const double pixel_to_meter = 0.025; // 1 pixel ~ 0.025 m
const double map_height_meter = 22.0; // m
const double map_width_meter = 22.0; // m

// RRT star constants
const double BOT_RADIUS = 1.0;
const double NODE_RADIUS = 0.1;
const double END_DIST_THRESHOLD = 0.1;
const double BOT_CLEARANCE = 1.5 * BOT_RADIUS;
const double BOT_TURN_RADIUS = 1.0;
const double RRTSTAR_NEIGHBOR_FACTOR = 0;
const bool BOT_FOLLOW_DUBIN = false;
// Point2D contain x, y coordinates
class Point2D
{
public:
	double x, y;
	Point2D();
	Point2D(double x, double y);
};
class Pose2D
{
public:
	Point2D position;
	double theta;
	Pose2D();
	Pose2D(double x, double y, double theta);
	Pose2D(Point2D position, double theta);
	Pose2D(Eigen::Vector2d position, double theta);
};
// BGR contain three channels of color (b, g, r)
class BGR
{
public: 
	int b, g, r; // store values of three channels
	/**
	* @brief Default constructor
	*/
	BGR();
	/**
	* @brief Construct BGR with three channel blue, green, red
	* @param blue value in blue channel
	* @param green value in green channel
	* @param red value in red channel
	*/
	BGR(int blue, int green, int red);
	/**
	* @brief Construct BGR by get color in an position of an image
	* @param x x-coordinate
	* @param y y-coordinate
	* @param image input image
	*/
	BGR(int x, int y, cv::Mat image);
};
class Point2DPixel
{
public: 
	int x, y; // Store coordinate in pixel
	/**
	* @brief Default constructor
	*/
	Point2DPixel();
	/**
	* @brief Construct Point2DPixel by convert from x, y in real
	* @param x x-coordinate
	* @param y y-coordinate
	* @param gain_x gain to add in x-coordinate
	* @param gain_y gain to add in y-coordinate
	* @param map_height height of map
	*/
	Point2DPixel(double x, double y, double gain_x, double gain_y, double map_height);
	/**
	* @brief Construct Point2DPixel by convert from Point2D in real
	* @param point point in real
	* @param gain_x gain to add in x-coordinate
	* @param gain_y gain to add in y-coordinate
	* @param map_height height of map
	*/
	Point2DPixel(Point2D point, double gain_x, double gain_y, double map_height);
	/**
	* @brief Construct Point2DPixel by convert from Eigen::Vector2d in real
	* @param point point in real
	* @param gain_x gain to add in x-coordinate
	* @param gain_y gain to add in y-coordinate
	* @param map_height height of map
	*/
	Point2DPixel(Eigen::Vector2d point, double gain_x, double gain_y, double map_height);
};
class Twist
{
public:
	double linear, angular;
	Twist();
	Twist(double linear, double angular);
};
struct Node
{
	std::vector<Node*> children;
	Node* parent;
	Pose2D pose;
	double cost;

};
// Structure for store DynamicWindow of robot
struct DynamicWindow
{
    DynamicWindow(double min_left, double max_left, double min_right, double max_right)
    {
        left_min_vel = min_left;
        left_max_vel = max_left;
        right_min_vel = min_right;
        right_max_vel = max_right;
    }
    DynamicWindow()
    {
        left_min_vel = right_min_vel = left_max_vel = right_min_vel = 0.0;
    }
	double left_min_vel , left_max_vel;
	double right_min_vel, right_max_vel;
};
// Structure for store wheel velocities
struct WheelVelocity
{
    double left_vel , right_vel;
    WheelVelocity(double left_vel_, double right_vel_)
    {
        left_vel = left_vel_;
        right_vel = right_vel_;
    }
    WheelVelocity()
    {
        left_vel = 0.0;
        right_vel = 0.0;
    }
};
struct LaserScanPoint
{
	double range; // range of laser scan
	double angle; // angle of laser scan
	Eigen::Vector2d point; // point of laser scan
	LaserScanPoint(){};
	LaserScanPoint(double range_, double angle_, Eigen::Vector2d point_, int cluster_id_)
	{
		range = range_;
		angle = angle_;
		point = point_;
	}
};
// The reachable velocity typedef
typedef std::vector<WheelVelocity> ReachableVelocity;
typedef std::vector<Point2D> Polygon;
typedef std::vector<Point2D> ReferencePath;
typedef std::vector<Point2D> Waypoints;
typedef std::vector<cv::Point> PathInPixel;
typedef std::vector<Point2D> PathInReal;
typedef std::pair<int, int> LineIndex;
typedef std::vector<Eigen::Vector2d> LineSegment;
typedef std::vector<Eigen::Vector2d> LaserPointCloud;
typedef std::vector<LaserScanPoint> LaserScanData;
typedef std::vector<LaserPointCloud> LaserPointCloudCluster;
// Common functions
bool comparisonFunction(const LaserScanPoint& point1, const LaserScanPoint& point2); 
/**
 * @brief Calculates distance between one point to line
 * 
 * @param point 
 * @param point_start point start on line
 * @param point_end point end on line
 * @return double 
 */
double distanceBetweenToLine(const Eigen::Vector2d& point, const Eigen::Vector2d &point_start, const Eigen::Vector2d &point_end);
/**
 * normalize the angle
 */
double normalize_angle(double angle);
/**
 * average two angles
 */
double average_angle(double angle1, double angle2);
/**
 * @brief Check a angle is between two angles
 * 
 * @param angle 
 * @param angle_right 
 * @param angle_left 
 * @return true 
 * @return false 
 */
bool checkAngleBetweenAngles(float angle, float angle_right, float angle_left);
/**
* @brief Extract 4 corners of an rectangle from pose of center, length and width of rectangle
* @param pose pose of center (x, y, theta)
* @param length length of rectangle
* @param width width of rectangle
* @output vector contains coordinate (x, y) of 4 corners
*/
std::vector<Point2D> extractCoorner(Pose2D pose, double length , double width);
/**
* @brief Apply cost map 2D for a map
* @param map original map
* @param cost radius for inflate map
* @param color_costmap color of cost map
* @param only_costmap for contain original map in cost map
* @output vector contains coordinate (x, y) of 4 corners
*/
cv::Mat computeCostMap(cv::Mat map, double cost, BGR color_costmap, bool only_costmap);
#endif 