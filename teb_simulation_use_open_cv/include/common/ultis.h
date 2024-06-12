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
#include <chrono>
// Constants
#define SMALL_NUM 0.00000001

// Map convert constants
const int meter_to_pixel = 40; // 1m ~ 10 pixels
const double pixel_to_meter = 0.025; // 1 pixel ~ 0.1 m
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

typedef std::vector<Point2D> Polygon;
typedef std::vector<Point2D> ReferencePath;
typedef std::vector<Point2D> Waypoints;
typedef std::vector<cv::Point> PathInPixel;
typedef std::vector<Point2D> PathInReal;

// Common functions
/**
 * normalize the angle
 */
double normalize_angle(double angle);
/**
 * average two angles
 */
double average_angle(double angle1, double angle2);
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