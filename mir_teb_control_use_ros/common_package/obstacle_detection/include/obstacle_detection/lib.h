#ifndef LIB_H
#define LIB_H

// OpenCV libraries
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
// Eigen libraries
#include <Eigen/Core>
#include <Eigen/Dense>
// C++ libraries
#include <cmath>
#include <vector>
#include <random> 
#include <iostream>
#include <list>
#define SMALL_NUM 0.00000001
// Constants for rectangle fit
const double extra_distance = 0.05;
// Line extractor constants
const int MINIMUM_POINTS_CHECK = 2;
const int MINIMUM_INDEX = 2;
const double MAXIMUM_GAP_DISTANCE = 0.5;
const double IN_LINE_THRESHOLD = 0.25;

typedef std::pair<int, int> PointPixel;
struct LaserScanPoint
{
	double range; // range of laser scan
	double angle; // angle of laser scan
	Eigen::Vector2d point; // point of laser scan
	LaserScanPoint(){};
	LaserScanPoint(double range_, double angle_, Eigen::Vector2d point_)
	{
		range = range_;
		angle = angle_;
		point = point_;
	}
};
typedef std::pair<int, int> LineIndex;
typedef std::vector<Eigen::Vector2d> LineSegment;
typedef std::vector<Eigen::Vector2d> LaserPointCloud;
typedef std::vector<LaserScanPoint> LaserScanData;
typedef std::vector<LaserScanData> LaserScanDataCluster;
typedef std::vector<LaserPointCloud> LaserPointCloudCluster;
// Common functions
/**
 * @brief Convert point from meter to pixel coordinates
 * 
 * @param point coordinate in meter
 * @param original original coordinate in meter
 * @param resolution resolution of map (m/cell)
 * @param map_width with of map (m)
 * @param map_height height of map (m)
 * @return cv::Point 
 */
cv::Point convertMeterToPixel(Eigen::Vector2d point, Eigen::Vector2d original, double resolution,double map_width, double map_height);

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
 * @brief 
 * 
 * @param x 
 * @param cellsize 
 * @return int 
 */
int contxy2disc(double x, double cellsize);

/**
 * @brief Get the Point Pixel object
 * 
 * @param x 
 * @param y 
 * @param origin_x 
 * @param origin_y 
 * @param resolution 
 * @param map_height_pixel 
 * @return PointPixel 
 */
PointPixel getPointPixel(double x, double y, double origin_x, double origin_y, double resolution, int map_height_pixel);
#endif 