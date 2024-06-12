#include "lib.h"

bool comparisonFunction(const LaserScanPoint &point1, const LaserScanPoint &point2)
{
    return point1.angle < point2.angle;
}

double distanceBetweenToLine(const Eigen::Vector2d &point, const Eigen::Vector2d &point_start, const Eigen::Vector2d &point_end)
{
    return abs((point_end.x() - point_start.x())*(point_start.y() - point.y()) - (point_start.x() - point.x())*(point_end.y() - point_start.y())) / (point_end - point_start).norm();
}

double normalize_angle(double angle)
{
    return atan2(sin(angle), cos(angle));
}

double average_angle(double angle1, double angle2)
{
	double x, y;

	x = cos(angle1) + cos(angle2);
	y = sin(angle1) + sin(angle2);
	if (x == 0 && y == 0)
		return 0;
	else
		return std::atan2(y, x);
}

bool checkAngleBetweenAngles(float angle, float angle_right, float angle_left)
{
    if (abs(angle_right - angle_left) <= M_PI)
    {
        if (angle_right <= angle && angle <= angle_left) return true;
        else return false;
    }
    else
    {
        if (angle_left < 0 && angle_right > 0)
        {
            angle_left += 2 * M_PI;
            if (angle < 0) angle += 2 * M_PI;
            if (angle_right <= angle && angle <= angle_left) return true;
            else return false;
        }
        if (angle_left > 0 && angle_right < 0)
        {
            angle_right += 2 * M_PI;
            if (angle < 0) angle += 2 * M_PI;
            if (angle_right <= angle && angle <= angle_left) return true;
            else return false;
        }
    }
    return false;
}
cv::Point convertMeterToPixel(Eigen::Vector2d point, Eigen::Vector2d original, double resolution, double map_width, double map_height)
{
    cv::Point point_pixel;
    point_pixel.x = (int)((point.x() - original.x() +  map_width/2) / resolution);
	point_pixel.y = (int)((map_height - point.y() + original.y() - map_height/2) / resolution);
    return point_pixel;
} 

int contxy2disc(double x, double cellsize)
{
    if (x >= 0)
    {
        return static_cast<int>(x / cellsize);
    } 
    else 
    {
        return static_cast<int>(x / cellsize) - 1;
    }
}

PointPixel getPointPixel(double x, double y, double origin_x, double origin_y, double resolution, int map_height_pixel)
{
    PointPixel point;
    point.first = contxy2disc(x - origin_x, resolution);
    point.second = map_height_pixel - contxy2disc(y-origin_y, resolution);

    return point;
}