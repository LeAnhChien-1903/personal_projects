#include "ultis.h"

Point2D::Point2D()
{
	this->x = 0.0;
	this->y = 0.0;
}

Point2D::Point2D(double x, double y)
{
	this->x = x;
	this->y = y;
}

Pose2D::Pose2D()
{
	this->position.x = 0.0;
	this->position.y = 0.0;
	this->theta = 0.0;
}

Pose2D::Pose2D(double x, double y, double theta)
{
	this->position.x = x;
	this->position.y = y;
	this->theta = theta;
}

Pose2D::Pose2D(Point2D position, double theta)
{
	this->position = position;
	this->theta = theta;
}

Pose2D::Pose2D(Eigen::Vector2d position, double theta)
{
	this->position = Point2D(position.x(), position.y());
	this->theta = theta;
}

BGR::BGR()
{
	this->b = 0;
	this->g = 0;
	this->r = 0;
}

BGR::BGR(int blue, int green, int red)
{
	this->b = blue;
	this->g = green;
	this->r = red;
}

BGR::BGR(int x, int y, cv::Mat image)
{
	this->b = image.at<cv::Vec3b>(y, x)[0];
	this->g = image.at<cv::Vec3b>(y, x)[1];
	this->r = image.at<cv::Vec3b>(y, x)[2];
}

Point2DPixel::Point2DPixel()
{
	this->x = -1;
	this->y = -1;
}

Point2DPixel::Point2DPixel(double x, double y, double gain_x, double gain_y, double map_height)
{
	this->x = (int)((x + gain_x) * meter_to_pixel);
	this->y = (int)(map_height- (y + gain_y) * meter_to_pixel);
}

Point2DPixel::Point2DPixel(Point2D point, double gain_x, double gain_y, double map_height)
{
	this->x = (int)((point.x + gain_x) * meter_to_pixel);
	this->y = (int)(map_height - (point.y + gain_y) * meter_to_pixel);
}

Point2DPixel::Point2DPixel(Eigen::Vector2d point, double gain_x, double gain_y, double map_height)
{
	this->x = (int)((point.x() + gain_x) * meter_to_pixel);
	this->y = (int)(map_height - (point.y() + gain_y) * meter_to_pixel);
}

Twist::Twist()
{
	this->linear = 0.0;
	this->angular = 0.0;
}

Twist::Twist(double linear, double angular)
{
	this->linear = linear;
	this->angular = angular;
}
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

	x = std::cos(angle1) + std::cos(angle2);
	y = std::sin(angle1) + std::sin(angle2);
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

std::vector<Point2D> extractCoorner(Pose2D pose, double length, double width)
{
	std::vector<Point2D> points;
	Point2D corner;
	double x = pose.position.x;
	double y = pose.position.y;
	double halfLength = length / 2;
	double halfWidth = width / 2;
	double sinAngle = sin(pose.theta);
	double cosAngle = cos(pose.theta);

	// Bottom left
	corner.x = x + (cosAngle * -halfLength) - (sinAngle * halfWidth);
	corner.y = y + (sinAngle * -halfLength) + (cosAngle * halfWidth);
	points.push_back(corner);

	// Top left corner
	corner.x = x + (cosAngle * -halfLength) - (sinAngle * -halfWidth);
	corner.y = y + (sinAngle * -halfLength) + (cosAngle * -halfWidth);
	points.push_back(corner);

	// Top right 
	corner.x = x + (cosAngle * halfLength) - (sinAngle * -halfWidth);
	corner.y = y + (sinAngle * halfLength) + (cosAngle * -halfWidth);
	points.push_back(corner);

	// Bottom right
	corner.x = x + (cosAngle * halfLength) - (sinAngle * halfWidth);
	corner.y = y + (sinAngle * halfLength) + (cosAngle * halfWidth);
	points.push_back(corner);

	return points;
}

cv::Mat computeCostMap(cv::Mat map, double cost, BGR color_costmap, bool only_costmap)
{
	cv::Mat result;
	map.copyTo(result);
	int radius = (int)(cost * meter_to_pixel);
	for (int x = 0; x < map.cols; x++)
	{
		for (int y = 0; y < map.rows; y++)
		{
			BGR color = BGR(x, y, map);
			if (color.b == 0 && color.g == 0 && color.r == 0)
			{
				cv::circle(result, cv::Point(x, y), radius, cv::Scalar(color_costmap.b, color_costmap.g, color_costmap.r), -1);
			}
		}
	}
	if (only_costmap == false)
	{
		for (int x = 0; x < map.cols; x++)
		{
			for (int y = 0; y < map.cols; y++)
			{
				BGR color = BGR(x, y, map);
				if (color.b == 0 && color.g == 0 && color.r == 0)
				{
					cv::circle(result, cv::Point(x, y), 0, cv::Scalar(0, 0, 0), -1);
				}
			}
		}
	}
	return result;
}

bool is_close(double a, double b, double epsilon)
{
    return std::fabs(a - b) < epsilon;
}

double average_angles(const std::vector<double> &angles)
{
    double x = 0, y = 0;
    for (std::vector<double>::const_iterator it = angles.begin(); it != angles.end(); ++it)
    {
        x += cos(*it);
        y += sin(*it);
    }
    if (x == 0 && y == 0)
        return 0;
    else
        return std::atan2(y, x);
}

bool smaller_than_abs(double i, double j)
{
	return std::fabs(i) < std::fabs(j); 
}

double fast_sigmoid(double x)
{
    return x / (1 + fabs(x));
}
