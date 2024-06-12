#include "ultis.h"

Point2DPixel convertMeterToPixel(const Point2D point, const double gain_x, const double gain_y, const int height)
{
	Point2DPixel new_point;
	new_point.x = (int) ((point.x + gain_x) * meter_to_pixel);
	new_point.y = (int) (height - (point.y + gain_y) * meter_to_pixel);

	return new_point;
}

BGR getColorInPoint(const cv::Mat image, const Point2DPixel point)
{
	BGR color;
	color.b = image.at<cv::Vec3b>(point.y, point.x)[0];
	color.g = image.at<cv::Vec3b>(point.y, point.x)[1];
	color.r = image.at<cv::Vec3b>(point.y, point.x)[2];

	return color;
}

double normalize_angle(double angle)
{
	return atan2(sin(angle), cos(angle));
}

std::vector<Point2D> extracRectangle(Pose2D pose, double length, double width)
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

cv::Mat computeCostMap(cv::Mat map, double cost, BGR color_map)
{
    cv::Mat result;
    map.copyTo(result);
    int radius = (int)(cost * meter_to_pixel);
    for (int x = 0; x < map.cols; x++)
    {
        for (int y = 0; y < map.rows; y++)
        {
            Point2DPixel point;
            point.x = x;
            point.y = y;
            BGR color = getColorInPoint(map, point);
            if (color.b == 0 && color.g == 0 && color.r == 0)
            {
                cv::circle(result, cv::Point(x, y), radius, cv::Scalar(color_map.b, color_map.g, color_map.r), -1);
            }
        }
    }
    for (int x = 0; x < map.cols; x++)
    {
        for (int y = 0; y < map.cols; y++)
        {
            Point2DPixel point;
            point.x = x;
            point.y = y;
            BGR color = getColorInPoint(map, point);
            if (color.b == 0 && color.g == 0 && color.r == 0)
            {
                cv::circle(result, cv::Point(x, y), 0, cv::Scalar(0, 0, 0), -1);
            }
        }
    }
    return result;
}

