#include "obstacles.h"


Obstacle::Obstacle(Point2D p, Velocity v, double r, double pt, double dt)
{
	this->position = p;
	this->vel = v;
	this->radius = r;
	this->predictTime = pt;
	this->dt = dt;
	this->width = 0.0;
	this->length = 0.0;
	Point2D pos = p;
}

Obstacle::~Obstacle(){}

void Obstacle::updatePosition(const cv::Mat map, const double gain_x, const double gain_y, const int height, const BGR color_map)
{
	Point2D new_position;
	new_position.x = this->position.x + this->vel.x * this->dt;
	new_position.y = this->position.y + this->vel.y * this->dt;
	if (this->checkCollideWithWall(new_position, map, gain_x, gain_y, height, color_map) == true)
	{
		this->vel.x = -this->vel.x;
		this->vel.y = -this->vel.y;
	}
	
	// Update obsatcle position
	this->position.x += this->vel.x * this->dt;
	this->position.y += this->vel.y * this->dt;
}

std::vector<Point2D> Obstacle::predictObstacleState()
{
	std::vector<Point2D> path;
	Point2D pos = this->position;
	for (double i = 0; i < this->predictTime; i += this->dt)
	{
		pos.x = pos.x + this->vel.x * this->dt;
		pos.y = pos.y + this->vel.y * this->dt;
		path.push_back(pos);
	}
	return path;
}

bool Obstacle::checkCollideWithWall(const Point2D point, const cv::Mat map, const double gain_x, const double gain_y, const int height, const BGR color_map)
{
	Point2D around_point;
	Point2DPixel point_pixel;
	around_point.x = point.x + this->radius;
	around_point.y = point.y;
	point_pixel = convertMeterToPixel(around_point, gain_x, gain_y, height);
	BGR color = getColorInPoint(map, point_pixel);
	if (color.b == color_map.b && color.g == color_map.g && color.r == color_map.r) return true;
	
	around_point.x = point.x - this->radius;
	around_point.y = point.y;
	point_pixel = convertMeterToPixel(around_point, gain_x, gain_y, height);
	color = getColorInPoint(map, point_pixel);
	if (color.b == color_map.b && color.g == color_map.g && color.r == color_map.r) return true;

	around_point.x = point.x;
	around_point.y = point.y + this->radius;
	point_pixel = convertMeterToPixel(around_point, gain_x, gain_y, height);
	color = getColorInPoint(map, point_pixel);
	if (color.b == color_map.b && color.g == color_map.g && color.r == color_map.r) return true;

	around_point.x = point.x;
	around_point.y = point.y - this->radius;
	point_pixel = convertMeterToPixel(around_point, gain_x, gain_y, height);
	color = getColorInPoint(map, point_pixel);
	if (color.b == color_map.b && color.g == color_map.g && color.r == color_map.r) return true;
	return false;
}

ObstacleList::ObstacleList(){}

void ObstacleList::initialization(cv::Mat map, const double predict_time, const double dt, const BGR color)
{
	this->map = map;
	this->predict_time = predict_time;
	this->dt = dt;
	this->gain_x = (double)map.rows * pixel_to_meter / 2;
	this->gain_y = (double)map.cols * pixel_to_meter / 2;
	this->height = this->map.cols;
	this->color_map = color;
	Point2D pos;
	pos.x = 10.0;
	pos.y = 7.0;
	Velocity vel;
	vel.x = 0.5;
	vel.y = 0.0;
	Obstacle obs(pos, vel, 0.5, predict_time, this->dt);
	this->obstacles.push_back(obs);
	pos.x = 10.0;
	pos.y = 0.0;
	vel.x = 0.3;
	vel.y = 0.0;
	Obstacle obs1(pos, vel, 0.5, predict_time, this->dt);
	this->obstacles.push_back(obs1);
	pos.x = 10.0;
	pos.y = -4.0;
	vel.x = 0.3;
	vel.y = 0.0;
	Obstacle obs2(pos, vel, 0.5, predict_time, this->dt);
	this->obstacles.push_back(obs2);
	pos.x = 10.0;
	pos.y = -7.0;
	vel.x = -0.5;
	vel.y = 0.0;
	Obstacle obs3(pos, vel, 0.5, predict_time, this->dt);
	this->obstacles.push_back(obs3);
	pos.x = 10.0;
	pos.y = -6.0;
	vel.x = 0.5;
	vel.y = 0.0;
	Obstacle obs4(pos, vel, 0.5, predict_time, this->dt);
	this->obstacles.push_back(obs4);
	pos.x = -5.0;
	pos.y = 11.0;
	vel.x = -0.5;
	vel.y = 0.0;
	Obstacle obs5(pos, vel, 0.5, predict_time, this->dt);
	this->obstacles.push_back(obs5);
	pos.x = 0.0;
	pos.y = 10.0;
	vel.x = -0.5;
	vel.y = 0.0;
	Obstacle obs6(pos, vel, 0.5, predict_time, this->dt);
	this->obstacles.push_back(obs6);
	pos.x = 10.0;
	pos.y = 2.0;
	vel.x = -0.5;
	vel.y = 0.0;
	Obstacle obs7(pos, vel, 0.5, predict_time, this->dt);
	this->obstacles.push_back(obs7);
	pos.x = 0.5;
	pos.y = -5.0;
	vel.x = 0.0;
	vel.y = 0.5;
	Obstacle obs8(pos, vel, 0.5, predict_time, this->dt);
	this->obstacles.push_back(obs8);
	pos.x = -0.5;
	pos.y = 5.0;
	vel.x = 0.0;
	vel.y = -0.6;
	Obstacle obs9(pos, vel, 0.5, predict_time, this->dt);
	this->obstacles.push_back(obs9);
	pos.x = 1.8;
	pos.y = 10.0;
	vel.x = 0.0;
	vel.y = -0.6;
	Obstacle obs10(pos, vel, 0.5, predict_time, this->dt);
	this->obstacles.push_back(obs10);
	pos.x = 01.5;
	pos.y = 0.0;
	vel.x = 0.0;
	vel.y = 0.0;
	Obstacle obs11(pos, vel, 0.5, predict_time, this->dt);
	this->obstacles.push_back(obs11);
}

ObstacleList::~ObstacleList() {}

std::vector<Obstacle> ObstacleList::getObstacleInObservation(const Pose2D robot_pose, const double observable_range)
{
	std::vector<Obstacle> obs_list;
	for (int i = 0; i < this->obstacles.size(); i++)
	{
		double dx = this->obstacles[i].position.x - robot_pose.position.x;
		double dy = this->obstacles[i].position.y - robot_pose.position.y;
		if (hypot(dx, dy) < observable_range)
		{
			obs_list.push_back(this->obstacles[i]);
		}
	}

	return obs_list;
}


void ObstacleList::updateObstaclePosition()
{
	for (int i = 0; i < this->obstacles.size(); i++)
	{
		this->obstacles[i].updatePosition(this->map, this->gain_x, this->gain_y, this->height, this->color_map);
	}
}

cv::Mat ObstacleList::obstacleVisualization()
{
	cv::Mat result;
	this->map.copyTo(result);
	for (int i = 0; i < this->obstacles.size(); i++)
	{
		Point2DPixel point = convertMeterToPixel(this->obstacles[i].position, this->gain_x, this->gain_y, this->height);
		cv::Point center(point.x, point.y);
		int radius = int(this->obstacles[i].radius * meter_to_pixel);
		cv::Scalar color(255, 0, 0);
		int thickness = -1;
		cv::circle(result, center, radius, color, thickness);
	}

	return result;
}
