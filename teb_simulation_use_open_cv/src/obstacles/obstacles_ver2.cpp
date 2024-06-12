#include "obstacles_ver2.h"

CircleObstacle::CircleObstacle(Eigen::Vector2d position_, Eigen::Vector2d velocity_, double radius_, double dt_)
{
    this->position = position_;
    this->velocity = velocity_;
    this->radius = radius_;
    this->dt = dt_;
}

CircleObstacle::~CircleObstacle()
{
}

void CircleObstacle::updatePosition(const cv::Mat cost_map, const double gain_x, const double gain_y, const int map_height)
{
    if (this->checkCollideWithWall(cost_map, gain_x, gain_y, map_height) == true)
    {
        this->velocity = - this->velocity;
    }
    this->position += this->velocity * this->dt;
}

bool CircleObstacle::checkCollideWithWall(const cv::Mat cost_map, const double gain_x, const double gain_y, const int map_height)
{
    Point2D around_point(this->position.x() + this->radius, this->position.y());
    Point2DPixel point_pixel(around_point, gain_x, gain_y, map_height);
    BGR point_color(point_pixel.x, point_pixel.y, cost_map);
    if (!(point_color.b == 255 && point_color.g == 255 && point_color.r == 255)) return true;
    
    Point2D around_point1(this->position.x() - this->radius, this->position.y());
    Point2DPixel point_pixel1(around_point1, gain_x, gain_y, map_height);
    BGR point_color1(point_pixel1.x, point_pixel1.y, cost_map);
    if (!(point_color1.b == 255 && point_color1.g == 255 && point_color1.r == 255)) return true;

	Point2D around_point2(this->position.x(), this->position.y() + this->radius);
    Point2DPixel point_pixel2(around_point2, gain_x, gain_y, map_height);
    BGR point_color2(point_pixel2.x, point_pixel2.y, cost_map);
    if (!(point_color2.b == 255 && point_color2.g == 255 && point_color2.r == 255)) return true;

    Point2D around_point3(this->position.x(), this->position.y() - this->radius);
    Point2DPixel point_pixel3(around_point3, gain_x, gain_y, map_height);
    BGR point_color3(point_pixel3.x, point_pixel3.y, cost_map);
    if (!(point_color3.b == 255 && point_color3.g == 255 && point_color3.r == 255)) return true;

    return false;
}

void CircleObstacle::visualize(cv::Mat &map, const double gain_x, const double gain_y, const int map_height, cv::Scalar color)
{
    Point2DPixel point(this->position, gain_x, gain_y, map_height);
    cv::circle(map, cv::Point(point.x, point.y), int(this->radius * meter_to_pixel), color, -1);
}

RectangleObstacle::RectangleObstacle(Eigen::Vector2d position_, Eigen::Vector2d velocity_, double length_, double width_, double dt_)
{
    this->position = position_;
    this->velocity = velocity_;
    this->length = length_;
    this->width = width_;
    this->dt = dt_;
    std::vector<Point2D> corners = extractCoorner(Pose2D(position_.x(), position_.y(), 0.0),length_, width_);
    for (int i = 0; i < corners.size(); ++i)
    {
        this->vertices.push_back(Eigen::Vector2d(corners[i].x, corners[i].y));
    }
}

void RectangleObstacle::updatePosition(const cv::Mat cost_map, const double gain_x, const double gain_y, const int map_height)
{
    if (this->checkCollideWithWall(cost_map, gain_x, gain_y, map_height) == true)
    {
        this->velocity = - this->velocity;
    }
    this->position += this->velocity * this->dt;
    this->vertices.clear();
    // Update vertices
    std::vector<Point2D> corners = extractCoorner(Pose2D(this->position.x(), this->position.y(), 0.0), this->length, this->width);
    for (int i = 0; i < corners.size(); ++i)
    {
        this->vertices.push_back(Eigen::Vector2d(corners[i].x, corners[i].y));
    }
}

bool RectangleObstacle::checkCollideWithWall(const cv::Mat cost_map, const double gain_x, const double gain_y, const int map_height)
{
    double radius = sqrt(pow(this->length/2, 2.0) + pow(this->width/2, 2.0));
    Point2D around_point(this->position.x() + radius, this->position.y());
    Point2DPixel point_pixel(around_point, gain_x, gain_y, map_height);
    BGR point_color(point_pixel.x, point_pixel.y, cost_map);
    if (!(point_color.b == 255 && point_color.g == 255 && point_color.r == 255)) return true;
    
    Point2D around_point1(this->position.x() - radius, this->position.y());
    Point2DPixel point_pixel1(around_point1, gain_x, gain_y, map_height);
    BGR point_color1(point_pixel1.x, point_pixel1.y, cost_map);
    if (!(point_color1.b == 255 && point_color1.g == 255 && point_color1.r == 255)) return true;

	Point2D around_point2(this->position.x(), this->position.y() + radius);
    Point2DPixel point_pixel2(around_point2, gain_x, gain_y, map_height);
    BGR point_color2(point_pixel2.x, point_pixel2.y, cost_map);
    if (!(point_color2.b == 255 && point_color2.g == 255 && point_color2.r == 255)) return true;

    Point2D around_point3(this->position.x(), this->position.y() - radius);
    Point2DPixel point_pixel3(around_point3, gain_x, gain_y, map_height);
    BGR point_color3(point_pixel3.x, point_pixel3.y, cost_map);
    if (!(point_color3.b == 255 && point_color3.g == 255 && point_color3.r == 255)) return true;

    return false;
}

void RectangleObstacle::visualize(cv::Mat &map, const double gain_x, const double gain_y, const int map_height, cv::Scalar color)
{
    std::vector<cv::Point> vertex_list;
    for (int i = 0; i < this->vertices.size(); ++i)
    {
        Point2DPixel vertex(this->vertices[i], gain_x, gain_y, map_height);
        vertex_list.push_back(cv::Point(vertex.x, vertex.y));
    }
    cv::fillConvexPoly(map, vertex_list, color);
}

ObstacleList::ObstacleList()
{
    this->circle_obs.clear();
    this->rectangle_obs.clear();
}

ObstacleList::ObstacleList(double dt_)
{
    // Circle obstacle
    CircleObstacle obs1(Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(0.0, 0.0), 0.6, dt_);
    this->circle_obs.push_back(obs1);
    CircleObstacle obs2(Eigen::Vector2d(0.0,-2.0), Eigen::Vector2d(0.0, 0.0), 0.6, dt_);
    this->circle_obs.push_back(obs2);
    // Rectangle obstacle
    RectangleObstacle obs3(Eigen::Vector2d(3.0, 2.0), Eigen::Vector2d(0.0, -0.2), 1.2, 1.2, dt_);
    this->rectangle_obs.push_back(obs3);
    RectangleObstacle obs4(Eigen::Vector2d(-3.0, -2.0), Eigen::Vector2d(0.0, 0.2), 1.2, 1.2, dt_);
    this->rectangle_obs.push_back(obs4);
}

void ObstacleList::initialize(double dt_)
{
    // Circle obstacle
    CircleObstacle obs1(Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(0.0, 0.0), 0.6, dt_);
    this->circle_obs.push_back(obs1);
    CircleObstacle obs2(Eigen::Vector2d(0.0,-2.0), Eigen::Vector2d(0.0, 0.0), 0.6, dt_);
    this->circle_obs.push_back(obs2);
    // Rectangle obstacle
    RectangleObstacle obs3(Eigen::Vector2d(3.0, 3.0), Eigen::Vector2d(0.0, -0.5), 1.2, 1.2, dt_);
    this->rectangle_obs.push_back(obs3);
    RectangleObstacle obs4(Eigen::Vector2d(-3.0, -3.0), Eigen::Vector2d(0.0, 0.5), 1.2, 1.2, dt_);
    this->rectangle_obs.push_back(obs4);
}

void ObstacleList::updateObstacleList(const cv::Mat cost_map, const double gain_x, const double gain_y, const int map_height)
{
    for (int i = 0; i < this->circle_obs.size(); ++i)
    {
        this->circle_obs[i].updatePosition(cost_map, gain_x, gain_y, map_height);
    }
    for (int i = 0; i < this->rectangle_obs.size(); ++i)
    {
        this->rectangle_obs[i].updatePosition(cost_map, gain_x, gain_y, map_height);
    }
}

void ObstacleList::visualizeObstaclesList(cv::Mat &map, const double gain_x, const double gain_y, const int map_height, cv::Scalar color)
{
    for (int i = 0; i < this->circle_obs.size(); ++i)
    {
        this->circle_obs[i].visualize(map, gain_x, gain_y, map_height, color);
    }
    for (int i = 0; i < this->rectangle_obs.size(); ++i)
    {
        this->rectangle_obs[i].visualize(map, gain_x, gain_y, map_height, color);
    }
}
