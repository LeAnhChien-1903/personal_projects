#ifndef OBSTACLES_VER2_H
#define OBSTACLES_VER2_H
#include "ultis.h"

class CircleObstacle
{
public:
	Eigen::Vector2d position; // Position of the obstacle
	Eigen::Vector2d velocity; // Velocity of the obstacle
	double radius, dt;
    /**
     * @brief Construct a new Circle Obstacle object
     * 
     * @param position_ position of the obstacle
     * @param velocity_ velocity of the obstacle
     * @param radius_ radius of the obstacle
     * @param dt_ the sample time 
     */
	CircleObstacle(Eigen::Vector2d position_, Eigen::Vector2d velocity_, double radius_, double dt_);
	/**
	 * @brief Destroy the Circle Obstacle object
	 * 
	 */
    ~CircleObstacle();
    /**
     * @brief Update the position of the obstacle with constant velocity
     * @param cost_map the map to check collide 
     * @param gain_x gain to add in x-coordinate
     * @param gain_y gain to add in y-coordinate
     * @param map_height height of map
     * @return true if the obstacle is collide with the obstacle in map
     */
	void updatePosition(const cv::Mat cost_map, const double gain_x, const double gain_y, const int map_height);
    /**
     * @brief Check if the obstacle is collide with the obstacle in map
     * 
     * @param cost_map the map to check collide 
     * @param gain_x gain to add in x-coordinate
     * @param gain_y gain to add in y-coordinate
     * @param map_height height of map
     * @return true if the obstacle is collide with the obstacle in map
     */
	bool checkCollideWithWall(const cv::Mat cost_map, const double gain_x, const double gain_y, const int map_height);
    /**
     * @brief Visualize the obstacle
     * 
     * @param map the map to visualize
     * @param gain_x gain to add in x-coordinate
     * @param gain_y gain to add in y-coordinate
     * @param map_height height map 
     * @param color the color of obstacles
     */
    void visualize(cv::Mat &map, const double gain_x, const double gain_y, const int map_height, cv::Scalar color);
};
class RectangleObstacle
{
public:
	public:
	Eigen::Vector2d position; // Position of the obstacle
	Eigen::Vector2d velocity; // Velocity of the obstacle
    std::vector<Eigen::Vector2d> vertices; // Vertices of the obstacle
	double length, width; // Length and width of the obstacle
    double dt; // Sample time
    /**
     * @brief Construct a new Rectangle Obstacle object
     * 
     * @param position_ the position of the obstacle
     * @param velocity_ the velocity of the obstacle
     * @param length_ the length of the obstacle
     * @param width_ the width of the obstacle
     * @param dt_ the sample time
     */
	RectangleObstacle(Eigen::Vector2d position_, Eigen::Vector2d velocity_, double length_, double width_, double dt_);
	/**
	 * @brief Destroy the Circle Obstacle object
	 * 
	 */
    ~RectangleObstacle(){};
    /**
     * @brief Update the position of the obstacle with constant velocity
     * @param cost_map the map to check collide 
     * @param gain_x gain to add in x-coordinate
     * @param gain_y gain to add in y-coordinate
     * @param map_height height of map
     * @return true if the obstacle is collide with the obstacle in map
     */
	void updatePosition(const cv::Mat cost_map, const double gain_x, const double gain_y, const int map_height);
    /**
     * @brief Check if the obstacle is collide with the obstacle in map
     * 
     * @param cost_map the map to check collide 
     * @param gain_x gain to add in x-coordinate
     * @param gain_y gain to add in y-coordinate
     * @param map_height height of map
     * @return true if the obstacle is collide with the obstacle in map
     */
	bool checkCollideWithWall(const cv::Mat cost_map, const double gain_x, const double gain_y, const int map_height);
    /**
     * @brief Visualize the obstacle
     * 
     * @param map the map to visualize
     * @param gain_x gain to add in x-coordinate
     * @param gain_y gain to add in y-coordinate
     * @param map_height height map 
     * @param color the color of obstacles
     */
    void visualize(cv::Mat &map, const double gain_x, const double gain_y, const int map_height, cv::Scalar color);
};

class ObstacleList
{
public:
    std::vector<CircleObstacle> circle_obs; // list of circle obstacles
    std::vector<RectangleObstacle> rectangle_obs; // list of rectangle obstacles    
    /**
     * @brief Construct a new Obstacle List object
     */
    ObstacleList();
    /**
     * @brief Construct a new Obstacle List object with specified sample time
     * 
     * @param dt_ the sample time
     */
    ObstacleList(double dt_);
    /**
     * @brief Destroy the Obstacle List object
     * 
     */
    ~ObstacleList(){};
    /**
     * @brief 
     * 
     * @param dt_ the sample time
     */
    void initialize(double dt_);
    /**
     * @brief Update the position of the obstacle with constant velocity
     * @param cost_map the map to check collide 
     * @param gain_x gain to add in x-coordinate
     * @param gain_y gain to add in y-coordinate
     * @param map_height height of map
     * @return true if the obstacle is collide with the obstacle in map
     */
    void updateObstacleList(const cv::Mat cost_map, const double gain_x, const double gain_y, const int map_height);
    /**
     * @brief Visualize the obstacle
     * 
     * @param map the map to visualize
     * @param gain_x gain to add in x-coordinate
     * @param gain_y gain to add in y-coordinate
     * @param map_height height map 
     * @param color the color of obstacles
     */
    void visualizeObstaclesList(cv::Mat &map, const double gain_x, const double gain_y, const int map_height, cv::Scalar color);

};

#endif