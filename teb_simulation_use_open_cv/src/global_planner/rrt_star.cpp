#include "rrt_star.h"

RRTSTAR::RRTSTAR(){}

void RRTSTAR::initialize(cv::Mat map, Pose2D start, Pose2D goal, BGR color, double radius)
{
	this->map = map;
	this->map_height_pixel = map.rows;
	this->map_width_pixel = map.cols;
	this->half_height_meter = double(map.rows * 0.5 / meter_to_pixel);
	this->half_width_meter = double(map.cols * 0.5 / meter_to_pixel);
	this->gain_x = double(this->map_width_pixel * pixel_to_meter / 2);
	this->gain_y = double(this->map_height_pixel * pixel_to_meter / 2);
	this->startPos = start;
	this->endPos = goal;
	this->color_map = color;
	this->robot_radius = radius;
	this->root = new Node;
	this->root->parent = NULL;
	this->root->pose = start;
	this->root->cost = 0.0;
	this->lastNode = root;
	this->nodes.push_back(root);
	this->step_size = 1.0;
	this->max_iter = 50000;
}

Node* RRTSTAR::getRandomNode()
{
	Node* ret;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(-1.0, 1.0);
	Pose2D pose;
	pose.position.x = dis(gen) * this->half_width_meter;
	pose.position.y = dis(gen) * this->half_height_meter;
	pose.theta = dis(gen) * CV_PI;

	ret = new Node;
	ret->pose = pose;
	return ret;
}
/**
 * @brief Get nearest node from a given configuration/position.
 * @param point
 * @return
 */
Node* RRTSTAR::nearest(Pose2D point)
{
	double minDist = INFINITY;
	Node* closest = NULL;
	for (int i = 0; i < (int)this->nodes.size(); i++)
	{
		double distance = this->distance(point.position, nodes[i]->pose.position);
		if (distance < minDist)
		{
			minDist = distance;
			closest = nodes[i];
		}
	}
	return closest;
}
/**
 * @brief Get neighborhood nodes of a given configuration/position.
 * @param point
 * @param radius
 * @param out_nodes
 * @return
 */
void RRTSTAR::near(Pose2D point, double radius, std::vector<Node*>& out_nodes)
{
	for (int i = 0; i < (int)this->nodes.size(); i++)
	{
		double dist = this->distance(point.position, this->nodes[i]->pose.position);
		if (dist < radius)
		{
			out_nodes.push_back(nodes[i]);
		}
	}
}
/**
 * @brief Helper method to find distance between two positions.
 * @param p
 * @param q
 * @return
 */
double RRTSTAR::distance(Point2D& p, Point2D& q)
{
	double dx = p.x - q.x;
	double dy = p.y - q.y;
	return hypot(dx, dy);
}
/**
 * @brief Return trajectory cost.
 * @param q
 * @return
 */
double RRTSTAR::cost(Node* q)
{
	return q->cost;
}
/**
 * @brief Compute path cost.
 * @param qFrom
 * @param qTo
 * @return
 */
double RRTSTAR::pathCost(Node* qFrom, Node* qTo)
{
	return this->distance(qTo->pose.position, qFrom->pose.position);
}
/**
 * @brief Add a node to the tree.
 * @param qNearest
 * @param qNew
 */
void RRTSTAR::add(Node* qNearest, Node* qNew)
{
	qNew->parent = qNearest;
	qNew->cost = qNearest->cost + this->pathCost(qNearest, qNew);
	qNearest->children.push_back(qNew);
	nodes.push_back(qNew);
	lastNode = qNew;
}
/**
 * @brief Check if the last node is close to the end position.
 * @return
 */
bool RRTSTAR::reached()
{
	if (this->distance(this->lastNode->pose.position, this->endPos.position) < END_DIST_THRESHOLD)
		return true;
	return false;
}

void RRTSTAR::setStepSize(double step)
{
	this->step_size = step;
}

void RRTSTAR::setMaxIterations(int iter)
{
	this->max_iter = iter;
}
/**
 * @brief Delete all nodes using DFS technique.
 * @param root
 */
void RRTSTAR::deleteNodes(Node* root)
{
	for (int i = 0; i < (int)this->root->children.size(); i++)
	{
		this->deleteNodes(root->children[i]);
	}
	delete root;
}
/**
 * @brief Find a configuration at a distance step_size from nearest node to random node.
 * @param q
 * @param qNearest
 * @return
 */
Vector3d RRTSTAR::newConfig(Node* q, Node* qNearest)
{
	Vector2d to(q->pose.position.x, q->pose.position.y);
	Vector2d from(qNearest->pose.position.x, qNearest->pose.position.y);
	Vector2d intermediate = to - from;
	intermediate = intermediate / intermediate.norm();
	Vector2d pos = from + step_size * intermediate;
	Vector3d ret(pos.x(), pos.y(), 0.0);
	return ret;
}

void RRTSTAR::implementAlgorithm()
{
	for (int i = 0; i < this->max_iter; i++)
	{
		Node* q = this->getRandomNode();
		if (q)
		{
			Node* qNearest = this->nearest(q->pose);
			if (this->distance(q->pose.position, qNearest->pose.position) > this->step_size)
			{
				Vector3d newConfigPosOrient = this->newConfig(q, qNearest);
				Pose2D newConfigPos;
				newConfigPos.position.x = newConfigPosOrient.x();
				newConfigPos.position.y = newConfigPosOrient.y();
				if (this->checkCollision(newConfigPos.position, qNearest->pose.position) == false)
				{
					Node* qNew = new Node;
					qNew->pose = newConfigPos;

					std::vector<Node*> Qnear;
					this->near(qNew->pose, this->step_size * RRTSTAR_NEIGHBOR_FACTOR, Qnear);
					Node* qMin = qNearest;
					double cmin = this->cost(qNearest) + this->pathCost(qNearest, qNew);
					for (int j = 0; j < Qnear.size(); j++)
					{
						Node* qNear = Qnear[j];
						if (this->checkCollision(qNear->pose.position, qNew->pose.position) == false &&
							this->cost(qNear) + this->pathCost(qNear, qNew) < cmin)
						{
							qMin = qNear;
							cmin = this->cost(qNear) + this->pathCost(qNear, qNew);
						}
					}
					this->add(qMin, qNew);
					for (int j = 0; j < Qnear.size(); j++)
					{
						Node* qNear = Qnear[j];
						if (this->checkCollision(qNew->pose.position, qNear->pose.position) &&
							(this->cost(qNew) + this->pathCost(qNew, qNear)) < this->cost(qNear))
						{
							Node* qParent = qNear->parent;
							// Remove edge between qParent and qNear
							qParent->children.erase(std::remove(qParent->children.begin(), qParent->children.end(), qNear), qParent->children.end());
							// Add edge between qNew and qNear
							qNear->cost = this->cost(qNew) + this->pathCost(qNew, qNear);
							qNear->parent = qNew;
							qNew->children.push_back(qNear);
						}
					}
				}
			}
		}
	}
	Node* q;
	if (this->reached())
	{
		q = this->lastNode;
	}
	else
	{
		// if not reached yet, then shortestPath will start from the closest node to end point.
		q = this->nearest(this->endPos);
	}
	// generate shortest path to destination.
	while (q != NULL) 
	{
		this->path.push_back(q);
		q = q->parent;
	}
	for (int i = (int) this->path.size() - 1; i >= 0; i--)
	{
		waypoints.push_back(this->path[i]->pose.position);
	}
	waypoints.push_back(this->endPos.position);
	this->getSmoothWaypoints();
}

bool RRTSTAR::checkCollision(Point2D start, Point2D end)
{
	double dx = end.x - start.x;
	double dy = end.y - start.y;
	double distance = hypot(dx, dy);
	double angle = atan2(dy, dx);
	for (double i = 0; i <= distance; i+= 0.1)
	{
		Point2D point;
		point.x = start.x + i * cos(angle);
		point.y = start.y + i * sin(angle);
		Point2DPixel point_pixel(point, this->gain_x, this->gain_y, this->map_height_pixel);
		BGR color(point_pixel.x, point_pixel.y, this->map);
		if (color.b == 0 && color.g == 0 && color.r == 0) return true;
		if (color.b == this->color_map.b && color.g == this->color_map.g && color.r == this->color_map.r) return true;
	}
	// Check collision with map
	for (double angle = 0; angle <= CV_2PI; angle += CV_PI / 20)
	{
		Point2D point;
		point.x = end.x + this->robot_radius * cos(angle);
		point.y = end.y + this->robot_radius * sin(angle);
		Point2DPixel point_pixel(point, this->gain_x, this->gain_y, this->map_height_pixel);
		BGR color(point_pixel.x, point_pixel.y, this->map);
		if (color.b == this->color_map.b && color.g == this->color_map.g && color.r == this->color_map.r) return true;
		if (color.b == 0 && color.g == 0 && color.r == 0) return true;
	}
	return false;
}

void RRTSTAR::getSmoothWaypoints()
{
	this->smoothWaypoints.push_back(this->startPos.position);
	for (int i = 1; i < this->waypoints.size(); i++)
	{
		if (i < this->waypoints.size() - 1)
		{
			if (this->checkCollision(this->smoothWaypoints.back(), this->waypoints[i]) == false &&
				this->checkCollision(this->smoothWaypoints.back(), this->waypoints[i + 1]) == true )
			{
				this->smoothWaypoints.push_back(this->waypoints[i]);
			}
		}
		else
		{
			if (this->checkCollision(this->smoothWaypoints.back(), this->waypoints[i]) == false)
			{
				this->smoothWaypoints.push_back(this->waypoints[i]);
			}
		}
		
	}
}
