#include "cubic_spline.h"

using namespace Eigen;
CubicSpline1D::CubicSpline1D() {}
CubicSpline1D::~CubicSpline1D() {}

void CubicSpline1D::initialization(const std::vector<double> s, const std::vector<double> x, const double ds)
{
	this->sample_input = s;
	this->x_input = x;
	VectorXd vec_s(s.size()), vec_x(x.size());
	for (int i = 0; i < s.size(); i++)
	{
		vec_s(i) = this->sample_input[i];
		vec_x(i) = this->x_input[i];
	}
	int numberX = int(s.size());
	VectorXd h = this->differenceInVector(vec_s);
	VectorXd a = vec_x;
	// Compute coefficient c
	MatrixXd A = this->calculateMatrixA(h, numberX);
	VectorXd B = this->calculateMatrixB(h, a, numberX);
	VectorXd c = A.inverse() * B;
	// Compute coefficient b and d
	VectorXd vb(numberX - 1), vd(numberX - 1);
	for (int i = 0; i < numberX - 1; i++)
	{
		vb(i) = 1 / h(i) * (a(i + 1) - a(i)) - h(i) / 3.0 * (2 * c(i) + c(i + 1));
		vd(i) = (c(i + 1) - c(i)) / (3 * h(i));
	}
	VectorXd b = vb;
	VectorXd d = vd;
	this->getCubicData(a, b, c, d, ds);
}

void CubicSpline1D::getCubicData(const MatrixXd a, const MatrixXd b, const MatrixXd c, const MatrixXd d, const double ds)
{
	for (int i = 0; i < this->sample_input.size() - 1; i++)
	{
		for (double time = this->sample_input[i]; time < this->sample_input[i + 1]; time += ds)
		{
			double dx = time - this->sample_input[i];
			double x_ = a(i) + b(i) * dx + c(i) * dx * dx + d(i) * dx * dx * dx;
			double dx_ = b(i) + 2.0 * c(i) * dx + 3.0 * d(i) * dx * dx;
			double ddx_ = 2.0 * c(i) + 6 * d(i) * dx;
			this->x.push_back(x_);
			this->dx.push_back(dx_);
			this->ddx.push_back(ddx_);
			this->s.push_back(time);
		}
	}
	this->x.push_back(this->x_input.back());
	this->dx.push_back(0.0);
	this->ddx.push_back(0.0);
	this->s.push_back(this->sample_input.back());
}

VectorXd CubicSpline1D::differenceInVector(const VectorXd x)
{
	VectorXd diff(x.size() - 1);
	for (int i = 0; i < diff.size(); i++)
	{
		diff(i) = x(i + 1) - x(i);
	}
	return diff;
}
MatrixXd CubicSpline1D::calculateMatrixA(const VectorXd h, const int numberX)
{
	// calc matrix A for spline coefficient c
	MatrixXd A(numberX, numberX);
	A.setZero();
	A(0, 0) = 1.0;
	for (int i = 0; i < numberX - 1; i++)
	{
		if (i != numberX - 2)
		{
			A(i + 1, i + 1) = 2.0 * (h(i) + h(i + 1));
		}
		A(i + 1, i) = h(i);
		A(i, i + 1) = h(i);
	}
	A(0, 1) = 0.0;
	A(numberX - 1, numberX - 2) = 0.0;
	A(numberX - 1, numberX - 1) = 1.0;
	return A;
}

VectorXd CubicSpline1D::calculateMatrixB(const VectorXd h, const VectorXd a, const int numberX)
{
	//calc matrix B for spline coefficient c
	VectorXd B(numberX);
	B.setZero();
	for (int i = 0; i < numberX - 2; i++)
	{
		B(i + 1) = 3.0 * (a(i + 2) - a(i + 1)) / h(i + 1) - 3.0 * (a(i + 1) - a(i)) / h(i);
	}
	return B;
}

CubicSpline2D::CubicSpline2D(){}

CubicSpline2D::~CubicSpline2D(){}

void CubicSpline2D::initialization(Waypoints waypoints, const double ds, const double meter_to_sample)
{
	this->ds = ds;
	std::vector<double> x, y;
	this->sample.push_back(0.0);
	for (int i = 0; i < waypoints.size(); i++)
	{
		x.push_back(waypoints[i].x);
		y.push_back(waypoints[i].y);
		if (i > 0)
		{
			double dx = waypoints[i].x - waypoints[i - 1].x;
			double dy = waypoints[i].y - waypoints[i - 1].y;
			double new_sample = ceil(this->sample.back() + hypot(dx, dy) * meter_to_sample);
			this->sample.push_back(new_sample);
		}
	}
	this->x_cubic.initialization(this->sample, x, ds);
	this->y_cubic.initialization(this->sample, y, ds);
}

ReferencePath CubicSpline2D::computeCubicPath()
{
	ReferencePath path;
	for (int i = 0; i < this->x_cubic.x.size(); i++)
	{
		Point2D point;
		point.x = this->x_cubic.x[i];
		point.y = this->y_cubic.x[i];
		path.push_back(point);
	}
	return path;
}
