#ifndef DBSCAN_H
#define DBSCAN_H

#include "ultis.h"

class DBSCAN
{
private:    
    unsigned int m_pointSize;
    unsigned int m_minPoints;
    float m_epsilon;
public:
    LaserScanData m_points;
    DBSCAN(unsigned int minPts, float eps, LaserScanData points){
        this->m_minPoints = minPts;
        this->m_epsilon = eps;
        this->m_points = points;
        this->m_pointSize = points.size();
    }
    ~DBSCAN(){}

    int run();
    std::vector<int> calculateCluster(LaserScanPoint point);
    int expandCluster(LaserScanPoint point, int clusterID);
    inline double calculateDistance(const LaserScanPoint pointCore, const LaserScanPoint pointTarget);
    int getTotalPointSize() {return m_pointSize;}
    int getMinimumClusterSize() {return m_minPoints;}
    int getEpsilonSize() {return m_epsilon;}
};
#endif