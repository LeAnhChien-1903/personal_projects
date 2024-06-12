#include "dbscan.h"

int DBSCAN::run()
{
    int clusterID = 1;
    LaserScanData::iterator iter;
    for(iter = m_points.begin(); iter != m_points.end(); ++iter)
    {
        if ( iter->cluster_id == UNCLASSIFIED )
        {
            if ( expandCluster(*iter, clusterID) != FAILURE )
            {
                clusterID += 1;
            }
        }
    }

    return 0;
}

std::vector<int> DBSCAN::calculateCluster(LaserScanPoint point)
{
    int index = 0;
    LaserScanData::iterator iter;
    std::vector<int> clusterIndex;
    for( iter = m_points.begin(); iter != m_points.end(); ++iter)
    {
        if ( calculateDistance(point, *iter) <= m_epsilon )
        {
            clusterIndex.push_back(index);
        }
        index++;
    }
    return clusterIndex;
}

int DBSCAN::expandCluster(LaserScanPoint point, int clusterID)
{
    std::vector<int> clusterSeeds = calculateCluster(point);

    if ( clusterSeeds.size() < m_minPoints )
    {
        point.cluster_id = NOISE;
        return FAILURE;
    }
    else
    {
        int index = 0, indexCorePoint = 0;
        std::vector<int>::iterator iterSeeds;
        for( iterSeeds = clusterSeeds.begin(); iterSeeds != clusterSeeds.end(); ++iterSeeds)
        {
            m_points.at(*iterSeeds).cluster_id = clusterID;
            if (m_points.at(*iterSeeds).point.x() == point.point.x() && m_points.at(*iterSeeds).point.x() == point.point.y())
            {
                indexCorePoint = index;
            }
            ++index;
        }
        clusterSeeds.erase(clusterSeeds.begin()+indexCorePoint);

        for(std::vector<int>::size_type i = 0, n = clusterSeeds.size(); i < n; ++i )
        {
            std::vector<int> clusterNeighbors = calculateCluster(m_points.at(clusterSeeds[i]));

            if ( clusterNeighbors.size() >= m_minPoints )
            {
                std::vector<int>::iterator iterNeighbors;
                for ( iterNeighbors = clusterNeighbors.begin(); iterNeighbors != clusterNeighbors.end(); ++iterNeighbors )
                {
                    if ( m_points.at(*iterNeighbors).cluster_id == UNCLASSIFIED || m_points.at(*iterNeighbors).cluster_id == NOISE )
                    {
                        if ( m_points.at(*iterNeighbors).cluster_id == UNCLASSIFIED )
                        {
                            clusterSeeds.push_back(*iterNeighbors);
                            n = clusterSeeds.size();
                        }
                        m_points.at(*iterNeighbors).cluster_id = clusterID;
                    }
                }
            }
        }
        return SUCCESS;
    }
}

inline double DBSCAN::calculateDistance(const LaserScanPoint pointCore, const LaserScanPoint pointTarget)
{
    return pow(pointCore.point.x() - pointTarget.point.x(), 2)+pow(pointCore.point.y() - pointTarget.point.y(),2);
}
