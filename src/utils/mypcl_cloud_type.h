/*
 * Ctrl-VIO: Continuous-Time Visual-Inertial Odometry for Rolling Shutter Cameras
 * Copyright (C) 2022 Xiaolei Lang
 * Copyright (C) 2022 Jiajun Lv
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace my_pcl
{

  struct PointXYZIRT
  {
    PCL_ADD_POINT4D;                // quad-word XYZ
    float intensity;                ///< laser intensity reading
    uint16_t ring;                  ///< laser ring number
    float time;                     ///< laser time reading
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // ensure proper alignment
  } EIGEN_ALIGN16;

  struct OusterPointXYZIRT
  {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    // uint16_t ambient;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  } EIGEN_ALIGN16;

  struct PointXYZT
  {
    PCL_ADD_POINT4D; /// quad-word XYZ
    float intensity;
    double timestamp;               /// laser timestamp
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // ensure proper alignment
  } EIGEN_ALIGN16;

  struct PointXYZIRPYT
  {
    PCL_ADD_POINT4D;
    PCL_ADD_INTENSITY;
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  } EIGEN_ALIGN16;

} // namespace my_pcl

// https://github.com/PointCloudLibrary/pcl/issues/3190
POINT_CLOUD_REGISTER_POINT_STRUCT(my_pcl::PointXYZIRT,          //
                                  (float, x, x)                 //
                                  (float, y, y)                 //
                                  (float, z, z)                 //
                                  (float, intensity, intensity) //
                                  (uint16_t, ring, ring)        //
                                  (float, time, time)           //
)

POINT_CLOUD_REGISTER_POINT_STRUCT(my_pcl::OusterPointXYZIRT,             //
                                  (float, x, x)                          //
                                  (float, y, y)                          //
                                  (float, z, z)                          //
                                  (float, intensity, intensity)          //
                                  (uint32_t, t, t)                       //
                                  (uint16_t, reflectivity, reflectivity) //
                                  (uint8_t, ring, ring)                  //
                                  // (uint16_t, ambient, ambient)            //
                                  (uint32_t, range, range)) //

POINT_CLOUD_REGISTER_POINT_STRUCT(my_pcl::PointXYZT,             //
                                  (float, x, x)                  //
                                  (float, y, y)                  //
                                  (float, z, z)                  //
                                  (float, intensity, intensity)  //
                                  (double, timestamp, timestamp) //
)

POINT_CLOUD_REGISTER_POINT_STRUCT(my_pcl::PointXYZIRPYT,        //
                                  (float, x, x)                 //
                                  (float, y, y)                 //
                                  (float, z, z)                 //
                                  (float, intensity, intensity) //
                                  (float, roll, roll)           //
                                  (float, pitch, pitch)         //
                                  (float, yaw, yaw)             //
                                  (double, time, time)          //
)

typedef pcl::PointXYZI VPoint;
typedef pcl::PointCloud<VPoint> VPointCloud;

typedef pcl::PointXYZ GPoint;
typedef pcl::PointCloud<GPoint> GPointCloud;

typedef my_pcl::PointXYZIRT RTPoint;
typedef pcl::PointCloud<RTPoint> RTPointCloud;

typedef my_pcl::OusterPointXYZIRT OusterPoint;
typedef pcl::PointCloud<OusterPoint> OusterPointCloud;

typedef my_pcl::PointXYZT PosPoint;
typedef pcl::PointCloud<PosPoint> PosCloud;

typedef my_pcl::PointXYZIRPYT PosePoint;
typedef pcl::PointCloud<PosePoint> PosePointCloud;

typedef pcl::PointXYZRGB ColorPoint;
typedef pcl::PointCloud<ColorPoint> ColorPointCloud;
