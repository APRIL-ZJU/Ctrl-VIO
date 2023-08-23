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

// #include "sophus_utils.hpp"
#include "parameter_struct.h"

namespace ctrlvio
{

  inline PoseData XYThetaToPoseData(double x, double y, double theta,
                                    double timestamp = 0)
  {
    PoseData pose;
    Eigen::Vector3d p(x, y, 0);
    Eigen::AngleAxisd rotation_vector(theta, Eigen::Vector3d::UnitZ());
    Eigen::Quaterniond q(rotation_vector);
    pose.timestamp = timestamp;
    pose.position = p;
    pose.orientation.setQuaternion(q);

    return pose;
  }

  inline PoseData SE3ToPoseData(SE3d se3_pose, double time = 0)
  {
    PoseData pose;
    pose.timestamp = time;
    pose.position = se3_pose.translation();
    pose.orientation = se3_pose.so3();
    return pose;
  }

  inline SE3d Matrix4fToSE3d(Eigen::Matrix4f matrix)
  {
    Eigen::Vector3d trans(matrix(0, 3), matrix(1, 3), matrix(2, 3));
    Eigen::Quaterniond q(matrix.block<3, 3>(0, 0).cast<double>());
    q.normalize();
    return SE3d(q, trans);
  }

  inline void SE3dToPositionEuler(SE3d se3_pose, Eigen::Vector3d &position,
                                  Eigen::Vector3d &euler)
  {
    position = se3_pose.translation();
    Eigen::Quaterniond q = se3_pose.unit_quaternion();
    euler = q.toRotationMatrix().eulerAngles(0, 1, 2);
    euler *= 180 / M_PI;
  }

} // namespace ctrlvio
