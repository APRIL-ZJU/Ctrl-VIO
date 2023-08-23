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

#include <sensor_msgs/PointCloud.h> // camera feature
#include <Eigen/Dense>
#include <iostream>
#include <map>

namespace ctrlvio
{

  using Vector2d = Eigen::Vector2d;
  using Vector3d = Eigen::Vector3d;
  using Vector7d = Eigen::Matrix<double, 7, 1>;
  using VectorXd = Eigen::VectorXd;

  using Matrix3d = Eigen::Matrix3d;
  using MatrixXd = Eigen::MatrixXd;

  class FeaturePerFrame
  {
  public:
    FeaturePerFrame(const Vector7d &_point, double td) : depth_from_lidar(-1)
    {
      point.x() = _point(0);
      point.y() = _point(1);
      point.z() = _point(2);
      uv.x() = _point(3);
      uv.y() = _point(4);
      velocity.x() = _point(5);
      velocity.y() = _point(6);
      cur_td = td;
    }

    // estimator_node.process --> Estimator::processImage
    // --> f_manager.addFeatureCheckParallax --> feature.push_back
    Vector3d point; // undistorted_pts.x; undistorted_pts.y; z = 1;
    Vector2d uv;
    Vector2d velocity;
    double cur_td;

    double depth_from_lidar;

    // MatrixXd A;
    // VectorXd b;
  };

  enum FeatureDepthFlag
  {
    Intial,    // 0 sliding window haven't solve yet
    SovelSucc, // 1 solve succ
    SolveFail  // 2 solve fail
  };

  class FeaturePerId
  {
  public:
    const int feature_id;
    int start_frame;
    std::vector<FeaturePerFrame> feature_per_frame;

    int used_num;

    // bool is_margin;
    double estimated_depth;
    FeatureDepthFlag solve_flag;
    bool lidar_depth_flag;

    FeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id),
          start_frame(_start_frame),
          used_num(0),
          estimated_depth(-1.0),
          solve_flag(Intial),
          lidar_depth_flag(false) {}

    int endFrame() const { return start_frame + feature_per_frame.size() - 1; };
  };

  inline void FeatureMsg2Image(
      const sensor_msgs::PointCloud::ConstPtr &img_msg,
      std::map<int, std::vector<std::pair<int, Vector7d>>> &image_out)
  {
    for (unsigned int i = 0; i < img_msg->points.size(); i++)
    {
      int v = img_msg->channels[0].values[i] + 0.5;
      int NUM_OF_CAM = 1;
      int feature_id = v / NUM_OF_CAM;
      int camera_id = v % NUM_OF_CAM;
      double x = img_msg->points[i].x;
      double y = img_msg->points[i].y;
      double z = img_msg->points[i].z;
      double p_u = img_msg->channels[1].values[i];
      double p_v = img_msg->channels[2].values[i];
      double velocity_x = img_msg->channels[3].values[i];
      double velocity_y = img_msg->channels[4].values[i];

      assert(z == 1);
      Vector7d xyz_uv_velocity;
      xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
      image_out[feature_id].emplace_back(camera_id, xyz_uv_velocity);
    }
  }

} // namespace ctrlvio