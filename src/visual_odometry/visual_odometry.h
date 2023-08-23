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

#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/PointCloud.h> // camera feature

#include <spline/trajectory.h>
#include <utils/tic_toc.h>
#include <utils/yaml_utils.h>

#include "feature_manager.h"
#include "integration_base.h"
#include "vio_initial.h"

namespace ctrlvio
{

  enum MarginalizationFlag
  {
    MARGIN_OLD = 0, //
    MARGIN_SECOND_NEW = 1
  };

  class VisualOdometry : public FeatureManager
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef std::shared_ptr<VisualOdometry> Ptr;
    using Vector7D = Eigen::Matrix<double, 7, 1>;

    VisualOdometry(const YAML::Node &node, Trajectory::Ptr traj)
        : trajectory_(std::move(traj))
    {
      readParameters(node);
      ClearState();
    }

    void InitWindow(VIOInitialization &vio_initialer, int64_t t_bag_start = 0);

    void ProcessIMU(double timestamp, const Eigen::Vector3d &linear_acceleration,
                    const Eigen::Vector3d &angular_velocity);

    void AddImageToWindow(const std::pair<int64_t, sensor_msgs::PointCloud::ConstPtr> &img_msg,
                          const Trajectory::Ptr &trajectory, double line_delay);

    void InitialFeatureDepthWithLiDAR();

    void SlideWindow();

    ///================== get functions ================== ///

    MarginalizationFlag GetMarginalizationFlag() { return marg_flag_; }

    const int64_t *GetTimestamps() const { return timestamps_; }

    const std::list<FeaturePerId> &GetFeatures() const { return this->feature; }

    // visualize landmarks in cloud message
    VPointCloud GetLandmarksInWindow() const;

    VPointCloud GetMarginCloud() const;

  private:
    static bool IsLandMarkStable(const FeaturePerId &fea)
    {
      if (!FeatureManager::isLandmarkCandidate(fea))
        return false;
      if (fea.start_frame > WINDOW_SIZE * 3.0 / 4.0)
        return false;
      if (fea.estimated_depth <= 0)
        return false;
      // TODO
      // if (it_per_id.solve_flag != SovelSucc)
      return true;
    }

    void ClearState();

    void SlideWindowNew();

    void SlideWindowOld();

    bool FailureDetection() const;

    void SaveDataForFailureDetection();

  private:
    Trajectory::Ptr trajectory_;

    MarginalizationFlag marg_flag_;

    int frame_count_;

  public:
    Eigen::Vector3d Ps_[(WINDOW_SIZE + 1)]; // notice: camera to world
    Eigen::Matrix3d Rs_[(WINDOW_SIZE + 1)];
    int64_t timestamps_[(WINDOW_SIZE + 1)];

    IntegrationBase *pre_integrations_[(WINDOW_SIZE + 1)];
    Eigen::Vector3d Bas_[(WINDOW_SIZE + 1)];
    Eigen::Vector3d Bgs_[(WINDOW_SIZE + 1)];

  private:
    // preintegration
    std::vector<double> imu_timestamps_buf_;
    std::vector<Eigen::Vector3d> linear_acceleration_buf_;
    std::vector<Eigen::Vector3d> angular_velocity_buf_;

    Eigen::Vector3d PI_;
    Eigen::Vector3d VI_;
    Eigen::Matrix3d RI_;
    Eigen::Vector3d Gravity_;

    double marg_old_t0_;
    Eigen::Matrix3d back_R0_;
    Eigen::Vector3d back_P0_;

    // For failure detection
    Eigen::Matrix3d last_R_, last_R0_;
    Eigen::Vector3d last_P_, last_P0_;
  };

} // namespace ctrlvio
