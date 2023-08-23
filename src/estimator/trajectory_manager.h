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
#include <estimator/trajectory_estimator.h>
#include <spline/trajectory.h>
#include <visual_odometry/feature_manager.h>
#include <visual_odometry/integration_base.h>
#include <visual_odometry/visual_odometry.h>
#include <visual_odometry/visual_struct.h>

#include <estimator/factor/analytic_diff/marginalization_factor.h>
#include <utils/opt_weight.h>

namespace ctrlvio
{

  class TrajectoryManager
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef std::shared_ptr<TrajectoryManager> Ptr;

    TrajectoryManager(const YAML::Node &node, Trajectory::Ptr trajectory);

    void InitFactorInfo(
        const ExtrinsicParam &Ep_CtoI, const double image_feature_weight = 0,
        const Eigen::Vector3d &local_velocity_weight = Eigen::Vector3d::Zero());

    void SetTrajectory(Trajectory::Ptr trajectory) { trajectory_ = trajectory; }

    void SetSystemState(const SystemState &sys_state);

    void SetOriginalPose(Eigen::Quaterniond q,
                         Eigen::Vector3d p = Eigen::Vector3d::Zero());

    void AddIMUData(const IMUData &data);

    bool UpdateVIOPrior(const std::list<FeaturePerId> &features,
                        const int64_t timestamps[],
                        IntegrationBase *pre_integrations[],
                        Eigen::VectorXd &feature_depths,
                        MarginalizationFlag marg_flag, double opt_min_time);

    void InitTrajectory(const Eigen::Vector3d &bg, const Eigen::Vector3d &ba);

    bool UpdateTrajectory(
        const std::list<FeaturePerId> &features, const int64_t timestamps[],
        IntegrationBase *pre_integrations[], Eigen::Vector3d Bgs[],
        Eigen::Vector3d Bas[], Eigen::VectorXd &feature_depths,
        MarginalizationFlag marg_flag, const int iteration = 50);

    void Log(std::string descri) const;

    void ExtendTrajectory(int64_t max_time);

    std::vector<int64_t> key_frame_time_vec_;

    Eigen::aligned_vector<IMUData> imu_data_;

  private:
    void RemoveIMUData(int64_t t_window_min);

    void feature2double(Eigen::VectorXd &dep)
    {
      for (int i = 0; i < dep.size(); i++)
        para_Feature[i][0] = dep(i);
    }

    void double2feature(Eigen::VectorXd &dep)
    {
      for (int i = 0; i < dep.size(); i++)
        dep(i) = para_Feature[i][0];
    }
    PoseData original_pose_;

    int64_t last_image_time_;

    double para_Feature[NUM_OF_F][1];

    // ==================== State ==================== //
    OptWeight opt_weight_;

    Eigen::Vector3d gravity_;

    Trajectory::Ptr trajectory_;

    std::map<int64_t, IMUBias> all_imu_bias_;

    // ==================== Marginazation ==================== //
    // vio system
    MarginalizationInfo::Ptr last_marginalization_info;
    std::vector<double *> last_marginalization_parameter_blocks;

    // ==================== discrete bias ==================== //
    //  double para_bg[WINDOW_SIZE + 1][3];
    //  double para_ba[WINDOW_SIZE + 1][3];

    // ==================== 4DoF ==================== //
    void double2vector(const int64_t *timestamps, const Eigen::Matrix3d &R0,
                       const Eigen::Vector3d &t0);

    Eigen::Matrix<double, 6, 1> sqrt_info_bias[WINDOW_SIZE];
    int64_t max_bef_ns;
    int max_bef_idx;
    int max_aft_idx;
  };

} // namespace ctrlvio
