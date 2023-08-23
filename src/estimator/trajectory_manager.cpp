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

#include <estimator/factor/analytic_diff/image_feature_factor.h>
#include <estimator/factor/analytic_diff/trajectory_value_factor.h>
#include <estimator/trajectory_manager.h>
#include <ros/assert.h>

#include <fstream>
std::fstream ldFile;

namespace ctrlvio
{
  TrajectoryManager::TrajectoryManager(const YAML::Node &node,
                                       Trajectory::Ptr trajectory)
      : last_image_time_(-1),
        opt_weight_(OptWeight(node)),
        trajectory_(trajectory),
        last_marginalization_info(nullptr)
  {
    std::string config_path = node["config_path"].as<std::string>();
    std::string imu_yaml = node["imu_yaml"].as<std::string>();
    YAML::Node imu_node = YAML::LoadFile(config_path + imu_yaml);
    key_frame_time_vec_.clear();

    InitFactorInfo(trajectory_->GetSensorEP(CameraSensor),
                   opt_weight_.image_weight, opt_weight_.local_velocity_info_vec);

    for (int i = 0; i < WINDOW_SIZE; i++)
    {
      sqrt_info_bias[i] = Eigen::Matrix<double, 6, 1>::Zero();
    }
  }

  void TrajectoryManager::InitFactorInfo(
      const ExtrinsicParam &Ep_CtoI, const double image_feature_weight,
      const Eigen::Vector3d &local_velocity_weight)
  {
    if (image_feature_weight > 1e-5)
    {
      Eigen::Matrix2d sqrt_info = image_feature_weight * Eigen::Matrix2d::Identity();
      analytic_derivative::ImageFeatureDelayFactor::S_CtoI = Ep_CtoI.so3;
      analytic_derivative::ImageFeatureDelayFactor::p_CinI = Ep_CtoI.p;
      analytic_derivative::ImageFeatureDelayFactor::sqrt_info = sqrt_info;
    }
  }

  void TrajectoryManager::SetSystemState(const SystemState &sys_state)
  {
    gravity_ = sys_state.g;
    all_imu_bias_[trajectory_->minTimeNs()] = sys_state.bias;
    SetOriginalPose(sys_state.q, sys_state.p);

    Eigen::Vector3d euler = R2ypr(sys_state.q.toRotationMatrix());
    std::cout << "SystemState:\n";
    std::cout << "\t- position: " << sys_state.p.transpose() << std::endl;
    std::cout << "\t- euler: " << euler.transpose() << std::endl;
    std::cout << "\t- gravity: " << gravity_.transpose() << std::endl;
    std::cout << "\t- gyr bias: " << sys_state.bias.gyro_bias.transpose() << std::endl;
    std::cout << "\t- acc bias: " << sys_state.bias.accel_bias.transpose() << std::endl;
  }

  void TrajectoryManager::SetOriginalPose(Eigen::Quaterniond q,
                                          Eigen::Vector3d p)
  {
    original_pose_.orientation.setQuaternion(q);
    original_pose_.position = p;
  }

  void TrajectoryManager::AddIMUData(const IMUData &data)
  {
    imu_data_.emplace_back(data);
    imu_data_.back().timestamp -= trajectory_->GetDataStartTime();
  }

  void TrajectoryManager::RemoveIMUData(int64_t t_window_min)
  {
    int64_t active_time = t_window_min;
    for (auto iter = imu_data_.begin(); iter != imu_data_.end();)
    {
      if (iter->timestamp < active_time)
      {
        iter = imu_data_.erase(iter);
      }
      else
      {
        break;
      }
    }
  }

  void TrajectoryManager::ExtendTrajectory(int64_t max_time)
  {
    double max_bef = trajectory_->maxTimeNs() * NS_TO_S;
    max_bef_ns = trajectory_->maxTimeNs();
    max_bef_idx = trajectory_->cpnum() - 1;

    SE3d last_knot = trajectory_->getLastKnot();
    trajectory_->extendKnotsTo(max_time, last_knot); // maxTime>=max_time
    double max_aft = trajectory_->maxTimeNs() * NS_TO_S;
    max_aft_idx = trajectory_->cpnum() - 1;

    LOG(INFO) << "[max_bef | max_aft] " << max_bef << " " << max_aft;
  }

  bool TrajectoryManager::UpdateVIOPrior(const std::list<FeaturePerId> &features,
                                         const int64_t timestamps[],
                                         IntegrationBase *pre_integrations[],
                                         Eigen::VectorXd &feature_depths,
                                         MarginalizationFlag marg_flag,
                                         double opt_min_time)
  {
    // get bias in the window
    std::map<int, double *> para_bg_vec;
    std::map<int, double *> para_ba_vec;
    for (int i = 0; i <= WINDOW_SIZE; ++i)
    {
      auto &bias = all_imu_bias_[timestamps[i]];
      para_bg_vec[i] = bias.gyro_bias.data();
      para_ba_vec[i] = bias.accel_bias.data();
    }

    TrajectoryEstimatorOptions option;
    option.show_residual_summary = true;

    // get the index of first control point at every keyframe time
    std::vector<size_t> key_startid_now;
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
      size_t start_idx = trajectory_->computeTIndexNs(timestamps[i]).second;
      key_startid_now.push_back(start_idx);
    }

    if (marg_flag == MARGIN_OLD)
    {
      option.is_marg_state = true;
      option.ctrl_to_be_opt_now = key_startid_now[0];
      option.ctrl_to_be_opt_later = key_startid_now[1];
    }
    else if (marg_flag == MARGIN_SECOND_NEW)
    {
      option.is_marg_state = false;
    }

    TrajectoryEstimator::Ptr estimator(
        new TrajectoryEstimator(trajectory_, option));
    if (marg_flag == MARGIN_OLD)
    {
      /// [1] Prior
      if (last_marginalization_info)
      {
        std::vector<double *> drop_param_set;
        for (int i = option.ctrl_to_be_opt_now; i < option.ctrl_to_be_opt_later; ++i)
        {
          drop_param_set.emplace_back(trajectory_->getKnotSO3(i).data());
          drop_param_set.emplace_back(trajectory_->getKnotPos(i).data());
        }
        drop_param_set.emplace_back(para_bg_vec[0]);
        drop_param_set.emplace_back(para_ba_vec[0]);

        std::vector<int> drop_set;
        for (int j = 0; j < (int)last_marginalization_parameter_blocks.size(); j++)
        {
          for (auto const &drop_param : drop_param_set)
          {
            if (last_marginalization_parameter_blocks[j] == drop_param)
            {
              drop_set.emplace_back(j);
              break;
            }
          }
        }

        if (drop_set.size() > 0)
        {
          MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);

          estimator->PrepareMarginalizationInfo(
              RType_Prior, marginalization_factor, NULL,
              last_marginalization_parameter_blocks, drop_set);
        }
        else
        {
          std::cout << "============== last_marginalization_info drop_set.size() "
                       "> 0 =============\n";
        }
      }

      /// [2] Image
      int feature_index = -1;
      for (const auto &it_per_id : features)
      {
        if (!FeatureManager::isLandmarkCandidate(it_per_id))
          continue;
        ++feature_index;

        int idx_i = it_per_id.start_frame;
        int idx_j = it_per_id.start_frame - 1;

        int64_t ti = timestamps[idx_i];
        int rowi = std::round(it_per_id.feature_per_frame[0].uv(1));
        Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        bool marg_this_factor = false;
        if (it_per_id.start_frame == 0 && (*para_Feature[feature_index]) > 0)
          marg_this_factor = true;

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
          if (++idx_j == idx_i)
            continue;

          int64_t tj = timestamps[idx_j];
          int rowj = std::round(it_per_frame.uv(1));
          Eigen::Vector3d pts_j = it_per_frame.point;

          estimator->AddImageFeatureDelayAnalytic(ti, rowi, pts_i, tj, rowj, pts_j,
                                                  para_Feature[feature_index], &trajectory_->line_delay, false, marg_this_factor);
        }
      }

      /// [3] IMU
      for (const auto &v : imu_data_)
      {
        if (v.timestamp < opt_min_time)
          continue;
        if (v.timestamp >= timestamps[1])
        {
          break;
        }

        bool marg_this_factor = true;
        int bias_idx = 0;
        estimator->AddIMUMeasurementAnalytic(
            v, gravity_, para_bg_vec[bias_idx], para_ba_vec[bias_idx],
            opt_weight_.imu_info_vec, marg_this_factor);
      }

      /// [4] Bias
      {
        int i = 0;
        int j = i + 1;
        bool marg_this_factor = true;
        estimator->AddBiasFactor(para_bg_vec[i], para_bg_vec[j], para_ba_vec[i],
                                 para_ba_vec[j], 1, sqrt_info_bias[i],
                                 marg_this_factor);
      }
      for (int i = 0; i < WINDOW_SIZE; i++)
        sqrt_info_bias[i] = Eigen::Matrix<double, 6, 1>::Zero();

      estimator->SaveMarginalizationInfo(last_marginalization_info,
                                         last_marginalization_parameter_blocks);
    }
    else if (marg_flag == MARGIN_SECOND_NEW) // marginalize nothing
    {
      if (last_marginalization_info)
      {
        for (int i = 0; i < (int)last_marginalization_parameter_blocks.size(); i++)
        {
          ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_bg_vec[WINDOW_SIZE - 1]);
          ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_ba_vec[WINDOW_SIZE - 1]);
        }
      }
    }

    if (option.show_residual_summary)
      estimator->GetResidualSummary().PrintSummary();

    return true;
  }

  void TrajectoryManager::InitTrajectory(const Eigen::Vector3d &bg, const Eigen::Vector3d &ba)
  {
    /// only optimize newly added control points
    int64_t opt_min_time = max_bef_ns;
    int64_t opt_max_time = trajectory_->maxTimeNs();

    TrajectoryEstimatorOptions option;
    option.lock_ab = true;
    option.lock_wb = true;
    option.show_residual_summary = true;
    TrajectoryEstimator::Ptr estimator(new TrajectoryEstimator(trajectory_, option));

    double bg_param[3] = {bg(0), bg(1), bg(2)};
    double ba_param[3] = {ba(0), ba(1), ba(2)};
    for (const auto &v : imu_data_)
    {
      if (v.timestamp >= opt_max_time)
        break;
      if (v.timestamp < opt_min_time)
        continue;

      estimator->AddIMUMeasurementAnalytic(v, gravity_, bg_param, ba_param, opt_weight_.imu_info_vec);
    }

    estimator->SetFixedIndex(max_bef_idx);
    ceres::Solver::Summary summary = estimator->Solve(8, false);
    LOG(INFO) << summary.BriefReport();
  }

  bool TrajectoryManager::UpdateTrajectory(
      const std::list<FeaturePerId> &features, const int64_t timestamps[],
      IntegrationBase *pre_integrations[], Eigen::Vector3d Bgs[],
      Eigen::Vector3d Bas[], Eigen::VectorXd &feature_depths,
      MarginalizationFlag marg_flag, const int iteration)
  {
    // optimization range
    int64_t opt_min_time = trajectory_->computeTIndexNs(timestamps[0]).second * trajectory_->getDtNs();
    int64_t opt_max_time = trajectory_->maxTimeNs();

    int min_idx = trajectory_->computeTIndexNs(timestamps[0]).second;
    Eigen::Matrix3d R0 = trajectory_->getKnotSO3(min_idx).unit_quaternion().toRotationMatrix();
    Eigen::Vector3d t0 = trajectory_->getKnotPos(min_idx);

    // get bias in the window
    std::map<int, double *> para_bg_vec;
    std::map<int, double *> para_ba_vec;
    for (int i = 0; i <= WINDOW_SIZE; ++i)
    {
      auto &bias = all_imu_bias_[timestamps[i]];
      bias.gyro_bias = Bgs[i];
      bias.accel_bias = Bas[i];

      para_bg_vec[i] = bias.gyro_bias.data();
      para_ba_vec[i] = bias.accel_bias.data();
    }
    // get inv depth
    feature2double(feature_depths);

    TrajectoryEstimatorOptions option;
    option.lock_ab = false;
    option.lock_wb = false;
    option.show_residual_summary = true;
    TrajectoryEstimator::Ptr estimator(new TrajectoryEstimator(trajectory_, option));

    /// [1] Prior
    if (last_marginalization_info)
    {
      estimator->AddMarginalizationFactor(last_marginalization_info,
                                          last_marginalization_parameter_blocks);
    }

    /// [2] Image
    int feature_index = -1;
    for (const auto &it_per_id : features)
    {
      if (!FeatureManager::isLandmarkCandidate(it_per_id))
        continue;
      ++feature_index;

      int idx_i = it_per_id.start_frame;
      int idx_j = it_per_id.start_frame - 1;

      int64_t ti = timestamps[idx_i];
      int rowi = std::round(it_per_id.feature_per_frame[0].uv(1));
      Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point;

      for (auto &it_per_frame : it_per_id.feature_per_frame)
      {
        if (++idx_j == idx_i)
          continue;

        int64_t tj = timestamps[idx_j];
        int rowj = std::round(it_per_frame.uv(1));
        Eigen::Vector3d pts_j = it_per_frame.point;

        estimator->AddImageFeatureDelayAnalytic(ti, rowi, pts_i, tj, rowj, pts_j, para_Feature[feature_index], &trajectory_->line_delay, false);
      }
    }

    /// [3] IMU
    for (const auto &v : imu_data_)
    {
      if (v.timestamp >= opt_max_time)
        break;
      if (v.timestamp < opt_min_time)
        continue;

      int bias_idx = 0;
      if (v.timestamp < timestamps[0])
      {
        bias_idx = 0;
      }
      else if (v.timestamp >= timestamps[WINDOW_SIZE])
      {
        bias_idx = WINDOW_SIZE;
      }
      else
      {
        for (int i = 1; i <= WINDOW_SIZE; ++i)
        {
          if (v.timestamp >= timestamps[i - 1] && v.timestamp < timestamps[i])
          {
            bias_idx = i - 1;
            break;
          }
        }
      }

      estimator->AddIMUMeasurementAnalytic(v, gravity_, para_bg_vec[bias_idx], para_ba_vec[bias_idx], opt_weight_.imu_info_vec);
    }

    /// [4] Bias
    Eigen::Matrix<double, 6, 6> noise_covariance = Eigen::Matrix<double, 6, 6>::Zero();
    noise_covariance.block<3, 3>(0, 0) = (opt_weight_.imu_noise.sigma_wb_discrete * opt_weight_.imu_noise.sigma_wb_discrete) * Eigen::Matrix3d::Identity();
    noise_covariance.block<3, 3>(3, 3) = (opt_weight_.imu_noise.sigma_ab_discrete * opt_weight_.imu_noise.sigma_ab_discrete) * Eigen::Matrix3d::Identity();
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
      int j = i + 1;
      Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Zero();

      int64_t left_time = timestamps[i], right_time = timestamps[j];
      for (int imu_idx = 1; imu_idx < imu_data_.size(); imu_idx++)
      {
        if (imu_data_.at(imu_idx - 1).timestamp < left_time)
          continue;
        if (imu_data_.at(imu_idx).timestamp >= right_time)
          break;

        double dt = (imu_data_[imu_idx].timestamp - imu_data_[imu_idx - 1].timestamp) * NS_TO_S;
        Eigen::Matrix<double, 6, 6> F = Eigen::Matrix<double, 6, 6>::Zero();
        F.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        F.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
        Eigen::Matrix<double, 6, 6> G = Eigen::Matrix<double, 6, 6>::Zero();
        G.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * dt;
        G.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * dt;
        covariance = F * covariance * F.transpose() + G * noise_covariance * G.transpose();
      }

      Eigen::Matrix<double, 6, 6> sqrt_info_mat = Eigen::LLT<Eigen::Matrix<double, 6, 6>>(covariance.inverse()).matrixL().transpose();
      sqrt_info_bias[i] << sqrt_info_mat(0, 0), sqrt_info_mat(1, 1), sqrt_info_mat(2, 2), sqrt_info_mat(3, 3), sqrt_info_mat(4, 4), sqrt_info_mat(5, 5);

      estimator->AddBiasFactor(para_bg_vec[i], para_bg_vec[j], para_ba_vec[i],
                               para_ba_vec[j], 1, sqrt_info_bias[i]);
    }

    ceres::Solver::Summary summary = estimator->Solve(iteration, false);

    LOG(INFO) << summary.BriefReport();

    double2feature(feature_depths);
    for (int i = 0; i <= WINDOW_SIZE; ++i)
    {
      auto &bias = all_imu_bias_[timestamps[i]];
      Bgs[i] = bias.gyro_bias;
      Bas[i] = bias.accel_bias;
    }

    // 4DoF transform
    // very important!
    double2vector(timestamps, R0, t0);

    // marginalization
    UpdateVIOPrior(features, timestamps, pre_integrations, feature_depths, marg_flag, opt_min_time);

    int64_t remove_time = timestamps[0] - 5.0 * S_TO_NS;
    if (remove_time < 0)
      remove_time = 0;
    RemoveIMUData(remove_time);

    if (marg_flag == MARGIN_OLD)
      key_frame_time_vec_.push_back(timestamps[0]);

    LOG(INFO) << "[line_delay] " << trajectory_->line_delay;

    return true;
  }

  void TrajectoryManager::double2vector(const int64_t *timestamps,
                                        const Eigen::Matrix3d &R0,
                                        const Eigen::Vector3d &t0)
  {
    int64_t opt_min_time = trajectory_->computeTIndexNs(timestamps[0]).second * trajectory_->getDtNs();
    int min_idx = trajectory_->computeTIndexNs(timestamps[0]).second;
    Eigen::Matrix3d R00 = trajectory_->getKnotSO3(min_idx).unit_quaternion().toRotationMatrix();
    Eigen::Vector3d t00 = trajectory_->getKnotPos(min_idx);

    Eigen::Vector3d euler_R0 = R2ypr(R0);
    Eigen::Vector3d euler_R00 = R2ypr(R00);
    double y_diff = euler_R0.x() - euler_R00.x();

    Eigen::Matrix3d rot_diff;
    Eigen::Vector3d tran_diff;
    rot_diff = ypr2R(Vector3d(y_diff, 0, 0));
    if (abs(abs(euler_R0.y()) - 90) < 1.0 || abs(abs(euler_R00.y()) - 90) < 1.0)
    {
      // std::cout << RED << "euler singular point!" << RESET << std::endl;
      rot_diff = R0 * R00.transpose();
    }
    tran_diff = t0 - rot_diff * t00;

    SE3d SE3_Rt(rot_diff, tran_diff);

    int start_idx = trajectory_->computeTIndexNs(opt_min_time).second;
    int end_idx = trajectory_->cpnum() - 1;
    for (int i = start_idx; i <= end_idx; i++)
    {
      trajectory_->setKnot(SE3_Rt * trajectory_->getKnot(i), i);
    }
  }

  void TrajectoryManager::Log(std::string descri) const
  {
    IMUBias last_bias = all_imu_bias_.rbegin()->second;
    LOG(INFO) << descri << " Gyro Bias: " << last_bias.gyro_bias[0] << ","
              << last_bias.gyro_bias[1] << "," << last_bias.gyro_bias[2]
              << "; Acce Bias: " << last_bias.accel_bias[0] << ","
              << last_bias.accel_bias[1] << "," << last_bias.accel_bias[2];
    //  trajectory_->print_knots();
  }

} // namespace ctrlvio
