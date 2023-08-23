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

#include "visual_odometry.h"

namespace ctrlvio
{

  void VisualOdometry::ClearState()
  {
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
      Rs_[i].setIdentity();
      Ps_[i].setZero();
      timestamps_[i] = 0;
      pre_integrations_[i] = nullptr;
      Bas_[i].setZero();
      Bgs_[i].setZero();
    }

    PI_.setZero();
    VI_.setZero();
    RI_.setZero();
    Gravity_ = Eigen::Vector3d(0, 0, std::fabs(GRAVITY_NORM));

    frame_count_ = 0;
  }

  void VisualOdometry::InitWindow(VIOInitialization &vio_initialer,
                                  int64_t t_bag_start)
  {
    // state
    PI_ = vio_initialer.Ps[WINDOW_SIZE];
    VI_ = vio_initialer.Vs[WINDOW_SIZE];
    RI_ = vio_initialer.Rs[WINDOW_SIZE];

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
      vio_initialer.timestamps[i] -= t_bag_start * NS_TO_S;
    }

    const auto &EP_CtoI = trajectory_->GetSensorEP(CameraSensor);
    Eigen::Matrix3d R_CtoI = EP_CtoI.q.toRotationMatrix();
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
      if (i == 0)
        timestamps_[i] = 0;
      else
        timestamps_[i] = vio_initialer.timestamps[i] * S_TO_NS;

      Ps_[i] = vio_initialer.Rs[i] * EP_CtoI.p + vio_initialer.Ps[i];
      Rs_[i] = vio_initialer.Rs[i] * R_CtoI;
      Bgs_[i] = vio_initialer.Bgs[i];
    }

    for (double &t_imu : vio_initialer.imu_timestamps_buf)
    {
      t_imu -= t_bag_start * NS_TO_S;
    }
    int last_imu_idx = 0;
    const auto &t_imu_buf = vio_initialer.imu_timestamps_buf;
    const auto &a_imu_buf = vio_initialer.linear_acceleration_buf;
    const auto &w_imu_buf = vio_initialer.angular_velocity_buf;
    for (frame_count_ = 1; frame_count_ <= WINDOW_SIZE; ++frame_count_)
    {
      double t_last = 0;
      for (int i = last_imu_idx; i < (int)t_imu_buf.size(); ++i)
      {
        if (t_imu_buf[i] < vio_initialer.timestamps[frame_count_ - 1])
          continue;
        if (t_imu_buf[i] > vio_initialer.timestamps[frame_count_])
        {
          last_imu_idx = (i - 1) > 0 ? (i - 1) : 0;
          break;
        }
        // process the imu data between keyframe
        if (!pre_integrations_[frame_count_])
        {
          pre_integrations_[frame_count_] = new IntegrationBase(
              a_imu_buf[i], w_imu_buf[i], Bas_[frame_count_], Bgs_[frame_count_]);
          t_last = t_imu_buf[i];
        }
        else
        {
          double dt = t_imu_buf[i] - t_last;
          t_last = t_imu_buf[i];
          pre_integrations_[frame_count_]->push_back(dt, a_imu_buf[i],
                                                     w_imu_buf[i]);
        }
      }
    }

    // save imu data from second new frame
    for (int i = 0; i < (int)t_imu_buf.size(); ++i)
    {
      if (t_imu_buf[i] < vio_initialer.timestamps[WINDOW_SIZE - 1])
        continue;

      imu_timestamps_buf_.push_back(t_imu_buf[i]);
      linear_acceleration_buf_.push_back(a_imu_buf[i]);
      angular_velocity_buf_.push_back(w_imu_buf[i]);
    }

    // feature
    this->last_track_num = vio_initialer.f_manager.last_track_num;
    for (auto const &fea : vio_initialer.f_manager.feature)
    {
      this->feature.push_back(fea);
    }

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
      std::cout << "keyframe time " << i << ": " << timestamps_[i] * NS_TO_S << std::endl;
    }
    std::cout << "feature : " << this->feature.size() << std::endl;

    // others
    frame_count_ = WINDOW_SIZE;
  }

  void VisualOdometry::ProcessIMU(double timestamp,
                                  const Eigen::Vector3d &linear_acceleration,
                                  const Eigen::Vector3d &angular_velocity)
  {
    if (!pre_integrations_[frame_count_])
    {
      pre_integrations_[frame_count_] =
          new IntegrationBase(linear_acceleration, angular_velocity,
                              Bas_[frame_count_], Bgs_[frame_count_]);
    }
    else
    {
      double dt = timestamp - imu_timestamps_buf_.back();

      // LOG(INFO) << "pre_integrations: " << pre_integrations_[frame_count_]->sum_dt
      //           << ", " << dt;

      pre_integrations_[frame_count_]->push_back(dt, linear_acceleration,
                                                 angular_velocity);

      int j = frame_count_;
      Vector3d un_acc_0 =
          RI_ * (linear_acceleration_buf_.back() - Bas_[j]) - Gravity_;
      Vector3d un_gyr =
          0.5 * (angular_velocity_buf_.back() + angular_velocity) - Bgs_[j];
      RI_ *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
      Vector3d un_acc_1 = RI_ * (linear_acceleration - Bas_[j]) - Gravity_;
      Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
      PI_ += dt * VI_ + 0.5 * dt * dt * un_acc;
      VI_ += dt * un_acc;
    }

    imu_timestamps_buf_.push_back(timestamp);
    linear_acceleration_buf_.push_back(linear_acceleration);
    angular_velocity_buf_.push_back(angular_velocity);
  }

  void VisualOdometry::AddImageToWindow(
      const std::pair<int64_t, sensor_msgs::PointCloud::ConstPtr> &img_msg, const Trajectory::Ptr &trajectory, double line_delay)
  {
    std::map<int, std::vector<std::pair<int, Vector7D>>> image;
    FeatureMsg2Image(img_msg.second, image);

    if (this->addFeatureCheckParallax(frame_count_, image, 0.0))
      marg_flag_ = MARGIN_OLD;
    else
      marg_flag_ = MARGIN_SECOND_NEW;

    timestamps_[frame_count_] = img_msg.first - trajectory_->GetDataStartTime();

    // TODO
    const auto &EP_CtoI = trajectory_->GetSensorEP(CameraSensor);
    Ps_[frame_count_] = RI_ * EP_CtoI.p + PI_;
    Rs_[frame_count_] = RI_ * EP_CtoI.q.toRotationMatrix();
    this->triangulate(Ps_, Rs_);
    // this->triangulateRS(timestamps_, trajectory, line_delay);
  }

  void VisualOdometry::SlideWindow()
  {
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
      SE3d cam_pose = trajectory_->GetCameraPose(timestamps_[i]);
      Ps_[i] = cam_pose.translation();
      Rs_[i] = cam_pose.unit_quaternion().toRotationMatrix();
    }

    // Eigen::Matrix<double, 9, 1> predict_error;
    // {
    //   IMUState last_state;
    //   trajectory_->GetIMUState(timestamps_[WINDOW_SIZE - 1], last_state);
    //   predict_error.block<3, 1>(0, 0) = PI_ - last_state.p;
    //   predict_error.block<3, 1>(3, 0) = VI_ - last_state.v;
    //   predict_error.block<3, 1>(6, 0) =
    //       Utility::R2ypr(RI_) - Utility::R2ypr(last_state.q.toRotationMatrix());

    //   LOG(INFO) << "predict_error: " << predict_error.transpose();
    //   std::cout << "predict_error: " << predict_error.transpose() << std::endl;
    // }

    IMUState imu_state;
    trajectory_->GetIMUState(timestamps_[WINDOW_SIZE], imu_state);
    PI_ = imu_state.p;
    VI_ = imu_state.v;
    RI_ = imu_state.q.toRotationMatrix();

    TicToc t_margin;
    if (marg_flag_ == MARGIN_OLD)
    {
      marg_old_t0_ = timestamps_[0] * NS_TO_S;
      back_P0_ = Ps_[0];
      back_R0_ = Rs_[0];

      for (int i = 0; i < WINDOW_SIZE; i++)
      {
        Ps_[i].swap(Ps_[i + 1]);
        Rs_[i].swap(Rs_[i + 1]);
        timestamps_[i] = timestamps_[i + 1];
        Bas_[i].swap(Bas_[i + 1]);
        Bgs_[i].swap(Bgs_[i + 1]);
        std::swap(pre_integrations_[i], pre_integrations_[i + 1]);
      }
      Ps_[WINDOW_SIZE] = Ps_[WINDOW_SIZE - 1];
      Rs_[WINDOW_SIZE] = Rs_[WINDOW_SIZE - 1];
      timestamps_[WINDOW_SIZE] = timestamps_[WINDOW_SIZE - 1];
      Bas_[WINDOW_SIZE] = Bas_[WINDOW_SIZE - 1];
      Bgs_[WINDOW_SIZE] = Bgs_[WINDOW_SIZE - 1];

      Eigen::Vector3d acc_0 = linear_acceleration_buf_.back();
      Eigen::Vector3d gyr_0 = angular_velocity_buf_.back();
      delete pre_integrations_[WINDOW_SIZE];
      pre_integrations_[WINDOW_SIZE] =
          new IntegrationBase(acc_0, gyr_0, Bas_[WINDOW_SIZE], Bgs_[WINDOW_SIZE]);

      SlideWindowOld();
    }
    else
    {
      for (int i = 1; i < (int)imu_timestamps_buf_.size(); i++)
      {
        if (imu_timestamps_buf_[i] < timestamps_[WINDOW_SIZE - 1] * NS_TO_S)
          continue;
        double dt = imu_timestamps_buf_[i] - imu_timestamps_buf_[i - 1];
        Vector3d acc = linear_acceleration_buf_[i];
        Vector3d gyr = angular_velocity_buf_[i];
        pre_integrations_[WINDOW_SIZE - 1]->push_back(dt, acc, gyr);
      }

      Ps_[WINDOW_SIZE - 1] = Ps_[WINDOW_SIZE];
      Rs_[WINDOW_SIZE - 1] = Rs_[WINDOW_SIZE];
      timestamps_[WINDOW_SIZE - 1] = timestamps_[WINDOW_SIZE];
      Bas_[WINDOW_SIZE - 1] = Bas_[WINDOW_SIZE];
      Bgs_[WINDOW_SIZE - 1] = Bgs_[WINDOW_SIZE];

      Eigen::Vector3d acc_0 = linear_acceleration_buf_.back();
      Eigen::Vector3d gyr_0 = angular_velocity_buf_.back();
      delete pre_integrations_[WINDOW_SIZE];
      pre_integrations_[WINDOW_SIZE] =
          new IntegrationBase(acc_0, gyr_0, Bas_[WINDOW_SIZE], Bgs_[WINDOW_SIZE]);

      SlideWindowNew();
    }

    double t0 = imu_timestamps_buf_.back();
    Eigen::Vector3d acc_0 = linear_acceleration_buf_.back();
    Eigen::Vector3d gyr_0 = angular_velocity_buf_.back();

    imu_timestamps_buf_.clear();
    linear_acceleration_buf_.clear();
    angular_velocity_buf_.clear();

    imu_timestamps_buf_.emplace_back(t0);
    linear_acceleration_buf_.emplace_back(acc_0);
    angular_velocity_buf_.emplace_back(gyr_0);
  }

  void VisualOdometry::SlideWindowNew()
  {
    this->removeFailures();
    this->removeFront(frame_count_);
  }

  void VisualOdometry::SlideWindowOld()
  {
    this->removeFailures();

    Eigen::Matrix3d R0 = back_R0_;
    Eigen::Vector3d P0 = back_P0_;
    Eigen::Matrix3d R1 = Rs_[0];
    Eigen::Vector3d P1 = Ps_[0];
    this->removeBackShiftDepth(R0, P0, R1, P1);
  }

  VPointCloud VisualOdometry::GetLandmarksInWindow() const
  {
    VPointCloud landmarks;

    for (auto &it_per_id : this->feature)
    {
      if (!IsLandMarkStable(it_per_id))
        continue;

      int i = it_per_id.start_frame;
      Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
      Eigen::Vector3d w_pts_i = Rs_[i] * pts_i + Ps_[i];

      VPoint p;
      p.x = w_pts_i(0);
      p.y = w_pts_i(1);
      p.z = w_pts_i(2);
      if (it_per_id.lidar_depth_flag)
        p.intensity = 200;
      else
        p.intensity = 50;
      landmarks.push_back(p);
    }

    LOG(INFO) << "vio landmarks/feature in window: " << landmarks.points.size()
              << "/" << this->feature.size() << std::endl;
    return landmarks;
  }

  VPointCloud VisualOdometry::GetMarginCloud() const
  {
    VPointCloud margin_cloud;

    for (auto &it_per_id : this->feature)
    {
      if (!IsLandMarkStable(it_per_id))
        continue;

      int used_num = it_per_id.feature_per_frame.size();
      if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
        continue;

      if (it_per_id.start_frame == 0 && it_per_id.feature_per_frame.size() <= 2 &&
          it_per_id.solve_flag == SovelSucc)
      {
        int cam_i = it_per_id.start_frame;
        Eigen::Vector3d pts_i =
            it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
        Eigen::Vector3d w_pts_i = Rs_[cam_i] * pts_i + Ps_[cam_i];

        VPoint p;
        p.x = w_pts_i(0);
        p.y = w_pts_i(1);
        p.z = w_pts_i(2);
        p.intensity = 200;
        margin_cloud.points.push_back(p);
      }
    }

    LOG(INFO) << "vio next_marg/feature in window: " << margin_cloud.points.size()
              << "/" << this->feature.size() << std::endl;
    return margin_cloud;
  }

  bool VisualOdometry::FailureDetection() const
  {
    if (this->last_track_num < 2)
    {
      LOG(INFO) << " little feature " << this->last_track_num;
      // return true;
    }
    // if (Bas_[WINDOW_SIZE].norm() > 2.5) {
    //   LOG(INFO) << " big IMU acc bias estimation " << Bas_[WINDOW_SIZE].norm();
    //   return true;
    // }
    // if (Bgs_[WINDOW_SIZE].norm() > 1.0) {
    //   LOG(INFO) << " big IMU gyr bias estimation " << Bgs_[WINDOW_SIZE].norm();
    //   return true;
    // }
    // TODO add more situations
    return false;
  }

  void VisualOdometry::SaveDataForFailureDetection()
  {
    last_R_ = Rs_[WINDOW_SIZE];
    last_P_ = Ps_[WINDOW_SIZE];
    last_R0_ = Rs_[0];
    last_P0_ = Ps_[0];
  }

} // namespace ctrlvio
