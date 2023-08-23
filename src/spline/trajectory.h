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

#include <glog/logging.h>
#include "../utils/parameter_struct.h"
#include <utils/mypcl_cloud_type.h>
#include "se3_spline.h"

namespace ctrlvio
{

  enum SensorType
  {
    IMUSensor = 0, //  qurey pose and EP
    CameraSensor
  };

  class TrajectoryManager;

  class Trajectory : public Se3Spline<SplineOrder, double>
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef std::shared_ptr<Trajectory> Ptr;

    Trajectory(double time_interval, double start_time = 0)
        : Se3Spline<SplineOrder, double>(time_interval * S_TO_NS, start_time * S_TO_NS),
          data_start_time_(-1),
          line_delay(0),
          linedelay_init(0)
    {
      this->extendKnotsTo(start_time, SO3d(Eigen::Quaterniond::Identity()), Eigen::Vector3d(0, 0, 0));
      std::cout << GREEN << "[init maxTime] " << this->maxTimeNs() * NS_TO_S << RESET << std::endl;
    }

    void SetLineDelay(double ld_init_, bool fix_ld_, double ld_lower_, double ld_upper_)
    {
      line_delay = ld_init_;
      linedelay_init = ld_init_;
      fix_ld = fix_ld_;
      ld_lower = ld_lower_;
      ld_upper = ld_upper_;
    }

    void SetSensorExtrinsics(const SensorType type,
                             const ExtrinsicParam &EP_StoI)
    {
      EP_StoI_[type] = EP_StoI;
    }

    ExtrinsicParam &GetSensorEP(const SensorType type)
    {
      return EP_StoI_.at(type);
    }

    std::map<SensorType, ExtrinsicParam> &GetSensorEPs() { return EP_StoI_; };

    void UpdateExtrinsics()
    {
      for (auto &sensor_EP : EP_StoI_)
      {
        auto &EP = sensor_EP.second;
        EP.se3.so3() = EP.so3;
        EP.se3.translation() = EP.p;
        EP.q = EP.so3.unit_quaternion();
      }
    }

    void GetIMUState(int64_t time, IMUState &imu_state) const;

    SE3d GetCameraPose(const int64_t timestamp) const
    {
      return GetSensorPose(timestamp, EP_StoI_.at(CameraSensor));
    }

    void SetDataStartTime(int64_t time) { data_start_time_ = time; }

    int64_t GetDataStartTime() { return data_start_time_; }

    double line_delay;
    double linedelay_init;
    bool fix_ld;
    double ld_lower;
    double ld_upper;

    std::string bag_name;

  protected:
    SE3d GetSensorPose(const int64_t time_ns,
                       const ExtrinsicParam &EP_StoI) const;

  private:
    int64_t data_start_time_;

    std::map<SensorType, ExtrinsicParam> EP_StoI_;

    friend TrajectoryManager;
  };

} // namespace ctrlvio
