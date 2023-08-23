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

#include <utils/yaml_utils.h>

#include <utils/parameter_struct.h>
#include <utils/eigen_utils.hpp>

namespace ctrlvio
{

  class IMUInitializer
  {
  public:
    typedef std::shared_ptr<IMUInitializer> Ptr;

    IMUInitializer(const YAML::Node &node);

    void FeedIMUData(const IMUData &imu_data);

    const Eigen::aligned_vector<IMUData> &GetIMUData() const
    {
      return imu_datas_;
    }

    bool InitialIMUState();

    bool StaticInitialIMUState();

    bool ActiveInitialIMUState();

    bool InitialDone() const { return initial_done_; }

    const IMUState GetIMUState() const { return imu_state_; } // I0toG

  private:
    IMUState imu_state_;

    Eigen::Vector3d gravity_;

    double window_length_;

    double imu_excite_threshold_;

    Eigen::aligned_vector<IMUData> imu_datas_;

    bool initial_done_;
  };

} // namespace ctrlvio
