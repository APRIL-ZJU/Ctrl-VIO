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

#include "trajectory.h"
#include <fstream>
#include <ros/ros.h>

namespace ctrlvio
{

  void Trajectory::GetIMUState(int64_t time, IMUState &imu_state) const
  {
    SE3d pose = this->poseNs(time);

    imu_state.timestamp = time;
    imu_state.q = pose.unit_quaternion();
    imu_state.p = pose.translation();
    imu_state.v = this->transVelWorld(time);
    // imu_state.bias;
    // imu_state.g;
  }

  SE3d Trajectory::GetSensorPose(const int64_t time_ns,
                                 const ExtrinsicParam &EP_StoI) const
  {
    int64_t t = time_ns;
    if (!(t >= this->minTimeNs() && t < this->maxTimeNs()))
    {
      LOG(WARNING) << t << "; not in [" << this->minTimeNs() << ", "
                   << this->maxTimeNs() << ")";
      //    t = this->maxTime() - 1e-9;
    }
    assert((t >= this->minTimeNs() && t < this->maxTimeNs()) &&
           "[GetSensorPose] querry time not in range.");

    SE3d pose_I_to_G = this->poseNs(t);
    SE3d pose_S_to_G = pose_I_to_G * EP_StoI.se3;
    return pose_S_to_G;
  }

} // namespace ctrlvio
