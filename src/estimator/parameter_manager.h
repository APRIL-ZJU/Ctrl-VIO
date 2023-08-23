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

#include <utils/parameter_struct.h>
#include <utils/yaml_utils.h>

namespace ctrlvio
{

  class ParamManager
  {
  public:
    typedef std::shared_ptr<ParamManager> Ptr;

    ParamManager(const YAML::Node &node)
    {
      if (node["CameraExtrinsics"])
      {
        EP_CtoI.Init(node["CameraExtrinsics"]);
      }
    }

    void SetSystemState(const IMUState &imu_state) { sys_state = imu_state; }

  public:
    SystemState sys_state;

    ExtrinsicParam EP_CtoI;
  };

} // namespace ctrlvio
