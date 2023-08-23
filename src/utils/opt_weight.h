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

// #include <yaml-cpp/yaml.h>
#include <glog/logging.h>
#include <utils/yaml_utils.h>
#include <Eigen/Core>
#include <cmath>

namespace ctrlvio
{
  struct IMUNoise
  {
    IMUNoise() {}

    IMUNoise(const YAML::Node &node)
    {
      imu_frequency = yaml::GetValue<double>(node, "imu_frequency");
      imu_rate_gyro = yaml::GetValue<double>(node, "imu_info_vec_rate_gyro");
      imu_rate_accel = yaml::GetValue<double>(node, "imu_info_vec_rate_accel");

      sigma_w = yaml::GetValue<double>(node, "gyroscope_noise_density");
      sigma_wb = yaml::GetValue<double>(node, "gyroscope_random_walk");
      sigma_a = yaml::GetValue<double>(node, "accelerometer_noise_density");
      sigma_ab = yaml::GetValue<double>(node, "accelerometer_random_walk");

      sigma_w_2 = std::pow(sigma_w, 2);
      sigma_wb_2 = std::pow(sigma_wb, 2);
      sigma_a_2 = std::pow(sigma_a, 2);
      sigma_ab_2 = std::pow(sigma_ab, 2);

      print();
    }

    void print()
    {
      printf("IMU Noise:\n");
      printf("\t- gyroscope_noise_density: %.6f\n", sigma_w);
      printf("\t- accelerometer_noise_density: %.5f\n", sigma_a);
      printf("\t- gyroscope_random_walk: %.7f\n", sigma_wb);
      printf("\t- accelerometer_random_walk: %.6f\n", sigma_ab);
    }

    double imu_frequency = 200.;

    double imu_rate_gyro = 1.0;
    double imu_rate_accel = 1.0;

    /// Gyroscope white noise (rad/s/sqrt(hz))
    double sigma_w = 1.6968e-04;

    /// Gyroscope white noise covariance
    double sigma_w_2 = std::pow(1.6968e-04, 2);

    /// Gyroscope random walk (rad/s^2/sqrt(hz))
    double sigma_wb = 1.9393e-05;

    /// Gyroscope random walk covariance
    double sigma_wb_2 = std::pow(1.9393e-05, 2);

    /// Accelerometer white noise (m/s^2/sqrt(hz))
    double sigma_a = 2.0000e-3;

    /// Accelerometer white noise covariance
    double sigma_a_2 = std::pow(2.0000e-3, 2);

    /// Accelerometer random walk (m/s^3/sqrt(hz))
    double sigma_ab = 3.0000e-03;

    /// Accelerometer random walk covariance
    double sigma_ab_2 = std::pow(3.0000e-03, 2);

    double sigma_wb_discrete = 0;
    double sigma_ab_discrete = 0;
  };

  struct OptWeight
  {
    IMUNoise imu_noise;

    Eigen::Matrix<double, 6, 1> imu_info_vec;
    Eigen::Matrix<double, 6, 1> bias_info_vec;

    double local_velocity;
    Eigen::Vector3d local_velocity_info_vec;

    double relative_rotation;
    Eigen::Vector3d relative_rotation_info_vec;

    double image_weight;
    double pre_integration_weight;

    OptWeight() {}

    OptWeight(const YAML::Node &node) { LoadWeight(node); }

    void LoadWeight(const YAML::Node &node)
    {
      imu_noise = IMUNoise(node);

      double sqrt_dt = std::sqrt(1.0 / imu_noise.imu_frequency);
      Eigen::Vector3d one3d = Eigen::Vector3d::Ones();

      Eigen::Matrix<double, 12, 1> Q_sqrt_inv;

      Q_sqrt_inv.block<3, 1>(0, 0) = 1.0 / (imu_noise.sigma_w) * one3d;
      Q_sqrt_inv.block<3, 1>(3, 0) = 1.0 / (imu_noise.sigma_a) * one3d;
      imu_info_vec = Q_sqrt_inv.block<6, 1>(0, 0); 
      imu_noise.sigma_wb_discrete = imu_noise.sigma_wb;
      imu_noise.sigma_ab_discrete = imu_noise.sigma_ab;

      local_velocity = yaml::GetValue<double>(node, "local_velocity");
      local_velocity_info_vec = local_velocity * one3d;
      relative_rotation = yaml::GetValue<double>(node, "relative_rotation");
      relative_rotation_info_vec = relative_rotation * one3d;

      image_weight = yaml::GetValue<double>(node, "image_weight");

      pre_integration_weight = yaml::GetValue<double>(node, "pre_integration_weight");

      print();
    }

    inline double w2(double w) { return w * w; }

    void print()
    {
      printf("OPTIMIZATION WEIGHT:\n");
      printf("\t- gyr_weight: %.3f\n", w2(imu_info_vec(0)));
      printf("\t- acc_weight: %.3f\n", w2(imu_info_vec(3)));
      printf("\t- bias_gyr_weight(NO /sqrt_dt): %.3f\n", w2(bias_info_vec(0)));
      printf("\t- bias_acc_weight(NO /sqrt_dt): %.3f\n", w2(bias_info_vec(3)));
      printf("\t- local_velocity : %.3f\n", w2(local_velocity));
      printf("\t- relative_rotation: %.3f\n", w2(relative_rotation));
      printf("\t- image_weight: %.3f\n", w2(image_weight));
      printf("\n");
    }
  };

} // namespace ctrlvio
