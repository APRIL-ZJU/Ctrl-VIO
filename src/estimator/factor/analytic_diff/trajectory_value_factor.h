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

#include <ceres/ceres.h>
#include <spline/spline_segment.h>
#include <utils/parameter_struct.h>

#include "split_spline_view.h"
#include "visual_odometry/integration_base.h"

namespace ctrlvio
{

  namespace analytic_derivative
  {

    // bias_gyr_i, bias_gyr_j, bias_acc_i, bias_acc_j
    class BiasFactor : public ceres::SizedCostFunction<6, 3, 3, 3, 3>
    {
    public:
      BiasFactor(double dt, const Eigen::Matrix<double, 6, 1> &sqrt_info)
      {
        double sqrt_dt = std::sqrt(dt);
        sqrt_info_.setZero();
        sqrt_info_.diagonal() = sqrt_info / sqrt_dt;
      }
      virtual bool Evaluate(double const *const *parameters, double *residuals,
                            double **jacobians) const
      {
        using Vec3d = Eigen::Matrix<double, 3, 1>;
        using Vec6d = Eigen::Matrix<double, 6, 1>;
        Eigen::Map<Vec3d const> bias_gyr_i(parameters[0]);
        Eigen::Map<Vec3d const> bias_gyr_j(parameters[1]);
        Eigen::Map<Vec3d const> bias_acc_i(parameters[2]);
        Eigen::Map<Vec3d const> bias_acc_j(parameters[3]);

        Vec6d res;
        res.block<3, 1>(0, 0) = bias_gyr_j - bias_gyr_i;
        res.block<3, 1>(3, 0) = bias_acc_j - bias_acc_i;

        Eigen::Map<Vec6d> residual(residuals);
        residual = sqrt_info_ * res;

        if (jacobians)
        {
          if (jacobians[0])
          {
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> jac_bg_i(
                jacobians[0]);
            jac_bg_i.setZero();
            jac_bg_i.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
            jac_bg_i.applyOnTheLeft(sqrt_info_);
          }
          if (jacobians[1])
          {
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> jac_bg_j(
                jacobians[1]);
            jac_bg_j.setZero();
            jac_bg_j.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
            jac_bg_j.applyOnTheLeft(sqrt_info_);
          }
          if (jacobians[2])
          {
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> jac_ba_i(
                jacobians[2]);
            jac_ba_i.setZero();
            jac_ba_i.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
            jac_ba_i.applyOnTheLeft(sqrt_info_);
          }
          if (jacobians[3])
          {
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> jac_ba_j(
                jacobians[3]);
            jac_ba_j.setZero();
            jac_ba_j.block<3, 3>(3, 0) = Eigen::Matrix3d::Identity();
            jac_ba_j.applyOnTheLeft(sqrt_info_);
          }
        }

        return true;
      }

    private:
      Eigen::Vector3d acc_i_, acc_j_;
      Eigen::Vector3d gyr_i_, gyr_j_;
      Eigen::Matrix<double, 6, 6> sqrt_info_;
    };

    class IMUFactor : public ceres::CostFunction, SplitSpineView
    {
    public:
      using SO3View = So3SplineView;
      using R3View = RdSplineView;
      using SplitView = SplitSpineView;

      using Vec3d = Eigen::Matrix<double, 3, 1>;
      using Vec6d = Eigen::Matrix<double, 6, 1>;
      using SO3d = Sophus::SO3<double>;

      IMUFactor(const IMUData &imu_data,
                const SplineSegmentMeta<SplineOrder> &spline_segment_meta,
                const Vec3d &gravity, const Vec6d &info_vec)
          : imu_data_(imu_data),
            spline_segment_meta_(spline_segment_meta),
            gravity_(gravity),
            info_vec_(info_vec)
      {
        set_num_residuals(6);

        size_t knot_num = this->spline_segment_meta_.NumParameters();
        for (size_t i = 0; i < knot_num; ++i)
        {
          mutable_parameter_block_sizes()->push_back(4);
        }
        for (size_t i = 0; i < knot_num; ++i)
        {
          mutable_parameter_block_sizes()->push_back(3);
        }
        mutable_parameter_block_sizes()->push_back(3); // gyro bias
        mutable_parameter_block_sizes()->push_back(3); // accel bias
      }

      virtual bool Evaluate(double const *const *parameters, double *residuals,
                            double **jacobians) const
      {
        typename SO3View::JacobianStruct J_rot_w;
        typename SO3View::JacobianStruct J_rot_a;
        typename R3View::JacobianStruct J_pos;

        typename SplitView::SplineIMUData spline_data;

        if (jacobians)
        {
          spline_data =
              SplitView::Evaluate(imu_data_.timestamp, spline_segment_meta_,
                                  parameters, gravity_, &J_rot_w, &J_rot_a, &J_pos);
        }
        else
        {
          spline_data = SplitView::Evaluate(
              imu_data_.timestamp, spline_segment_meta_, parameters, gravity_);
        }
        size_t knot_num = this->spline_segment_meta_.NumParameters();
        Eigen::Map<Vec3d const> gyro_bias(parameters[2 * knot_num]);
        Eigen::Map<Vec3d const> accel_bias(parameters[2 * knot_num + 1]);

        Eigen::Map<Vec6d> residual(residuals);
        residual.block<3, 1>(0, 0) =
            spline_data.gyro - (imu_data_.gyro - gyro_bias);
        residual.block<3, 1>(3, 0) =
            spline_data.accel - (imu_data_.accel - accel_bias);

        residual = (info_vec_.asDiagonal() * residual).eval();

        if (!jacobians)
        {
          return true;
        }

        if (jacobians)
        {
          for (size_t i = 0; i < knot_num; ++i)
          {
            if (jacobians[i])
            {
              Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> jac_kont_R(
                  jacobians[i]);
              jac_kont_R.setZero();
            }
            if (jacobians[i + knot_num])
            {
              Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> jac_kont_p(
                  jacobians[i + knot_num]);
              jac_kont_p.setZero();
            }
          }
        }

        /// rotation control point
        for (size_t i = 0; i < knot_num; i++)
        {
          size_t idx = i;
          if (jacobians[idx])
          {
            Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> jac_knot_R(
                jacobians[idx]);
            jac_knot_R.setZero();
            /// for gyro residual
            jac_knot_R.block<3, 3>(0, 0) = J_rot_w.d_val_d_knot[i];
            /// for accel residual
            jac_knot_R.block<3, 3>(3, 0) = J_rot_a.d_val_d_knot[i];

            jac_knot_R = (info_vec_.asDiagonal() * jac_knot_R).eval();
          }
        }

        /// translation control point
        for (size_t i = 0; i < knot_num; i++)
        {
          size_t idx = knot_num + i;
          if (jacobians[idx])
          {
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> jac_knot_p(
                jacobians[idx]);
            jac_knot_p.setZero();

            /// for accel residual
            jac_knot_p.block<3, 3>(3, 0) =
                J_pos.d_val_d_knot[i] * spline_data.R_inv.matrix();
            jac_knot_p = (info_vec_.asDiagonal() * jac_knot_p).eval();
          }
        }

        /// bias
        if (jacobians[2 * knot_num])
        {
          Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> jac_bw(
              jacobians[2 * knot_num]);
          Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> jac_ba(
              jacobians[2 * knot_num + 1]);

          jac_bw.setZero();
          jac_bw.block<3, 3>(0, 0).diagonal() = info_vec_.head(3);

          jac_ba.setZero();
          jac_ba.block<3, 3>(3, 0).diagonal() = info_vec_.tail(3);
        }

        return true;
      }

      IMUData imu_data_;
      SplineSegmentMeta<SplineOrder> spline_segment_meta_;
      Vec3d gravity_;
      Vec6d info_vec_;
    };

  } // namespace analytic_derivative

} // namespace ctrlvio
