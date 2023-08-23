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

#include <estimator/factor/analytic_diff/rd_spline_view.h>
#include <estimator/factor/analytic_diff/so3_spline_view.h>

namespace ctrlvio
{
  namespace analytic_derivative
  {

    struct SplitSpineView : public So3SplineView, public RdSplineView
    {
      static constexpr int N = SplineOrder; // Order of the spline.
      static constexpr int DEG = N - 1;     // Degree of the spline.
      static constexpr int DIM = 3;         // Dimension of euclidean vector space.

      using MatN = Eigen::Matrix<double, N, N>;
      using VecN = Eigen::Matrix<double, N, 1>;
      using Vec3 = Eigen::Matrix<double, 3, 1>;
      using Mat3 = Eigen::Matrix<double, 3, 3>;

      using VecD = Eigen::Matrix<double, DIM, 1>;
      using MatD = Eigen::Matrix<double, DIM, DIM>;

      using SO3 = Sophus::SO3<double>;

      using SO3View = So3SplineView;
      using R3View = RdSplineView;

      // struct SplineIMUData
      // {
      //   double timestamp;
      //   Eigen::Vector3d gyro;
      //   VecD accel;
      //   SO3 R_inv;
      // };
      struct SplineIMUData
      {
        int64_t time_ns;
        Eigen::Vector3d gyro;
        VecD accel;
        SO3 R_inv;
        size_t start_idx;
      };

      SplitSpineView() {}

      static SplineIMUData Evaluate(
          const int64_t time_ns, const SplineSegmentMeta<N> &splne_meta,
          double const *const *knots, const Vec3 &gravity,
          typename SO3View::JacobianStruct *J_rot_w = nullptr,
          typename SO3View::JacobianStruct *J_rot_a = nullptr,
          typename R3View::JacobianStruct *J_pos = nullptr)
      {
        SplineIMUData spline_imu_data;
        spline_imu_data.time_ns = time_ns;

        std::pair<double, size_t> ui = splne_meta.computeTIndexNs(time_ns);
        size_t s = ui.second;
        double u = ui.first;

        size_t R_knot_offset = s;
        size_t P_knot_offset = s + splne_meta.NumParameters();
        spline_imu_data.start_idx = s;

        VecN Up, lambda_a;
        R3View::template baseCoeffsWithTimeR3<2>(Up, u);
        lambda_a = splne_meta.pow_inv_dt[2] * R3View::blending_matrix_ * Up;

        VecN Ur, lambda_R;
        SO3View::template baseCoeffsWithTime<0>(Ur, u);
        lambda_R = SO3View::blending_matrix_ * Ur;

        VecN Uw, lambda_w;
        SO3View::template baseCoeffsWithTime<1>(Uw, u);
        lambda_w = splne_meta.pow_inv_dt[1] * SO3View::blending_matrix_ * Uw;

        VecD accelerate;
        accelerate.setZero();

        if (J_pos)
          J_pos->start_idx = s;
        for (int i = 0; i < N; i++)
        {
          Eigen::Map<VecD const> p(knots[P_knot_offset + i]);
          accelerate += lambda_a[i] * p;

          if (J_pos)
            J_pos->d_val_d_knot[i] = lambda_a[i];
        }

        Vec3 d_vec[DEG];       // d_1, d_2, d_3
        SO3 A_rot_inv[DEG];    // A_1_inv, A_2_inv, A_3_inv
        SO3 A_accum_inv;       // A_3_inv * A2_inv * A1_inv
        Mat3 A_post_inv[N];    // A_3_inv*A2_inv*A1_inv, A_3_inv*A2_inv, A_3_inv, I
        Mat3 Jr_dvec_inv[DEG]; // Jr_inv(d1), Jr_inv(d2), Jr_inv(d3)
        Mat3 Jr_kdelta[DEG];   // Jr(-kd1), Jr(-kd2), Jr(-kd3)

        A_post_inv[N - 1] = A_accum_inv.matrix(); // Identity Matrix
        /// 2 1 0
        for (int i = DEG - 1; i >= 0; i--)
        {
          Eigen::Map<SO3 const> R0(knots[R_knot_offset + i]);
          Eigen::Map<SO3 const> R1(knots[R_knot_offset + i + 1]);

          d_vec[i] = (R0.inverse() * R1).log();

          Vec3 k_delta = lambda_R[i + 1] * d_vec[i];
          A_rot_inv[i] = Sophus::SO3d::exp(-k_delta);
          A_accum_inv *= A_rot_inv[i];

          if (J_rot_w || J_rot_a)
          {
            A_post_inv[i] = A_accum_inv.matrix();

            Sophus::rightJacobianInvSO3(d_vec[i], Jr_dvec_inv[i]);
            Sophus::rightJacobianSO3(-k_delta, Jr_kdelta[i]);
          }
        }

        /// Omega(j)
        Vec3 omega[N]; // w(1), w(2), w(3), w(4)
        {
          omega[0] = Vec3::Zero();
          for (int i = 0; i < DEG; i++)
          {
            omega[i + 1] = A_rot_inv[i] * omega[i] + lambda_w[i + 1] * d_vec[i];
          }
          spline_imu_data.gyro = omega[3];

          Eigen::Map<SO3 const> Ri(knots[R_knot_offset]);
          SO3 R_inv = A_accum_inv * Ri.inverse();

          spline_imu_data.accel = R_inv * (accelerate + gravity);
          spline_imu_data.R_inv = R_inv;
        }

        if (J_rot_w)
        {
          J_rot_w->start_idx = s;
          for (int i = 0; i < N; i++)
          {
            J_rot_w->d_val_d_knot[i].setZero();
          }

          // d(omega) / d(d_j)
          Mat3 d_omega_d_delta[DEG]; // w(4)/d1, w(4)/d2, w(4)/d3
          d_omega_d_delta[0] = lambda_w[1] * A_post_inv[1];
          for (int i = 1; i < DEG; i++)
          {
            d_omega_d_delta[i] = lambda_R[i + 1] * A_post_inv[i] *
                                     SO3::hat(omega[i]) * Jr_kdelta[i] +
                                 lambda_w[i + 1] * A_post_inv[i + 1];
          }

          for (int i = 0; i < DEG; i++)
          {
            J_rot_w->d_val_d_knot[i] -=
                d_omega_d_delta[i] * Jr_dvec_inv[i].transpose();
            J_rot_w->d_val_d_knot[i + 1] += d_omega_d_delta[i] * Jr_dvec_inv[i];
          }
        }

        if (J_rot_a)
        {
          // for accelerate jacobian
          Mat3 R_accum[DEG - 1]; // R_i, R_i*A_1, R_i*A_1*A_2
          Eigen::Map<SO3 const> R0(knots[R_knot_offset]);
          R_accum[0] = R0.matrix();
          /// 1 2
          for (int i = 1; i < DEG; i++)
          {
            R_accum[i] = R_accum[i - 1] * A_rot_inv[i - 1].matrix().transpose();
          }

          J_rot_a->start_idx = s;
          for (int i = 0; i < N; i++)
          {
            J_rot_a->d_val_d_knot[i].setZero();
          }

          Mat3 lhs =
              spline_imu_data.R_inv.matrix() * SO3::hat(accelerate + gravity);
          J_rot_a->d_val_d_knot[0] += lhs * R_accum[0];
          for (int i = 0; i < DEG; i++)
          {
            Mat3 d_a_d_delta = lambda_R[i + 1] * lhs * R_accum[i] * Jr_kdelta[i];

            J_rot_a->d_val_d_knot[i] -= d_a_d_delta * Jr_dvec_inv[i].transpose();
            J_rot_a->d_val_d_knot[i + 1] += d_a_d_delta * Jr_dvec_inv[i];
          }
        }

        return spline_imu_data;
      }
    };

  }; // namespace analytic_derivative

} // namespace ctrlvio