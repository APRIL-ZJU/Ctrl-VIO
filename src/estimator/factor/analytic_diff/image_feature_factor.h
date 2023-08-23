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

#include <estimator/factor/analytic_diff/split_spline_view.h>

namespace ctrlvio
{
  namespace analytic_derivative
  {
    class ImageFeatureDelayFactor : public ceres::CostFunction,
                                    So3SplineView,
                                    RdSplineView
    {
    public:
      using SO3View = So3SplineView;
      using R3View = RdSplineView;

      using Vec3d = Eigen::Matrix<double, 3, 1>;
      using SO3d = Sophus::SO3<double>;

      ImageFeatureDelayFactor(const int64_t t_i, const int rowi, const Eigen::Vector3d &p_i,
                              const int64_t t_j, const int rowj, const Eigen::Vector3d &p_j,
                              const SplineMeta<SplineOrder> &spline_meta)
          : t_i_(t_i), rowi_(rowi), p_i_(p_i), t_j_(t_j), rowj_(rowj), p_j_(p_j), spline_meta_(spline_meta)
      {
        set_num_residuals(2);

        size_t kont_num = spline_meta.NumParameters();
        for (size_t i = 0; i < kont_num; ++i)
        {
          mutable_parameter_block_sizes()->push_back(4); // rotation control point
        }
        for (size_t i = 0; i < kont_num; ++i)
        {
          mutable_parameter_block_sizes()->push_back(3); // translation control point
        }
        mutable_parameter_block_sizes()->push_back(1); // inverse depth
        mutable_parameter_block_sizes()->push_back(1); // line delay
      }

      virtual bool Evaluate(double const *const *parameters, double *residuals,
                            double **jacobians) const
      {
        typename SO3View::JacobianStruct J_R[2];
        typename R3View::JacobianStruct J_p[2];

        size_t Knot_offset = 2 * spline_meta_.NumParameters();
        double d_inv = parameters[Knot_offset][0];
        double l_delay = parameters[Knot_offset + 1][0];
        int64_t l_delay_ns = l_delay * S_TO_NS; // "row * l_delay_ns >= 0.039 * S_TO_NS" is not permitted

        size_t kont_num = spline_meta_.NumParameters();
        // assert(spline_meta_.segments.size() == 2);
        // assert(t_i_ + rowi_ * l_delay_ns < spline_meta_.segments[0].MaxTimeNs());
        // assert(t_j_ + rowj_ * l_delay_ns < spline_meta_.segments[1].MaxTimeNs());

        size_t R_offset[2] = {0, 0};
        size_t P_offset[2] = {0, 0};
        size_t seg_idx[2] = {0, 0};
        {
          double u;
          spline_meta_.ComputeSplineIndex(t_i_ + rowi_ * l_delay_ns, R_offset[0], u);
          spline_meta_.ComputeSplineIndex(t_j_ + rowj_ * l_delay_ns, R_offset[1], u);

          size_t segment0_knot_num = spline_meta_.segments.at(0).NumParameters();
          for (int i = 0; i < 2; ++i)
          {
            if (R_offset[i] >= segment0_knot_num)
            {
              seg_idx[i] = 1;
              R_offset[i] = segment0_knot_num;
            }
            else
            {
              R_offset[i] = 0;
            }
            P_offset[i] = R_offset[i] + kont_num;
          }
        }

        /// compute at t_i
        Vec3d x_ci = p_i_ / d_inv;
        Vec3d p_Ii = S_CtoI * x_ci + p_CinI;
        SO3d S_IitoG;
        Vec3d p_IiinG = Vec3d::Zero();
        Vec3d Omega_Ii = Vec3d::Zero();
        Vec3d v_IiinG = Vec3d::Zero();
        if (jacobians)
        {
          Omega_Ii = SO3View::VelocityBody(t_i_ + rowi_ * l_delay_ns, spline_meta_.segments.at(seg_idx[0]),
                                           parameters + R_offset[0], nullptr);
          v_IiinG = R3View::velocity(t_i_ + rowi_ * l_delay_ns, spline_meta_.segments.at(seg_idx[0]),
                                     parameters + P_offset[0], nullptr);
          // rhs = p_Ii
          S_IitoG = SO3View::EvaluateRp(t_i_ + rowi_ * l_delay_ns, spline_meta_.segments.at(seg_idx[0]),
                                        parameters + R_offset[0], &J_R[0]);
          p_IiinG = R3View::evaluate(t_i_ + rowi_ * l_delay_ns, spline_meta_.segments.at(seg_idx[0]),
                                     parameters + P_offset[0], &J_p[0]);
        }
        else
        {
          S_IitoG = SO3View::EvaluateRp(t_i_ + rowi_ * l_delay_ns, spline_meta_.segments.at(seg_idx[0]),
                                        parameters + R_offset[0], nullptr);
          p_IiinG = R3View::evaluate(t_i_ + rowi_ * l_delay_ns, spline_meta_.segments.at(seg_idx[0]),
                                     parameters + P_offset[0], nullptr);
        }

        /// compute at t_j
        Vec3d p_G = S_IitoG * p_Ii + p_IiinG;
        SO3d S_GtoIj;
        Vec3d p_IjinG = Vec3d::Zero();
        Vec3d Omega_Ij = Vec3d::Zero();
        Vec3d v_IjinG = Vec3d::Zero();
        if (jacobians)
        {
          Omega_Ij = SO3View::VelocityBody(t_j_ + rowj_ * l_delay_ns, spline_meta_.segments.at(seg_idx[1]),
                                           parameters + R_offset[1], nullptr);
          v_IjinG = R3View::velocity(t_j_ + rowj_ * l_delay_ns, spline_meta_.segments.at(seg_idx[1]),
                                     parameters + P_offset[1], nullptr);
          // rhs = p_G - p_IjinG
          S_GtoIj = SO3View::EvaluateRTp(t_j_ + rowj_ * l_delay_ns, spline_meta_.segments.at(seg_idx[1]),
                                         parameters + R_offset[1], &J_R[1]);
          p_IjinG = R3View::evaluate(t_j_ + rowj_ * l_delay_ns, spline_meta_.segments.at(seg_idx[1]),
                                     parameters + P_offset[1], &J_p[1]);
        }
        else
        {
          S_GtoIj = SO3View::EvaluateRTp(t_j_ + rowj_ * l_delay_ns, spline_meta_.segments.at(seg_idx[1]),
                                         parameters + R_offset[1], nullptr);
          p_IjinG = R3View::evaluate(t_j_ + rowj_ * l_delay_ns, spline_meta_.segments.at(seg_idx[1]),
                                     parameters + P_offset[1], nullptr);
        }
        SO3d S_ItoC = S_CtoI.inverse();
        SO3d S_GtoCj = S_ItoC * S_GtoIj;
        Vec3d x_j = S_GtoCj * (p_G - p_IjinG) - S_ItoC * p_CinI;
        // Vec3d p_M =
        //     S_CtoI.inverse() * ((S_GtoIj * (p_G - p_IjinG)) - p_CinI);

        Eigen::Map<Eigen::Vector2d> residual(residuals);
        double depth_j_inv = 1.0 / x_j.z();
        residual = (x_j * depth_j_inv).head<2>() - p_j_.head<2>();

        if (jacobians)
        {
          for (size_t i = 0; i < kont_num; ++i)
          {
            if (jacobians[i])
            {
              Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> jac_kont_R(jacobians[i]);
              jac_kont_R.setZero();
            }
            if (jacobians[i + kont_num])
            {
              Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jac_kont_p(jacobians[i + kont_num]);
              jac_kont_p.setZero();
            }
          }
        }

        if (jacobians)
        {
          Eigen::Matrix<double, 2, 3> J_v;
          J_v.block<2, 2>(0, 0) = depth_j_inv * Eigen::Matrix2d::Identity();
          J_v.block<2, 1>(0, 2) = -depth_j_inv * depth_j_inv * x_j.head<2>();

          Eigen::Matrix<double, 2, 3> jac_lhs_R[2];
          Eigen::Matrix<double, 2, 3> jac_lhs_P[2];

          // jacobians related to t_i frame M is coincide with frame Cj)
          jac_lhs_R[0] = -J_v * (S_GtoCj * S_IitoG).matrix() * SO3::hat(p_Ii);
          jac_lhs_P[0] = J_v * S_GtoCj.matrix();

          // jacobians related to t_j
          jac_lhs_R[1] = J_v * S_GtoCj.matrix() * SO3::hat(p_G - p_IjinG);
          jac_lhs_P[1] = -J_v * S_GtoCj.matrix();

          /// [1] jacobians of control point
          for (int seg = 0; seg < 2; ++seg)
          {
            /// rotation control point
            size_t pre_idx_R = R_offset[seg] + J_R[seg].start_idx;
            for (size_t i = 0; i < SplineOrder; i++)
            {
              size_t idx = pre_idx_R + i;
              if (jacobians[idx])
              {
                Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> jac_kont_R(jacobians[idx]);
                Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_temp;
                /// 2*3 3*3
                J_temp = jac_lhs_R[seg] * J_R[seg].d_val_d_knot[i];
                J_temp = (sqrt_info * J_temp).eval();

                jac_kont_R.block<2, 3>(0, 0) += J_temp;
              }
            }

            /// translation control point
            size_t pre_idx_P = P_offset[seg] + J_p[seg].start_idx;
            for (size_t i = 0; i < SplineOrder; i++)
            {
              size_t idx = pre_idx_P + i;
              if (jacobians[idx])
              {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jac_kont_p(jacobians[idx]);

                Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_temp;
                /// 1*1 2*3
                J_temp = J_p[seg].d_val_d_knot[i] * jac_lhs_P[seg];
                J_temp = (sqrt_info * J_temp).eval();

                jac_kont_p += J_temp;
              }
            }
          }

          /// [2] jacobians of inverse depth
          if (jacobians[Knot_offset])
          {
            Eigen::Map<Eigen::Matrix<double, 2, 1>> jac_depth_inv(jacobians[Knot_offset]);
            jac_depth_inv.setZero();

            Vec3d J_Xm_d = -(S_GtoCj * S_IitoG * S_CtoI).matrix() * x_ci / d_inv;
            /// 2*3 3*1
            jac_depth_inv.block<2, 1>(0, 0) = J_v * J_Xm_d;
            jac_depth_inv = (sqrt_info * jac_depth_inv).eval();
          }

          /// [3] jacobians of line delay
          if (jacobians[Knot_offset + 1])
          {
            Eigen::Map<Eigen::Matrix<double, 2, 1>> jac_line_delay(jacobians[Knot_offset + 1]);
            jac_line_delay.setZero();

            Vec3d J_x = S_GtoIj * (rowi_ * v_IiinG - rowj_ * v_IjinG)                                 // for translation
                        + rowj_ * SO3::hat(Omega_Ij).transpose() * S_GtoIj.matrix() * (p_G - p_IjinG) // for R_j
                        + rowi_ * S_GtoIj.matrix() * S_IitoG.matrix() * SO3::hat(Omega_Ii) * p_Ii;    // for R_i
            J_x = (S_ItoC * J_x).eval();

            /// 2*3 3*1
            jac_line_delay.block<2, 1>(0, 0) = J_v * J_x;
            jac_line_delay = (sqrt_info * jac_line_delay).eval();
          }
        }

        residual = (sqrt_info * residual).eval();
        return true;
      }

      static inline Eigen::Matrix2d sqrt_info = 450. / 1.5 * Eigen::Matrix2d::Identity();

      static inline SO3d S_CtoI;
      static inline Vec3d p_CinI;

    private:
      int64_t t_i_;
      int rowi_;
      Eigen::Vector3d p_i_;
      int64_t t_j_;
      int rowj_;
      Eigen::Vector3d p_j_;

      SplineMeta<SplineOrder> spline_meta_;
    };

  } // namespace analytic_derivative

} // namespace ctrlvio
