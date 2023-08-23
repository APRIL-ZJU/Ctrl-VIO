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
#include <ceres/covariance.h>
#include <estimator/factor/ceres_local_param.h>
#include <utils/parameter_struct.h>

#include "trajectory_estimator_options.h"
#include "visual_odometry/integration_base.h"

#include <estimator/factor/analytic_diff/image_feature_factor.h>
#include <estimator/factor/analytic_diff/marginalization_factor.h>
#include <estimator/factor/analytic_diff/trajectory_value_factor.h>

namespace ctrlvio
{

  struct ResidualSummary
  {
    std::map<ResidualType, std::vector<double>> err_type_sum;
    std::map<ResidualType, int> err_type_number;

    std::string descri_info;

    ResidualSummary(std::string descri = "") : descri_info(descri)
    {
      for (auto typ = RType_Pose; typ <= RType_Prior;
           typ = ResidualType(typ + 1))
      {
        err_type_sum[typ].push_back(0);
        err_type_number[typ] = 0;
      }
    }

    void AddResidualInfo(ResidualType r_type,
                         const ceres::CostFunction *cost_function,
                         const std::vector<double *> &param_vec);

    void PrintSummary() const;
  };

  class TrajectoryEstimator
  {
    static ceres::Problem::Options DefaultProblemOptions()
    {
      ceres::Problem::Options options;
      options.loss_function_ownership = ceres::TAKE_OWNERSHIP;
      options.local_parameterization_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
      return options;
    }

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef std::shared_ptr<TrajectoryEstimator> Ptr;

    TrajectoryEstimator(Trajectory::Ptr trajectory,
                        TrajectoryEstimatorOptions &option);

    ~TrajectoryEstimator()
    {
      // 手动删除new的变量
      delete analytic_local_parameterization_;
      // delete marginalization_info_;
    }

    /// 固定这帧之前的控制点
    void SetKeyScanConstant(double max_time);

    /// 直接指定固定控制点的索引
    void SetFixedIndex(int idx) { fixed_control_point_index_ = idx; }

    // [factor] 添加位置测量(起始位姿，固定轨迹的起始点)
    void AddStartTimePose(const PoseData &pose);

    // [factor] 轨迹固定因子
    void AddStaticSegment(const std::vector<size_t> &static_ctrl_segment);

    // [factor] 添加位置测量
    void AddPoseMeasurementAnalytic(const PoseData &pose_data,
                                    const Eigen::Matrix<double, 6, 1> &info_vec);
    // [factor] 添加IMU测量
    void AddIMUMeasurementAnalytic(const IMUData &imu_data,
                                   const Eigen::Vector3d &gravity,
                                   double *gyro_bias, double *accel_bias,
                                   const Eigen::Matrix<double, 6, 1> &info_vec,
                                   bool marg_this_factor = false);

    // [factor] 添加bias因子
    void AddBiasFactor(double *bias_gyr_i, double *bias_gyr_j, double *bias_acc_i,
                       double *bias_acc_j, double dt,
                       const Eigen::Matrix<double, 6, 1> &info_vec,
                       bool marg_this_factor = false);

    // [factor] 添加预积分因子 ti(ta), tj(tb)
    void AddPreIntegrationAnalytic(int64_t ti, int64_t tj,
                                   IntegrationBase *pre_integration,
                                   double *gyro_bias_i, double *gyro_bias_j,
                                   double *accel_bias_i, double *accel_bias_j,
                                   bool marg_this_factor = false);

    // [factor] 图像重投影因子
    void AddImageFeatureAnalytic(const double ti, const Eigen::Vector3d &pi,
                                 const double tj, const Eigen::Vector3d &pj,
                                 double *inv_depth, bool fixed_depth = false,
                                 bool marg_this_fearure = false);

    // [factor] 图像重投影因子
    void AddImageFeatureDelayAnalytic(const int64_t ti, const int rowi, const Eigen::Vector3d &pi,
                                      const int64_t tj, const int rowj, const Eigen::Vector3d &pj,
                                      double *inv_depth, double *line_delay, bool fixed_depth = false,
                                      bool marg_this_fearure = false);

    void AddDelayAnalytic(double *line_delay);

    void AddImageFeatureDelayTestAnalytic(const double ti, const int rowi, const Eigen::Vector3d &pi,
                                          const double tj, const int rowj, const Eigen::Vector3d &pj,
                                          double *inv_depth, double *line_delay, bool fixed_depth = false,
                                          bool marg_this_fearure = false);

    void AddImageFeatureNew(const double ti, const int rowi, const Eigen::Vector3d &pi,
                            const double tj, const int rowj, const Eigen::Vector3d &pj,
                            double *inv_depth, bool fixed_depth = false,
                            bool marg_this_fearure = false);

    // [factor] 先验因子
    void AddMarginalizationFactor(
        MarginalizationInfo::Ptr last_marginalization_info,
        std::vector<double *> &last_marginalization_parameter_blocks);

    void AddCallback(const std::vector<std::string> &descriptions,
                     const std::vector<size_t> &block_size,
                     std::vector<double *> &param_block);

    ceres::Solver::Summary Solve(int max_iterations = 50, bool progress = false,
                                 int num_threads = -1);

    // 为边缘化做准备
    void PrepareMarginalizationInfo(ResidualType r_type,
                                    ceres::CostFunction *cost_function,
                                    ceres::LossFunction *loss_function,
                                    std::vector<double *> &parameter_blocks,
                                    std::vector<int> &drop_set);

    // 保存先验信息
    void SaveMarginalizationInfo(MarginalizationInfo::Ptr &marg_info_out,
                                 std::vector<double *> &marg_param_blocks_out);

    const ResidualSummary &GetResidualSummary() const
    {
      return residual_summary_;
    }

    std::shared_ptr<ceres::Problem> problem_;

  private:
    void AddControlPoints(const SplineMeta<SplineOrder> &spline_meta,
                          std::vector<double *> &vec, bool addPosKnot = false);

    // 为边缘化做准备
    void PrepareMarginalizationInfo(ResidualType r_type,
                                    const SplineMeta<SplineOrder> &spline_meta,
                                    ceres::CostFunction *cost_function,
                                    ceres::LossFunction *loss_function,
                                    std::vector<double *> &parameter_blocks,
                                    std::vector<int> &drop_set_wo_ctrl_point);

    bool IsParamUpdated(const double *values) const;

  private:
    TrajectoryEstimatorOptions options;
    Trajectory::Ptr trajectory_;

    // std::shared_ptr<ceres::Problem> problem_;
    ceres::LocalParameterization *analytic_local_parameterization_;

    int fixed_control_point_index_;

    // Marginalization
    MarginalizationInfo::Ptr marginalization_info_;

    // for debug
    ResidualSummary residual_summary_;

    bool callback_needs_state_;
    std::vector<std::unique_ptr<ceres::IterationCallback>> callbacks_;
  };

} // namespace ctrlvio
