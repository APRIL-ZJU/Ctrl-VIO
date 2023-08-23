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

#include <ceres/ceres.h>
#include <ceres/covariance.h>
#include <ceres/dynamic_cost_function.h>
#include <utils/ceres_callbacks.h>

#include <estimator/trajectory_estimator.h>

#include <memory>
#include <thread>

#include <iostream>
#include <variant>

namespace ctrlvio
{

  void ResidualSummary::AddResidualInfo(ResidualType r_type,
                                        const ceres::CostFunction *cost_function,
                                        const std::vector<double *> &param_vec)
  {
    int num_residuals = cost_function->num_residuals();
    Eigen::MatrixXd residuals;
    residuals.setZero(num_residuals, 1);
    cost_function->Evaluate(param_vec.data(), residuals.data(), nullptr);

    auto &error_sum = err_type_sum[r_type];
    // initial error as 0
    while ((int)error_sum.size() < num_residuals)
    {
      error_sum.push_back(0);
    }
    for (int i = 0; i < num_residuals; i++)
    {
      error_sum[i] += std::fabs(residuals(i, 0));
    }
    err_type_number[r_type]++;

    if (RType_PreIntegration == r_type)
    {
      auto &&log = COMPACT_GOOGLE_LOG_INFO;
      log.stream() << "imu_residuals :";
      for (int i = 0; i < num_residuals; i++)
      {
        log.stream() << std::fabs(residuals(i, 0)) << ", ";
      }
      log.stream() << "\n";
    }
  }

  void ResidualSummary::PrintSummary() const
  {
    if (err_type_sum.empty())
      return;

    auto &&log = COMPACT_GOOGLE_LOG_INFO;
    log.stream() << "ResidualSummary :" << descri_info << "\n";
    // look through every residual info
    for (auto typ = RType_Pose; typ <= RType_Prior; typ = ResidualType(typ + 1))
    {
      double num = err_type_number.at(typ);
      if (num > 0)
      {
        log.stream() << "\t- " << ResidualTypeStr[int(typ)] << ": num = " << num
                     << "; err_ave = ";

        auto &error_sum = err_type_sum.at(typ);
        for (int i = 0; i < (int)error_sum.size(); ++i)
        {
          log.stream() << error_sum[i] / num << ", ";
          if ((i + 1) % 10 == 0)
            log.stream() << "\n\t\t\t\t";
        }
        log.stream() << std::endl;
      }
    }
  }

  TrajectoryEstimator::TrajectoryEstimator(Trajectory::Ptr trajectory,
                                           TrajectoryEstimatorOptions &option)
      : trajectory_(trajectory), fixed_control_point_index_(-1)
  {
    this->options = option;
    problem_ = std::make_shared<ceres::Problem>(DefaultProblemOptions());

    analytic_local_parameterization_ =
        new LieAnalyticLocalParameterization<SO3d>();

    // if (option.is_marg_state) {
    marginalization_info_ = std::make_shared<MarginalizationInfo>();
    // } else {
    //   marginalization_info_ = nullptr;
    // }
  }

  void TrajectoryEstimator::AddControlPoints(
      const SplineMeta<SplineOrder> &spline_meta, std::vector<double *> &vec,
      bool addPosKnot)
  {
    for (auto const &seg : spline_meta.segments)
    {
      size_t start_idx = trajectory_->computeTIndexNs(seg.t0_ns).second;
      for (size_t i = start_idx; i < (start_idx + seg.NumParameters()); ++i)
      {
        if (addPosKnot)
        {
          vec.emplace_back(trajectory_->getKnotPos(i).data());
          problem_->AddParameterBlock(vec.back(), 3);
        }
        else
        {
          vec.emplace_back(trajectory_->getKnotSO3(i).data());
          problem_->AddParameterBlock(vec.back(), 4,
                                      analytic_local_parameterization_);
        }
        if (options.lock_traj || (fixed_control_point_index_ >= 0 &&
                                  i <= size_t(fixed_control_point_index_)))
        {
          problem_->SetParameterBlockConstant(vec.back());
        }
      }
    }
  }

  void TrajectoryEstimator::PrepareMarginalizationInfo(
      ResidualType r_type, ceres::CostFunction *cost_function,
      ceres::LossFunction *loss_function, std::vector<double *> &parameter_blocks,
      std::vector<int> &drop_set)
  {
    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
        r_type, cost_function, NULL, parameter_blocks, drop_set);
    marginalization_info_->addResidualBlockInfo(residual_block_info);
  }

  void TrajectoryEstimator::PrepareMarginalizationInfo(
      ResidualType r_type, const SplineMeta<SplineOrder> &spline_meta,
      ceres::CostFunction *cost_function, ceres::LossFunction *loss_function,
      std::vector<double *> &parameter_blocks,
      std::vector<int> &drop_set_wo_ctrl_point)
  {
    // add contrl point id to drop set
    std::vector<int> drop_set = drop_set_wo_ctrl_point;
    if (options.ctrl_to_be_opt_later > options.ctrl_to_be_opt_now)
    {
      std::vector<int> ctrl_id;
      trajectory_->GetCtrlIdxs(spline_meta, ctrl_id);
      for (int i = 0; i < (int)ctrl_id.size(); ++i)
      {
        if (ctrl_id[i] < options.ctrl_to_be_opt_later)
        {
          drop_set.emplace_back(i);
          drop_set.emplace_back(i + spline_meta.NumParameters());
        }
      }
    }

    // 对之后的优化没有约束的因子直接丢就行,因为留下来也没有约束作用
    // if (drop_set.size() > 0)
    {
      ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
          r_type, cost_function, loss_function, parameter_blocks, drop_set);
      marginalization_info_->addResidualBlockInfo(residual_block_info);
    }
  }

  void TrajectoryEstimator::SaveMarginalizationInfo(
      MarginalizationInfo::Ptr &marg_info_out,
      std::vector<double *> &marg_param_blocks_out)
  {
    // prepare the schur complement
    marginalization_info_->preMarginalize();
    bool ret = marginalization_info_->marginalize();

    if (ret)
    {
      marg_info_out = marginalization_info_;
      marg_param_blocks_out = marginalization_info_->getParameterBlocks();
    }
    else
    {
      marg_info_out = nullptr;
      marg_param_blocks_out.clear();
    }

    marginalization_info_ = std::make_shared<MarginalizationInfo>();
  }

  bool TrajectoryEstimator::IsParamUpdated(const double *values) const
  {
    if (problem_->HasParameterBlock(values) &&
        !problem_->IsParameterBlockConstant(const_cast<double *>(values)))
    {
      return true;
    }
    else
    {
      return false;
    }
  }

  void TrajectoryEstimator::AddIMUMeasurementAnalytic(
      const IMUData &imu_data, const Eigen::Vector3d &gravity, double *gyro_bias,
      double *accel_bias, const Eigen::Matrix<double, 6, 1> &info_vec,
      bool marg_this_factor)
  {
    SplineMeta<SplineOrder> spline_meta;
    trajectory_->CaculateSplineMeta({{imu_data.timestamp, imu_data.timestamp}},
                                    spline_meta);
    ceres::CostFunction *cost_function = new analytic_derivative::IMUFactor(
        imu_data, spline_meta.segments.at(0), gravity, info_vec);

    std::vector<double *> vec;
    AddControlPoints(spline_meta, vec);
    AddControlPoints(spline_meta, vec, true);
    vec.emplace_back(gyro_bias);
    vec.emplace_back(accel_bias);

    if (options.lock_wb)
    {
      problem_->AddParameterBlock(gyro_bias, 3);
      problem_->SetParameterBlockConstant(gyro_bias);
    }
    if (options.lock_ab)
    {
      problem_->AddParameterBlock(accel_bias, 3);
      problem_->SetParameterBlockConstant(accel_bias);
    }

    problem_->AddResidualBlock(cost_function, NULL, vec);

    if (options.is_marg_state && marg_this_factor)
    {
      std::vector<int> drop_set_wo_ctrl_point;
      int Knot_size = 2 * spline_meta.NumParameters();
      drop_set_wo_ctrl_point.emplace_back(Knot_size);     // gyro_bias
      drop_set_wo_ctrl_point.emplace_back(Knot_size + 1); // accel_bias
      PrepareMarginalizationInfo(RType_IMU, spline_meta, cost_function, NULL, vec,
                                 drop_set_wo_ctrl_point);
    }

    if (options.show_residual_summary)
    {
      residual_summary_.AddResidualInfo(RType_IMU, cost_function, vec);
    }
  }

  void TrajectoryEstimator::AddBiasFactor(
      double *bias_gyr_i, double *bias_gyr_j, double *bias_acc_i,
      double *bias_acc_j, double dt, const Eigen::Matrix<double, 6, 1> &info_vec,
      bool marg_this_factor)
  {
    analytic_derivative::BiasFactor *cost_function =
        new analytic_derivative::BiasFactor(dt, info_vec);

    std::vector<double *> vec;
    vec.emplace_back(bias_gyr_i);
    vec.emplace_back(bias_gyr_j);
    vec.emplace_back(bias_acc_i);
    vec.emplace_back(bias_acc_j);
    problem_->AddResidualBlock(cost_function, NULL, vec);

    if (options.is_marg_state && marg_this_factor)
    {
      // bias_gyr_i,bias_acc_i
      std::vector<int> drop_set = {0, 2};
      PrepareMarginalizationInfo(RType_Bias, cost_function, NULL, vec, drop_set);
    }

    if (options.show_residual_summary)
    {
      residual_summary_.AddResidualInfo(RType_Bias, cost_function, vec);
    }
  }

  void TrajectoryEstimator::AddImageFeatureDelayAnalytic(const int64_t ti, const int rowi, const Eigen::Vector3d &pi,
                                                         const int64_t tj, const int rowj, const Eigen::Vector3d &pj,
                                                         double *inv_depth, double *line_delay, bool fixed_depth,
                                                         bool marg_this_fearure)
  {
    SplineMeta<SplineOrder> spline_meta;
    trajectory_->CaculateSplineMeta({{ti, ti + 0.039 * S_TO_NS}, {tj, tj + 0.039 * S_TO_NS}}, // t_padding（ld_upper * image_height < 0.039）（0.039 < 0.04）
                                    spline_meta);

    using Functor = analytic_derivative::ImageFeatureDelayFactor;
    ceres::CostFunction *cost_function = new Functor(ti, rowi, pi, tj, rowj, pj, spline_meta);

    std::vector<double *> vec;
    AddControlPoints(spline_meta, vec);
    AddControlPoints(spline_meta, vec, true);
    vec.emplace_back(inv_depth);
    vec.emplace_back(line_delay);

    problem_->AddParameterBlock(line_delay, 1);
    if (trajectory_->fix_ld)
      problem_->SetParameterBlockConstant(line_delay);
    else
    {
      problem_->SetParameterLowerBound(line_delay, 0, trajectory_->ld_lower);
      problem_->SetParameterUpperBound(line_delay, 0, trajectory_->ld_upper);
    }

    ceres::LossFunction *loss_function;
    double cauchy_loss = marg_this_fearure ? 1 : 2;
    loss_function = new ceres::CauchyLoss(cauchy_loss);
    problem_->AddResidualBlock(cost_function, loss_function, vec);

    if (options.is_marg_state && marg_this_fearure)
    {
      std::vector<int> drop_set_wo_ctrl_point;
      drop_set_wo_ctrl_point.emplace_back(vec.size() - 2);
      PrepareMarginalizationInfo(RType_Image, spline_meta, cost_function,
                                 loss_function, vec, drop_set_wo_ctrl_point);
    }
  }

  void TrajectoryEstimator::AddMarginalizationFactor(
      MarginalizationInfo::Ptr last_marginalization_info,
      std::vector<double *> &last_marginalization_parameter_blocks)
  {
    MarginalizationFactor *marginalization_factor =
        new MarginalizationFactor(last_marginalization_info);
    problem_->AddResidualBlock(marginalization_factor, NULL,
                               last_marginalization_parameter_blocks);

    if (options.show_residual_summary)
    {
      residual_summary_.AddResidualInfo(RType_Prior, marginalization_factor,
                                        last_marginalization_parameter_blocks);
    }
  }

  void TrajectoryEstimator::AddCallback(
      const std::vector<std::string> &descriptions,
      const std::vector<size_t> &block_size, std::vector<double *> &param_block)
  {
    // Add callback for debug
    std::unique_ptr<CheckStateCallback> cb =
        std::make_unique<CheckStateCallback>();
    for (int i = 0; i < (int)block_size.size(); ++i)
    {
      cb->addCheckState(descriptions[i], block_size[i], param_block[i]);
    }

    callbacks_.push_back(std::move(cb));
    // If any callback requires state, the flag must be set
    callback_needs_state_ = true;
  }

  ceres::Solver::Summary TrajectoryEstimator::Solve(int max_iterations,
                                                    bool progress,
                                                    int num_threads)
  {
    ceres::Solver::Options options;

    options.minimizer_type = ceres::TRUST_REGION;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = progress;
    options.max_num_iterations = num_threads;

    if (num_threads < 1)
    {
      num_threads = 1; // std::thread::hardware_concurrency(); // mine is 8
    }
    options.num_threads = num_threads;
    options.max_num_iterations = max_iterations;

    if (callbacks_.size() > 0)
    {
      for (auto &cb : callbacks_)
      {
        options.callbacks.push_back(cb.get());
      }

      if (callback_needs_state_)
        options.update_state_every_iteration = true;
    }

    ceres::Solver::Summary summary;
    ceres::Solve(options, problem_.get(), &summary);

    trajectory_->UpdateExtrinsics();

    if (this->options.show_residual_summary)
    {
      residual_summary_.PrintSummary();
    }

    return summary;
  }

} // namespace ctrlvio
