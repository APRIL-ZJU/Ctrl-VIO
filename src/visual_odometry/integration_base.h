#pragma once

#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include <sophus_lib/so3.hpp>

#include <utils/parameter_struct.h>
#include "parameters.h"
#include "utility.h"

namespace ctrlvio
{

    using namespace Eigen;

    enum StateOrder
    {
        O_P = 0, // just for pre-integration factor
        O_R = 3,
        O_V = 6,
        O_BA = 9,
        O_BG = 12
    };

    enum NoiseOrder
    {
        O_AN = 0, // just for pre-integration factor
        O_GN = 3,
        O_AW = 6,
        O_GW = 9
    };

    class IntegrationBase
    {
    public:
        IntegrationBase() = delete;

        IntegrationBase(const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                        const Eigen::Vector3d &_linearized_ba,
                        const Eigen::Vector3d &_linearized_bg)
            : sum_dt{0.0},
              acc_0{_acc_0},
              gyr_0{_gyr_0},
              linearized_acc{_acc_0},
              linearized_gyr{_gyr_0},
              delta_p{Eigen::Vector3d::Zero()},
              delta_q{Eigen::Quaterniond::Identity()},
              delta_v{Eigen::Vector3d::Zero()},
              linearized_ba{_linearized_ba},
              linearized_bg{_linearized_bg},
              jacobian{Eigen::Matrix<double, 15, 15>::Identity()},
              covariance{Eigen::Matrix<double, 15, 15>::Zero()}
        {
            noise_covariance = Eigen::Matrix<double, 18, 18>::Zero();
            noise_covariance.block<3, 3>(0, 0) =
                (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
            noise_covariance.block<3, 3>(3, 3) =
                (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
            noise_covariance.block<3, 3>(6, 6) =
                (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
            noise_covariance.block<3, 3>(9, 9) =
                (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
            noise_covariance.block<3, 3>(12, 12) =
                (ACC_W * ACC_W) * Eigen::Matrix3d::Identity();
            noise_covariance.block<3, 3>(15, 15) =
                (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();
        }

        void push_back(double dt, const Eigen::Vector3d &acc,
                       const Eigen::Vector3d &gyr)
        {
            dt_buf.push_back(dt);
            acc_buf.push_back(acc);
            gyr_buf.push_back(gyr);
            propagate(dt, acc, gyr); 
        }

        void repropagate(const Eigen::Vector3d &_linearized_ba,
                         const Eigen::Vector3d &_linearized_bg)
        {
            sum_dt = 0.0;
            acc_0 = linearized_acc;
            gyr_0 = linearized_gyr;
            delta_p.setZero();
            delta_q.setIdentity();
            delta_v.setZero();
            linearized_ba = _linearized_ba;
            linearized_bg = _linearized_bg;
            jacobian.setIdentity();
            covariance.setZero();
            for (int i = 0; i < static_cast<int>(dt_buf.size()); i++)
                propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
        }

        void midPointIntegration(
            // INPUT
            double _dt, const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
            const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
            const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q,
            const Eigen::Vector3d &delta_v, const Eigen::Vector3d &linearized_ba,
            const Eigen::Vector3d &linearized_bg, // OUTPUT
            Eigen::Vector3d &result_delta_p, Eigen::Quaterniond &result_delta_q,
            Eigen::Vector3d &result_delta_v, Eigen::Vector3d &result_linearized_ba,
            Eigen::Vector3d &result_linearized_bg, bool update_jacobian)
        {
            Vector3d un_gyr = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
            result_delta_q = delta_q * Utility::deltaQ(un_gyr * _dt);

            Vector3d un_acc_0 = delta_q * (_acc_0 - linearized_ba);
            Vector3d un_acc_1 = result_delta_q * (_acc_1 - linearized_ba);
            Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

            result_delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * _dt * _dt;
            result_delta_v = delta_v + un_acc * _dt;

            result_linearized_ba = linearized_ba;
            result_linearized_bg = linearized_bg;

            if (update_jacobian)
            {
                Vector3d w_x = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
                Vector3d a_0_x = _acc_0 - linearized_ba;
                Vector3d a_1_x = _acc_1 - linearized_ba;
                Matrix3d R_w_x, R_a_0_x, R_a_1_x;

                R_w_x << 0, -w_x(2), w_x(1), w_x(2), 0, -w_x(0), -w_x(1), w_x(0), 0;
                R_a_0_x << 0, -a_0_x(2), a_0_x(1), a_0_x(2), 0, -a_0_x(0), -a_0_x(1),
                    a_0_x(0), 0;
                R_a_1_x << 0, -a_1_x(2), a_1_x(1), a_1_x(2), 0, -a_1_x(0), -a_1_x(1),
                    a_1_x(0), 0;

                Matrix3d MatI = Matrix3d::Identity();
                Matrix3d dR_last = delta_q.toRotationMatrix();
                Matrix3d dR_now = result_delta_q.toRotationMatrix();

                MatrixXd F = MatrixXd::Zero(15, 15);
                F.block<3, 3>(0, 0) = MatI;
                F.block<3, 3>(0, 3) =
                    -0.25 * dR_last * R_a_0_x * _dt * _dt +
                    -0.25 * dR_now * R_a_1_x * (MatI - R_w_x * _dt) * _dt * _dt;
                F.block<3, 3>(0, 6) = MatI * _dt;
                F.block<3, 3>(0, 9) = -0.25 * (dR_last + dR_now) * _dt * _dt;
                F.block<3, 3>(0, 12) = 0.25 * dR_now * R_a_1_x * _dt * _dt * _dt;
                F.block<3, 3>(3, 3) = MatI - R_w_x * _dt;
                F.block<3, 3>(3, 12) = -1.0 * MatI * _dt;
                F.block<3, 3>(6, 3) =
                    -0.5 * dR_last * R_a_0_x * _dt +
                    -0.5 * dR_now * R_a_1_x * (MatI - R_w_x * _dt) * _dt;
                F.block<3, 3>(6, 6) = MatI;
                F.block<3, 3>(6, 9) = -0.5 * (dR_last + dR_now) * _dt;
                F.block<3, 3>(6, 12) = 0.5 * dR_now * R_a_1_x * _dt * _dt;
                F.block<3, 3>(9, 9) = MatI;
                F.block<3, 3>(12, 12) = MatI;

                MatrixXd G = MatrixXd::Zero(15, 18);
                G.block<3, 3>(0, 0) = 0.25 * dR_last * _dt * _dt;
                G.block<3, 3>(0, 3) = -0.125 * dR_now * R_a_1_x * _dt * _dt * _dt;
                G.block<3, 3>(0, 6) = 0.25 * dR_now * _dt * _dt;
                G.block<3, 3>(0, 9) = G.block<3, 3>(0, 3);
                G.block<3, 3>(3, 3) = 0.5 * MatI * _dt;
                G.block<3, 3>(3, 9) = 0.5 * MatI * _dt;
                G.block<3, 3>(6, 0) = 0.5 * dR_last * _dt;
                G.block<3, 3>(6, 3) = -0.25 * dR_now * R_a_1_x * _dt * _dt;
                G.block<3, 3>(6, 6) = 0.5 * dR_now * _dt;
                G.block<3, 3>(6, 9) = G.block<3, 3>(6, 3);
                G.block<3, 3>(9, 12) = MatI * _dt;
                G.block<3, 3>(12, 15) = MatI * _dt;

                jacobian = F * jacobian;

                covariance =
                    F * covariance * F.transpose() + G * noise_covariance * G.transpose();
            }
        }

        void propagate(double _dt, const Eigen::Vector3d &_acc_1,
                       const Eigen::Vector3d &_gyr_1)
        {
            dt = _dt;
            acc_1 = _acc_1;
            gyr_1 = _gyr_1;

            Vector3d result_delta_p;
            Quaterniond result_delta_q;
            Vector3d result_delta_v;
            Vector3d result_linearized_ba;
            Vector3d result_linearized_bg;

            midPointIntegration(dt, acc_0, gyr_0, acc_1, gyr_1, delta_p, delta_q,
                                delta_v, linearized_ba, linearized_bg, result_delta_p,
                                result_delta_q, result_delta_v, result_linearized_ba,
                                result_linearized_bg, 1);

            delta_p = result_delta_p;
            delta_q = result_delta_q;
            delta_v = result_delta_v;
            linearized_ba = result_linearized_ba;
            linearized_bg = result_linearized_bg;
            sum_dt += dt;
            acc_0 = acc_1;
            gyr_0 = gyr_1;
        }

        template <typename T>
        Eigen::Matrix<T, 15, 1> evaluate333(
            const Eigen::Matrix<T, 3, 1> &Pi, const Sophus::SO3<T> &Si,
            const Eigen::Matrix<T, 3, 1> &Vi, const Eigen::Matrix<T, 3, 1> &Bai,
            const Eigen::Matrix<T, 3, 1> &Bgi, const Eigen::Matrix<T, 3, 1> &Pj,
            const Sophus::SO3<T> &Sj, const Eigen::Matrix<T, 3, 1> &Vj,
            const Eigen::Matrix<T, 3, 1> &Baj, const Eigen::Matrix<T, 3, 1> &Bgj)
        {
            using Vec3T = Eigen::Matrix<T, 3, 1>;
            using Mat3T = Eigen::Matrix<T, 3, 3>;
            using SO3T = Sophus::SO3<T>;

            Eigen::Matrix<T, 15, 15> jac = jacobian.template cast<T>();
            Mat3T dp_dba = jac.template block<3, 3>(O_P, O_BA);
            Mat3T dp_dbg = jac.template block<3, 3>(O_P, O_BG);

            Mat3T dq_dbg = jac.template block<3, 3>(O_R, O_BG);

            Mat3T dv_dba = jac.template block<3, 3>(O_V, O_BA);
            Mat3T dv_dbg = jac.template block<3, 3>(O_V, O_BG);

            Vec3T dba = Bai - linearized_ba.template cast<T>();
            Vec3T dbg = Bgi - linearized_bg.template cast<T>();

            Vec3T corrected_delta_p =
                delta_p.template cast<T>() + dp_dba * dba + dp_dbg * dbg;
            Eigen::Quaternion<T> corrected_delta_q =
                delta_q.template cast<T>() * SO3T::exp(dq_dbg * dbg).unit_quaternion();
            Vec3T corrected_delta_v =
                delta_v.template cast<T>() + dv_dba * dba + dv_dbg * dbg;

            SO3T S_j_to_i(corrected_delta_q);
            Vec3T res_log = (S_j_to_i.inverse() * Si.inverse() * Sj).log();

            Vec3T gravity;
            gravity << T(0), T(0), T(GRAVITY_NORM);

            Eigen::Matrix<T, 15, 1> residuals;
            residuals.template block<3, 1>(O_P, 0) =
                Si.inverse() *
                    (T(0.5 * sum_dt * sum_dt) * gravity + Pj - Pi - Vi * T(sum_dt)) -
                corrected_delta_p;
            residuals.template block<3, 1>(O_R, 0) = res_log;
            residuals.template block<3, 1>(O_V, 0) =
                Si.inverse() * (gravity * T(sum_dt) + Vj - Vi) - corrected_delta_v;
            residuals.template block<3, 1>(O_BA, 0) = Baj - Bai;
            residuals.template block<3, 1>(O_BG, 0) = Bgj - Bgi;

            return residuals;
        }

        double dt;
        double sum_dt;
        Eigen::Vector3d acc_0, gyr_0;
        Eigen::Vector3d acc_1, gyr_1;

        const Eigen::Vector3d linearized_acc, linearized_gyr;
        std::vector<double> dt_buf;
        std::vector<Eigen::Vector3d> acc_buf;
        std::vector<Eigen::Vector3d> gyr_buf;

        Eigen::Vector3d delta_p;
        Eigen::Quaterniond delta_q;
        Eigen::Vector3d delta_v;
        Eigen::Vector3d linearized_ba;
        Eigen::Vector3d linearized_bg;

        Eigen::Matrix<double, 15, 15> jacobian;     
        Eigen::Matrix<double, 15, 15> covariance;       
        Eigen::Matrix<double, 18, 18> noise_covariance; 
    };
} // namespace ctrlvio