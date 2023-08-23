#pragma once
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <cstdlib>
#include <deque>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

struct SFMFeature
{
  int id;     // feature_id
  bool state; 

  std::vector<std::pair<int, Eigen::Vector2d>> observation;

  Eigen::Vector3d position;
};

struct ReprojectionError3D
{
  ReprojectionError3D(double observed_u, double observed_v)
      : observed_u(observed_u), observed_v(observed_v) {}

  template <typename T>
  bool operator()(const T *const camera_R, const T *const camera_T,
                  const T *point, T *residuals) const
  {
    T p[3];
    ceres::QuaternionRotatePoint(camera_R, point, p);
    p[0] += camera_T[0];
    p[1] += camera_T[1];
    p[2] += camera_T[2];
    T xp = p[0] / p[2];
    T yp = p[1] / p[2];
    residuals[0] = xp - T(observed_u);
    residuals[1] = yp - T(observed_v);
    return true;
  }

  static ceres::CostFunction *Create(const double observed_x,
                                     const double observed_y)
  {
    return (new ceres::AutoDiffCostFunction<ReprojectionError3D, 2, 4, 3, 3>(
        new ReprojectionError3D(observed_x, observed_y)));
  }

  double observed_u;
  double observed_v;
};

class GlobalSFM
{
public:
  GlobalSFM() {}

  bool construct(const int frame_num, const int ref_frame_idx,
                 const int cur_fixed_idx, std::vector<SFMFeature> &sfm_f,
                 Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[]) const;

  /**
   * @brief Uses sfm to get initial estimate for the poses and landmarks
   *
   * @param frame_num current frame idx in sliding window
   * @param l previous frame idx which contians enough correspondance and
   * parallex with current frame
   * @param relative_R rotation from frame [frame_num] to frame [l]
   * @param relative_T position from frame [frame_num] to frame [l]
   * @param q_out rotation from camera to global
   * @param T_out position of camera in global frame
   * @param sfm_f all measurements and estiamted position of landmarks
   * @param sfm_tracked_points map between landmark ID to position estimate
   * @return Returns false if it fails to sfm
   */
  bool construct_orignal(
      int frame_num, int l, const Eigen::Matrix3d relative_R,
      const Eigen::Vector3d relative_T, Eigen::Quaterniond q_out[],
      Eigen::Vector3d T_out[], std::vector<SFMFeature> &sfm_f,
      std::map<int, Eigen::Vector3d> &sfm_tracked_points) const;

private:
  bool solveFrameByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial,
                       int i, std::vector<SFMFeature> &sfm_f) const;

  void triangulatePoint(const Eigen::Matrix<double, 3, 4> &Pose0,
                        const Eigen::Matrix<double, 3, 4> &Pose1,
                        const Eigen::Vector2d &point0,
                        const Eigen::Vector2d &point1,
                        Eigen::Vector3d &point_3d) const;

  void triangulateTwoFrames(const int frame0,
                            const Eigen::Matrix<double, 3, 4> &Pose0,
                            const int frame1,
                            const Eigen::Matrix<double, 3, 4> &Pose1,
                            std::vector<SFMFeature> &sfm_f) const;
};