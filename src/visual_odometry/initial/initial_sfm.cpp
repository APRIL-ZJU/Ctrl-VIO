#include "initial_sfm.h"

using namespace std;
using namespace Eigen;

void GlobalSFM::triangulatePoint(const Eigen::Matrix<double, 3, 4> &Pose0,
                                 const Eigen::Matrix<double, 3, 4> &Pose1,
                                 const Vector2d &point0, const Vector2d &point1,
                                 Vector3d &point_3d) const
{
  Matrix4d design_matrix = Matrix4d::Zero();
  design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
  design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
  design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
  design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
  Vector4d triangulated_point =
      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
  point_3d(0) = triangulated_point(0) / triangulated_point(3);
  point_3d(1) = triangulated_point(1) / triangulated_point(3);
  point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

bool GlobalSFM::solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i,
                                vector<SFMFeature> &sfm_f) const
{
  vector<cv::Point2f> pts_2_vector;
  vector<cv::Point3f> pts_3_vector;

  for (int j = 0; j < (int)sfm_f.size(); j++)
  {
    if (sfm_f[j].state != true)
      continue;
    Vector2d point2d;
    for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
    {
      if (sfm_f[j].observation[k].first == i)
      {
        Vector2d img_pts = sfm_f[j].observation[k].second;
        cv::Point2f pts_2(img_pts(0), img_pts(1));
        pts_2_vector.push_back(pts_2);
        cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1],
                          sfm_f[j].position[2]);
        pts_3_vector.push_back(pts_3);
        break;
      }
    }
  }

  if (int(pts_2_vector.size()) < 15)
  {
    printf("unstable features tracking, please slowly move you device!\n");
    if (int(pts_2_vector.size()) < 10)
      return false;
  }

  cv::Mat r, rvec, t, D, tmp_r;
  cv::eigen2cv(R_initial, tmp_r);
  cv::Rodrigues(tmp_r, rvec);
  cv::eigen2cv(P_initial, t);
  cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);

  bool pnp_succ;
  pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
  if (!pnp_succ)
  {
    return false;
  }

  cv::Rodrigues(rvec, r);
  MatrixXd R_pnp;
  cv::cv2eigen(r, R_pnp);
  MatrixXd T_pnp;
  cv::cv2eigen(t, T_pnp);
  R_initial = R_pnp;
  P_initial = T_pnp;
  return true;
}

void GlobalSFM::triangulateTwoFrames(const int frame0,
                                     const Eigen::Matrix<double, 3, 4> &Pose0,
                                     const int frame1,
                                     const Eigen::Matrix<double, 3, 4> &Pose1,
                                     vector<SFMFeature> &sfm_f) const
{
  if (frame0 == frame1)
  {
    std::cout << "frame0 " << frame0 << "; frame1: " << frame1 << std::endl;
  }
  assert(frame0 != frame1);

  for (int j = 0; j < (int)sfm_f.size(); j++)
  {
    if (sfm_f[j].state == true)
      continue;
    bool has_0 = false, has_1 = false;
    Vector2d point0;
    Vector2d point1;

    for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
    {
      if (sfm_f[j].observation[k].first == frame0)
      {
        point0 = sfm_f[j].observation[k].second;
        has_0 = true;
      }
      if (sfm_f[j].observation[k].first == frame1)
      {
        point1 = sfm_f[j].observation[k].second;
        has_1 = true;
      }
    }
    if (has_0 && has_1)
    {
      Vector3d point_3d;
      triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
      sfm_f[j].state = true;
      sfm_f[j].position = point_3d;
    }
  }
}

bool GlobalSFM::construct(const int frame_num, const int ref_frame_idx,
                          const int cur_fixed_idx,
                          std::vector<SFMFeature> &sfm_f, Eigen::Vector3d Ps[],
                          Eigen::Matrix3d Rs[]) const
{
  std::map<int, Eigen::Vector3d> sfm_tracked_points;

  assert(ref_frame_idx < cur_fixed_idx &&
         "ref frame idx large than current fixed frame.");

  Eigen::Matrix<double, 3, 4> Pose[frame_num];
  for (int i = 0; i <= cur_fixed_idx; i++)
  {
    Pose[i].block<3, 3>(0, 0) = Rs[i];
    Pose[i].block<3, 1>(0, 3) = Ps[i];
  }

  for (int i = ref_frame_idx; i < cur_fixed_idx; i++)
  {
    triangulateTwoFrames(i, Pose[i], cur_fixed_idx, Pose[cur_fixed_idx], sfm_f);
  }
  for (int i = ref_frame_idx + 1; i < cur_fixed_idx; i++)
  {
    triangulateTwoFrames(ref_frame_idx, Pose[ref_frame_idx], i, Pose[i], sfm_f);
  }

  if (ref_frame_idx > 0)
  {
    for (int i = 0; i < ref_frame_idx; i++)
    {
      triangulateTwoFrames(ref_frame_idx, Pose[ref_frame_idx], i, Pose[i],
                           sfm_f);
    }
  }

  if (cur_fixed_idx < frame_num)
  {
    for (int i = cur_fixed_idx + 1; i < frame_num; i++)
    {
      // solve pnp
      Matrix3d R_initial = Rs[i - 1];
      Vector3d P_initial = Ps[i - 1];
      if (!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
        return false;
      Rs[i] = R_initial;
      Ps[i] = P_initial;
      // triangulate
      triangulateTwoFrames(i, Pose[i], ref_frame_idx, Pose[ref_frame_idx],
                           sfm_f);
    }
  }

#if false
  // triangulate all other points
  for (int j = 0; j <(int) sfm_f.size(); j++) {
    if (sfm_f[j].state == true) continue;
    if ((int)sfm_f[j].observation.size() >= 2) {
      Vector2d point0, point1;
      int frame_0 = sfm_f[j].observation[0].first;
      point0 = sfm_f[j].observation[0].second;
      int frame_1 = sfm_f[j].observation.back().first;
      point1 = sfm_f[j].observation.back().second;
      Vector3d point_3d;
      triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
      sfm_f[j].state = true;
      sfm_f[j].position = point_3d;
    }
  }
#endif
  // TODO: Full BA to opitimiaze the poses and landmarks
  return true;
}

bool GlobalSFM::construct_orignal(
    int frame_num, int l, const Eigen::Matrix3d relative_R,
    const Eigen::Vector3d relative_T, Eigen::Quaterniond q_out[],
    Eigen::Vector3d T_out[], std::vector<SFMFeature> &sfm_f,
    std::map<int, Eigen::Vector3d> &sfm_tracked_points) const
{
  int feature_num = sfm_f.size();

  q_out[l].setIdentity();
  T_out[l].setZero();
  q_out[frame_num - 1] = Quaterniond(relative_R);
  T_out[frame_num - 1] = relative_T;

  Matrix3d c_Rotation[frame_num];
  Vector3d c_Translation[frame_num];
  Eigen::Matrix<double, 3, 4> Pose[frame_num];

  std::vector<int> known_idx = {l, frame_num - 1};
  for (auto const &i : known_idx)
  {
    c_Rotation[i] = q_out[i].inverse().toRotationMatrix();
    c_Translation[i] = -1 * (c_Rotation[i] * T_out[i]);
    Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
    Pose[i].block<3, 1>(0, 3) = c_Translation[i];
  }

  for (int i = l; i < frame_num - 1; i++) // frame_num - 1 = frame_count
  {
    if (i > l)
    {
      Matrix3d R_initial = c_Rotation[i - 1];
      Vector3d P_initial = c_Translation[i - 1];
      if (!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
        return false;

      c_Rotation[i] = R_initial;
      c_Translation[i] = P_initial;
      Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
      Pose[i].block<3, 1>(0, 3) = c_Translation[i];
    }
    triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
  }
  for (int i = l + 1; i < frame_num - 1; i++)
    triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);

  for (int i = l - 1; i >= 0; i--)
  {
    Matrix3d R_initial = c_Rotation[i + 1];
    Vector3d P_initial = c_Translation[i + 1];
    if (!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
      return false;

    c_Rotation[i] = R_initial;
    c_Translation[i] = P_initial;
    Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
    Pose[i].block<3, 1>(0, 3) = c_Translation[i];
    triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
  }

  for (int j = 0; j < feature_num; j++)
  {
    if (sfm_f[j].state == true)
      continue;
    if ((int)sfm_f[j].observation.size() >= 2)
    {
      Vector2d point0, point1;
      int frame_0 = sfm_f[j].observation[0].first;
      point0 = sfm_f[j].observation[0].second;
      int frame_1 = sfm_f[j].observation.back().first;
      point1 = sfm_f[j].observation.back().second;
      Vector3d point_3d;
      triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
      sfm_f[j].state = true;
      sfm_f[j].position = point_3d;
    }
  }

  ceres::Problem problem;
  ceres::LocalParameterization *local_parameterization =
      new ceres::QuaternionParameterization();

  double c_rotation[frame_num][4];
  double c_translation[frame_num][3];
  for (int i = 0; i < frame_num; ++i)
  {
    // double array for ceres
    Quaterniond q(c_Rotation[i]);
    c_rotation[i][0] = q.w();
    c_rotation[i][1] = q.x();
    c_rotation[i][2] = q.y();
    c_rotation[i][3] = q.z();
    c_translation[i][0] = c_Translation[i].x();
    c_translation[i][1] = c_Translation[i].y();
    c_translation[i][2] = c_Translation[i].z();
    problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
    problem.AddParameterBlock(c_translation[i], 3);
    if (i == l)
    {
      problem.SetParameterBlockConstant(c_rotation[i]);
    }
    if (i == l || i == frame_num - 1)
    {
      problem.SetParameterBlockConstant(c_translation[i]);
    }
  }

  for (int i = 0; i < feature_num; i++)
  {
    if (sfm_f[i].state != true)
      continue;
    for (int j = 0; j < int(sfm_f[i].observation.size()); j++)
    {
      int frame_idx = sfm_f[i].observation[j].first;
      ceres::CostFunction *cost_function = ReprojectionError3D::Create(sfm_f[i].observation[j].second.x(),
                                                                       sfm_f[i].observation[j].second.y());

      problem.AddResidualBlock(cost_function, NULL, c_rotation[frame_idx], c_translation[frame_idx], sfm_f[i].position.data());
    }
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.max_solver_time_in_seconds = 0.2;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;
  if (summary.termination_type == ceres::CONVERGENCE ||
      summary.final_cost < 5e-03)
  {
    std::cout << "vision only BA converge" << std::endl;
  }
  else
  {
    std::cout << "vision only BA not converge " << std::endl;
    return false;
  }

  for (int i = 0; i < frame_num; i++)
  {
    q_out[i].w() = c_rotation[i][0];
    q_out[i].x() = c_rotation[i][1];
    q_out[i].y() = c_rotation[i][2];
    q_out[i].z() = c_rotation[i][3];
    q_out[i] = q_out[i].inverse();
  }
  for (int i = 0; i < frame_num; i++)
  {
    T_out[i] = -1 * (q_out[i] * Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
  }

  for (int i = 0; i < (int)sfm_f.size(); i++)
  {
    if (sfm_f[i].state)
    {
      sfm_tracked_points[sfm_f[i].id] = sfm_f[i].position;
    }
  }
  return true;
}
