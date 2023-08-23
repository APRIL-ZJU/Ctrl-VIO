#include "feature_manager.h"
#include <glog/logging.h>

using namespace std;
using SE3d = Sophus::SE3<double>;
using SO3d = Sophus::SO3<double>;

namespace ctrlvio
{

  void FeatureManager::clearState() { feature.clear(); }

  int FeatureManager::getFeatureCount()
  {
    int cnt = 0;
    for (auto &it : feature)
    {
      it.used_num = it.feature_per_frame.size();
      if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)
      {
        cnt++;
      }
    }
    return cnt;
  }

  // image —— feature_id: vector<(cam_id, (x,y,z,u,v,vx,vy))>
  bool FeatureManager::addFeatureCheckParallax(
      int frame_count, const map<int, vector<pair<int, Vector7d>>> &image,
      double td)
  {
    ROS_DEBUG("input feature: %d", (int)image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum = 0;
    int parallax_num = 0;
    last_track_num = 0;

    for (auto &id_pts : image)
    {
      int feature_id = id_pts.first;
      auto it = find_if(feature.begin(), feature.end(),
                        [feature_id](const FeaturePerId &it)
                        {
                          return it.feature_id == feature_id;
                        });

      FeaturePerFrame f_per_fra(id_pts.second[0].second, td);

      if (it == feature.end())
      {
        feature.push_back(FeaturePerId(feature_id, frame_count));
        feature.back().feature_per_frame.push_back(f_per_fra);
      }
      else if (it->feature_id == feature_id)
      {
        it->feature_per_frame.push_back(f_per_fra);
        last_track_num++;
      }
    }

    if (frame_count < 2 || last_track_num < 20)
      return true;

    for (auto &it_per_id : feature)
    {
      if (it_per_id.start_frame <= frame_count - 2 &&
          it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >=
              frame_count - 1)
      {
        parallax_sum += compensatedParallax2(it_per_id, frame_count);
        parallax_num++;
      }
    }

    if (parallax_num == 0)
    {
      return true;
    }
    else
    {
      LOG(INFO) << "parallax_sum: " << parallax_sum
                << "; parallax_num: " << parallax_num;
      LOG(INFO) << "current parallax: " << parallax_sum / parallax_num
                << "; MIN_PARALLAX: " << MIN_PARALLAX;
      return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
  }

  vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(
      int frame_count_l, int frame_count_r) const
  {
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature)
    {
      if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
      {
        Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
        int idx_l = frame_count_l - it.start_frame;
        int idx_r = frame_count_r - it.start_frame;

        a = it.feature_per_frame[idx_l].point;

        b = it.feature_per_frame[idx_r].point;

        corres.push_back(make_pair(a, b));
      }
    }
    return corres;
  }

  VectorXd FeatureManager::getDepthVector()
  {
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
      it_per_id.used_num = it_per_id.feature_per_frame.size();
      if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
        continue;

      dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
    }
    return dep_vec;
  }

  void FeatureManager::setDepth(const VectorXd &x)
  {
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
      if (!isLandmarkCandidate(it_per_id))
        continue;

      it_per_id.estimated_depth = 1.0 / x(++feature_index);
      if (it_per_id.estimated_depth < 0)
      {
        it_per_id.solve_flag = SolveFail;
      }
      else
        it_per_id.solve_flag = SovelSucc;

      // LOG(INFO) << "feature id " << it_per_id->feature_id << " , start_frame "
      //           << it_per_id->start_frame << ", depth "
      //           << it_per_id->estimated_depth;
    }
  }

  void FeatureManager::removeFailures()
  {
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
      it_next++;
      if (it->solve_flag == SolveFail)
        feature.erase(it);
    }
  }

  void FeatureManager::clearDepth(const VectorXd &x)
  {
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
      it_per_id.used_num = it_per_id.feature_per_frame.size();
      if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
        continue;

      it_per_id.estimated_depth = 1.0 / x(++feature_index);
    }
  }

#if 1
  void FeatureManager::triangulate(Vector3d Ps[], Matrix3d Rs[])
  {
    for (auto &it_per_id : feature)
    {
      it_per_id.used_num = it_per_id.feature_per_frame.size();
      if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
        continue;
      if (it_per_id.estimated_depth > 0)
        continue;

      Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
      int svd_idx = 0;
      // Tcl_ci
      int imu_i = it_per_id.start_frame;
      int imu_j = imu_i - 1;

      Eigen::Matrix<double, 3, 4> P0;
      Eigen::Vector3d t0 = Ps[imu_i];
      Eigen::Matrix3d R0 = Rs[imu_i];
      P0.leftCols<3>() = Eigen::Matrix3d::Identity();
      P0.rightCols<1>() = Eigen::Vector3d::Zero();

      for (auto &it_per_frame : it_per_id.feature_per_frame)
      {
        imu_j++;

        Eigen::Vector3d t1 = Ps[imu_j];
        Eigen::Matrix3d R1 = Rs[imu_j];
        Eigen::Vector3d t = R0.transpose() * (t1 - t0);
        Eigen::Matrix3d R = R0.transpose() * R1;
        Eigen::Matrix<double, 3, 4> P; // R0 to R1
        P.leftCols<3>() = R.transpose();
        P.rightCols<1>() = -R.transpose() * t;
        Eigen::Vector3d f = it_per_frame.point.normalized();
        svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
        svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);
      }

      ROS_ASSERT(svd_idx == svd_A.rows());
      Eigen::Vector4d svd_V =
          Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV)
              .matrixV()
              .rightCols<1>();
      it_per_id.estimated_depth = svd_V[2] / svd_V[3];

      if (it_per_id.estimated_depth < 0.1)
      {
        it_per_id.estimated_depth = INIT_DEPTH;
      }
    }
  }
#endif

  void FeatureManager::triangulate(Matrix3d Rs[], Vector3d Ps[], const Matrix3d &ric, const Eigen::Vector3d &tic)
  {
    for (auto &it_per_id : feature)
    {
      it_per_id.used_num = it_per_id.feature_per_frame.size();
      if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
        continue;
      if (it_per_id.estimated_depth > 0)
        continue;

      Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
      int svd_idx = 0;
      // Tcl_ci
      int imu_i = it_per_id.start_frame;
      int imu_j = imu_i - 1;
      Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic;
      Eigen::Matrix3d R0 = Rs[imu_i] * ric;

      for (auto &it_per_frame : it_per_id.feature_per_frame)
      {
        imu_j++;
        Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic;
        Eigen::Matrix3d R1 = Rs[imu_j] * ric;
        Eigen::Vector3d t = R0.transpose() * (t1 - t0);
        Eigen::Matrix3d R = R0.transpose() * R1;

        Eigen::Matrix<double, 3, 4> P;
        P.leftCols<3>() = R.transpose();
        P.rightCols<1>() = -R.transpose() * t;
        Eigen::Vector3d f = it_per_frame.point.normalized();

        svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
        svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

        if (imu_i == imu_j)
          continue;
      }

      ROS_ASSERT(svd_idx == svd_A.rows());
      Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
      double svd_method = svd_V[2] / svd_V[3];
      it_per_id.estimated_depth = svd_method;

      if (it_per_id.estimated_depth < 0.1)
      {
        it_per_id.estimated_depth = INIT_DEPTH;
      }
    }
  }

#if 0
  void FeatureManager::triangulateRS(double timestamps[], const Trajectory::Ptr &trajectory, double line_delay)
  {
    Eigen::Vector3d tic = Eigen::Vector3d(0.00699407, -0.0570823, -0.0422772);
    Eigen::Matrix3d ric;
    ric << -0.00276873, -0.999936, -0.0110011,
        -0.999987, 0.00281495, -0.00418819,
        0.00421888, 0.0109894, -0.999931;

    for (auto &it_per_id : feature)
    {
      if (!isLandmarkCandidate(it_per_id))
        continue;
      if (it_per_id.estimated_depth > 0)
        continue;

      Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
      int svd_idx = 0;
      // Tcl_ci
      int imu_i = it_per_id.start_frame;
      int imu_j = imu_i - 1;
      double ti = timestamps[imu_i] + std::round(it_per_id.feature_per_frame[0].uv(1)) * line_delay;
      SE3d pose_i = trajectory->pose(ti);
      Eigen::Vector3d P_i = pose_i.translation();
      Eigen::Matrix3d R_i = pose_i.so3().unit_quaternion().toRotationMatrix();

      Eigen::Vector3d t0 = P_i + R_i * tic;
      Eigen::Matrix3d R0 = R_i * ric;  

      for (auto &it_per_frame : it_per_id.feature_per_frame)
      {
        imu_j++;
        double tj = timestamps[imu_j] + std::round(it_per_frame.uv(1)) * line_delay;
        SE3d pose_j = trajectory->pose(tj);
        Eigen::Vector3d P_j = pose_j.translation();
        Eigen::Matrix3d R_j = pose_j.so3().unit_quaternion().toRotationMatrix();

        Eigen::Vector3d t1 = P_j + R_j * tic;        
        Eigen::Matrix3d R1 = R_j * ric;        
        Eigen::Vector3d t = R0.transpose() * (t1 - t0); 
        Eigen::Matrix3d R = R0.transpose() * R1;        

        Eigen::Matrix<double, 3, 4> P;
        P.leftCols<3>() = R.transpose();
        P.rightCols<1>() = -R.transpose() * t;
        Eigen::Vector3d f = it_per_frame.point.normalized(); 

        svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
        svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

        if (imu_i == imu_j) 
          continue;
      }

      Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
      double svd_method = svd_V[2] / svd_V[3]; 
      it_per_id.estimated_depth = svd_method;  
      if (it_per_id.estimated_depth < 0.1)
      {
        it_per_id.estimated_depth = INIT_DEPTH;
      }
    }
  }
#endif

  void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R,
                                            Eigen::Vector3d marg_P,
                                            Eigen::Matrix3d new_R,
                                            Eigen::Vector3d new_P)
  {
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
      it_next++;

      if (it->start_frame != 0)
        it->start_frame--;
      else
      {
        Eigen::Vector3d uv_i = it->feature_per_frame[0].point;
        it->feature_per_frame.erase(it->feature_per_frame.begin());
        if (it->feature_per_frame.size() < 2)
        {
          feature.erase(it);
          continue;
        }
        else
        {
          // double depth = it->feature_per_frame.front().depth_from_lidar;
          // if (depth > 0) {
          //   it->estimated_depth = depth;
          //   it->lidar_depth_flag = true;
          //   continue;
          // }
          Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
          Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
          Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
          double dep_j = pts_j(2);
          if (dep_j > 0)
            it->estimated_depth = dep_j;
          else
            it->estimated_depth = INIT_DEPTH;
        }
      }
    }
  }

  void FeatureManager::removeBack()
  {
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
      it_next++;

      if (it->start_frame != 0)
        it->start_frame--;
      else
      {
        it->feature_per_frame.erase(it->feature_per_frame.begin());
        if (it->feature_per_frame.size() == 0)
          feature.erase(it);
      }
    }
  }

  void FeatureManager::removeFront(int frame_count)
  {
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
      it_next++;
      if (it->start_frame == frame_count)
      {
        it->start_frame--;
      }
      else
      {
        if (it->endFrame() < frame_count - 1)
          continue;

        int j = frame_count - 1 - it->start_frame;
        it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
        if (it->feature_per_frame.size() == 0)
          feature.erase(it);
      }
    }
  }

  double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id,
                                              int frame_count)
  {
    const FeaturePerFrame &frame_i =
        it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j =
        it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    double ans = 0;
    Vector3d p_j = frame_j.point;

    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;

    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    p_i_comp = p_i;
    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    ans = max(
        ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
  }
} // namespace ctrlvio
