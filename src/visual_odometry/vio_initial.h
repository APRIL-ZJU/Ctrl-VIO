#pragma once

#include <sensor_msgs/PointCloud.h> // camera feature

#include "feature_manager.h"
#include "integration_base.h"
#include "parameters.h"
#include "utility.h"

#include "initial/initial_alignment.h"
#include "initial/initial_sfm.h"
#include "initial/solve_5pts.h"

namespace ctrlvio
{

  using namespace std;
  using namespace Eigen;

  enum Slide_Flag
  {
    SLIDE_OLD = 0,
    SLIDE_SECOND_NEW = 1
  };

  class VIOInitialization
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Vector7D = Eigen::Matrix<double, 7, 1>;

    VIOInitialization()
    {
      frame_count = 0;

      for (int i = 0; i < WINDOW_SIZE + 1; i++)
      {
        timestamps[i] = 0;
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();

        // pre_integrations[i] = nullptr;
      }
      tmp_pre_integration = nullptr;

      p_CinI.setZero();
      q_CtoI.setIdentity();
      t_offset_CtoI = 0;

      initial_timestamp = 0;
      initial_done = false;

      ros::NodeHandle nh;
      pub_landmarks =
          nh.advertise<sensor_msgs::PointCloud>("/vio_initial_landmarks", 5);
    }

    void SetExtrinsicParam(Eigen::Vector3d _p_CinI, Eigen::Quaterniond _q_CtoI,
                           double _t_offset_CtoI = 0)
    {
      p_CinI = _p_CinI;
      q_CtoI = _q_CtoI;
      t_offset_CtoI = _t_offset_CtoI;
      std::cout << "[SetExtrinsicParam] p_CinI: " << p_CinI.transpose()
                << "; q_CtoI: " << q_CtoI.coeffs().transpose() << std::endl;
    }

    void ProcessIMU(double dt, const Eigen::Vector3d &linear_acceleration,
                    const Eigen::Vector3d &angular_velocity);

    void ProcessImage(const sensor_msgs::PointCloud::ConstPtr &img_msg,
                      double traj_start_time /*= 0*/);

    bool InitialDone() const { return initial_done; }

    void PublishSFMLandmarks();

    void GetWindowData();

  private:
    bool InitialStructure();

    bool VisualInitialAlign();

    // bool relativePose(Matrix3d& relative_R, Vector3d& relative_T, int& l) const;
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);

    void SlideWindow();

    Slide_Flag slide_flag;

    int frame_count;

  public:
    double timestamps[(WINDOW_SIZE + 1)];
    Eigen::Vector3d Ps[(WINDOW_SIZE + 1)];
    Eigen::Vector3d Vs[(WINDOW_SIZE + 1)];
    Eigen::Matrix3d Rs[(WINDOW_SIZE + 1)];
    Eigen::Vector3d Bas[(WINDOW_SIZE + 1)];
    Eigen::Vector3d Bgs[(WINDOW_SIZE + 1)];

    // IntegrationBase* pre_integrations[(WINDOW_SIZE + 1)];
    std::vector<double> imu_timestamps_buf;
    Eigen::aligned_vector<Eigen::Vector3d> linear_acceleration_buf;
    Eigen::aligned_vector<Eigen::Vector3d> angular_velocity_buf;

    std::map<double, ImageFrame> all_image_frame;
    IntegrationBase *tmp_pre_integration;

    Eigen::Vector3d p_CinI;
    Eigen::Quaterniond q_CtoI;
    double t_offset_CtoI;

    FeatureManager f_manager;

    double initial_timestamp;
    Eigen::Vector3d gravity;

    MotionEstimator m_estimator;

    bool initial_done;

    ros::Publisher pub_landmarks;
  };

} // namespace ctrlvio