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

#include <eigen_conversions/eigen_msg.h>
#include <estimator/odometry_manager.h>
#include <pcl_conversions/pcl_conversions.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace ctrlvio
{

  OdometryManager::OdometryManager(const YAML::Node &node, ros::NodeHandle &nh)
      : image_topic_(""),
        cur_imu_timestamp_(-1),
        is_initialized_(false)
  {
    param_manager_ = std::make_shared<ParamManager>(node);
    ld_init_ = node["ld_init"].as<double>();
    fix_ld_ = node["fix_ld"].as<bool>();
    ld_lower_ = node["ld_lower"].as<double>();
    ld_upper_ = node["ld_upper"].as<double>();

    double knot_distance = node["knot_distance"].as<double>();
    trajectory_ = std::make_shared<Trajectory>(knot_distance);
    trajectory_->SetLineDelay(ld_init_, fix_ld_, ld_lower_, ld_upper_);
    trajectory_->SetSensorExtrinsics(SensorType::CameraSensor,
                                     param_manager_->EP_CtoI);
    vio_initializer_.SetExtrinsicParam(param_manager_->EP_CtoI.p,
                                       param_manager_->EP_CtoI.q,
                                       param_manager_->EP_CtoI.t_offset);

    std::string config_path = node["config_path"].as<std::string>();
    std::string bag_path = node["bag_path"].as<std::string>();
    bag_path_ = bag_path;
    CreateCacheFolder(config_path, bag_path);

    boost::filesystem::path path_bag(bag_path);
    trajectory_->bag_name = path_bag.stem().string();

    std::string cam_yaml = config_path + node["cam_yaml"].as<std::string>();
    std::cout << "load " << cam_yaml << std::endl;
    YAML::Node cam_node = YAML::LoadFile(cam_yaml);
    image_topic_ = cam_node["image_topic"].as<std::string>();
    feature_tracker_node_ =
        std::make_shared<feature_tracker::FeatureTrackerNode>(cam_yaml, false);
    visual_odom_ = std::make_shared<VisualOdometry>(node, trajectory_);

    trajectory_manager_ = std::make_shared<TrajectoryManager>(node, trajectory_);

    std::string imu_yaml = node["imu_yaml"].as<std::string>();
    YAML::Node imu_node = YAML::LoadFile(config_path + imu_yaml);
    imu_initializer_ = std::make_shared<IMUInitializer>(imu_node);
    imu_topic_ = imu_node["imu_topic"].as<std::string>();

    odom_viewer_.SetPublisher(nh);
    LoadBag(node);

    std::cout << std::fixed << std::setprecision(4);
    LOG(INFO) << std::fixed << std::setprecision(4);
  }

  bool OdometryManager::CreateCacheFolder(const std::string &config_path,
                                          const std::string &bag_path)
  {
    boost::filesystem::path path_cfg(config_path);
    boost::filesystem::path path_bag(bag_path);
    if (path_bag.extension() != ".bag")
      return false;

    std::string bag_name_ = path_bag.stem().string();

    std::string cache_path_parent_ = path_cfg.parent_path().string();
    // cache_path_ = cache_path_parent_ + "/data/" + bag_name_;
    cache_path_ = cache_path_parent_ + "/data";
    boost::filesystem::create_directory(cache_path_);
    return true;
  }

  void OdometryManager::LoadBag(const YAML::Node &node)
  {
    std::string bag_path = node["bag_path"].as<std::string>();
    double bag_start = node["bag_start"].as<double>();
    double bag_durr = node["bag_durr"].as<double>();

    std::vector<std::string> topics;
    // IMU
    topics.push_back(imu_topic_);
    // Camera
    if (image_topic_ != "")
      topics.push_back(image_topic_);

    bag_.open(bag_path, rosbag::bagmode::Read);

    rosbag::View view_full;
    view_full.addQuery(bag_);
    ros::Time time_start = view_full.getBeginTime();
    time_start += ros::Duration(bag_start);
    ros::Time time_finish = bag_durr < 0 ? view_full.getEndTime() : time_start + ros::Duration(bag_durr);
    view_.addQuery(bag_, rosbag::TopicQuery(topics), time_start, time_finish);
    if (view_.size() == 0)
    {
      ROS_ERROR("No messages to play on specified topics.  Exiting.");
      ros::shutdown();
      return;
    }
    LOG(INFO) << "load bag " << bag_path << " with "
              << (time_finish - time_start).toSec() << " seconds";
  }

  void OdometryManager::RunBag()
  {
    LOG(INFO) << "RunBag ....";
    for (const rosbag::MessageInstance &m : view_)
    {
      if (!ros::ok())
      {
        return;
        ros::shutdown();
      }

      /// IMU cache
      if (m.getTopic() == imu_topic_)
      {
        sensor_msgs::Imu::ConstPtr imu_msg = m.instantiate<sensor_msgs::Imu>();
        if (imu_msg != NULL)
        {
          TicToc t_imu;
          IMUMsgHandle(imu_msg);
          LOG(INFO) << "[IMUMsgHandle cost] " << t_imu.toc() << " ms.";
        }
      }
      /// image cache
      else if (m.getTopic() == image_topic_)
      {
        auto img_msg = m.instantiate<sensor_msgs::Image>();
        if (img_msg != NULL)
        {
          TicToc t_img;
          ImageMsgHandle(img_msg);
          LOG(INFO) << "[ImageMsgHandle cost] " << t_img.toc() << " ms.";
        }
      }

      /// core
      PerformCtrlVIO();
    }
  }

  void OdometryManager::PerformCtrlVIO()
  {
    std::pair<Eigen::aligned_vector<IMUData>,
              std::pair<int64_t, sensor_msgs::PointCloud::ConstPtr>>
        vi_pair({}, std::make_pair(-1, nullptr));
    if (!GetMsgsForProcessing(vi_pair))
      return;

    ProcessVIOData(vi_pair);
  }

  void OdometryManager::ProcessVIOData(
      const std::pair<Eigen::aligned_vector<IMUData>,
                      std::pair<int64_t, sensor_msgs::PointCloud::ConstPtr>> &vio_pair)
  {
    int64_t data_start_time = trajectory_->GetDataStartTime();
    if (!is_initialized_)
      data_start_time = 0;

    int64_t current_img_time = vio_pair.second.first - data_start_time;
    static int64_t current_imu_time = -1;
    if (current_imu_time < 0)
    {
      int64_t t_imu = vio_pair.first.front().timestamp - data_start_time;
      current_imu_time = t_imu;
    }

    /// [1] preintegration
    Vector3d accel = Vector3d::Zero();
    Vector3d gyro = Vector3d::Zero();
    for (const auto &imu_data : vio_pair.first)
    {
      int64_t t_imu = imu_data.timestamp - data_start_time;
      if (t_imu <= current_img_time)
      {
        // imu data older than image
        int64_t dt = t_imu - current_imu_time;
        assert(dt >= 0);
        current_imu_time = t_imu;

        accel = imu_data.accel;
        gyro = imu_data.gyro;
      }
      else
      {
        int64_t dt_1 = current_img_time - current_imu_time;
        int64_t dt_2 = t_imu - current_img_time;
        current_imu_time = current_img_time;
        assert(dt_1 >= 0);
        assert(dt_2 >= 0);
        assert(dt_1 + dt_2 > 0);
        double w1 = 1.0 * dt_2 / (1.0 * (dt_1 + dt_2));
        double w2 = 1.0 * dt_1 / (1.0 * (dt_1 + dt_2));
        accel = (w1 * accel + w2 * imu_data.accel).eval();
        gyro = (w1 * gyro + w2 * imu_data.gyro).eval();
      }
      if (vio_initializer_.InitialDone())
      {
        visual_odom_->ProcessIMU(current_imu_time * NS_TO_S, accel, gyro);
      }
      else
      {
        vio_initializer_.ProcessIMU(current_imu_time * NS_TO_S, accel, gyro);
      }
    }

    /// [2-1] if not initialized, try initialization
    if (!vio_initializer_.InitialDone())
    {
      vio_initializer_.ProcessImage(vio_pair.second.second, 0);
      // once initialized
      if (vio_initializer_.InitialDone())
      {
        SetInitialState();

        // change into relative time
        data_start_time = trajectory_->GetDataStartTime();
        current_img_time -= data_start_time;
        current_imu_time -= data_start_time;
        visual_odom_->InitWindow(vio_initializer_, data_start_time);              // copy the whole window
        trajectory_manager_->ExtendTrajectory(current_img_time + 0.04 * S_TO_NS); // 1024 * 0.0000294737 < 0.04 | extend trajectory to cover all data
      }
    }
    /// [2-2] if initialized, extend trajectory and add image
    else
    {
      trajectory_manager_->ExtendTrajectory(current_img_time + 0.04 * S_TO_NS);
      visual_odom_->AddImageToWindow(vio_pair.second, trajectory_, trajectory_->line_delay);
    }

    /// [3] if initialized, perform vio optimization
    if (vio_initializer_.InitialDone())
    {
      static bool first_opt_flag = true;
      if (first_opt_flag)
      {
        first_opt_flag = false;
      }
      else
      {
        // optimize trajectory only by IMU (for prediction)
        trajectory_manager_->InitTrajectory(visual_odom_->Bgs_[WINDOW_SIZE], visual_odom_->Bas_[WINDOW_SIZE]);
      }

      Eigen::VectorXd depth_inv_vec = visual_odom_->getDepthVector();

      // optimize trajectory by both IMU and image
      trajectory_manager_->UpdateTrajectory(
          visual_odom_->GetFeatures(), visual_odom_->GetTimestamps(),
          visual_odom_->pre_integrations_, visual_odom_->Bgs_, visual_odom_->Bas_,
          depth_inv_vec, visual_odom_->GetMarginalizationFlag(), 15);

      visual_odom_->setDepth(depth_inv_vec);

      visual_odom_->SlideWindow();

      odom_viewer_.PublishImageLandmarks(visual_odom_->GetLandmarksInWindow(),
                                         visual_odom_->GetMarginCloud());
      odom_viewer_.PublishVioKeyFrame(visual_odom_->Ps_);
      PublishCloudAndTrajectory();
      auto pose = trajectory_->GetCameraPose(trajectory_->maxTimeNs() - 0.05 * S_TO_NS);
      odom_viewer_.PublishTF(pose.unit_quaternion(), pose.translation(), "camera", "map");
      ROS_INFO("estimated line delay: %fus", trajectory_->line_delay * 1e6);

      std::vector<int> show_idx = {1, WINDOW_SIZE};
      for (int &i : show_idx)
        LOG(INFO) << "[Visual] " << i
                  << " Gyro Bias: " << visual_odom_->Bgs_[i][0] << ","
                  << visual_odom_->Bgs_[i][1] << "," << visual_odom_->Bgs_[i][2]
                  << "; Acce Bias: " << visual_odom_->Bas_[i][0] << ","
                  << visual_odom_->Bas_[i][1] << "," << visual_odom_->Bas_[i][2];
    }
  }

  void OdometryManager::IMUMsgToIMUData(const sensor_msgs::Imu::ConstPtr &imu_msg,
                                        IMUData &data) const
  {
    data.timestamp = imu_msg->header.stamp.toSec() * S_TO_NS;
    data.gyro =
        Eigen::Vector3d(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y,
                        imu_msg->angular_velocity.z);
    data.accel = Eigen::Vector3d(imu_msg->linear_acceleration.x,
                                 imu_msg->linear_acceleration.y,
                                 imu_msg->linear_acceleration.z);
    Eigen::Vector4d q(imu_msg->orientation.w, imu_msg->orientation.x,
                      imu_msg->orientation.y, imu_msg->orientation.z);
    if (std::fabs(q.norm() - 1) < 0.01)
    {
      data.orientation = SO3d(Eigen::Quaterniond(q[0], q[1], q[2], q[3]));
    }
  }

  void OdometryManager::IMUMsgHandle(const sensor_msgs::Imu::ConstPtr &imu_msg)
  {
    // if (cur_imu_timestamp_ > 0) {
    //   if (imu_msg->header.stamp.toSec() - cur_imu_timestamp_ < 0.01) return;
    // }
    cur_imu_timestamp_ = imu_msg->header.stamp.toSec() * S_TO_NS;

    IMUData data;
    IMUMsgToIMUData(imu_msg, data);

    // for vio
    imu_buf_.emplace_back(data);
    if (!is_initialized_)
    {
      imu_initializer_->FeedIMUData(data);
    }
    else
    {
      trajectory_manager_->AddIMUData(data);
    }
  }

  bool OdometryManager::GetMsgsForProcessing(std::pair<Eigen::aligned_vector<IMUData>,
                                                       std::pair<int64_t, sensor_msgs::PointCloud::ConstPtr>> &vi_pair)
  {
    auto &img_feature_buf = feature_tracker_node_->GetImageFeatureBuf();

    // [0] after vio initialization, remove the data before data_start_time
    int64_t data_start_time = trajectory_->GetDataStartTime();
    if (is_initialized_)
    {
      while (!img_feature_buf.empty() && img_feature_buf.front().first < data_start_time)
      {
        img_feature_buf.pop();
      }
      for (auto it = imu_buf_.begin(); it != imu_buf_.end();)
      {
        if (it->timestamp < data_start_time)
          it = imu_buf_.erase(it);
        else
          it++;
      }
    }

    // [1] no imu or no image is not allowed
    if (img_feature_buf.empty() || imu_buf_.empty())
      return false;
    if (cur_imu_timestamp_ < 0 && img_feature_buf.size() > 100)
    {
      LOG(WARNING) << "No IMU data. CHECK imu topic";
    }
    if (cur_imu_timestamp_ < 0)
    {
      return false;
    }

    // [2] after vio initialization, the latest IMU should be later than the predicted traj_max_time
    if (is_initialized_)
    {
      int64_t traj_max_time = trajectory_->maxTimeNs() + 0.2 * S_TO_NS; // predicted traj_max_time
      if (cur_imu_timestamp_ < traj_max_time + data_start_time)
      {
        return false;
      }
    }

    // [3] the current image should be surrounded by IMU
    Eigen::aligned_vector<IMUData> IMUs;
    sensor_msgs::PointCloud::ConstPtr image_msg = nullptr;
    int64_t t_image = img_feature_buf.front().first;
    if (!(imu_buf_.back().timestamp > t_image))
    {
      return false;
    }
    if (!(imu_buf_.front().timestamp < t_image))
    {
      img_feature_buf.pop();
      return false;
    }

    // [4] pair the vio data
    image_msg = img_feature_buf.front().second;
    img_feature_buf.pop();
    while (imu_buf_.front().timestamp < t_image)
    {
      IMUs.emplace_back(imu_buf_.front());
      auto iter = imu_buf_.begin();
      iter = imu_buf_.erase(iter);
    }
    IMUs.emplace_back(imu_buf_.front()); // for interpolation
    assert(!IMUs.empty());
    vi_pair = std::make_pair(IMUs, std::make_pair(t_image, image_msg));

    bool has_new_msg = (image_msg != nullptr);
    return has_new_msg;
  }

  void OdometryManager::SetInitialState()
  {
    is_initialized_ = true;

    SystemState initial_state;
    initial_state.p = vio_initializer_.Ps[0];
    initial_state.q = Eigen::Quaterniond(vio_initializer_.Rs[0]);
    initial_state.g << 0, 0, GRAVITY_NORM;
    initial_state.bias.gyro_bias = vio_initializer_.Bgs[0];
    initial_state.bias.accel_bias = Eigen::Vector3d::Zero();
    param_manager_->SetSystemState(initial_state);
    trajectory_manager_->SetSystemState(initial_state);

    int64_t t_image0 = vio_initializer_.timestamps[0] * S_TO_NS;
    trajectory_->SetDataStartTime(t_image0);

    const auto &imu_data_buf = imu_initializer_->GetIMUData();
    for (auto const &imu_data : imu_data_buf)
    {
      if (imu_data.timestamp < t_image0)
        continue;
      trajectory_manager_->AddIMUData(imu_data);
    }

    SO3d R0(initial_state.q);
    for (size_t i = 0; i <= trajectory_->numKnots(); i++) // only 4 control points at the very beginning
    {
      trajectory_->setKnotSO3(R0, i);
    }

    std::cout << YELLOW << "use [vio_initializer] set trajectory start time as: " << trajectory_->GetDataStartTime() << RESET << std::endl;
    assert(trajectory_->GetDataStartTime() > 0 && "data start time < 0");
  }

  void OdometryManager::PublishCloudAndTrajectory()
  {
    odom_viewer_.PublishSplineTrajectory(trajectory_, trajectory_->minTimeNs(), trajectory_->maxTimeNs(), 0.02 * S_TO_NS);
  }
} // namespace ctrlvio
