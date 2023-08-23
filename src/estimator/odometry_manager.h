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

#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>

/// read rosbag
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <boost/foreach.hpp>

#include <visual_odometry/vio_initial.h>
#include <visual_odometry/visual_feature/feature_tracker_node.h>
#include <visual_odometry/visual_odometry.h>

#include <estimator/odometry_viewer.h>
#include <estimator/parameter_manager.h>
#include <estimator/trajectory_manager.h>

#include <inertial/inertial_initializer.h>

#include <utils/sophus_utils.hpp>

namespace ctrlvio
{

  class OdometryManager
  {
  public:
    OdometryManager(const YAML::Node &node, ros::NodeHandle &nh);

    void LoadBag(const YAML::Node &node);

    void RunBag();

    void PerformCtrlVIO();

  protected:
    bool CreateCacheFolder(const std::string &config_path,
                           const std::string &bag_path);

    void ProcessVIOData(const std::pair<Eigen::aligned_vector<IMUData>,
                                        std::pair<int64_t, sensor_msgs::PointCloud::ConstPtr>> &vio_pair);

    void IMUMsgToIMUData(const sensor_msgs::Imu::ConstPtr &imu_msg,
                         IMUData &data) const;

    void IMUMsgHandle(const sensor_msgs::Imu::ConstPtr &imu_msg);

    void ImageMsgHandle(const sensor_msgs::Image::ConstPtr &image_msg)
    {
      feature_tracker_node_->img_callback(image_msg);
    }

    bool GetMsgsForProcessing(std::pair<Eigen::aligned_vector<IMUData>,
                                        std::pair<int64_t, sensor_msgs::PointCloud::ConstPtr>> &vi_pair);

    void SetInitialState();

    void PublishCloudAndTrajectory();

  public:
    std::string imu_topic_;
    std::string image_topic_;

  protected:
    rosbag::Bag bag_;
    rosbag::View view_;

    ros::Subscriber sub_imu_;
    ros::Subscriber sub_image_;

    int64_t cur_imu_timestamp_;
    Eigen::aligned_vector<IMUData> imu_buf_; // vio
    // From FeatureTrackerNode
    // std::vector<sensor_msgs::PointCloud::ConstPtr> img_feature_buf_;

    ParamManager::Ptr param_manager_;

    bool is_initialized_;
    IMUInitializer::Ptr imu_initializer_;

    Trajectory::Ptr trajectory_;
    TrajectoryManager::Ptr trajectory_manager_;

    feature_tracker::FeatureTrackerNode::Ptr feature_tracker_node_;
    VisualOdometry::Ptr visual_odom_;
    VIOInitialization vio_initializer_;

    OdometryViewer odom_viewer_;

    std::string cache_path_;

  private:
    double ld_init_;
    bool fix_ld_;
    double ld_lower_;
    double ld_upper_;

    std::string bag_path_;
  };

} // namespace ctrlvio
