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

#include <eigen_conversions/eigen_msg.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <ctrlvio/imu_array.h>
#include <ctrlvio/pose_array.h>

#include <pcl_conversions/pcl_conversions.h>

#include <spline/trajectory.h>
#include <utils/parameter_struct.h>
#include <visual_odometry/feature_manager.h>

namespace ctrlvio
{
  class OdometryViewer
  {
  private:
    // PublishViconData
    ros::Publisher pub_trajectory_raw_;
    ros::Publisher pub_trajectory_est_;

    // PublishIMUData
    ros::Publisher pub_imu_raw_array_;
    ros::Publisher pub_imu_est_array_;

    // PublishSplineTrajectory
    ros::Publisher pub_spline_trajectory_;
    ros::Publisher pub_spline_ctrl_;
    ros::Publisher pub_spline_ctrl_cloud_;

    // PublishImageLandmarks
    ros::Publisher pub_img_landmarks_;
    ros::Publisher pub_img_marg_cloud_;
    ros::Publisher pub_vio_keyframe_;

  public:
    void SetPublisher(ros::NodeHandle &nh)
    {
      /// Vicon data
      pub_trajectory_raw_ = nh.advertise<ctrlvio::pose_array>("/path_raw", 10);
      pub_trajectory_est_ = nh.advertise<ctrlvio::pose_array>("/path_est", 10);
      /// IMU fitting results
      pub_imu_raw_array_ = nh.advertise<ctrlvio::imu_array>("/imu_raw_array", 10);
      pub_imu_est_array_ = nh.advertise<ctrlvio::imu_array>("/imu_est_array", 10);

      /// spline trajectory
      pub_spline_trajectory_ =
          nh.advertise<nav_msgs::Path>("/spline/trajectory", 10);
      pub_spline_ctrl_ = nh.advertise<nav_msgs::Path>("/spline/ctrl_path", 10);
      pub_spline_ctrl_cloud_ =
          nh.advertise<sensor_msgs::PointCloud2>("/spline/ctrl_cloud", 10);

      pub_img_landmarks_ =
          nh.advertise<sensor_msgs::PointCloud2>("/vio/landmarks", 10);
      pub_img_marg_cloud_ =
          nh.advertise<sensor_msgs::PointCloud2>("/vio/marg_cloud", 10);
      pub_vio_keyframe_ =
          nh.advertise<sensor_msgs::PointCloud2>("/vio/keyframe", 10);

      std::cout << "[SetPublisher] init done.\n";
    }

    void PublishSplineTrajectory(Trajectory::Ptr trajectory, int64_t min_time,
                                 int64_t max_time, int64_t dt)
    {
      ros::Time time_now = ros::Time::now();
      ros::Time t_temp;

      if (pub_spline_trajectory_.getNumSubscribers() != 0)
      {
        std::vector<geometry_msgs::PoseStamped> poses_geo;
        for (int64_t t = min_time; t < max_time; t += dt)
        {
          SE3d pose = trajectory->poseNs(t);
          geometry_msgs::PoseStamped poseIinG;
          poseIinG.header.stamp = t_temp.fromSec(t * NS_TO_S);
          poseIinG.header.frame_id = "/map";
          tf::pointEigenToMsg(pose.translation(), poseIinG.pose.position);
          tf::quaternionEigenToMsg(pose.unit_quaternion(),
                                   poseIinG.pose.orientation);
          poses_geo.push_back(poseIinG);
        }

        nav_msgs::Path traj_path;
        traj_path.header.stamp = time_now;
        traj_path.header.frame_id = "/map";
        traj_path.poses = poses_geo;
        pub_spline_trajectory_.publish(traj_path);
      }

#if 0
      if (pub_spline_ctrl_.getNumSubscribers() != 0)
      {
        std::vector<geometry_msgs::PoseStamped> poses_ctrl;
        for (size_t i = 0; i < trajectory->numKnots(); ++i)
        {
          double t = min_time + i * trajectory->getDt();
          geometry_msgs::PoseStamped geo_ctrl;
          geo_ctrl.header.stamp = t_temp.fromSec(t);
          geo_ctrl.header.frame_id = "/map";
          tf::pointEigenToMsg(trajectory->getKnotPos(i), geo_ctrl.pose.position);
          tf::quaternionEigenToMsg(trajectory->getKnotSO3(i).unit_quaternion(),
                                   geo_ctrl.pose.orientation);
          poses_ctrl.push_back(geo_ctrl);
        }

        nav_msgs::Path traj_ctrl;
        traj_ctrl.header.stamp = time_now;
        traj_ctrl.header.frame_id = "/map";
        traj_ctrl.poses = poses_ctrl;
        pub_spline_ctrl_.publish(traj_ctrl);
      }
#endif

      if (pub_spline_ctrl_cloud_.getNumSubscribers() != 0)
      {
        VPointCloud ctrl_cloud;
        for (size_t i = 0; i < trajectory->numKnots(); ++i)
        {
          const Eigen::Vector3d &p = trajectory->getKnotPos(i);
          VPoint ctrl_p;
          ctrl_p.x = p[0];
          ctrl_p.y = p[1];
          ctrl_p.z = p[2];
          ctrl_p.intensity = 100;

          ctrl_cloud.push_back(ctrl_p);
        }

        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(ctrl_cloud, cloud_msg);
        cloud_msg.header.stamp = ros::Time::now();
        cloud_msg.header.frame_id = "/map";
        pub_spline_ctrl_cloud_.publish(cloud_msg);
      }
    }

    void PublishImageLandmarks(const VPointCloud &landmarks,
                               const VPointCloud &marg_cloud)
    {
      if (pub_img_landmarks_.getNumSubscribers() != 0 && !landmarks.empty())
      {
        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(landmarks, cloud_msg);
        cloud_msg.header.stamp = ros::Time::now();
        cloud_msg.header.frame_id = "/map";

        pub_img_landmarks_.publish(cloud_msg);
      }

      if (pub_img_marg_cloud_.getNumSubscribers() != 0 && !marg_cloud.empty())
      {
        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(marg_cloud, cloud_msg);
        cloud_msg.header.stamp = ros::Time::now();
        cloud_msg.header.frame_id = "/map";

        pub_img_marg_cloud_.publish(cloud_msg);
      }
    }

    void PublishVioKeyFrame(const Eigen::Vector3d Ps[])
    {
      if (pub_vio_keyframe_.getNumSubscribers() == 0)
        return;

      VPointCloud keyframe_cloud;
      for (size_t i = 0; i <= WINDOW_SIZE; i++)
      {
        VPoint p;
        p.x = Ps[i].x();
        p.y = Ps[i].y();
        p.z = Ps[i].z();
        p.intensity = 200;
        keyframe_cloud.push_back(p);
      }

      sensor_msgs::PointCloud2 cloud_msg;
      pcl::toROSMsg(keyframe_cloud, cloud_msg);
      cloud_msg.header.stamp = ros::Time::now();
      cloud_msg.header.frame_id = "/map";

      pub_vio_keyframe_.publish(cloud_msg);
    }

    void PublishTF(Eigen::Quaterniond quat, Eigen::Vector3d pos,
                   std::string from_frame, std::string to_frame)
    {
      static tf::TransformBroadcaster tbr;
      tf::Transform transform;
      transform.setOrigin(tf::Vector3(pos[0], pos[1], pos[2]));
      tf::Quaternion tf_q(quat.x(), quat.y(), quat.z(), quat.w());
      transform.setRotation(tf_q);
      tbr.sendTransform(tf::StampedTransform(transform, ros::Time::now(),
                                             to_frame, from_frame));
    }
  };
} // namespace ctrlvio
