
#include "feature_tracker_node.h"

#define SHOW_UNDISTORTION 0

using namespace std;

namespace feature_tracker
{

  FeatureTrackerNode::FeatureTrackerNode(std::string &config_file,
                                         bool offline_mode)
      : first_image_time(0),
        last_image_time(0),
        first_image_flag(true),
        init_pub(false),
        pub_count(1),
        offline_mode_(offline_mode)
  {
    LOG(INFO) << "[FeatureTrackerNode] load  " << config_file;
    readParameters(config_file);

    for (int i = 0; i < NUM_OF_CAM; i++)
      trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);

    if (FISHEYE)
    {
      for (int i = 0; i < NUM_OF_CAM; i++)
      {
        trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
        if (!trackerData[i].fisheye_mask.data)
        {
          LOG(INFO) << "load mask fail";
          ROS_BREAK();
        }
        else
          LOG(INFO) << "load mask success";
      }
    }

    LOG(INFO) << "FeatureTrackerNode subscribe image topic " << IMAGE_TOPIC;

    ros::NodeHandle nh;
    if (!offline_mode_)
    {
      sub_img = nh.subscribe(IMAGE_TOPIC, 100, &FeatureTrackerNode::img_callback,
                             this, ros::TransportHints().tcpNoDelay());
    }

    pub_img = nh.advertise<sensor_msgs::PointCloud>("image_feature", 1000);
    pub_match = nh.advertise<sensor_msgs::Image>("feature_img", 1000);
    pub_restart = nh.advertise<std_msgs::Bool>("restart", 1000);
  }

  void FeatureTrackerNode::img_callback(
      const sensor_msgs::ImageConstPtr &img_msg)
  {
    if (first_image_flag)
    {
      first_image_flag = false;
      first_image_time = img_msg->header.stamp.toSec();
      last_image_time = img_msg->header.stamp.toSec();
      return;
    }
    if (img_msg->header.stamp.toSec() - last_image_time > 1.0 ||
        img_msg->header.stamp.toSec() < last_image_time)
    {
      LOG(INFO) << "image discontinue! reset the feature tracker!";
      first_image_flag = true;
      last_image_time = 0;
      pub_count = 1;
      std_msgs::Bool restart_flag;
      restart_flag.data = true;
      pub_restart.publish(restart_flag);
      return;
    }

    last_image_time = img_msg->header.stamp.toSec();

    double cur_freq =
        1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time);
    if (round(cur_freq) <= LK_DESIRED_FREQ)
    {
      PUB_THIS_FRAME = true;
      // reset the frequency control
      if (abs(cur_freq - LK_DESIRED_FREQ) < 0.01 * LK_DESIRED_FREQ)
      {
        first_image_time = img_msg->header.stamp.toSec();
        pub_count = 0;
      }
    }
    else
      PUB_THIS_FRAME = false;

    TicToc t_r;
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1" || img_msg->encoding == "mono8")
    {
      sensor_msgs::Image img;
      img.header = img_msg->header;
      img.height = img_msg->height;
      img.width = img_msg->width;
      img.is_bigendian = img_msg->is_bigendian;
      img.step = img_msg->step;
      img.data = img_msg->data;
      img.encoding = "mono8";
      ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else if (img_msg->encoding == "8UC3")
    {
      sensor_msgs::Image img;
      img.header = img_msg->header;
      img.height = img_msg->height;
      img.width = img_msg->width;
      img.is_bigendian = img_msg->is_bigendian;
      img.step = img_msg->step;
      img.data = img_msg->data;
      img.encoding = "bgr8";
      ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
    }
    else
    {
      ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
    }
    cv::Mat show_img = ptr->image;
    if (show_img.channels() != 1)
    {
      cv::cvtColor(show_img, show_img, CV_RGB2GRAY);
      trackerData[0].readImage(show_img, img_msg->header.stamp.toSec());
    }
    else
    {
      trackerData[0].readImage(ptr->image.rowRange(ROW * 0, ROW * (0 + 1)),
                               img_msg->header.stamp.toSec());
    }
#if SHOW_UNDISTORTION
    trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
    for (unsigned int i = 0;; i++)
    {
      bool completed = false;
      completed |= trackerData[0].updateID(i);
      if (!completed)
        break;
    }

    if (PUB_THIS_FRAME)
    {
      pub_count++;
      sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
      sensor_msgs::ChannelFloat32 id_of_point;
      sensor_msgs::ChannelFloat32 u_of_point;
      sensor_msgs::ChannelFloat32 v_of_point;
      sensor_msgs::ChannelFloat32 velocity_x_of_point;
      sensor_msgs::ChannelFloat32 velocity_y_of_point;

      feature_points->header = img_msg->header;
      feature_points->header.frame_id = "world";

      auto &cur_un_pts = trackerData[0].cur_un_pts;
      auto &cur_pts = trackerData[0].cur_pts;
      auto &ids = trackerData[0].ids;
      auto &pts_velocity = trackerData[0].pts_velocity;
      for (unsigned int i = 0; i < ids.size(); i++)
      {
        if (trackerData[0].track_cnt[i] > 1)
        {
          int p_id = ids[i];
          geometry_msgs::Point32 p;
          p.x = cur_un_pts[i].x;
          p.y = cur_un_pts[i].y;
          p.z = 1;

          feature_points->points.push_back(p);       
          id_of_point.values.push_back(p_id);        
          u_of_point.values.push_back(cur_pts[i].x);
          v_of_point.values.push_back(cur_pts[i].y);
          velocity_x_of_point.values.push_back(pts_velocity[i].x);
          velocity_y_of_point.values.push_back(pts_velocity[i].y);
        }
      }

      feature_points->channels.push_back(id_of_point);
      feature_points->channels.push_back(u_of_point);
      feature_points->channels.push_back(v_of_point);
      feature_points->channels.push_back(velocity_x_of_point);
      feature_points->channels.push_back(velocity_y_of_point);

      if (!init_pub) 
      {
        init_pub = true;
      }
      else
        image_feature_buf_.push(std::make_pair(img_msg->header.stamp.toSec() * 1e9, feature_points));

      if (SHOW_TRACK)
      {
        ptr = cv_bridge::cvtColor(ptr, "bgr8");

        for (unsigned int i = 0; i < trackerData[0].cur_pts.size(); i++)
        {
          double len = std::min(1.0, 1.0 * trackerData[0].track_cnt[i] / 20);
          cv::circle(ptr->image, trackerData[0].cur_pts[i], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
        }
        pub_match.publish(ptr->toImageMsg());
      }
    }
    LOG(INFO) << "whole feature tracker processing costs: " << t_r.toc() << " ms";
  }

  template <typename T>
  T readParam(ros::NodeHandle &n, std::string name)
  {
    T ans;
    if (n.getParam(name, ans))
    {
      ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
      ROS_ERROR_STREAM("Failed to load " << name);
      n.shutdown();
    }
    return ans;
  }
} // namespace feature_tracker

int main(int argc, char **argv)
{
  ros::init(argc, argv, "feature_tracker_node");
  ros::NodeHandle nh("~");
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME,
                                 ros::console::levels::Info);

  std::string config_file;
  config_file = feature_tracker::readParam<std::string>(nh, "config_file");

  feature_tracker::FeatureTrackerNode feature_tracker_node(config_file, false);
  /*
  if (SHOW_TRACK)
      cv::namedWindow("vis", cv::WINDOW_NORMAL);
  */
  ros::spin();
  return 0;
}

// new points velocity is 0, pub or not?
// track cnt > 1 pub?
