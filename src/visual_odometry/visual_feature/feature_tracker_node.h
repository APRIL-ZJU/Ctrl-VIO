#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Bool.h>

#include <cv_bridge/cv_bridge.h>

#include "feature_tracker.h"

namespace feature_tracker
{

  class FeatureTrackerNode
  {
  public:
    typedef std::shared_ptr<FeatureTrackerNode> Ptr;

    FeatureTrackerNode(std::string &config_file, bool offline_mode = true);

    void img_callback(const sensor_msgs::ImageConstPtr &img_msg);

    std::queue<std::pair<int64_t, sensor_msgs::PointCloud::ConstPtr>> &GetImageFeatureBuf()
    {
      return image_feature_buf_;
    }

  private:
    ros::Subscriber sub_img;
    ros::Publisher pub_img, pub_match;
    ros::Publisher pub_restart;

    FeatureTracker trackerData[NUM_OF_CAM];
    double first_image_time;
    double last_image_time;

    bool first_image_flag;

    bool init_pub;
    int pub_count;

    bool offline_mode_;

    std::queue<std::pair<int64_t, sensor_msgs::PointCloud::ConstPtr>> image_feature_buf_;
  };
} // namespace feature_tracker