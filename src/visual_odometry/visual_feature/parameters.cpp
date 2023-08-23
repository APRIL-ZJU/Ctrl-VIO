#include "parameters.h"
#include <opencv2/core.hpp>

namespace feature_tracker
{

  std::string IMAGE_TOPIC;
  std::vector<std::string> CAM_NAMES;
  int ROW;
  int COL;
  int FOCAL_LENGTH;

  int MAX_CNT;
  int MIN_DIST;

  int LK_DESIRED_FREQ;
  double F_THRESHOLD;
  int SHOW_TRACK;
  int EQUALIZE;

  int FISHEYE;
  int FLOW_BACK;
  int REJECTWF;
  double FB_THRESHOLD;
  std::string FISHEYE_MASK;

  bool PUB_THIS_FRAME = false;
  int STEREO_TRACK = false;

  void readParameters(std::string &config_file)
  {
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened())
    {
      std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    fsSettings["image_topic"] >> IMAGE_TOPIC;
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    FOCAL_LENGTH = fsSettings["focal_length"];

    LK_DESIRED_FREQ = fsSettings["freq"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    EQUALIZE = fsSettings["equalize"];

    FISHEYE = fsSettings["fisheye"];
    if (FISHEYE == 1)
    {
      fsSettings["fisheye_mask_path"] >> FISHEYE_MASK;
    }
    CAM_NAMES.push_back(config_file);

    if (LK_DESIRED_FREQ == 0)
      LK_DESIRED_FREQ = 100;

    FLOW_BACK = fsSettings["flow_back"];
    REJECTWF = fsSettings["reject_wf"];
    FB_THRESHOLD = fsSettings["fb_threshold"];

    fsSettings.release();
  }

} // namespace feature_tracker
