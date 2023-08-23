#pragma once

#include <execinfo.h>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <queue>

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "camera_models/CameraFactory.h"
#include "camera_models/CataCamera.h"
#include "camera_models/PinholeCamera.h"
#include "parameters.h"

#include <glog/logging.h>
#include <utils/tic_toc.h>

namespace feature_tracker
{
  bool inBorder(const cv::Point2f &pt);

  void reduceVector(std::vector<cv::Point2f> &v, std::vector<uchar> status);
  void reduceVector(std::vector<int> &v, std::vector<uchar> status);

  class FeatureTracker
  {
  public:
    FeatureTracker() {}

    void readImage(const cv::Mat &_img, double _cur_time);

    void readIntrinsicParameter(const std::string &calib_file);

    bool updateID(unsigned int i);

    void showUndistortion(const std::string &name);

    cv::Mat mask;    
    cv::Mat fisheye_mask; 

    double prev_time, cur_time;
    cv::Mat prev_img, cur_img, forw_img;                 
    std::vector<cv::Point2f> prev_pts, cur_pts, forw_pts;    
    std::vector<cv::Point2f> prev_un_pts, cur_un_pts;          
    std::map<int, cv::Point2f> prev_un_pts_map, cur_un_pts_map; 
    std::vector<cv::Point2f> pts_velocity;    
    std::vector<cv::Point2f> n_pts;                        
    std::vector<int> ids;                    
    std::vector<int> track_cnt;                           

    camodocal::CameraPtr m_camera; 

    static int n_id;

  private:
    void applyMask();

    void addPoints(std::vector<cv::Point2f> &n_pts);

    void addPoints();

    void reducePoints(std::vector<uchar> &status);

    void rejectWithF();

    void undistortedPoints();
  };
} // namespace feature_tracker