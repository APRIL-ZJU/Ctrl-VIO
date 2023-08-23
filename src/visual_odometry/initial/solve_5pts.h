#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

class MotionEstimator {
 public:
  bool solveRelativeRT(
      const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> &corres,
      Eigen::Matrix3d &R, Eigen::Vector3d &T) const;
};
