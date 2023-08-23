#pragma once
#include <ros/ros.h>
#include <eigen3/Eigen/Dense>

#include <map>
#include "../integration_base.h"

namespace ctrlvio
{

  using namespace Eigen;
  using namespace std;

  class ImageFrame
  {
  public:
    ImageFrame(){};
    ImageFrame(
        const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &_points,
        double _t)
        : t{_t}, is_key_frame{false}, pre_integration(nullptr)
    {
      points = _points;
    };

    double t;
    bool is_key_frame;
    std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>
        points;

    Eigen::Matrix3d R;
    Eigen::Vector3d T;

    ctrlvio::IntegrationBase *pre_integration;
  };

  class VisualIMUAlignment
  {
  public:
    VisualIMUAlignment(Eigen::Vector3d _p_CtoI) : p_CtoI(_p_CtoI) {}

    bool TryAlign(map<double, ImageFrame> &all_image_frame, Vector3d *Bgs,
                  Vector3d &g, VectorXd &x);

  private:
    void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame,
                            Vector3d *Bgs);

    MatrixXd TangentBasis(Vector3d &g0);

    void RefineGravity(map<double, ImageFrame> &all_image_frame, Vector3d &g,
                       VectorXd &x);

    bool LinearAlignment(map<double, ImageFrame> &all_image_frame, Vector3d &g,
                         VectorXd &x);

    Eigen::Vector3d p_CtoI;
  };
} // namespace ctrlvio
