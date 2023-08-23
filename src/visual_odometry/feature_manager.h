#pragma once

#include <algorithm>
#include <list>
#include <numeric>
#include <vector>

#include <ros/assert.h>
#include <ros/console.h>

#include "parameters.h"
#include "visual_struct.h"
#include "../spline/trajectory.h"

#include <sophus_lib/so3.hpp>
#include <sophus_lib/se3.hpp>

namespace ctrlvio
{

  class FeatureManager
  {
  public:
    FeatureManager() {}

    void clearState();

    int getFeatureCount();

    bool addFeatureCheckParallax(
        int frame_count,
        const std::map<int, std::vector<std::pair<int, Vector7d>>> &image,
        double td);

    std::vector<std::pair<Vector3d, Vector3d>> getCorresponding(
        int frame_count_l, int frame_count_r) const;

    VectorXd getDepthVector();

    void setDepth(const VectorXd &x);

    void removeFailures();

    void clearDepth(const VectorXd &x);

    void triangulate(Matrix3d Rs[], Vector3d Ps[], const Matrix3d &ric, const Eigen::Vector3d &tic);

    void triangulate(Vector3d Ps[], Matrix3d Rs[]);                                          
    void triangulateRS(double timestamps[], const Trajectory::Ptr &trajectory, double line_delay);

    void removeBackShiftDepth(Matrix3d marg_R, Vector3d marg_P, Matrix3d new_R,
                              Vector3d new_P);

    void removeBack();

    void removeFront(int frame_count);

    static bool isLandmarkCandidate(const FeaturePerId &feature)
    {
      int used_num = feature.feature_per_frame.size();
      if (used_num >= 2 && feature.start_frame < WINDOW_SIZE - 2)
        return true;
      else
        return false;
    }

    std::list<FeaturePerId> feature;
    int last_track_num;

  private:
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
  };

} // namespace ctrlvio
