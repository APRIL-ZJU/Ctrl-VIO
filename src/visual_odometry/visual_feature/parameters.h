#pragma once
#include <ros/ros.h>

namespace feature_tracker
{

    extern int ROW;        
    extern int COL;        
    extern int FOCAL_LENGTH; 
    const int NUM_OF_CAM = 1; 

    ///===== feature_tracker_node parameter ==== ///
    extern std::string IMAGE_TOPIC;        
    extern std::string FISHEYE_MASK;          
    extern std::vector<std::string> CAM_NAMES;

    extern int LK_DESIRED_FREQ; 
    extern int SHOW_TRACK; 
    extern int STEREO_TRACK;   

    // also in FeatureTracker
    extern bool PUB_THIS_FRAME;

    ///===== FeatureTracker parameter ==== ///
    extern double F_THRESHOLD; // for findFundamentalMat in rejectWithF
    extern int MAX_CNT;       
    extern int MIN_DIST;     
    extern int EQUALIZE;     
    extern int FISHEYE;     
    extern int FLOW_BACK;    
    extern int REJECTWF;
    extern double FB_THRESHOLD; 

    void readParameters(std::string &config_file);

} // namespace feature_tracker
