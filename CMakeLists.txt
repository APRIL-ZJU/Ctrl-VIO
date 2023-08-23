cmake_minimum_required(VERSION 3.0.2)
project(ctrlvio)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -msse4.2")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  eigen_conversions
  pcl_conversions
  pcl_ros
  cv_bridge
  roslib
  rosbag
  tf
  message_generation
)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
# find_package(Sophus REQUIRED)
find_package(OpenCV 3 REQUIRED)

#find_package(yaml-cpp REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML_CPP REQUIRED yaml-cpp>=0.5)

add_message_files(
  FILES
  imu_array.msg
  pose_array.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs 
  sensor_msgs
)

catkin_package(
  INCLUDE_DIRS src
  CATKIN_DEPENDS std_msgs
)

include_directories(
  src
  ${catkin_INCLUDE_DIRS}
  ${YAML_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  # ${Sophus_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

list(APPEND thirdparty_libraries
        ${YAML_CPP_LIBRARIES}
        ${Boost_LIBRARIES}
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        ${OpenCV_LIBS}
        ${CERES_LIBRARIES}
)

######################
### Library
######################
add_library(spline_lib src/spline/trajectory.cpp)
target_link_libraries(spline_lib ${thirdparty_libraries})

file(GLOB visual_feature_files
    "src/visual_odometry/visual_feature/*.cpp"
    "src/visual_odometry/visual_feature/camera_models/*.cc"
)
add_library(feature_tracker_lib ${visual_feature_files})
target_link_libraries(feature_tracker_lib ${thirdparty_libraries})

file(GLOB visual_odometry_files
    "src/visual_odometry/*.cpp"
    "src/visual_odometry/initial/*.cpp"
)
add_library(vio_lib ${visual_odometry_files})
target_link_libraries(vio_lib spline_lib ${thirdparty_libraries})

######################
### Estimator
######################
add_executable(odometry_node 
   src/app/odometry_node.cpp
   src/inertial/inertial_initializer.cpp
   src/estimator/trajectory_manager.cpp
   src/estimator/trajectory_estimator.cpp
   src/estimator/odometry_manager.cpp
   src/estimator/factor/analytic_diff/marginalization_factor.cpp
 )
 target_link_libraries(odometry_node 
   spline_lib 
   vio_lib 
   feature_tracker_lib 
   ${thirdparty_libraries}
 )
