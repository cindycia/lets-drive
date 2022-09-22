cmake_minimum_required(VERSION 2.8.3)
project(sac_discrete)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  msg_builder
  roscpp
  rospy
  std_msgs
  nav_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs msg_builder roscpp rospy std_msgs
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

