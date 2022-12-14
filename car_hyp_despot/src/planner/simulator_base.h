#pragma once

#include <core/globals.h>
#include <ros/node_handle.h>
#include <string>

#include "msg_builder/imitation_data.h"

class WorldBeliefTracker;
class WorldModel;
class WorldStateTracker;

using namespace despot;

class SimulatorBase  {
protected:
	double time_scale_;
	double cmd_speed_;
	double lane_decision_;

	ACT_TYPE buffered_action_;
	double speed_in_search_state_;
public:
	ros::NodeHandle& nh;

	double real_speed;

	static WorldModel world_model;

	static bool agents_data_ready;
	static bool agents_path_data_ready;

public:
	SimulatorBase(ros::NodeHandle&_nh):
		nh(_nh), time_scale_(1.0), buffered_action_(0),
		cmd_speed_(0), real_speed(0), lane_decision_(0) {}
};


static int action_map[3]={2,0,1};
