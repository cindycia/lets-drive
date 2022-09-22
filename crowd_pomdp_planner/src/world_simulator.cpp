#include <despot/util/logging.h>
#include "world_model.h"
#include "context_pomdp.h"
#include "param.h"

#include "ros/ros.h"
#include <std_msgs/Int32.h>

#include <msg_builder/car_info.h>
#include <msg_builder/peds_info.h>
#include <msg_builder/ped_info.h>

#include <nav_msgs/OccupancyGrid.h>
#include <std_msgs/Bool.h>
#include <std_srvs/Empty.h>
#include <msg_builder/TrafficAgentArray.h>
#include <msg_builder/AgentPathArray.h>
#include <msg_builder/Lanes.h>
#include <msg_builder/Obstacles.h>
#include <msg_builder/PomdpCmd.h>
#include <msg_builder/SendStateAction.h>
#include <msg_builder/Belief.h>
#include <msg_builder/SimplePath.h>
#include <msg_builder/SimplePathSet.h>
#include "world_simulator.h"
#include "threaded_print.h"

using namespace despot;


#include "neural_prior.h"
#include "carla/client/Client.h"
#include "carla/geom/Vector2D.h"
#include "carla/sumonetwork/SumoNetwork.h"
#include "carla/occupancy/OccupancyMap.h"
#include "carla/segments/SegmentMap.h"
namespace cc = carla::client;
namespace cg = carla::geom;
namespace sumo = carla::sumonetwork;
namespace occu = carla::occupancy;
namespace segm = carla::segments;

ros::ServiceClient WorldSimulator::data_client_;
ros::ServiceClient WorldSimulator::fetch_data_srv_;


#undef LOG
#define LOG(lv) \
if (despot::logging::level() < despot::logging::ERROR || despot::logging::level() < lv) ; \
else despot::logging::stream(lv)
#include <despot/util/logging.h>

static sumo::SumoNetwork network_;
static occu::OccupancyMap network_occupancy_map_;
static segm::SegmentMap network_segment_map_;
static nav_msgs::OccupancyGrid raw_map_;

bool SimulatorBase::agents_data_ready = false;
bool SimulatorBase::agents_path_data_ready = false;

double pub_frequency = 9.0;

double navposeToHeadingDir(const geometry_msgs::Pose & msg) {
	/* get yaw angle [-pi, pi)*/
	tf::Pose pose;
	tf::poseMsgToTF(msg, pose);

	double yaw;
	yaw = tf::getYaw(pose.getRotation());
	if (yaw < 0)
		yaw += 2 * 3.1415926;
	return yaw;
}

WorldSimulator::WorldSimulator(ros::NodeHandle& _nh, DSPOMDP* model,
		unsigned seed, std::string map_location, int summit_port) :
		SimulatorBase(_nh), worldModel(SimulatorBase::world_model), model_(
		model), last_acc_(-1), goal_reached_(false),
		paths_time_stamp_(-1), car_time_stamp_(0), agents_time_stamp_(0), 
		safe_action_(0), time_scale(1.0), topic_state_ready(false),
		car_data_ready(false), World() {

	map_location_ = map_location;
	summit_port_ = summit_port;
	worldModel.summit_port = summit_port_;
	worldModel.map_location = map_location_;
	worldModel.ConnectCarla();

	worldModel.InitGamma();

	ResetWorldModel();
}

WorldSimulator::~WorldSimulator() {
	msg_builder::PomdpCmd cmd;
	cmd.target_speed = 0.0;
	cmd.cur_speed = real_speed;
	cmd.acc = -ModelParams::ACC_SPEED;
	cmd.steer = 0;
	cmdPub_.publish(cmd);
}


void WorldSimulator::ConnectCarla(){
	// Connect with CARLA server
//	logi << "Connecting with carla world at port " << summit_port_ << endl;
//    auto client = cc::Client("127.0.0.1", summit_port_);
//    client.SetTimeout(10s);
//    auto world = client.GetWorld();
//
//    // Define bounds.
//    cg::Vector2D scenario_center, scenario_min, scenario_max, geo_min, geo_max;
//	if (map_location_ == "map"){
//		scenario_center = cg::Vector2D(825, 1500);
//		scenario_min = cg::Vector2D(450, 1100);
//		scenario_max = cg::Vector2D(1200, 1900);
//		geo_min = cg::Vector2D(1.2894000, 103.7669000);
//		geo_max = cg::Vector2D(1.3088000, 103.7853000);
//	}
//	else if (map_location_ == "meskel_square"){
//		scenario_center = cg::Vector2D(450, 400);
//		scenario_min = cg::Vector2D(350, 300);
//		scenario_max = cg::Vector2D(550, 500);
//		geo_min = cg::Vector2D(9.00802, 38.76009);
//		geo_max = cg::Vector2D(9.01391, 38.76603);
//	}
//	else if (map_location_ == "magic"){
//		scenario_center = cg::Vector2D(180, 220);
//		scenario_min = cg::Vector2D(80, 120);
//		scenario_max = cg::Vector2D(280, 320);
//		geo_min = cg::Vector2D(51.5621800, -1.7729100);
//		geo_max = cg::Vector2D(51.5633900, -1.7697300);
//	}
//	else if (map_location_ == "highway"){
//		scenario_center = cg::Vector2D(100, 400);
//		scenario_min = cg::Vector2D(0, 300);
//		scenario_max = cg::Vector2D(200, 500);
//		geo_min = cg::Vector2D(1.2983800, 103.7777000);
//		geo_max = cg::Vector2D(1.3003700, 103.7814900);
//	}
//	else if (map_location_ == "chandni_chowk"){
//		scenario_center = cg::Vector2D(380, 250);
//		scenario_min = cg::Vector2D(260, 830);
//		scenario_max = cg::Vector2D(500, 1150);
//		geo_min = cg::Vector2D(28.653888, 77.223296);
//		geo_max = cg::Vector2D(28.660295, 77.236850);
//	}
//	else if (map_location_ == "shi_men_er_lu"){
//		scenario_center = cg::Vector2D(1010, 1900);
//		scenario_min = cg::Vector2D(780, 1700);
//		scenario_max = cg::Vector2D(1250, 2100);
//		geo_min = cg::Vector2D(31.229828, 121.438702);
//		geo_max = cg::Vector2D(31.242810, 121.464944);
//	}
//	else if (map_location_ == "beijing"){
//		scenario_center = cg::Vector2D(2080, 1860);
//		scenario_min = cg::Vector2D(490, 1730);
//		scenario_max = cg::Vector2D(3680, 2000);
//		geo_min = cg::Vector2D(39.8992818, 116.4099687);
//		geo_max = cg::Vector2D(39.9476116, 116.4438916);
//	}

    std::string homedir = getenv("HOME");
	auto summit_root = homedir + "/summit/";
	string file_flag = summit_root + "Data/" + map_location_;
	logi << "[ConnectCarla] SUMMIT map file_flag " << file_flag + ".net.xml" << endl;
	logi << "[ConnectCarla] SUMMIT map file_flag " << file_flag + ".network.wkt" << endl;

	network_ = sumo::SumoNetwork::Load(file_flag + ".net.xml");
	network_occupancy_map_ = occu::OccupancyMap::Load(file_flag + ".network.wkt");
	network_segment_map_ = network_.CreateSegmentMap();

	logi << "[ConnectCarla] SUMMIT size(network_segment_map_)=" << sizeof(network_segment_map_) << endl;

}

/**
 * [Essential]
 * Establish connection to simulator or system
 */
bool WorldSimulator::Connect() {
	cerr << "DEBUG: Connecting with world" << endl;

	if (Globals::config.state_source == STATE_FROM_SIM) {

//		agent_sub_ = nh.subscribe("agent_array", 1,	&WorldSimulator::AgentArrayCallback, this);
//		agent_path_sub_ = nh.subscribe("agent_path_array", 1, &WorldSimulator::AgentPathArrayCallback, this);
		world_Sub_ = nh.subscribe("world_agents", 1, &WorldSimulator::WorldAgentsCallBack, this);

		cmdPub_ = nh.advertise<msg_builder::PomdpCmd>("cmd_action_pomdp", 1);
		unlabelled_belief_pub_ = nh.advertise<msg_builder::Belief>("unlabelled_belief", 10);
		decision_path_pub_ = nh.advertise<nav_msgs::Path>("plan", 1);

		ego_sub_ = nh.subscribe("ego_state", 1, &WorldSimulator::EgoStateCallBack, this);
		ego_dead_sub_ = nh.subscribe("ego_dead", 1,	&WorldSimulator::EgoDeadCallBack, this);

//		steerSub_ = nh.subscribe("purepursuit_cmd_steer", 1, &WorldSimulator::CmdSteerCallback, this);
//		lane_change_Sub_ = nh.subscribe("gamma_lane_decision", 1, &WorldSimulator::LaneChangeCallback, this);

//		last_lane_change_Sub_ = nh.subscribe("last_lane_decision", 1,
//				&WorldSimulator::LastLaneChangeCallback, this);

		ConnectCarla();

		ros::spinOnce();

		auto start = Time::now();
		bool agent_data_ok = false, car_data_ok = false, agent_flag_ok = false;
		while (Globals::ElapsedTime(start) < 30.0) {
			auto agent_data = ros::topic::waitForMessage<
					msg_builder::WorldAgents>("world_agents",
					ros::Duration(1));
			if (agent_data && !agent_data_ok) {
				logi << "world_agents get at the " << Globals::ElapsedTime()
						<< "th second" << endl;
				agent_data_ok = true;
			}

			ros::spinOnce();

			auto car_data = ros::topic::waitForMessage<msg_builder::car_info>(
					"ego_state", ros::Duration(1));
			if (car_data && !car_data_ok) {
				logi << "ego_state get at the " << Globals::ElapsedTime()
						<< "th second" << endl;
				car_data_ok = true;
			}
			ros::spinOnce();

			auto agent_ready_bool = ros::topic::waitForMessage<std_msgs::Bool>(
					"agents_ready", ros::Duration(1));
			if (agent_ready_bool && !agent_flag_ok) {
				logi << "agents ready at get at the " << Globals::ElapsedTime()
						<< "th second" << endl;
				agent_flag_ok = true;
			}
			ros::spinOnce();

			if (agent_data_ok && car_data_ok && agent_flag_ok)
				break;
		}

		if (Globals::ElapsedTime(start) >= 30.0)
			ERR("No agent array messages received after 30 seconds.");
	} else if (Globals::config.state_source == STATE_FROM_TOPIC) {
		ConnectCarla();
		unlabelled_belief_sub_ = nh.subscribe("unlabelled_belief", 1,
				&WorldSimulator::UnlabelledBeliefCallBack, this);
		fetch_data_srv_ = nh.serviceClient<std_srvs::Empty>("/fetch_data");

		while(true) {
			bool up = fetch_data_srv_.waitForExistence(ros::Duration(1));
			if (up) {
				logi << "/fetch_data service is ready" << endl;
				break;
			}
			else
				logi << "waiting for /fetch_data service to be ready..." << endl;
		}

		logi << "Subscribers and Publishers created at the "
				<< Globals::ElapsedTime() << "th second" << endl;
	}

	ros::NodeHandle n("~");
	data_client_ = n.serviceClient<msg_builder::SendStateAction>("/send_state_action");
	while(true) {
		bool up = data_client_.waitForExistence(ros::Duration(1));
		if (up) {
			logi << "/send_state_action service is ready" << endl;
			break;
		}
		else
			logi << "waiting for /send_state_action service to be ready..." << endl;
	}
	
	logi << "Subscribers and Publishers created at the "
			<< Globals::ElapsedTime() << "th second" << endl;

	return true;
}

/**
 * [Essential]
 * Initialize or reset the (simulation) environment, return the start state if applicable
 */
State* WorldSimulator::Initialize() {

	cerr << "DEBUG: Initializing world in WorldSimulator::Initialize" << endl;

	safe_action_ = 2;
	cmd_speed_ = 0.0;
	lane_decision_ = 0.0;
	goal_reached_ = false;
	last_acc_ = 0;

	if(SolverPrior::nn_priors.size() == 0){
		ERR("No nn_prior exist");
	}

	for (int i = 0; i < SolverPrior::nn_priors.size(); i++) {
		PedNeuralSolverPrior * nn_prior =
				static_cast<PedNeuralSolverPrior *>(SolverPrior::nn_priors[i]);
		nn_prior->raw_map_ = raw_map_;
		nn_prior->map_received = true;
		nn_prior->Init();
	}
	return NULL;
}

std::map<double, AgentStruct&> WorldSimulator::GetSortedAgents() {
	std::map<double, AgentStruct&> sorted_agents;
	for (std::map<int, AgentStruct>::iterator it = exo_agents_.begin();
			it != exo_agents_.end(); ++it) {
		AgentStruct& agent = it->second;
		double dis_to_car = COORD::EuclideanDistance(car_.pos, agent.pos);
		sorted_agents.insert ( std::pair<double, AgentStruct&>(dis_to_car, agent) );
	}
	return sorted_agents;
}

/**
 * [Optional]
 * To help construct initial belief to print debug informations in Logger
 */
State* WorldSimulator::GetCurrentState() {
	logi << "Spinning once for GetCurrentState" << endl;
	ros::spinOnce(); // get the lasted states of the world

	if (path_from_decision_.size() == 0 && car_data_ready) {
		Path * p = worldModel.ParseLanePath(car_.pos, car_.heading_dir, LaneCode::KEEP);
		p->CopyTo(path_from_decision_);
		world_model.SetPath(path_from_decision_);
	}

	if (car_data_ready)
		updateLanes(car_.pos);

	if (Globals::config.state_source == STATE_FROM_SIM) {
//		if (path_from_topic_.size() == 0) {
//			logi << "[GetCurrentState] path topic not ready yet..." << endl;
//			return NULL;
//		}

		current_state_.car = car_;

		int n = 0;
		std::map<double, AgentStruct&> sorted_agents = GetSortedAgents();
		for (auto it = sorted_agents.begin();
				it != sorted_agents.end(); ++it) {
			current_state_.agents[n] = it->second;
			n++;
			if (n >= ModelParams::N_PED_WORLD)
				break;
		}
		current_state_.num = n;
		current_state_.time_stamp = min(car_time_stamp_, agents_time_stamp_);

		if (logging::level() >= logging::DEBUG) {
			logi << "current world state:" << endl;
			static_cast<ContextPomdp*>(model_)->PrintWorldState(current_state_);
		}
		logi << " current state time stamp " << current_state_.time_stamp << endl;
		return static_cast<State*>(&current_state_);
	} else if (Globals::config.state_source == STATE_FROM_TOPIC)
		return NULL;
}


double WorldSimulator::StepReward(PomdpStateWorld& state, ACT_TYPE action, double ttc) {
	double reward = 0.0;

//	if (worldModel.IsGlobalGoal(state.car)) {
//		reward = ModelParams::GOAL_REWARD;
//		return reward;
//	}

	ContextPomdp* ContextPomdp_model = static_cast<ContextPomdp*>(model_);

	if (!worldModel.IsInMap(state.car)) {
		reward = ContextPomdp_model->CrashPenalty(state);
		return reward;
	}

	if (state.car.vel > 1.0 && worldModel.InRealCollision(state, 120.0)) { /// collision occurs only when car is moving
		reward = ContextPomdp_model->CrashPenalty(state);
		return reward;
	}

	// TTC penalty
	reward += ContextPomdp_model->TTCPenalty(ttc, state);
	// TTC penalty
	reward += ContextPomdp_model->InvalidActionPenalty(action, state);
	// Smoothness control
	reward += ContextPomdp_model->ActionPenalty(action);

	// Speed control: Encourage higher speed
	reward += ContextPomdp_model->MovementPenalty(state,
			ContextPomdp_model->GetLane(action));
	return reward;
}

bool WorldSimulator::Emergency(PomdpStateWorld* curr_state) {
//	double mindist = numeric_limits<double>::infinity();
//	for (std::map<int, AgentStruct>::iterator it = exo_agents_.begin();
//			it != exo_agents_.end(); ++it) {
//		AgentStruct& agent = it->second;
//		double d = COORD::EuclideanDistance(car_.pos, agent.pos);
//		if (d < mindist)
//			mindist = d;
//	}
//	cout << "Emergency mindist = " << mindist << endl;
//	return (mindist < 1.5);

	return false;
	return curr_state->car.vel > 0.0 && worldModel.InCollision(*curr_state, 60.0);
}

bool WorldSimulator::Terminal(PomdpStateWorld* curr_state) {
//	if (worldModel.IsGlobalGoal(curr_state->car)) {
//		cout << "--------------------------- goal reached ----------------------------"
//			 << endl;
//		cout << "" << endl;
//		goal_reached_ = true;
//		return true;
//	}
	if (Globals::config.state_source == STATE_FROM_SIM) {

		if (!worldModel.IsInMap(curr_state->car)) {
			cerr << "--------------------------- out_of_map ----------------------------"
					 << endl;
			return true;
		}

		int collision_peds_id;
		if (curr_state->car.vel > 0.5 * time_scale
				&& worldModel.InRealCollision(*curr_state, collision_peds_id,
						120.0)) {
			cerr << "--------------------------- collision = 1 ----------------------------"
				 << endl;
			cerr << "collision ped: " << collision_peds_id << endl;
	//		return true;
		}

		if (worldModel.path.size() > 0
				&& COORD::EuclideanDistance(curr_state->car.pos, worldModel.path[0]) > 4.0) {
			cerr << "---------------------------- Path offset too high ----------------------------"
					<< endl;
			return true;
		}
	}

	return false;
}
/**
 * [Essential]
 * send action, receive reward, obs, and terminal
 * @param action Action to be executed in the real-world system
 * @param obs    Observation sent back from the real-world system
 */
bool WorldSimulator::ExecuteAction(ACT_TYPE action, OBS_TYPE& obs) {

	if (action == -1)
		return false;

	if (Globals::config.state_source == STATE_FROM_SIM) {
		logi << "ExecuteAction at the " << Globals::ElapsedTime() << "th second"
				<< endl;

		/* Update state */
		PomdpStateWorld* curr_state =
				static_cast<PomdpStateWorld*>(GetCurrentState());

		if (logging::level() >= logging::DEBUG) {
			logi << "Executing action for state:" << endl;
			static_cast<ContextPomdp*>(model_)->PrintWorldState(*curr_state);
		}

		double acc, lane;

		logv << "[WorldSimulator::" << __FUNCTION__
				<< "] Update steering and target speed" << endl;

		bool emergency = ReviseAction(curr_state, action);

		logv << "[WorldSimulator::" << __FUNCTION__
				<< "] Update lane and target speed" << endl;

		// Updating cmd steering and speed. They will be send to vel_pulisher by timer
		UpdateCmds(buffered_action_, emergency);
		UpdatePath(buffered_action_);

		PublishCmdAction(buffered_action_);
		PublishPath();

		double ttc = static_cast<ContextPomdp*>(model_)->TimeToCollision(
				curr_state, buffered_action_);

		double step_reward = StepReward(*curr_state, buffered_action_, ttc);
		cout << "action **= " << buffered_action_ << endl;
		cout << "reward **= " << step_reward << endl;
		cout << "ttc **= " << ttc << endl;

		safe_action_ = action;

		sendStateActionData(*curr_state, buffered_action_, step_reward,
			cmd_speed_, ttc);

		PublishUnlabelledBelief();

		if(Terminal(curr_state)) {
			ERR("Termination of episode due to terminal state.");
		}

		/* Receive observation.
		 * Caution: obs info is not up-to-date here. Don't use it
		 */

		logv << "[WorldSimulator::" << __FUNCTION__ << "] Generate obs" << endl;
		obs = static_cast<ContextPomdp*>(model_)->StateToIndex(GetCurrentState());
	} else if (Globals::config.state_source == STATE_FROM_TOPIC) {
		PomdpStateWorld curr_state;
		PomdpState* state_from_topic = static_cast<PomdpState*>(
				SolverPrior::nn_priors[0]->unlabelled_belief_[0]);
		curr_state.assign(*state_from_topic);

		bool emergency = ReviseAction(&curr_state, action);

		double ttc = static_cast<ContextPomdp*>(model_)->TimeToCollision(
					&curr_state, action);

		double step_reward = StepReward(curr_state, buffered_action_, ttc);
		cout << "action **= " << action << endl;
		cout << "reward **= " << step_reward << endl;
		cout << "ttc **= " << ttc << endl;

		safe_action_ = action;

		sendStateActionData(curr_state, buffered_action_, step_reward,
			cmd_speed_, ttc);

		ros::spinOnce();
	}
	return goal_reached_;
}

void WorldSimulator::UpdateCmds(ACT_TYPE action, bool emergency) {
	if (logging::level() >= logging::INFO) {
		worldModel.path.Text();
	}

	logi << "[update_cmds_naive] Buffering action " << action << endl;
	float acc = static_cast<ContextPomdp*>(model_)->GetAcceleration(action);
	double speed_step = ModelParams::ACC_SPEED / ModelParams::CONTROL_FREQ;

	cmd_speed_ = real_speed;
	if (acc > 0.0)
		cmd_speed_ = real_speed + speed_step;
	else if (acc < 0.0) {
		float scale = 1.0;
		if (emergency)
			scale = 2.0;
		cmd_speed_ = real_speed - scale * speed_step;
	}
	cmd_speed_ = max(min(cmd_speed_, ModelParams::VEL_MAX), 0.0);
	
	lane_decision_ = static_cast<ContextPomdp*>(model_)->GetLane(action);
	worldModel.last_decision_lane = static_cast<ContextPomdp*>(model_)->GetLaneID(action);
	logi << "Executing action:" << action << " lane/acc/speed = "
			<< lane_decision_ << "/" << acc << "/" << cmd_speed_ << endl;
}

void WorldSimulator::PublishCmdAction(const ros::TimerEvent &e) {
	// for timer
	PublishCmdAction(buffered_action_);
}

void WorldSimulator::PublishCmdAction(ACT_TYPE action) {
	msg_builder::PomdpCmd cmd;
	cmd.target_speed = cmd_speed_;
	cmd.cur_speed = real_speed;
	cmd.acc = static_cast<ContextPomdp*>(model_)->GetAcceleration(action);
	cmd.lane = lane_decision_;
	cout << "[PublishCmdAction] time stamp" << Globals::ElapsedTime() << endl;
	cout << "[PublishCmdAction] target speed " << cmd_speed_ << " target lane "
			<< lane_decision_ << endl;
	cmdPub_.publish(cmd);
}


void WorldSimulator::UpdatePath(ACT_TYPE action) {
	if (action == -1) {

	} else {
		int lane = static_cast<ContextPomdp*>(model_)->GetLaneID(action);
		int pos = worldModel.root_path_id_map[lane];
		worldModel.root_path_id_map.clear();
		const Path *p = WorldModel::path_tree_->at(pos);
		p->CopyTo(path_from_decision_);
		worldModel.ExtendPath(car_.pos, path_from_decision_, 20.0);
		world_model.SetPath(path_from_decision_);

		if (path_from_decision_.GetLength() < 3.0)
			ERR("No path available !!!");
	}
}

void WorldSimulator::PublishPath() {
	ros::Time plan_time = ros::Time::now();
	nav_msgs::Path path_msg;
	for(const COORD& p: path_from_decision_) {
		geometry_msgs::PoseStamped pose;
		pose.header.stamp = plan_time;
		pose.header.frame_id = 'map';
		pose.pose.position.x = p.x;
		pose.pose.position.y = p.y;
		pose.pose.position.z = 0.0;
		pose.pose.orientation.x = 0.0;
		pose.pose.orientation.y = 0.0;
		pose.pose.orientation.z = 0.0;
		pose.pose.orientation.w = 1.0;
		path_msg.poses.push_back(pose);
	}
	decision_path_pub_.publish(path_msg);
}

void CalBBExtents(COORD pos, double heading_dir, vector<COORD>& bb,
		double& extent_x, double& extent_y) {
	COORD forward_vec = COORD(cos(heading_dir), sin(heading_dir));
	COORD sideward_vec = COORD(-sin(heading_dir), cos(heading_dir));

	for (auto& point : bb) {
		extent_x = max((point - pos).dot(sideward_vec), extent_x);
		extent_y = max((point - pos).dot(forward_vec), extent_y);
	}
}

void CalBBExtents(AgentStruct& agent, std::vector<COORD>& bb,
		double heading_dir) {
	if (agent.type == AgentType::ped) {
		agent.bb_extent_x = 0.3;
		agent.bb_extent_y = 0.3;
	} else {
		agent.bb_extent_x = 0.0;
		agent.bb_extent_y = 0.0;
	}
	CalBBExtents(agent.pos, heading_dir, bb, agent.bb_extent_x,
			agent.bb_extent_y);
}

void WorldSimulator::WorldAgentsCallBack(msg_builder::WorldAgents data) {
	AgentArrayCallback(data.agents);
	AgentPathArrayCallback(data.agent_paths);
}

void WorldSimulator::AgentArrayCallback(msg_builder::TrafficAgentArray data) {
	agents_topic_ = data;
	double data_sec = data.header.stamp.sec;  // std_msgs::time
	double data_nsec = data.header.stamp.nsec;
	double data_time_sec = data_sec + data_nsec * 1e-9;
	agents_time_stamp_ = data_time_sec;
	DEBUG(
			string_sprintf("receive %d agents at time %f", data.agents.size(),
					Globals::ElapsedTime()));

	exo_agents_.clear();
	for (msg_builder::TrafficAgent& agent : data.agents) {
		std::string agent_type = agent.type;
		int id = agent.id;
		exo_agents_[id] = AgentStruct();
		exo_agents_[id].id = id;
		if (agent_type == "car")
			exo_agents_[id].type = AgentType::car;
		else if (agent_type == "bike")
			exo_agents_[id].type = AgentType::car;
		else if (agent_type == "ped")
			exo_agents_[id].type = AgentType::ped;
		else
			ERR(string_sprintf("Unsupported type %s", agent_type));

		exo_agents_[id].pos = COORD(agent.pose.position.x,
				agent.pose.position.y);
		exo_agents_[id].vel = COORD(agent.vel.x, agent.vel.y);
		exo_agents_[id].speed = exo_agents_[id].vel.Length();
		exo_agents_[id].heading_dir = navposeToHeadingDir(agent.pose);

		std::vector<COORD> bb;
		for (auto& corner : agent.bbox.points) {
			bb.emplace_back(corner.x, corner.y);
		}
		CalBBExtents(exo_agents_[id], bb, exo_agents_[id].heading_dir);

		assert(exo_agents_[id].bb_extent_x > 0);
		assert(exo_agents_[id].bb_extent_y > 0);
	}

	if (logging::level() >= logging::DEBUG)
		worldModel.PrintPathMap();

	SimulatorBase::agents_data_ready = true;
}

void WorldSimulator::AgentPathArrayCallback(msg_builder::AgentPathArray data) {
	agents_path_topic_ = data;
	double data_sec = data.header.stamp.sec;  // std_msgs::time
	double data_nsec = data.header.stamp.nsec;
	double data_time_sec = data_sec + data_nsec * 1e-9;
	paths_time_stamp_ = data_time_sec;
	DEBUG(
			string_sprintf("receive %d agent paths at time %f",
					data.agents.size(), Globals::ElapsedTime()));

	worldModel.id_map_belief_reset.clear();
	worldModel.id_map_paths.clear();
	worldModel.id_map_num_paths.clear();

	for (msg_builder::AgentPaths& agent : data.agents) {
		std::string agent_type = agent.type;
		int id = agent.id;

		auto it = exo_agents_.find(id);
		if (it != exo_agents_.end()) {
			if (agent_type == "ped")
				exo_agents_[id].cross_dir = agent.cross_dirs[0];

			worldModel.id_map_belief_reset[id] = agent.reset_intention;
			worldModel.id_map_paths[id] = worldModel.ParsePathCandidates(
					agent.path_start_rp.edge,
					agent.path_start_rp.lane,
					agent.path_start_rp.segment,
					agent.path_start_rp.offset,
					agent_type);
			worldModel.id_map_num_paths[id] = worldModel.id_map_paths[id].size();
		}
	}

	SimulatorBase::agents_path_data_ready = true;
}

double xylength(geometry_msgs::Point32 p) {
	return sqrt(p.x * p.x + p.y * p.y);
}

void WorldSimulator::EgoDeadCallBack(const std_msgs::Bool ego_dead) {
	ERR("Ego vehicle killed in ego_vehicle.py");
}

bool initial = true;
void WorldSimulator::EgoStateCallBack(
		const msg_builder::car_info::ConstPtr car) {
	
	logi << "get car state at t=" << Globals::ElapsedTime() << endl;
	const msg_builder::car_info& ego_car = *car;
	car_.pos = COORD(ego_car.car_pos.x, ego_car.car_pos.y);
	car_.heading_dir = ego_car.car_yaw;
	car_.vel = ego_car.car_speed;

	real_speed = COORD(ego_car.car_vel.x, ego_car.car_vel.y).Length();
	if (real_speed > ModelParams::VEL_MAX * 1.3) {
		ERR(
				string_sprintf(
						"Unusual car vel (too large): %f. Check the speed controller for possible problems (VelPublisher.cpp)",
						real_speed));
	}

	logi << "get car meta info" << endl;

	if (initial) {
//		ERR("Pause");

		ModelParams::CAR_FRONT = COORD(
				ego_car.front_axle_center.x - ego_car.car_pos.x,
				ego_car.front_axle_center.y - ego_car.car_pos.y).Length();
		ModelParams::CAR_REAR = COORD(
				ego_car.rear_axle_center.y - ego_car.car_pos.y,
				ego_car.rear_axle_center.y - ego_car.car_pos.y).Length();
		ModelParams::CAR_WHEEL_DIST = ModelParams::CAR_FRONT
				+ ModelParams::CAR_REAR;

		ModelParams::MAX_STEER_ANGLE = ego_car.max_steer_angle / 180.0 * M_PI;

		ModelParams::CAR_WIDTH = 0;
		ModelParams::CAR_LENGTH = 0;

		double car_yaw = car_.heading_dir;
		COORD tan_dir(-sin(car_yaw), cos(car_yaw));
		COORD along_dir(cos(car_yaw), sin(car_yaw));
		for (auto& point : ego_car.car_bbox.points) {
			COORD p(point.x - ego_car.car_pos.x, point.y - ego_car.car_pos.y);
			double proj = p.dot(tan_dir);
			ModelParams::CAR_WIDTH = max(ModelParams::CAR_WIDTH, fabs(proj));
			proj = p.dot(along_dir);
			ModelParams::CAR_LENGTH = max(ModelParams::CAR_LENGTH, fabs(proj));
		}
		ModelParams::CAR_WIDTH = ModelParams::CAR_WIDTH * 2;
		ModelParams::CAR_LENGTH = ModelParams::CAR_LENGTH * 2;
		ModelParams::CAR_FRONT = ModelParams::CAR_LENGTH / 2.0;
		initial = false;
	}

	logi << "get car bounding gbox" << endl;

	for (auto& n_prior: SolverPrior::nn_priors)
		if (n_prior != NULL)
			static_cast<PedNeuralSolverPrior*>(n_prior)->update_ego_car_shape(
				ego_car.car_bbox.points);

	car_data_ready = true;
	logi << "car_info call_back done" << endl;
}

void WorldSimulator::sendStateActionData(PomdpStateWorld& planning_state,
		ACT_TYPE action, float reward, float cmd_vel, float ttc) {
	
	if (Globals::config.use_prior)
		if (!SolverPrior::nn_priors[0]->policy_ready())
			return;

	msg_builder::SendStateAction message;

	at::Tensor state_tensor = SolverPrior::nn_priors[0]->root_input();
	message.request.state = std::vector<unsigned char>(
			state_tensor.data<SRV_DATA_TYPE>(),
			state_tensor.data<SRV_DATA_TYPE>() + state_tensor.numel());

	at::Tensor semantic_tensor = SolverPrior::nn_priors[0]->root_semantic_input();
	message.request.semantic = std::vector<SRV_DATA_TYPE>(
			semantic_tensor.data<SRV_DATA_TYPE>(),
			semantic_tensor.data<SRV_DATA_TYPE>() + semantic_tensor.numel());

	// std::vector<double> policy_probs = SolverPrior::nn_priors[0]->GetPolicyProbs();
	// double value = SolverPrior::nn_priors[0]->GetValue();

	// message.request.action_probs =
	// 		std::vector<SRV_DATA_TYPE>(policy_probs.begin(), policy_probs.end());
	message.request.action = action;
	// message.request.value = value;
	message.request.vel = car_.vel;

	message.request.reward = StepReward(planning_state, action, ttc);
	message.request.value_col_factor = SolverPrior::despot_value_col_factor;
	message.request.value_ncol_factor = SolverPrior::despot_value_ncol_factor;

	bool terminal = Terminal(&planning_state);
	message.request.is_terminal = terminal;
	message.request.ttc = ttc;

	logd << "calling service send_state_action" << endl;

	if (!data_client_.call(message))
	{
		ERR("query of send_state_action service failed!!!");
	}

	if (terminal) {
		DEBUG("Sending terminal state.");
		if (!data_client_.call(message))
		{
			ERR("query of send_state_action service failed!!!");
		}

		sleep(1); // wait for the reward to be sent
	}
}

float publish_map_rad = 100.0;

void WorldSimulator::updateLanes(COORD car_pos){
	auto start = Time::now();

	auto OM_bound = occu::OccupancyMap(
	            cg::Vector2D(car_pos.x - publish_map_rad, car_pos.y - publish_map_rad),
	            cg::Vector2D(car_pos.x + publish_map_rad, car_pos.y + publish_map_rad));

	logi << "[updateLanes] network_segment_map_ size: " << network_segment_map_.GetSegments().size() << endl;

	auto local_lanes = network_segment_map_.Intersection(OM_bound);

	logi << "[updateLanes] num of lanes found for car_pos "
			<< car_pos.x << "," << car_pos.y << ": "
			<< local_lanes.GetSegments().size() << endl;
	worldModel.local_lane_segments_.resize(0);

	for (auto& lane_seg : local_lanes.GetSegments()){
		msg_builder::LaneSeg lane_seg_tmp;
		lane_seg_tmp.start.x = lane_seg.start.x;
		lane_seg_tmp.start.y = lane_seg.start.y;
		lane_seg_tmp.end.x = lane_seg.end.x;
		lane_seg_tmp.end.y = lane_seg.end.y;

		worldModel.local_lane_segments_.push_back(lane_seg_tmp);
	}
	logi << "[updateLanes] used time " << Globals::ElapsedTime(start) << endl;

	if (worldModel.local_lane_segments_.size() == 0)
		ERR("no lane segment in range");
}

void WorldSimulator::updateObs(COORD car_pos){
	auto OM_bound = occu::OccupancyMap(
		            cg::Vector2D(car_pos.x - publish_map_rad, car_pos.y - publish_map_rad),
		            cg::Vector2D(car_pos.x + publish_map_rad, car_pos.y + publish_map_rad));

	auto local_obstacles = OM_bound.Difference(network_occupancy_map_);
	worldModel.local_obs_contours_.resize(0);

	for (auto& polygon : local_obstacles.GetPolygons()){
		geometry_msgs::Polygon polygon_tmp;
		auto& outer_contour = polygon[0];
		for (auto& point: outer_contour){
			geometry_msgs::Point32 tmp;
			tmp.x = point.x;
			tmp.y = point.y;
			tmp.z = 0.0;
			polygon_tmp.points.push_back(tmp);
		}
		worldModel.local_obs_contours_.push_back(polygon_tmp);
	}
}

void WorldSimulator::ResetWorldModel() {
	worldModel.ResetPathTree();
	UpdatePath(-1);
	agents_topic_bak_ = agents_topic_;
	agents_path_topic_bak_ = agents_path_topic_;

	for (auto& prior: SolverPrior::nn_priors) {
		prior->CleanUnlabelledBelief();
	}
}

bool WorldSimulator::ReviseAction(PomdpStateWorld* curr_state, ACT_TYPE action) {
	bool emergency = false;
	if (Emergency(curr_state)) {
		lane_decision_ = LaneCode::KEEP;
		cmd_speed_ = -1;
		emergency = true;
		buffered_action_ = static_cast<ContextPomdp*>(model_)->GetActionID(LaneCode::KEEP, AccCode::DEC);
		cout
				<< "--------------------------- emergency ----------------------------"
				<< endl;
	} else {
		buffered_action_ = action;
	}
	return emergency;
}

void WorldSimulator::PublishUnlabelledBelief() {

	std::map<int, int> topic_agent_map;
	for (auto& agent: agents_path_topic_bak_.agents) {
		topic_agent_map[agent.id] = 0;
	}

	for (SolverPrior* prior: SolverPrior::nn_priors) {
		if (prior->unlabelled_belief_.size()==0)
			continue;

		msg_builder::Belief belief_msg;

//		std::map<int, std::vector<Path>> included_paths;
		for (State* particle: prior->unlabelled_belief_) {
			const PomdpState* s = static_cast<PomdpState*>(particle);
			const CarStruct& car = s->car;
			msg_builder::State particle_msg;
			particle_msg.car.pos.x = car.pos.x;
			particle_msg.car.pos.y = car.pos.y;
			particle_msg.car.pos.z = 0.0;
			particle_msg.car.heading_dir = car.heading_dir;
			particle_msg.car.path_idx = -1;
			particle_msg.car.vel = car.vel;
			particle_msg.num = s->num;
			for (int i = 0; i < s->num; i++) {
				const AgentStruct& agent = s->agents[i];

				if (topic_agent_map.find(agent.id) == topic_agent_map.end()) {
					tout << "agent " << agent.id << " (pos " << i <<" in " << s->num <<") is not in backed-up world agents topic"
					<< endl;
				}

				msg_builder::AgentStruct agent_msg;
				agent_msg.bb_extent_x = agent.bb_extent_x;
				agent_msg.bb_extent_y = agent.bb_extent_y;
				agent_msg.cross_dir = agent.cross_dir;
				agent_msg.heading_dir = agent.heading_dir;
				agent_msg.id = agent.id;
				agent_msg.intention = agent.intention;

//				if (included_paths.find(agent.id) == included_paths.end() ) {
//					included_paths.insert(
//						std::make_pair(agent.id, worldModel.id_map_paths_bak[agent.id]));
//				}

				agent_msg.mode = agent.mode;
				agent_msg.pos.x = agent.pos.x;
				agent_msg.pos.y = agent.pos.y;
				agent_msg.pos.z = 0.0;
				agent_msg.pos_along_path = agent.pos_along_path;
				agent_msg.speed = agent.speed;
				agent_msg.type = agent.type;
				agent_msg.vel.x = agent.vel.x;
				agent_msg.vel.y = agent.vel.y;
				agent_msg.vel.z = 0.0;
				particle_msg.agents.push_back(agent_msg);
			}
			belief_msg.particles.push_back(particle_msg);
		}

		if (prior->unlabelled_hist_images_.size() > 0) {
			at::Tensor image = prior->unlabelled_hist_images_[0];
			detectNAN(image);
			belief_msg.state_tensor_0 = std::vector<SRV_DATA_TYPE>(
					image.data<SRV_DATA_TYPE>(), image.data<SRV_DATA_TYPE>() + image.numel());
			image = prior->unlabelled_hist_images_[1];
			detectNAN(image);
			belief_msg.state_tensor_1 = std::vector<SRV_DATA_TYPE >(
					image.data<SRV_DATA_TYPE>(), image.data<SRV_DATA_TYPE>() + image.numel());
			image = prior->unlabelled_hist_images_[2];
			detectNAN(image);
			belief_msg.state_tensor_2 = std::vector<SRV_DATA_TYPE>(
					image.data<SRV_DATA_TYPE>(), image.data<SRV_DATA_TYPE>() + image.numel());
			image = prior->unlabelled_hist_images_[3];
			detectNAN(image);
			belief_msg.state_tensor_3 = std::vector<SRV_DATA_TYPE>(
					image.data<SRV_DATA_TYPE>(), image.data<SRV_DATA_TYPE>() + image.numel());
		}

		for (float semantic: prior->unlabelled_semantic_) {
			belief_msg.semantic_tensor.push_back(semantic);
		}

		belief_msg.meta.car_front = ModelParams::CAR_FRONT;
		belief_msg.meta.car_rear = ModelParams::CAR_REAR;
		belief_msg.meta.car_width = ModelParams::CAR_WIDTH;
		belief_msg.meta.car_length = ModelParams::CAR_REAR;
		belief_msg.meta.car_wheel_dist = ModelParams::CAR_WHEEL_DIST;
		belief_msg.meta.max_steer_angle = ModelParams::MAX_STEER_ANGLE;

//		for( auto it = included_paths.begin(); it != included_paths.end(); ++it )
//		{
//			int agent_id = it->first;
//			std::vector<Path>& agent_path_list = it->second;
//			msg_builder::SimplePathSet path_list_msg;
//			path_list_msg.agent_id = agent_id;
//			for(Path& path : agent_path_list) {
//				msg_builder::SimplePath path_msg;
//				for (COORD& point: path) {
//					geometry_msgs::Point32 p32;
//					p32.x = point.x;
//					p32.y = point.y;
//					p32.z = 0.0;
//					path_msg.points.push_back(p32);
//				}
//				path_list_msg.pathlist.push_back(path_msg);
//			}
//			belief_msg.id_path_map.push_back(path_list_msg);
//		}

		belief_msg.agents = agents_topic_bak_;
		belief_msg.agent_paths = agents_path_topic_bak_;

		belief_msg.map_loc = map_location_;
		belief_msg.depth = prior->unlabelled_belief_depth_;
		unlabelled_belief_pub_.publish(belief_msg);
	}
}

void WorldSimulator::UnlabelledBeliefCallBack(msg_builder::Belief data) {
	if (data.map_loc != map_location_) // only process points relative to the current map_loc
		return;

	if (topic_state_ready)
		return;

	SolverPrior* prior = SolverPrior::nn_priors[0];
	cout << "get unlabeled belief at search depth " << data.depth << endl;

	for (State* s: prior->unlabelled_belief_)
		model_->Free(s);
	prior->unlabelled_belief_.resize(0);
	prior->unlabelled_hist_images_.resize(0);
	prior->unlabelled_semantic_.resize(0);

	cout << "get " << data.particles.size() << " particles" << endl;

	for (msg_builder::State& particle_msg: data.particles) {
		PomdpState* s = static_cast<PomdpState*>(model_->Allocate());
		CarStruct& car = s->car;
		car.pos.x = particle_msg.car.pos.x;
		car.pos.y = particle_msg.car.pos.y;
		car.heading_dir = particle_msg.car.heading_dir;
		car.path_idx = -1;
		car.vel = particle_msg.car.vel;
		s->num = particle_msg.num;
		for (int i = 0; i < s->num; i++) {
			AgentStruct& agent = s->agents[i];
			msg_builder::AgentStruct& agent_msg = particle_msg.agents[i];
			agent.bb_extent_x = agent_msg.bb_extent_x;
			agent.bb_extent_y = agent_msg.bb_extent_y;
			agent.cross_dir = agent_msg.cross_dir;
			agent.heading_dir = agent_msg.heading_dir;
			agent.id = agent_msg.id;
			agent.intention = agent_msg.intention;
			agent.mode = agent_msg.mode;
			agent.pos.x = agent_msg.pos.x;
			agent.pos.y = agent_msg.pos.y;
			agent.pos_along_path = agent_msg.pos_along_path;
			agent.speed = agent_msg.speed;
			agent.type = AgentType(agent_msg.type);
			agent.vel.x = agent_msg.vel.x;
			agent.vel.y = agent_msg.vel.y;
		}
		prior->unlabelled_belief_.push_back(s);
	}

	torch::Tensor image =
			torch::from_blob(data.state_tensor_0.data(), {IMSIZE, IMSIZE}, TORCH_DATA_TYPE).clone();
	detectNAN(image);
	prior->unlabelled_hist_images_.push_back(image);
	image = torch::from_blob(data.state_tensor_1.data(), {IMSIZE, IMSIZE}, TORCH_DATA_TYPE).clone();
	detectNAN(image);
	prior->unlabelled_hist_images_.push_back(image);
	image = torch::from_blob(data.state_tensor_2.data(), {IMSIZE, IMSIZE}, TORCH_DATA_TYPE).clone();
	detectNAN(image);
	prior->unlabelled_hist_images_.push_back(image);
	image = torch::from_blob(data.state_tensor_3.data(), {IMSIZE, IMSIZE}, TORCH_DATA_TYPE).clone();
	detectNAN(image);
	prior->unlabelled_hist_images_.push_back(image);

	for (float vel: data.semantic_tensor)
		prior->unlabelled_semantic_.push_back(vel);

	AgentArrayCallback(data.agents);
	AgentPathArrayCallback(data.agent_paths);

//	worldModel.id_map_belief_reset.clear();
//	worldModel.id_map_paths.clear();
//	worldModel.id_map_num_paths.clear();
//
//	for (msg_builder::AgentPaths& agent : data.agent_paths) {
//		std::string agent_type = agent.type;
//		int id = agent.id;
//
//		if (agent_type == "ped")
//			exo_agents_[id].cross_dir = agent.cross_dirs[0];
//
//		worldModel.id_map_belief_reset[id] = agent.reset_intention;
//		worldModel.id_map_paths[id] = worldModel.ParsePathCandidates(
//				agent.path_start_rp.edge,
//				agent.path_start_rp.lane,
//				agent.path_start_rp.segment,
//				agent.path_start_rp.offset,
//				agent_type);
//		worldModel.id_map_num_paths[id] = worldModel.id_map_paths[id].size();
//	}

	msg_builder::State& particle0_msg = data.particles[0];
	car_.heading_dir = particle0_msg.car.heading_dir;
	car_.path_idx = -1;
	car_.pos.x = particle0_msg.car.pos.x;
	car_.pos.y = particle0_msg.car.pos.y;
	car_.vel = particle0_msg.car.vel;
	real_speed = car_.vel;

	ModelParams::CAR_FRONT = data.meta.car_front;
	ModelParams::CAR_REAR = data.meta.car_rear;
	ModelParams::CAR_WIDTH = data.meta.car_width;
	ModelParams::CAR_REAR = data.meta.car_length;
	ModelParams::CAR_WHEEL_DIST = data.meta.car_wheel_dist;
	ModelParams::MAX_STEER_ANGLE = data.meta.max_steer_angle;

	car_data_ready = true;
	topic_state_ready = true;
}

void WorldSimulator::TriggerUnlabelledBeliefTopic() {
	logd << "calling service fetch_data" << endl;
	std_srvs::Empty empty;
	if (!fetch_data_srv_.call(empty))
	{
		ERR("query of fetch_data service failed!!!");
	}
}

