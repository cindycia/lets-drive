#include <interface/world.h>
#include <string>
#include <ros/ros.h>
#include "context_pomdp.h"
#include "param.h"

#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/PointCloud.h>

#include <rosgraph_msgs/Clock.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PolygonStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include "param.h"
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <nav_msgs/GetPlan.h>

#include <msg_builder/car_info.h>
#include <msg_builder/peds_info.h>
#include <msg_builder/TrafficAgentArray.h>
#include <msg_builder/AgentPathArray.h>
#include <msg_builder/AgentPathArray.h>
#include <msg_builder/WorldAgents.h>
#include <msg_builder/Belief.h>

#include "std_msgs/Float32.h"
#include <std_msgs/Bool.h>

#include "simulator_base.h"


using namespace despot;

class WorldSimulator: public SimulatorBase, public World {
private:
	DSPOMDP* model_;
	WorldModel& worldModel;

	double car_time_stamp_;
	double agents_time_stamp_;
	double paths_time_stamp_;
	CarStruct car_;
	std::map<int, AgentStruct> exo_agents_;

	msg_builder::TrafficAgentArray agents_topic_;
	msg_builder::AgentPathArray agents_path_topic_;
	msg_builder::TrafficAgentArray agents_topic_bak_;
	msg_builder::AgentPathArray agents_path_topic_bak_;

	PomdpStateWorld current_state_;
	Path path_from_topic_;
	Path path_from_decision_;

	int safe_action_;
	bool goal_reached_;
	double last_acc_;
	int last_laneID_;

	ros::Publisher cmdPub_, unlabelled_belief_pub_, ego_pub_, decision_path_pub_;
	ros::Subscriber ego_sub_, ego_dead_sub_, pathSub_, agent_sub_, agent_path_sub_, world_Sub_;
	ros::Subscriber steerSub_, lane_change_Sub_, last_lane_change_Sub_, unlabelled_belief_sub_;

	std::string map_location_;
	int summit_port_;

public:
	double time_scale;
	bool topic_state_ready;
	bool car_data_ready;

public:
	static ros::ServiceClient data_client_;
	static ros::ServiceClient fetch_data_srv_;

public:
	WorldSimulator(ros::NodeHandle& _nh, DSPOMDP* model, unsigned seed,
			std::string map_location, int summit_port);
	~WorldSimulator();

public:

	bool Connect();
	void ConnectCarla();
	State* Initialize();
	std::map<double, AgentStruct&> GetSortedAgents();
	State* GetCurrentState();
	bool ExecuteAction(ACT_TYPE action, OBS_TYPE& obs);
	double StepReward(PomdpStateWorld& state, ACT_TYPE action, double ttc);
	bool Emergency(PomdpStateWorld* curr_state);
	bool Terminal(PomdpStateWorld* world_state);

	void UpdateCmds(ACT_TYPE action, bool emergency = false);
	void UpdatePath(ACT_TYPE action);
	void PublishCmdAction(const ros::TimerEvent &e);
	void PublishCmdAction(ACT_TYPE);
	void PublishPath();
	void sendStateActionData(PomdpStateWorld& planning_state, ACT_TYPE safeAction,
			float reward, float vel, float ttc);
	void PublishUnlabelledBelief();

	void EgoDeadCallBack(const std_msgs::Bool ego_dead);
	void EgoStateCallBack(const msg_builder::car_info::ConstPtr car);
	void AgentArrayCallback(msg_builder::TrafficAgentArray data);
	void AgentPathArrayCallback(msg_builder::AgentPathArray data);
	void WorldAgentsCallBack(msg_builder::WorldAgents data);
	void UnlabelledBeliefCallBack(msg_builder::Belief data);
	void TriggerUnlabelledBeliefTopic();

	void updateLanes(COORD car_pos);
	void updateObs(COORD car_pos);

	void ResetWorldModel();

public:
	bool ReviseAction(PomdpStateWorld* curr_state, ACT_TYPE action);
};

