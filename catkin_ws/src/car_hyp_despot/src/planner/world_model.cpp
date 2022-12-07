#include<limits>
#include<cmath>
#include<cstdlib>
#include <numeric>

#include <despot/GPUcore/thread_globals.h>
#include <despot/core/globals.h>
#include <despot/solver/despot.h>

#include"math_utils.h"
#include"coord.h"
#include "path.h"
#include "world_model.h"
#include "context_pomdp.h"
#include "threaded_print.h"

#undef LOG
#define LOG(lv) \
if (despot::logging::level() < despot::logging::ERROR || despot::logging::level() < lv) ; \
else despot::logging::stream(lv)
#include <despot/util/logging.h>

using namespace std;
std::chrono::time_point<std::chrono::system_clock> init_time;

bool use_vel_from_summit = true;
int include_stop_intention = 0;
int include_curvel_intention = 1;
bool initial_update_step = true;


int ClosestInt(double v) {
	if ((v - (int) v) < 0.5)
		return (int) v;
	else
		return (int) v + 1;
}

int FloorIntRobust(double v) {
	if ((v - (int) v) > 1 - 1e-5)
		return (int) v + 1;
	else
		return (int) v;
}

WorldModel::WorldModel() :
		freq(ModelParams::CONTROL_FREQ), in_front_angle_cos(
				cos(ModelParams::IN_FRONT_ANGLE_DEG / 180.0 * M_PI)),
				last_decision_lane(LaneCode::KEEP) {
	if (DESPOT::Debug_mode)
		ModelParams::NOISE_GOAL_ANGLE = 0.000001;
	init_time = Time::now();
}


WorldModel::~WorldModel() {
	traffic_agent_sim_.clear();
	traffic_agent_sim_.resize(0);
}

void WorldModel::SetPath(Path path) {
	this->path = path;
	ModelParams::GOAL_TRAVELLED = path.GetLength();
}


ACT_TYPE WorldModel::DefaultStatePolicy(const State* _state) const {
	logd << "[" << __FUNCTION__ << "] Get state info" << endl;
	const PomdpState *state = static_cast<const PomdpState*>(_state);
	double mindist_along = numeric_limits<double>::infinity();
	double mindist_tang = numeric_limits<double>::infinity();
	const COORD& carpos = state->car.pos;
	double carheading = state->car.heading_dir;
	double carvel = state->car.vel;

	double lane = 0;


	logd << "[" << __FUNCTION__ << "] Calculate acceleration" << endl;

	double acceleration;
	// Closest pedestrian in front
	for (int i = 0; i < state->num; i++) {
		const AgentStruct & p = state->agents[i];

		float infront_angle = ModelParams::IN_FRONT_ANGLE_DEG;
		if (Globals::config.pruning_constant > 100.0)
			infront_angle = 60.0;
		if (!InFront(p.pos, state->car, infront_angle))
			continue;

		double d_along = COORD::DirectedDistance(carpos, p.pos, carheading);
		double d_tang = COORD::DirectedDistance(carpos, p.pos,
				carheading + M_PI / 2.0);

		if (p.type == AgentType::car) {
			d_along = d_along - 2.0;
			d_tang = d_tang - 1.0;
		}

		mindist_along = min(mindist_along, d_along);
		mindist_tang = min(mindist_tang, d_tang);
	}

	if (mindist_along < 2 || mindist_tang < 1.0) {
		acceleration = (carvel <= 0.01) ? 0 : -ModelParams::ACC_SPEED;
	} else if (mindist_along < 4 && mindist_tang < 2.0) {
		if (carvel > ModelParams::VEL_MAX / 1.5 + 1e-4)
			acceleration = -ModelParams::ACC_SPEED;
		else if (carvel < ModelParams::VEL_MAX / 2.0 - 1e-4)
			acceleration = ModelParams::ACC_SPEED;
		else
			acceleration = 0.0;
	} else
		acceleration =
				carvel >= ModelParams::VEL_MAX - 1e-4 ?
						0 : ModelParams::ACC_SPEED;

	logd << "[" <<__FUNCTION__ << "] Calculate action ID" << endl;
	return ContextPomdp::GetActionID(lane, acceleration);
}

bool WorldModel::InFront(const COORD ped_pos, const CarStruct car,
		double infront_angle_deg) const {
	if (infront_angle_deg == -1)
		infront_angle_deg = ModelParams::IN_FRONT_ANGLE_DEG;
	if (infront_angle_deg >= 180.0) {
		// inFront check is disabled
		return true;
	}

	double d0 = COORD::EuclideanDistance(car.pos, ped_pos);
	COORD dir(cos(car.heading_dir), sin(car.heading_dir));
	double dot = dir.dot(ped_pos - car.pos);
	double cosa = (d0 > 0) ? dot / (d0) : 0;
	assert(cosa <= 1.0 + 1E-8 && cosa >= -1.0 - 1E-8);
	return cosa > in_front_angle_cos;
}

int WorldModel::DefaultPolicy(const std::vector<State*>& particles) {
	const PomdpState *state = static_cast<const PomdpState*>(particles[0]);
	return DefaultStatePolicy(state);
}

int WorldModel::MinStepToGoal(const PomdpState& state) {
	double d = path.GetLength(path.Nearest(state.car.pos));
	if (d < 0)
		d = 0;
	return int(ceil(d / (ModelParams::VEL_MAX / freq)));
}

void WorldModel::AgentStep(AgentStruct &agent, Random& random) {
	double noise = random.NextGaussian() * ModelParams::NOISE_GOAL_ANGLE;
	if (goal_mode == "cur_vel") {
		AgentStepCurVel(agent, 1, noise);
	} else if (goal_mode == "path") {
		AgentStepPath(agent, 1, noise);
	}
}

void WorldModel::AgentStep(AgentStruct &agent, double& random) {

	double noise = sqrt(-2 * log(random));
	if (FIX_SCENARIO != 1 && !CPUDoPrint) {
		random = QuickRandom::RandGeneration(random);
	}
	noise *= cos(2 * M_PI * random) * ModelParams::NOISE_GOAL_ANGLE;

	if (goal_mode == "cur_vel") {
		AgentStepCurVel(agent, 1, noise);
	} else if (goal_mode == "path") {
		AgentStepPath(agent, 1, noise);
	}
}


void WorldModel::AgentStepCurVel(AgentStruct& agent, int step, double noise) {
	if (noise != 0) {
		double a = agent.vel.GetAngle();
		a += noise;
		COORD move(a, step * agent.vel.Length() / freq, 0);
		agent.pos.x += move.x;
		agent.pos.y += move.y;
	} else {
		agent.pos.x += agent.vel.x * (float(step) / freq);
		agent.pos.y += agent.vel.y * (float(step) / freq);
	}
}

void WorldModel::AgentStepPath(AgentStruct& agent, int step, double noise,
		bool doPrint) {
	if (agent.type == AgentType::ped)
		PedStepPath(agent, step, noise, doPrint);
	else if (agent.type == AgentType::car)
		VehStepPath(agent, step, noise, doPrint);
	else
		ERR(string_sprintf("unsupported agent mode %d", agent.type));
}

void WorldModel::PedStepPath(AgentStruct& agent, int step, double noise,
		bool doPrint) {
	auto& path_candidates = PathCandidates(agent.id);

	int intention = doPrint? 0: agent.intention;
	int old_path_pos = agent.pos_along_path;

	if (intention < path_candidates.size()) {
		auto& path = path_candidates[intention];
		agent.pos_along_path = path.Forward(agent.pos_along_path,
				agent.speed * (float(step) / freq));
		COORD new_pos = path[agent.pos_along_path];

		if (noise != 0) {
			COORD goal_vec = new_pos - agent.pos;
			double a = goal_vec.GetAngle() + noise;
			COORD move(a, step * agent.speed / freq, 0);
			agent.pos.x += move.x;
			agent.pos.y += move.y;
		} else {
			agent.pos = new_pos;
		}
		agent.vel = (new_pos - agent.pos) * freq;

		if (doPrint && agent.pos_along_path == old_path_pos)
			logd << "[AgentStepPath]: agent " << agent.id
					<< " no move: path length " << path.size() << " speed "
					<< agent.speed << " forward distance "
					<< agent.speed * (float(step) / freq) << endl;
	} else if (IsCurVelIntention(intention, agent.id)) {
		agent.pos = agent.pos + agent.vel * (1.0 / freq);
	}
}

void WorldModel::VehStepPath(AgentStruct& agent, int step, double noise,
		bool doPrint) {
	auto& path_candidates = PathCandidates(agent.id);
	int intention = doPrint? 0: agent.intention;
	int old_path_pos = agent.pos_along_path;

	if (intention < path_candidates.size()) {
		COORD old_pos = agent.pos;
		auto& path = path_candidates[intention];
		int pursuit_pos = path.Forward(agent.pos_along_path, PURSUIT_LEN);
		COORD pursuit_point = path[pursuit_pos];

		double steering = PControlAngle<AgentStruct>(agent, pursuit_point) + noise;
		BicycleModel(agent, steering, agent.speed);

		agent.pos_along_path = path.Nearest(agent.pos);
		agent.vel = (agent.pos - old_pos) * freq;

		if (doPrint && agent.pos_along_path == old_path_pos)
			logd << "[AgentStepPath]: agent " << agent.id
					<< " no move: path length " << path.size() << " speed "
					<< agent.speed << " forward distance "
					<< agent.speed * (float(step) / freq) << endl;
	} else if (IsCurVelIntention(intention, agent.id)) {
		agent.pos = agent.pos + agent.vel * (1.0 / freq);
	}
}


COORD WorldModel::GetGoalPos(const AgentStruct& agent, int intention_id) {

	if (intention_id == -1)
		intention_id = agent.intention;

	if (goal_mode == "path")
		return GetGoalPosFromPaths(agent.id, intention_id, agent.pos_along_path,
				agent.pos, agent.vel, agent.type, agent.cross_dir);
	else if (goal_mode == "cur_vel")
		return agent.pos + agent.vel.Scale(PURSUIT_LEN);
	else {
		RasieUnsupportedGoalMode(__FUNCTION__);
		return COORD(0.0, 0.0);
	}
}

COORD WorldModel::GetGoalPosFromPaths(int agent_id, int intention_id,
		int pos_along_path, const COORD& agent_pos, const COORD& agent_vel,
		AgentType type, bool agent_cross_dir) {
	auto& path_candidates = PathCandidates(agent_id);
	if (intention_id < path_candidates.size()) {
		auto& path = path_candidates[intention_id];
		if (path.size() == 0)
			return agent_pos;
		COORD pursuit = path[path.Forward(pos_along_path, PURSUIT_LEN)];
		return pursuit;
	} else if (IsCurVelIntention(intention_id, agent_id)) {
		COORD dir(agent_vel);
		dir.AdjustLength(PURSUIT_LEN);
		return agent_pos + dir;
	} else if (IsStopIntention(intention_id, agent_id)) { // stop intention
		return COORD(agent_pos.x, agent_pos.y);
	} else {
		ERR(string_sprintf(
				"Intention ID %d excesses # intentions %d for agent %d of type %d\n",
				intention_id, GetNumIntentions(agent_id), agent_id,
				type));
		return COORD(0.0, 0.0);
	}
}

bool WorldModel::IsStopIntention(int intention, int agent_id) {
	if (include_stop_intention)
		return intention == GetNumIntentions(agent_id) - include_stop_intention;
	else
		return false;
}

bool WorldModel::IsCurVelIntention(int intention, int agent_id) {
	if (include_curvel_intention) {
		if (NumPaths(agent_id) == 0)
			return intention == GetNumIntentions(agent_id)
						- include_curvel_intention
						- include_stop_intention;
		else
			return false;
	}
	else
		return false;
}

int WorldModel::NumPaths(int agent_id) {
	auto it = id_map_num_paths.find(agent_id);
	if (it == id_map_num_paths.end()) {
		return 0;
	}
	return it->second;
}

void WorldModel::PrintPathMap() {
	cout << "id_map_paths: ";
	for (auto it = id_map_paths.begin(); it != id_map_paths.end(); ++it) {
		cout << " (" << it->first << ", l_" << it->second.size() << ")";
	}
	cout << endl;
}

std::vector<Path>& WorldModel::PathCandidates(int agent_id) {
	auto it = id_map_paths.find(agent_id);
	if (it == id_map_paths.end()) {
		std::cout << __FILE__ << ": agent id " << agent_id
				<< " not found in id_map_paths" << std::endl;
//		PrintPathMap();
		id_map_paths[agent_id] = std::vector<Path>();
		it = id_map_paths.find(agent_id);
	}
	return it->second;
}

int WorldModel::GetNumIntentions(int agent_id) {
	if (goal_mode == "path") {
		int num_paths = NumPaths(agent_id);
		if (num_paths > 0)
			return NumPaths(agent_id) + include_stop_intention;
		else
			return include_curvel_intention + include_stop_intention;
	}
	else if (goal_mode == "cur_vel")
		return 2;
	else
		return 0;
}

void WorldModel::ValidateIntention(int agent_id, int intention_id,
		const char* msg, int line) {
	if (intention_id >= GetNumIntentions(agent_id)) {
		int num_path = NumPaths(agent_id);
		int num_intention = GetNumIntentions(agent_id);
		ERR(
				string_sprintf(
						"%s@%d: Agent %d Intention ID excess # intentions: %d, # intention = %d, # path = %d",
						msg, line, agent_id, intention_id, num_intention, num_path));
	}
}

bool WorldModel::NeedBeliefReset(int agent_id) {
	auto it = id_map_belief_reset.find(agent_id);
	if (it == id_map_belief_reset.end()) {
		return false;
	}
	return it->second;
}

/**
 * H: center of the head of the car
 * N: a point right in front of the car
 * M: an arbitrary point
 *
 * Check whether M is in the safety zone
 */
bool InCollision(double Px, double Py, double Cx, double Cy, double Ctheta,
		bool expand = true);
bool inCarlaCollision(double ped_x, double ped_y, double car_x, double car_y,
		double Ctheta, double car_extent_x, double car_extent_y, bool flag = 0);

bool InRealCollision(double Px, double Py, double Cx, double Cy, double Ctheta,
		bool expand = true);
std::vector<COORD> ComputeRect(COORD pos, double heading,
		double ref_to_front_side, double ref_to_back_side,
		double ref_front_side_angle, double ref_back_side_angle);
bool InCollision(std::vector<COORD> rect_1, std::vector<COORD> rect_2);

bool WorldModel::InCollision(const PomdpState& state) const {
	const COORD& car_pos = state.car.pos;

	int i = 0;
	for (auto& agent : state.agents) {
		if (i >= state.num)
			break;
		i++;

		if (!InFront(agent.pos, state.car))
			continue;

		if (agent.type == AgentType::ped) {
			const COORD& pedpos = agent.pos;
			if (::InCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y,
					state.car.heading_dir)) {
				return true;
			}
		} else if (agent.type == AgentType::car) {
			if (CheckCarWithVehicle(state.car, agent, 0))
				return true;
		} else {
			ERR(string_sprintf("unsupported agent type"));
		}
	}

	return false;
}

bool WorldModel::InRealCollision(const PomdpStateWorld& state, int &id,
		double infront_angle) const {
	id = -1;
	if (infront_angle == -1) {
		infront_angle = ModelParams::IN_FRONT_ANGLE_DEG;
	}

	const COORD& car_pos = state.car.pos;

	int i = 0;
	for (auto& agent : state.agents) {
		if (i >= state.num)
			break;
		i++;

		if (!InFront(agent.pos, state.car, infront_angle))
			continue;

		if (agent.type == AgentType::ped) {
			const COORD& pedpos = agent.pos;
			if (::InRealCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y,
					state.car.heading_dir)) {
				id = agent.id;
			}
		} else if (agent.type == AgentType::car) {
			if (CheckCarWithVehicle(state.car, agent, 1)) {
				id = agent.id;
			}
		} else {
			ERR(string_sprintf("unsupported agent type"));
		}

		if (id != -1) {
			logd << "[WorldModel::InRealCollision] car_pos: (" << car_pos.x
					<< "," << car_pos.y << "), heading: ("
					<< std::cos(state.car.heading_dir) << ","
					<< std::sin(state.car.heading_dir) << "), agent_pos: ("
					<< agent.pos.x << "," << agent.pos.y << ")\n";
			return true;
		}
	}

	return false;
}

bool WorldModel::InRealCollision(const PomdpStateWorld& state,
		double infront_angle_deg) const {
	const COORD& car_pos = state.car.pos;
	int id = -1;
	if (infront_angle_deg == -1) {
		infront_angle_deg = ModelParams::IN_FRONT_ANGLE_DEG;
	}

	int i = 0;
	for (auto& agent : state.agents) {
		if (i >= state.num)
			break;
		i++;

		if (!InFront(agent.pos, state.car, infront_angle_deg))
			continue;

		if (agent.type == AgentType::ped) {
			const COORD& pedpos = agent.pos;
			if (::InRealCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y,
					state.car.heading_dir)) {
				id = agent.id;
			}
		} else if (agent.type == AgentType::car) {
			if (CheckCarWithVehicle(state.car, agent, 1)) {
				id = agent.id;
			}
		} else {
			ERR(string_sprintf("unsupported agent type"));
		}

		if (id != -1) {
			logd << "[WorldModel::InRealCollision] car_pos: (" << car_pos.x
					<< "," << car_pos.y << "), heading: ("
					<< std::cos(state.car.heading_dir) << ","
					<< std::sin(state.car.heading_dir) << "), agent_pos: ("
					<< agent.pos.x << "," << agent.pos.y << ")\n";
			return true;
		}
	}

	return false;
}

bool WorldModel::InCollision(const PomdpStateWorld& state, double in_front_angle_deg) const {
	const COORD& car_pos = state.car.pos;

	if (in_front_angle_deg == -1) {
		in_front_angle_deg = ModelParams::IN_FRONT_ANGLE_DEG;
	}

	int i = 0;
	for (auto& agent : state.agents) {
		if (i >= state.num)
			break;
		i++;

		if (!InFront(agent.pos, state.car, in_front_angle_deg))
			continue;

		if (agent.type == AgentType::ped) {
			const COORD& pedpos = agent.pos;
			if (::InCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y,
					state.car.heading_dir)) {
				return true;
			}
		} else if (agent.type == AgentType::car) {
			if (CheckCarWithVehicle(state.car, agent, 0))
				return true;
		} else {
			ERR(string_sprintf("unsupported agent type"));
		}
	}

	return false;
}

bool WorldModel::InCollision(const PomdpState& state, int &id) const {
	id = -1;
	const COORD& car_pos = state.car.pos;

	int i = 0;
	for (auto& agent : state.agents) {
		if (i >= state.num)
			break;
		i++;

		if (!InFront(agent.pos, state.car))
			continue;

		if (agent.type == AgentType::ped) {
			const COORD& pedpos = agent.pos;
			if (::InCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y,
					state.car.heading_dir)) {
				id = agent.id;
				return true;
			}

		} else if (agent.type == AgentType::car) {
			if (CheckCarWithVehicle(state.car, agent, 0)) {
				id = agent.id;
				return true;
			}
		} else {
			ERR(string_sprintf("unsupported agent type"));
		}
	}

	if (id != -1) {
		return true;
	}

	return false;
}

bool WorldModel::InCollision(const PomdpStateWorld& state, int &id) const {
	id = -1;
	const COORD& car_pos = state.car.pos;

	int i = 0;
	for (auto& agent : state.agents) {
		if (i >= state.num)
			break;
		i++;

		if (!InFront(agent.pos, state.car))
			continue;

		if (agent.type == AgentType::ped) {
			const COORD& pedpos = agent.pos;
			if (::InCollision(pedpos.x, pedpos.y, car_pos.x, car_pos.y,
					state.car.heading_dir)) {
				id = agent.id;
			}

		} else if (agent.type == AgentType::car) {
			if (CheckCarWithVehicle(state.car, agent, 0)) {
				id = agent.id;
			}
		} else {
			ERR(string_sprintf("unsupported agent type"));
		}

		if (id != -1)
			logd << "[WorldModel::InRealCollision] car_pos: (" << car_pos.x
					<< "," << car_pos.y << "), heading: ("
					<< std::cos(state.car.heading_dir) << ","
					<< std::sin(state.car.heading_dir) << "), agent_pos: ("
					<< agent.pos.x << "," << agent.pos.y << ")\n";
	}

	if (id != -1) {
		return true;
	}

	return false;
}

bool WorldModel::CheckCarWithVehicle(const CarStruct& car,
		const AgentStruct& veh, int flag) const {
	COORD tan_dir(-sin(veh.heading_dir), cos(veh.heading_dir)); // along_dir rotates by 90 degree counter-clockwise
	COORD along_dir(cos(veh.heading_dir), sin(veh.heading_dir));

	COORD test;

	bool result = false;
	double veh_bb_extent_x = veh.bb_extent_x;
	double veh_bb_extent_y = veh.bb_extent_y;

	double car_bb_extent_x = ModelParams::CAR_WIDTH / 2.0;
	double car_bb_extent_y = ModelParams::CAR_FRONT;

	test = veh.pos + tan_dir * veh_bb_extent_x + along_dir * veh_bb_extent_y;
	if (::inCarlaCollision(test.x, test.y, car.pos.x, car.pos.y,
			car.heading_dir, car_bb_extent_x, car_bb_extent_y, flag)) {
		result = true;
	}
	test = veh.pos - tan_dir * veh_bb_extent_x + along_dir * veh_bb_extent_y;
	if (::inCarlaCollision(test.x, test.y, car.pos.x, car.pos.y,
			car.heading_dir, car_bb_extent_x, car_bb_extent_y, flag)) {
		result = true;
	}
	test = veh.pos + tan_dir * veh_bb_extent_x - along_dir * veh_bb_extent_y;
	if (::inCarlaCollision(test.x, test.y, car.pos.x, car.pos.y,
			car.heading_dir, car_bb_extent_x, car_bb_extent_y, flag)) {
		result = true;
	}
	test = veh.pos - tan_dir * veh_bb_extent_x - along_dir * veh_bb_extent_y;
	if (::inCarlaCollision(test.x, test.y, car.pos.x, car.pos.y,
			car.heading_dir, car_bb_extent_x, car_bb_extent_y, flag)) {
		result = true;
	}

	tan_dir = COORD(-sin(car.heading_dir), cos(car.heading_dir));
	along_dir = COORD(cos(car.heading_dir), sin(car.heading_dir));

	test = car.pos + tan_dir * (car_bb_extent_x) + along_dir * car_bb_extent_y;
	if (::inCarlaCollision(test.x, test.y, veh.pos.x, veh.pos.y,
			veh.heading_dir, veh_bb_extent_x, veh_bb_extent_y, flag)) {
		result = true;
	}
	test = car.pos - tan_dir * (car_bb_extent_x) + along_dir * car_bb_extent_y;
	if (::inCarlaCollision(test.x, test.y, veh.pos.x, veh.pos.y,
			veh.heading_dir, veh_bb_extent_x, veh_bb_extent_y, flag)) {
		result = true;
	}
	test = car.pos + tan_dir * (car_bb_extent_x) - along_dir * car_bb_extent_y;
	if (::inCarlaCollision(test.x, test.y, veh.pos.x, veh.pos.y,
			veh.heading_dir, veh_bb_extent_x, veh_bb_extent_y, flag)) {
		result = true;
	}
	test = car.pos - tan_dir * (car_bb_extent_x) - along_dir * car_bb_extent_y;
	if (::inCarlaCollision(test.x, test.y, veh.pos.x, veh.pos.y,
			veh.heading_dir, veh_bb_extent_x, veh_bb_extent_y, flag)) {
		result = true;
	}

	if (flag == 1 && result) {
		cout << "collision point";
		cout << " " << veh.pos.x << " " << veh.pos.y;
		test = veh.pos + tan_dir * veh_bb_extent_x
				+ along_dir * veh_bb_extent_y;
		cout << " " << test.x << " " << test.y;
		test = veh.pos - tan_dir * veh_bb_extent_x
				+ along_dir * veh_bb_extent_y;
		cout << " " << test.x << " " << test.y;
		test = veh.pos + tan_dir * veh_bb_extent_x
				- along_dir * veh_bb_extent_y;
		cout << " " << test.x << " " << test.y;
		test = veh.pos - tan_dir * veh_bb_extent_x
				- along_dir * veh_bb_extent_y;
		cout << " " << test.x << " " << test.y;

		cout << " " << car.pos.x << " " << car.pos.y;
		test = car.pos + tan_dir * (car_bb_extent_x)
				+ along_dir * car_bb_extent_y;
		cout << " " << test.x << " " << test.y;
		test = car.pos - tan_dir * (car_bb_extent_x)
				+ along_dir * car_bb_extent_y;
		cout << " " << test.x << " " << test.y;
		test = car.pos + tan_dir * (car_bb_extent_x)
				- along_dir * car_bb_extent_y;
		cout << " " << test.x << " " << test.y;
		test = car.pos - tan_dir * (car_bb_extent_x)
				- along_dir * car_bb_extent_y;
		cout << " " << test.x << " " << test.y << endl;
	}

	return result;
}

bool WorldModel::CheckCarWithVehicleReal(const CarStruct& car,
		const AgentStruct& veh, int flag) const {

	COORD tan_dir(-sin(veh.heading_dir), cos(veh.heading_dir)); // along_dir rotates by 90 degree counter-clockwise
	COORD along_dir(cos(veh.heading_dir), sin(veh.heading_dir));

	COORD test;

	bool result = false;
	double veh_bb_extent_x = veh.bb_extent_x * 0.95;
	double veh_bb_extent_y = veh.bb_extent_y * 0.95;

	double car_bb_extent_x = ModelParams::CAR_WIDTH / 2.0;
	double car_bb_extent_y = ModelParams::CAR_FRONT;

	test = veh.pos + tan_dir * veh_bb_extent_x + along_dir * veh_bb_extent_y;
	if (::inCarlaCollision(test.x, test.y, car.pos.x, car.pos.y,
			car.heading_dir, car_bb_extent_x, car_bb_extent_y, flag)) {
		result = true;
	}
	test = veh.pos - tan_dir * veh_bb_extent_x + along_dir * veh_bb_extent_y;
	if (::inCarlaCollision(test.x, test.y, car.pos.x, car.pos.y,
			car.heading_dir, car_bb_extent_x, car_bb_extent_y, flag)) {
		result = true;
	}
	test = veh.pos + tan_dir * veh_bb_extent_x - along_dir * veh_bb_extent_y;
	if (::inCarlaCollision(test.x, test.y, car.pos.x, car.pos.y,
			car.heading_dir, car_bb_extent_x, car_bb_extent_y, flag)) {
		result = true;
	}
	test = veh.pos - tan_dir * veh_bb_extent_x - along_dir * veh_bb_extent_y;
	if (::inCarlaCollision(test.x, test.y, car.pos.x, car.pos.y,
			car.heading_dir, car_bb_extent_x, car_bb_extent_y, flag)) {
		result = true;
	}

	tan_dir = COORD(-sin(car.heading_dir), cos(car.heading_dir));
	along_dir = COORD(cos(car.heading_dir), sin(car.heading_dir));

	test = car.pos + tan_dir * (car_bb_extent_x) + along_dir * car_bb_extent_y;
	if (::inCarlaCollision(test.x, test.y, veh.pos.x, veh.pos.y,
			veh.heading_dir, veh_bb_extent_x, veh_bb_extent_y, flag)) {
		result = true;
	}
	test = car.pos - tan_dir * (car_bb_extent_x) + along_dir * car_bb_extent_y;
	if (::inCarlaCollision(test.x, test.y, veh.pos.x, veh.pos.y,
			veh.heading_dir, veh_bb_extent_x, veh_bb_extent_y, flag)) {
		result = true;
	}
	test = car.pos + tan_dir * (car_bb_extent_x) - along_dir * car_bb_extent_y;
	if (::inCarlaCollision(test.x, test.y, veh.pos.x, veh.pos.y,
			veh.heading_dir, veh_bb_extent_x, veh_bb_extent_y, flag)) {
		result = true;
	}
	test = car.pos - tan_dir * (car_bb_extent_x) - along_dir * car_bb_extent_y;
	if (::inCarlaCollision(test.x, test.y, veh.pos.x, veh.pos.y,
			veh.heading_dir, veh_bb_extent_x, veh_bb_extent_y, flag)) {
		result = true;
	}

	if (flag == 1 && result) {
		cout << "collision point";
		cout << " " << veh.pos.x << " " << veh.pos.y;
		test = veh.pos + tan_dir * veh_bb_extent_x
				+ along_dir * veh_bb_extent_y;
		cout << " " << test.x << " " << test.y;
		test = veh.pos - tan_dir * veh_bb_extent_x
				+ along_dir * veh_bb_extent_y;
		cout << " " << test.x << " " << test.y;
		test = veh.pos + tan_dir * veh_bb_extent_x
				- along_dir * veh_bb_extent_y;
		cout << " " << test.x << " " << test.y;
		test = veh.pos - tan_dir * veh_bb_extent_x
				- along_dir * veh_bb_extent_y;
		cout << " " << test.x << " " << test.y;

		cout << " " << car.pos.x << " " << car.pos.y;
		test = car.pos + tan_dir * (car_bb_extent_x)
				+ along_dir * car_bb_extent_y;
		cout << " " << test.x << " " << test.y;
		test = car.pos - tan_dir * (car_bb_extent_x)
				+ along_dir * car_bb_extent_y;
		cout << " " << test.x << " " << test.y;
		test = car.pos + tan_dir * (car_bb_extent_x)
				- along_dir * car_bb_extent_y;
		cout << " " << test.x << " " << test.y;
		test = car.pos - tan_dir * (car_bb_extent_x)
				- along_dir * car_bb_extent_y;
		cout << " " << test.x << " " << test.y << endl;
	}

	return result;
}

bool WorldModel::CheckCarWithObsLine(const CarStruct& car, COORD start_point,
		COORD end_point, int flag) const {
	const double step = ModelParams::OBS_LINE_STEP;

	double d = COORD::EuclideanDistance(start_point, end_point);
	double dx = (end_point.x - start_point.x) / d;
	double dy = (end_point.y - start_point.y) / d;
	double sx = start_point.x;
	double sy = start_point.y;

	double t = 0, ti = 0;
	while (t < ti + d) {
		double u = t - ti;
		double nx = sx + dx * u;
		double ny = sy + dy * u;

		if (flag == 0) {
			if (::InCollision(nx, ny, car.pos.x, car.pos.y, car.heading_dir,
					false))
				return true;
		} else if (flag == 1) {
			if (::InRealCollision(nx, ny, car.pos.x, car.pos.y, car.heading_dir,
					false))
				return true;
		}

		if (t == ti + d - 0.01)
			break;
		else
			t = min(t + step, ti + d - 0.01);
	}

	return false;
}

bool WorldModel::IsGlobalGoal(const CarStruct& car) const {
	if (car.path_idx == -1)
		return false;

	double d = WorldModel::path_tree_->length_left(car.pos, car.path_idx);

	return (d < ModelParams::GOAL_TOLERANCE);
}

#include "Vector2.h"

std::vector<RVO::Vector2> WorldModel::GetBoundingBoxCorners(
		AgentStruct& agent) {
	RVO::Vector2 forward_vec = RVO::Vector2(cos(agent.heading_dir),
			sin(agent.heading_dir));
	RVO::Vector2 sideward_vec = RVO::Vector2(-sin(agent.heading_dir),
			cos(agent.heading_dir)); // rotate 90 degree counter-clockwise

	double side_len = agent.bb_extent_x;
	double forward_len = agent.bb_extent_y;

	if (agent.type == AgentType::ped) {
		side_len = 0.23;
		forward_len = 0.23;
	}

	RVO::Vector2 pos = RVO::Vector2(agent.pos.x, agent.pos.y);

	return std::vector<RVO::Vector2>(
			{ pos + forward_vec * forward_len + sideward_vec * side_len, pos
					- forward_vec * forward_len + sideward_vec * side_len, pos
					- forward_vec * forward_len - sideward_vec * side_len, pos
					+ forward_vec * forward_len - sideward_vec * side_len });
}

std::vector<RVO::Vector2> WorldModel::GetBoundingBoxCorners(
		RVO::Vector2 forward_vec, RVO::Vector2 sideward_vec, RVO::Vector2 pos,
		double forward_len, double side_len) {

	return std::vector<RVO::Vector2>(
			{ pos + forward_vec * forward_len + sideward_vec * side_len, pos
					- forward_vec * forward_len + sideward_vec * side_len, pos
					- forward_vec * forward_len - sideward_vec * side_len, pos
					+ forward_vec * forward_len - sideward_vec * side_len });
}

void WorldModel::InitGamma() {
	if (!Globals::config.use_multi_thread_)
		Globals::config.NUM_THREADS = 1;

	int NumThreads = Globals::config.NUM_THREADS;

	traffic_agent_sim_.resize(NumThreads);
	for (int tid = 0; tid < NumThreads; tid++) {
		traffic_agent_sim_[tid] = new RVO::RVOSimulator();
		traffic_agent_sim_[tid]->setTimeStep(1.0f / ModelParams::CONTROL_FREQ);
		traffic_agent_sim_[tid]->setAgentDefaults(5.0f, 5, 1.5f, 1.5f, PED_SIZE,
				2.5f);
	}

	default_car_ = AgentParams::getDefaultAgentParam("Car");
	default_bike_ = AgentParams::getDefaultAgentParam("Bike");
	default_ped_ = AgentParams::getDefaultAgentParam("People");
}

void WorldModel::EnsureMeanDirExist(int agent_id, int intention_id) {
	auto it = ped_mean_dirs.find(agent_id);
	if (it == ped_mean_dirs.end()) {
		vector<COORD> dirs(GetNumIntentions(agent_id));
		ped_mean_dirs[agent_id] = dirs;
	} else if (it->second.size() != GetNumIntentions(agent_id)) { // path list size has been updated
		ped_mean_dirs[agent_id].resize(GetNumIntentions(agent_id));
	}

	if (intention_id != -1 && intention_id >= ped_mean_dirs[agent_id].size()) {
		ERR(string_sprintf(
			"Encountering overflowed intention id %d at agent %d\n ",
			intention_id, agent_id));
	}
}

void WorldModel::PrepareAttentiveAgentMeanDirs(const State* state) {
	const PomdpStateWorld* crowd_state = static_cast<const PomdpStateWorld*>(state);
	const CarStruct& car = crowd_state->car;
	const auto& agents = crowd_state->agents;
	int num_agents = crowd_state->num;
	logd << "num_agents in belief tracker: " << num_agents << endl;

	if (num_agents == 0)
		return;

	for (int i = 0; i < crowd_state->num; i++) {
		EnsureMeanDirExist(agents[i].id, -1);
	}

	if (initial_update_step){
		initial_update_step = false;
		for (int i = 0; i < crowd_state->num; i++) {
			int id = agents[i].id;
			for (int intention_id = 0; intention_id < GetNumIntentions(id);	intention_id++)
				ped_mean_dirs[id][intention_id] = COORD(0.0, 0.0);
		}
	}
	else {
		int threadID = GetThreadID();

		traffic_agent_sim_[threadID]->clearAllAgents();

		std::vector<int> agent_ids;
		int gamma_id = 0;
		for (int i = 0; i < crowd_state->num; i++) {
			AddGammaAgent(agents[i], gamma_id);
			gamma_id++;
		}
		AddEgoGammaAgent(num_agents, car);

		traffic_agent_sim_[threadID]->doPreStep();

		gamma_id = 0;
		for (int i = 0; i < crowd_state->num; i++) {
			auto& agent = agents[i];
			int agent_id = agent.id;
			// For each ego agent
			for (int intention_id = 0; intention_id < GetNumIntentions(agent_id);
					intention_id++) {
				RVO::Vector2 ori_pos =
						traffic_agent_sim_[threadID]->getAgentPosition(gamma_id);
				// Leave other pedestrians to have default preferred velocity
				ValidateIntention(agent_id, intention_id, __FUNCTION__, __LINE__);
				auto goal_pos = GetGoalPos(agent, intention_id);
				RVO::Vector2 goal(goal_pos.x, goal_pos.y);
				if (abs(goal - traffic_agent_sim_[threadID]->getAgentPosition(gamma_id)) < 0.5) {
					// Agent is within 0.5 meters of its goal, set preferred velocity to zero
					traffic_agent_sim_[threadID]->setAgentPrefVelocity(gamma_id,
							RVO::Vector2(0.0f, 0.0f));
				} else {
					// Agent is far away from its goal, set preferred velocity as unit vector towards agent's goal.
					double spd = agent.speed;
					traffic_agent_sim_[threadID]->setAgentPrefVelocity(gamma_id,
							normalize(goal - traffic_agent_sim_[threadID]->getAgentPosition(gamma_id)) * spd);
				}

				traffic_agent_sim_[threadID]->doStepForOneAgent(gamma_id);

				COORD dir;
				dir.x = traffic_agent_sim_[threadID]->getAgentPosition(gamma_id).x()
						- agent.pos.x;
				dir.y = traffic_agent_sim_[threadID]->getAgentPosition(gamma_id).y()
						- agent.pos.y;

				logv << "[PrepareAttentiveAgentMeanDirs] num agents in ped_mean_dirs="
						<< ped_mean_dirs.size() << " intention_list len for agent " << agent_id << "="
						<< ped_mean_dirs[agent_id].size() << "\n";
				ped_mean_dirs[agent_id][intention_id] = dir;

				// reset back agent position
				traffic_agent_sim_[threadID]->setAgentPosition(gamma_id, ori_pos);
			}
			gamma_id++;
		}
	}
}


std::string WorldModel::GetLaneText(LaneCode code) const {
	switch (code) {
	case LaneCode::KEEP: return "keep lane";
	case LaneCode::LEFT: return "left lane";
	case LaneCode::RIGHT: return "right lane";
	}
}

