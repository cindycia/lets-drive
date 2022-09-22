#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <unordered_map>

#include <core/builtin_lower_bounds.h>
#include <core/builtin_policy.h>
#include <core/builtin_upper_bounds.h>
#include <core/prior.h>
#include <GPUcore/thread_globals.h>
#include <interface/default_policy.h>
#include <interface/world.h>
#include <math_utils.h>
#include <solver/despot.h>
#include <util/seeds.h>
#include <despot/util/logging.h>

#include <GammaParams.h>

#include "config.h"
#include "coord.h"
#include "threaded_print.h"
#include "debug_util.h"
#include "disabled_util.h"
#include "neural_prior.h"
#include "crowd_belief.h"
#include "world_model.h"
#include "context_pomdp.h"
#include "simulator_base.h"

using namespace despot;


double path_look_ahead = 5.0;

#undef LOG
#define LOG(lv) \
if (despot::logging::level() < despot::logging::ERROR || despot::logging::level() < lv) ; \
else despot::logging::stream(lv)
#include <despot/util/logging.h>

static std::map<uint64_t, std::vector<int>> Obs_hash_table;
static PomdpState hashed_state;
WorldModel SimulatorBase::world_model;


class ContextPomdpParticleLowerBound: public ParticleLowerBound {
private:
	const ContextPomdp *context_pomdp_;
public:
	ContextPomdpParticleLowerBound(const DSPOMDP *model) :
			ParticleLowerBound(model), context_pomdp_(
					static_cast<const ContextPomdp *>(model)) {
	}

	virtual ValuedAction Value(const std::vector<State *> &particles) const {
		PomdpState *state = static_cast<PomdpState *>(particles[0]);
		int min_step = numeric_limits<int>::max();
		auto &carpos = state->car.pos;
		double carvel = state->car.vel;

		// Find minimum number of steps for car-pedestrian collision
		for (int i = 0; i < state->num; i++) {
			auto &p = state->agents[i];

			if (!context_pomdp_->world_model.InFront(p.pos, state->car))
				continue;
			int step = min_step;
			if (p.speed + carvel > 1e-5)
				step = int(ceil(ModelParams::CONTROL_FREQ * max(
						COORD::EuclideanDistance(carpos, p.pos)
						- ModelParams::CAR_FRONT - CAR_FRONT_MARGIN,
						0.0) / ((p.speed + carvel))));

			if (DoPrintCPU)
				printf("   step,min_step, p.speed + carvel=%d %d %f\n", step,
						min_step, p.speed + carvel);
			min_step = min(step, min_step);
		}

		double value = 0;
		ACT_TYPE default_act;
		if (/*Globals::config.use_prior*/false) {
			double move_penalty = context_pomdp_->MovementPenalty(*state);

			// Case 1, no pedestrian: Constant car speed
			value = move_penalty / (1 - Globals::Discount());
			// Case 2, with pedestrians: Constant car speed, head-on collision with nearest neighbor
			if (min_step != numeric_limits<int>::max()) {
				double crash_penalty = context_pomdp_->CrashPenalty(*state);
				value = (move_penalty) * (1 - Globals::Discount(min_step))
						/ (1 - Globals::Discount())
						+ crash_penalty * Globals::Discount(min_step);
				if (DoPrintCPU)
					printf("   min_step,crash_penalty, value=%d %f %f\n",
							min_step, crash_penalty, value);
			}

			if (DoPrintCPU)
				printf("   min_step,num_peds,move_penalty, value=%d %d %f %f\n",
						min_step, state->num, move_penalty, value);
			// default action, go straight with current speed
			default_act = context_pomdp_->GetActionID(LaneCode::KEEP, AccCode::MTN);
		} else     // Joint-Action-POMDP
		{
			float stay_cost = ModelParams::REWARD_FACTOR_VEL
					* -1.0; // full vel penalty
			int dec_step = round(
					carvel / ModelParams::ACC_SPEED
							* ModelParams::CONTROL_FREQ);

			if (dec_step > min_step) {
				value = context_pomdp_->CrashPenalty(*state);
			} else {
				// 2. stay forever
				value += stay_cost / (1 - Globals::Discount());
			}
			// 1. Decelerate until collision or full stop
			for (int step = 0; step < min(dec_step, min_step); step++) { // -1.0 is action penalty
				value = -1.0 + stay_cost + value * Globals::Discount();
			}

			// default action, go straight and decelerate
			default_act = context_pomdp_->GetActionID(LaneCode::KEEP, AccCode::DEC);

			logd << "base lower bound: " << value << " / " << endl;
		}

		return ValuedAction(default_act, State::Weight(particles) * value);
	}

	virtual FactoredValuedAction FactoredValue(const std::vector<State *> &particles) const {
		PomdpState *state = static_cast<PomdpState *>(particles[0]);
		int min_step = numeric_limits<int>::max();
		auto &carpos = state->car.pos;
		double carvel = state->car.vel;

		// Find minimum number of steps for car-pedestrian collision
		for (int i = 0; i < state->num; i++) {
			auto &p = state->agents[i];

			if (!context_pomdp_->world_model.InFront(p.pos, state->car))
				continue;
			int step = min_step;
			if (p.speed + carvel > 1e-5)
				step = int(ceil(ModelParams::CONTROL_FREQ * max(
						COORD::EuclideanDistance(carpos, p.pos)
						- ModelParams::CAR_FRONT - CAR_FRONT_MARGIN,
						0.0) / ((p.speed + carvel))));

			if (DoPrintCPU)
				printf("   step,min_step, p.speed + carvel=%d %d %f\n", step,
						min_step, p.speed + carvel);
			min_step = min(step, min_step);
		}

		double value[3] = {0, 0, 0};
		ACT_TYPE default_act;

		float stay_cost = ModelParams::REWARD_FACTOR_VEL * -1.0; // full vel penalty
		int dec_step = round(carvel / ModelParams::ACC_SPEED * ModelParams::CONTROL_FREQ);

		if (dec_step > min_step) {
			value[RWD_COL] = context_pomdp_->CrashPenalty(*state);
		} else {
			// 2. stay forever
			value[RWD_NCOL] += stay_cost / (1 - Globals::Discount());
		}
		// 1. Decelerate until collision or full stop
		for (int step = 0; step < min(dec_step, min_step); step++) { // -1.0 is action penalty
			value[RWD_NCOL] = -1.0 + stay_cost + value[RWD_NCOL] * Globals::Discount();
			value[RWD_COL] = value[RWD_COL] * Globals::Discount();
		}

		value[RWD_TOTAL] += value[RWD_COL] + value[RWD_NCOL];
		// Normalize factored value
		value[RWD_COL] = col_value_transform(value[RWD_COL]);
		value[RWD_NCOL] = noncol_value_transform(value[RWD_NCOL]);

//		if (value[RWD_COL] < 0)
//			ERR("RWD_COL < 0 in lb heuristics");

		// default action, go straight and decelerate
		default_act = context_pomdp_->GetActionID(LaneCode::KEEP, AccCode::DEC);

		logd << "[ContextPomdpParticleLowerBound] base lower bound: "
				<< value[RWD_TOTAL] << " "
				<< value[RWD_NCOL] << " "
				<< value[RWD_COL] << " "
				<< "num_particles " << particles.size() << endl;

		return FactoredValuedAction(default_act,
				{State::Weight(particles) * value[RWD_TOTAL],
				State::Weight(particles) * value[RWD_NCOL],
				State::Weight(particles) * value[RWD_COL]});
	}
};

class ContextPomdpSmartScenarioLowerBound: public DefaultPolicy {
protected:
	const ContextPomdp *context_pomdp_;

public:
	ContextPomdpSmartScenarioLowerBound(const DSPOMDP *model,
			ParticleLowerBound *bound) :
			DefaultPolicy(model, bound), context_pomdp_(
					static_cast<const ContextPomdp *>(model)) {
	}

	int Action(const std::vector<State *> &particles, RandomStreams &streams,
			History &history) const {
		return context_pomdp_->world_model.DefaultPolicy(particles);
	}
};

class ContextPomdpSmartParticleUpperBound: public ParticleUpperBound {
protected:
	const ContextPomdp *context_pomdp_;
public:
	ContextPomdpSmartParticleUpperBound(const DSPOMDP *model) :
			context_pomdp_(static_cast<const ContextPomdp *>(model)) {
	}

	double Value(const State &s) const {
		const PomdpState &state = static_cast<const PomdpState &>(s);
		double max_step_reward = 0.1;
		return max_step_reward / (1.0 - Globals::Discount());
//		int min_step = context_pomdp_->world_model.MinStepToGoal(state);
//		return -ModelParams::TIME_REWARD * min_step
//				+ ModelParams::GOAL_REWARD * Globals::Discount(min_step);
	}
};

ContextPomdp::ContextPomdp() : world_model(SimulatorBase::world_model),
		random_(Random((unsigned) Seeds::Next())) {
	InitGAMMASetting();
}

SolverPrior *ContextPomdp::CreateSolverPrior(World *world, std::string name,
		bool update_prior) const {
	SolverPrior *prior = NULL;

	if (name == "NEURAL") {
		prior = new PedNeuralSolverPrior(this, world_model);
	}

	logv << "DEBUG: Getting initial state " << endl;

	const State *init_state = world->GetCurrentState();

	logv << "DEBUG: Adding initial state " << endl;

	if (init_state != NULL && update_prior) {
		prior->Add(-1, init_state);
		State *init_search_state_ = CopyForSearch(init_state); //the state is used in search
		prior->Add_in_search(-1, init_search_state_);
		logi << __FUNCTION__ << " add history search state of ts "
				<< static_cast<PomdpState *>(init_search_state_)->time_stamp
				<< endl;
	}

	return prior;
}

void ContextPomdp::InitGAMMASetting() {
	use_gamma_in_search = true;
	use_gamma_in_simulation = true;
	use_simplified_gamma = false;

	if (use_simplified_gamma) {
		use_gamma_in_search = true;
		use_gamma_in_simulation = true;

		GammaParams::use_polygon = false;
		GammaParams::consider_kinematics = false;
		GammaParams::use_dynamic_att = false;
	}
}

const std::vector<int> &ContextPomdp::ObserveVector(const State &state_) const {
	const PomdpState &state = static_cast<const PomdpState &>(state_);
	static std::vector<int> obs_vec;

	obs_vec.resize(state.num * 2 + 3);

	int i = 0;
	obs_vec[i++] = int(state.car.pos.x / ModelParams::POS_RLN);
	obs_vec[i++] = int(state.car.pos.y / ModelParams::POS_RLN);

	obs_vec[i++] = int((state.car.vel + 1e-5) / ModelParams::VEL_RLN); //add some noise to make 1.5/0.003=50

	for (int j = 0; j < state.num; j++) {
		obs_vec[i++] = int(state.agents[j].pos.x / ModelParams::POS_RLN);
		obs_vec[i++] = int(state.agents[j].pos.y / ModelParams::POS_RLN);
	}

	return obs_vec;
}

uint64_t ContextPomdp::Observe(const State &state) const {
	hash<std::vector<int>> myhash;
	return myhash(ObserveVector(state));
}

std::vector<State *> ContextPomdp::ConstructParticles(
		std::vector<PomdpState> &samples) const {
	int num_particles = samples.size();
	std::vector<State *> particles;
	for (int i = 0; i < samples.size(); i++) {
		PomdpState *particle = static_cast<PomdpState *>(Allocate(-1,
				1.0 / num_particles));
		(*particle) = samples[i];
		particle->SetAllocated();
		particle->weight = 1.0 / num_particles;
		particles.push_back(particle);
	}

	return particles;
}

// Very high cost for collision
double ContextPomdp::CrashPenalty(const PomdpState &state) const // , int closest_ped, double closest_dist) const {
		{
	// double ped_vel = state.agent[closest_ped].vel;
	return ModelParams::CRASH_PENALTY
		* (state.car.vel + ModelParams::REWARD_BASE_CRASH_VEL);
}

// Very high cost for collision
double ContextPomdp::CrashPenalty(const PomdpStateWorld &state) const // , int closest_ped, double closest_dist) const {
		{
	// double ped_vel = state.agent[closest_ped].vel;
	return ModelParams::CRASH_PENALTY
			* (state.car.vel + ModelParams::REWARD_BASE_CRASH_VEL);
}

double ContextPomdp::TTCPenalty(double ttc, const PomdpStateWorld& state) const {
	if (ttc > ModelParams::VEL_MAX / ModelParams::ACC_SPEED + 1.0) // 2 seconds
		return 0.0;
	else if (ttc < 1e-5) // collision penalty should be used instead
		return 0.0;
	else {
		double max_penalty = 1.0 * ModelParams::CONTROL_FREQ;
		return -std::pow((1.0 / ttc) * 9.0 / max_penalty, 2);
	}

}

double ContextPomdp::InvalidActionPenalty(int action, const PomdpStateWorld& state) const {
	int lane = GetLaneID(action);
	if (!world_model.LaneExist(state.car.pos, state.car.heading_dir, lane)) {
		double fake_ttc = 0.3;
		double max_penalty = 1.0 * ModelParams::CONTROL_FREQ;
		return -std::pow((1.0 / fake_ttc) * 9.0 / max_penalty, 2);
	}

	return 0.0;
}

// Avoid frequent dec or acc
double ContextPomdp::ActionPenalty(int action) const {
	double reward = 0.0;
	if (GetAccelerationID(action) == AccCode::DEC) //DEC
		reward -= 0.1;

	if (abs(GetLane(action)) > 0.01) // lane change
		reward -= ModelParams::REWARD_FACTOR_VEL;

	return reward;
}

// Less penalty for longer distance travelled
double ContextPomdp::MovementPenalty(const PomdpState &state) const {
	return min(ModelParams::REWARD_FACTOR_VEL
			* (state.car.vel - ModelParams::VEL_MAX) / ModelParams::VEL_MAX, 0.0);
}

double ContextPomdp::MovementPenalty(const PomdpState &state, float lane) const {
	return min(ModelParams::REWARD_FACTOR_VEL
				* (state.car.vel - ModelParams::VEL_MAX) / ModelParams::VEL_MAX, 0.0);
}

// Less penalty for longer distance travelled
double ContextPomdp::MovementPenalty(const PomdpStateWorld &state) const {
	return min(ModelParams::REWARD_FACTOR_VEL
				* (state.car.vel - ModelParams::VEL_MAX) / ModelParams::VEL_MAX, 0.0);
}

double ContextPomdp::MovementPenalty(const PomdpStateWorld &state,
		float lane) const {
	return min(ModelParams::REWARD_FACTOR_VEL
				* (state.car.vel - ModelParams::VEL_MAX) / ModelParams::VEL_MAX, 0.0);
}

double ContextPomdp::Reward(const State& _state, ACT_TYPE action) const {
	const PomdpState &state = static_cast<const PomdpState &>(_state);
	double reward = 0.0;
//	if (world_model.IsGlobalGoal(state.car)) {
//		reward = ModelParams::GOAL_REWARD;
//
//		cout << "assigning goal reward " << reward << endl;
//		return reward;
//	}

	// Safety control: collision; Terminate upon collision
	if (!world_model.IsInMap(state.car)) {
		reward = CrashPenalty(state);
		return reward;
	}

	int col_agent = 0;
	if (state.car.vel > 0.001 && world_model.InCollision(state, col_agent)) /// collision occurs only when car is moving
			{
		reward = CrashPenalty(state);

		cout << "assigning collision reward " << reward << endl;
		return reward;
	}

	// Smoothness control
	double acc_reward = ActionPenalty(action);
	reward += acc_reward;
	cout << "assigning action reward " << acc_reward << endl;

	// Speed control: Encourage higher speed
	double lane = GetLane(action);
	double move_reward = MovementPenalty(state, lane);

	cout << "assigning move reward " << acc_reward << endl;
	cout << "Scaling factor=" << ModelParams::REWARD_FACTOR_VEL << ", car_vel="
			<< state.car.vel << ", VEL_MAX=" << ModelParams::VEL_MAX << endl;

	reward += move_reward;

	return reward;
}

bool ContextPomdp::Step(State &state_, double rNum, int action, double &reward,
		uint64_t &obs) const {
	logd << "Step function" << endl;
	PomdpState &state = static_cast<PomdpState &>(state_);
	reward = 0.0;

	////// NOTE: Using true random number to make results in different qnodes different ////
	rNum = Random::RANDOM.NextDouble();

	if (FIX_SCENARIO == 1 || DESPOT::Print_nodes) {
		if (CPUDoPrint && state_.scenario_id == CPUPrintPID) {
			printf("(CPU) Before step: scenario%d \n", state_.scenario_id);
			printf("action= %d \n", action);
			PomdpState *context_pomdp_state = static_cast<PomdpState *>(&state_);
			printf("Before step:\n");
			printf("car_pos= %f,%f", context_pomdp_state->car.pos.x,
					context_pomdp_state->car.pos.y);
			printf("car_heading=%f\n", context_pomdp_state->car.heading_dir);
			printf("car_vel= %f\n", context_pomdp_state->car.vel);
			for (int i = 0; i < context_pomdp_state->num; i++) {
				printf("agent %d pox_x= %f pos_y=%f\n", i,
						context_pomdp_state->agents[i].pos.x,
						context_pomdp_state->agents[i].pos.y);
			}
		}
	}
	// Terminate upon reaching goal
	if (world_model.IsGlobalGoal(state.car)) {
		world_model.ExtendCurLane(&state);

//		reward = ModelParams::GOAL_REWARD;
//		logv << "assigning goal reward " << reward << endl;
//		ERR("Goal (end of path) should not be reached!!!");
//		return true;
	}

	// Safety control: collision; Terminate upon collision
	if (!world_model.IsInMap(state.car, true)) {
		reward = CrashPenalty(state);
		return true;
	}

	int col_agent = 0;
	if (state.car.vel > 0.001 && world_model.InCollision(state, col_agent)) /// collision occurs only when car is moving
			{
		reward = CrashPenalty(state);

		logv << "assigning collision reward " << reward << endl;
		return true;
	}

	// Smoothness control
	reward += ActionPenalty(action);

	// Speed control: Encourage higher speed
	double lane = GetLane(action);
	if (Globals::config.use_prior)
		reward += MovementPenalty(state);
	else {
		reward += MovementPenalty(state, lane);
	}

	// State transition
	if (Globals::config.use_multi_thread_) {
		QuickRandom::SetSeed(INIT_QUICKRANDSEED,
				Globals::MapThread(this_thread::get_id()));
	} else
		QuickRandom::SetSeed(INIT_QUICKRANDSEED, 0);

	logv << "[ContextPomdp::" << __FUNCTION__ << "] Refract action" << endl;
	double acc = GetAcceleration(action);

	world_model.RobStep(state.car, lane, rNum);
	world_model.RobVelStep(state.car, acc, rNum);

	state.time_stamp = state.time_stamp + 1.0 / ModelParams::CONTROL_FREQ;

	if (use_gamma_in_search) {
		// Attentive pedestrians
		world_model.GammaAgentStep(state.agents, rNum, state.num, state.car);
		for (int i = 0; i < state.num; i++) {
			//Distracted pedestrians
			if (state.agents[i].mode == AGENT_DIS)
				world_model.AgentStep(state.agents[i], rNum);
		}
	} else {
		for (int i = 0; i < state.num; i++) {
			world_model.AgentStep(state.agents[i], rNum);
			if (isnan(state.agents[i].pos.x))
				ERR("state.agents[i].pos.x is NAN");
		}
	}

	if (CPUDoPrint && state.scenario_id == CPUPrintPID) {
		if (true) {
			PomdpState *context_pomdp_state = static_cast<PomdpState *>(&state_);
			printf("(CPU) After step: scenario=%d \n",
					context_pomdp_state->scenario_id);
			printf("rand=%f, action=%d \n", rNum, action);
			printf("After step:\n");
			printf("Reward=%f\n", reward);

			printf("car_pos= %f,%f", context_pomdp_state->car.pos.x,
					context_pomdp_state->car.pos.y);
			printf("car_heading=%f\n", context_pomdp_state->car.heading_dir);
			printf("car vel= %f\n", context_pomdp_state->car.vel);
			for (int i = 0; i < context_pomdp_state->num; i++) {
				printf("agent %d pox_x= %f pos_y=%f\n", i,
						context_pomdp_state->agents[i].pos.x,
						context_pomdp_state->agents[i].pos.y);
			}
		}
	}

	// Observation
	obs = Observe(state);
	return false;
}

bool ContextPomdp::Step(State& state_, double rNum, ACT_TYPE action, double reward[], uint64_t& obs) const {
	logd << "Step function" << endl;
	PomdpState &state = static_cast<PomdpState &>(state_);
	for (int i = 0; i < NumRewardFactors(); i++)
		reward[i] = 0.0;

	////// NOTE: Using true random number to make results in different qnodes different ////
	rNum = Random::RANDOM.NextDouble();

	// Terminate upon reaching goal
	if (world_model.IsGlobalGoal(state.car)) {
		world_model.ExtendCurLane(&state);
	}

	// Safety control: collision; Terminate upon collision
	if (!world_model.IsInMap(state.car, true)) {
		reward[RWD_TOTAL] = CrashPenalty(state);
		reward[RWD_NCOL] = 0.0;
		reward[RWD_COL] = col_value_transform(reward[RWD_TOTAL]);

//		if (reward[RWD_COL] < 0)
//			ERR("RWD_COL < 0 in step function");
		return true;
	}

	int col_agent = 0;
	if (state.car.vel > 0.001 && world_model.InCollision(state, col_agent)) /// collision occurs only when car is moving
			{
		reward[RWD_TOTAL] = CrashPenalty(state);
		reward[RWD_NCOL] = 0.0;
		reward[RWD_COL] = col_value_transform(reward[RWD_TOTAL]);
//		if (reward[RWD_COL] < 0)
//			ERR("RWD_COL < 0 in step function");
		return true;
	}

	// Smoothness control
	reward[RWD_NCOL] += ActionPenalty(action);

	// Speed control: Encourage higher speed
	double lane = GetLane(action);
	if (Globals::config.use_prior)
		reward[RWD_NCOL] += MovementPenalty(state);
	else {
		reward[RWD_NCOL] += MovementPenalty(state, lane);
	}

	reward[RWD_TOTAL] = reward[RWD_NCOL];
	reward[RWD_NCOL] = noncol_value_transform(reward[RWD_NCOL]);
	reward[RWD_COL] = 0.0;

	// State transition
	if (Globals::config.use_multi_thread_) {
		QuickRandom::SetSeed(INIT_QUICKRANDSEED,
				Globals::MapThread(this_thread::get_id()));
	} else
		QuickRandom::SetSeed(INIT_QUICKRANDSEED, 0);

	logv << "[ContextPomdp::" << __FUNCTION__ << "] Refract action" << endl;
	double acc = GetAcceleration(action);

	world_model.RobStep(state.car, lane, rNum);
	world_model.RobVelStep(state.car, acc, rNum);

	state.time_stamp = state.time_stamp + 1.0 / ModelParams::CONTROL_FREQ;

	if (use_gamma_in_search) {
		// Attentive pedestrians
		world_model.GammaAgentStep(state.agents, rNum, state.num, state.car);
		for (int i = 0; i < state.num; i++) {
			//Distracted pedestrians
			if (state.agents[i].mode == AGENT_DIS)
				world_model.AgentStep(state.agents[i], rNum);
		}
	} else {
		for (int i = 0; i < state.num; i++) {
			world_model.AgentStep(state.agents[i], rNum);
			if (isnan(state.agents[i].pos.x))
				ERR("state.agents[i].pos.x is NAN");
		}
	}

	// Observation
	obs = Observe(state);
	return false;
}

bool ContextPomdp::Step(PomdpStateWorld &state, double rNum, int action,
		double &reward, uint64_t &obs) const {

	reward = 0.0;

	// Terminate upon reaching goal
//	if (world_model.IsGlobalGoal(state.car)) {
//		reward = ModelParams::GOAL_REWARD;
//
//		logv << "assigning goal reward " << reward << endl;
//		return true;
//	}

	if (!world_model.IsInMap(state.car)) {
		reward = CrashPenalty(state);
		return true;
	}

	if (state.car.vel > 0.001 && world_model.InRealCollision(state, 120.0)) /// collision occurs only when car is moving
			{
		reward = CrashPenalty(state);
		return true;
	}

	// Smoothness control
	reward += ActionPenalty(action);

	// Speed control: Encourage higher speed
	double lane = GetLane(action);
	if (Globals::config.use_prior)
		reward += MovementPenalty(state);
	else
		reward += MovementPenalty(state, lane);

	// State transition
	Random random(rNum);
	logv << "[ContextPomdp::" << __FUNCTION__ << "] Refract action" << endl;
	double acc = GetAcceleration(action);

	world_model.RobStep(state.car, lane, random);
	world_model.RobVelStep(state.car, acc, random);

	if (use_gamma_in_simulation) {
		// Attentive pedestrians
		double zero_rand = 0.0;
		world_model.GammaAgentStep(state.agents, zero_rand, state.num, state.car);
		// Distracted pedestrians
		for (int i = 0; i < state.num; i++) {
			if (state.agents[i].mode == AGENT_DIS)
				world_model.AgentStep(state.agents[i], random);
		}
	} else {
		for (int i = 0; i < state.num; i++)
			world_model.AgentStep(state.agents[i], random);
	}
	return false;
}

void ContextPomdp::ForwardAndVisualize(const State *sample, int step) const {
	PomdpState *next_state = static_cast<PomdpState *>(Copy(sample));

	for (int i = 0; i < step; i++) {
		// forward
		next_state = PredictAgents(next_state);

		// print
		PrintStateCar(*next_state, string_sprintf("predicted_car_%d", i));
		PrintStateAgents(*next_state, string_sprintf("predicted_agents_%d", i));
	}
}

PomdpState* ContextPomdp::PredictAgents(const PomdpState *ped_state, int acc) const {
	PomdpState* predicted_state = static_cast<PomdpState*>(Copy(ped_state));

	ACT_TYPE action = GetActionID(LaneCode::KEEP, AccCode::MTN);

	OBS_TYPE dummy_obs;
	double dummy_reward;

	double rNum = Random::RANDOM.NextDouble();
	bool terminal = Step(*predicted_state, rNum, action, dummy_reward, dummy_obs);

	if (terminal)
		logi << "[PredictAgents] Reach terminal state" << endl;

	return predicted_state;
}

double ContextPomdp::TimeToCollision(const PomdpStateWorld *ped_state, int acc) const {
	PomdpStateWorld* predicted_state = static_cast<PomdpStateWorld*>(Copy(ped_state));
	predicted_state->num = min(predicted_state->num, ModelParams::N_PED_IN);
	ACT_TYPE action = GetActionID(LaneCode::KEEP, AccCode::MTN);

	OBS_TYPE dummy_obs;
	double dummy_reward;
	bool terminal = false;
	double ttc = 1.0 / ModelParams::CONTROL_FREQ;

	for (int i = 0; i < predicted_state->num; i++){
		auto& agent = predicted_state->agents[i];
		agent.mode = AGENT_DIS;
		agent.intention = 0;
	}

	int step = 0;
	while(step < 15) {
		terminal = Step(*predicted_state, 0.0, action, dummy_reward, dummy_obs);
		if (!terminal)
			ttc += 1.0 / ModelParams::CONTROL_FREQ;
		else
			break;
		step ++;
	}

	return ttc;
}

double ContextPomdp::ObsProb(uint64_t obs, const State &s, int action) const {
	return obs == Observe(s);
}

std::vector<std::vector<double>> ContextPomdp::GetBeliefVector(
		const std::vector<State *> particles) const {
	std::vector<std::vector<double>> belief_vec;
	return belief_vec;
}

Belief *ContextPomdp::InitialBelief(const State *state, string type) const {

	//Uniform initial distribution
	CrowdBelief *belief = new CrowdBelief(this);
	return belief;
}

ValuedAction ContextPomdp::GetBestAction() const {
	return ValuedAction(0,
			ModelParams::CRASH_PENALTY
			* (ModelParams::VEL_MAX	+ ModelParams::REWARD_BASE_CRASH_VEL));
}

double ContextPomdp::GetMaxReward() const {
	return 0;
}

ScenarioLowerBound *ContextPomdp::CreateScenarioLowerBound(string name,
		string particle_bound_name) const {
	name = "SMART";
	ScenarioLowerBound *lb;
	if (name == "TRIVIAL") {
		lb = new TrivialParticleLowerBound(this);
	} else if (name == "RANDOM") {
		lb = new RandomPolicy(this, new ContextPomdpParticleLowerBound(this));
	} else if (name == "SMART") {
		Globals::config.rollout_type = "INDEPENDENT";
		cout << "[LowerBound] Smart policy independent rollout" << endl;
		lb = new ContextPomdpSmartScenarioLowerBound(this,
				new ContextPomdpParticleLowerBound(this));
	} else {
		cerr << "[LowerBound] Unsupported scenario lower bound: " << name
				<< endl;
		exit(0);
	}

	return lb;
}

ParticleUpperBound *ContextPomdp::CreateParticleUpperBound(string name) const {
	name = "SMART";
	if (name == "TRIVIAL") {
		return new TrivialParticleUpperBound(this);
	} else if (name == "SMART") {
		return new ContextPomdpSmartParticleUpperBound(this);
	} else {
		cerr << "Unsupported particle upper bound: " << name << endl;
		exit(0);
	}
}

ScenarioUpperBound *ContextPomdp::CreateScenarioUpperBound(string name,
		string particle_bound_name) const {
	// name = "SMART";
	name = "TRIVIAL";
	ScenarioUpperBound *ub;
	if (name == "TRIVIAL") {
		cout << "[UpperBound] Trivial upper bound" << endl;
		ub = new TrivialParticleUpperBound(this);
	} else if (name == "SMART") {
		cout << "[UpperBound] Smart upper bound" << endl;
		ub = new ContextPomdpSmartParticleUpperBound(this);
	} else {
		cerr << "[UpperBound] Unsupported scenario upper bound: " << name
				<< endl;
		exit(0);
	}
	return ub;
}

/// output the probability of the intentions of the pedestrians
void ContextPomdp::Statistics(const std::vector<PomdpState *> particles) const {
	return;
	double goal_count[10][10] = { { 0 } };
	cout << "Current Belief" << endl;
	if (particles.size() == 0)
		return;

	PrintState(*particles[0]);
	PomdpState *state_0 = particles[0];
	for (int i = 0; i < particles.size(); i++) {
		PomdpState *state = particles[i];
		for (int j = 0; j < state->num; j++) {
			goal_count[j][state->agents[j].intention] += particles[i]->weight;
		}
	}

	for (int j = 0; j < state_0->num; j++) {
		cout << "Ped " << j << " Belief is ";
		for (int i = 0; i < world_model.GetNumIntentions(state_0->agents[j].id);
				i++) {
			cout << (goal_count[j][i] + 0.0) << " ";
		}
		cout << endl;
	}
}

void ContextPomdp::PrintState(const State &s, string msg, ostream &out) const {

	if (DESPOT::Debug_mode)
		return;

	if (msg == "")
		out << "Search state:\n";
	else
		cout << msg << endl;

	PrintState(s, out);
}

void ContextPomdp::PrintState(const State &s, ostream &out) const {

	if (DESPOT::Debug_mode)
		return;

	out << "Address: " << &s << endl;

	if (static_cast<const PomdpStateWorld *>(&s) != NULL) {

		if (static_cast<const PomdpStateWorld *>(&s)->num
				> ModelParams::N_PED_IN) {
			PrintWorldState(static_cast<const PomdpStateWorld &>(s), out);
			return;
		}
	}

	const PomdpState &state = static_cast<const PomdpState &>(s);
	auto &carpos = state.car.pos;

	out << "car pos / heading / vel = " << "(" << carpos.x << ", " << carpos.y
			<< ") / " << state.car.heading_dir << " / " << state.car.vel
			<< " car dim " << ModelParams::CAR_WIDTH << " "
			<< ModelParams::CAR_FRONT * 2 << endl;
	out << state.num << " pedestrians " << endl;
	for (int i = 0; i < state.num; i++) {
		out << "agent " << i
				<< ": id / pos / speed / vel / intention / dist2car / infront =  "
				<< state.agents[i].id << " / " << "(" << state.agents[i].pos.x
				<< ", " << state.agents[i].pos.y << ") / "
				<< state.agents[i].speed << " / " << "("
				<< state.agents[i].vel.x << ", " << state.agents[i].vel.y
				<< ") / " << state.agents[i].intention << " / "
				<< COORD::EuclideanDistance(state.agents[i].pos, carpos)
				<< " / " << world_model.InFront(state.agents[i].pos, state.car)
				<< " (mode) " << state.agents[i].mode << " (type) "
				<< state.agents[i].type << " (bb) "
				<< state.agents[i].bb_extent_x << " "
				<< state.agents[i].bb_extent_y << " (cross) "
				<< state.agents[i].cross_dir << " (heading) "
				<< state.agents[i].heading_dir << endl;
	}

	double min_dist = -1;
	if (state.num > 0)
		min_dist = COORD::EuclideanDistance(carpos, state.agents[0].pos);
	out << "MinDist: " << min_dist << endl;

	ValidateState(state, __FUNCTION__);
}

void ContextPomdp::PrintStateCar(const State &s, std::string msg,
		ostream &out) const {
	const PomdpState &state = static_cast<const PomdpState &>(s);
	out << msg << " ";
	out << state.car.pos.x << " " << state.car.pos.y << " "
			<< state.car.heading_dir << endl;
}

PomdpState last_state;
void ContextPomdp::PrintStateAgents(const State &s, std::string msg,
		ostream &out) const {

	const PomdpState &state = static_cast<const PomdpState &>(s);

	out << msg << " ";
	for (int i = 0; i < state.num; i++) {
		out << state.agents[i].pos.x << " " << state.agents[i].pos.y << " "
				<< state.agents[i].heading_dir << " "
				<< state.agents[i].bb_extent_x << " "
				<< state.agents[i].bb_extent_y << " ";
	}
	out << endl;

//	out << "vel ";
//	for (int i = 0; i < state.num; i++) {
//		out << COORD::EuclideanDistance(last_state.agents[i].pos, state.agents[i].pos) * ModelParams::CONTROL_FREQ << " ";
//	}
//	out << endl;
//
//	out << "cur_vel intention ";
//	for (int i = 0; i < state.num; i++) {
//		out << world_model.IsCurVelIntention(state.agents[i].intention, state.agents[i].id) << " ";
//	}
//	out << endl;
//
//	out << "step intention ";
//	for (int i = 0; i < state.num; i++) {
//		out << world_model.IsStopIntention(state.agents[i].intention, state.agents[i].id) << " ";
//	}
//	out << endl;

	last_state = state;
}

void ContextPomdp::PrintWorldState(const PomdpStateWorld &state,
		ostream &out) const {
	out << "World state:\n";
	auto &carpos = state.car.pos;
	out << "car pos / heading / vel = " << "(" << carpos.x << ", " << carpos.y
			<< ") / " << state.car.heading_dir << " / " << state.car.vel
			<< " car dim " << ModelParams::CAR_WIDTH << " "
			<< ModelParams::CAR_FRONT * 2 << endl;
	out << state.num << " pedestrians " << endl;

	double min_dist = -1;
	int mindist_id = 0;

	for (int i = 0; i < state.num; i++) {
		if (COORD::EuclideanDistance(state.agents[i].pos, carpos) < min_dist) {
			min_dist = COORD::EuclideanDistance(state.agents[i].pos, carpos);
			mindist_id = i;
		}

		string intention_str = "";
		if (world_model.IsCurVelIntention(state.agents[i].intention, state.agents[i].id))
			intention_str = "cur_vel";
		else if (world_model.IsStopIntention(state.agents[i].intention, state.agents[i].id))
			intention_str = "stop";
		else
			intention_str = "path_" + to_string(state.agents[i].intention);

		string mode_str = "";
		if (state.agents[i].mode == AGENT_DIS)
			mode_str = "dis";
		else
			mode_str = "att";

		out << "agent " << i
				<< ": id / pos / speed / vel / intention / dist2car / infront =  "
				<< state.agents[i].id << " / " << "(" << state.agents[i].pos.x
				<< ", " << state.agents[i].pos.y << ") / "
				<< state.agents[i].speed << " / " << "("
				<< state.agents[i].vel.x << ", " << state.agents[i].vel.y
				<< ") / " << intention_str << " / "
				<< COORD::EuclideanDistance(state.agents[i].pos, carpos)
				<< " / " << world_model.InFront(state.agents[i].pos, state.car)
				<< " (mode) " << mode_str << " (type) "
				<< state.agents[i].type << " (bb) "
				<< state.agents[i].bb_extent_x << " "
				<< state.agents[i].bb_extent_y << " (cross) "
				<< state.agents[i].cross_dir << " (heading) "
				<< state.agents[i].heading_dir << endl;

		world_model.PathCandidates(state.agents[i].id);
	}

	if (state.num > 0)
		min_dist = COORD::EuclideanDistance(carpos,
				state.agents[mindist_id].pos);

	out << "MinDist: " << min_dist << endl;
}

void ContextPomdp::PrintObs(const State &state, uint64_t obs, ostream &out) const {
	out << obs << endl;
}

void ContextPomdp::PrintAction(int action, ostream &out) const {
	out << action << endl;
}

void ContextPomdp::PrintBelief(const Belief &belief, ostream &out) const {

}

/// output the probability of the intentions of the pedestrians
void ContextPomdp::PrintParticles(const vector<State *> particles,
		ostream &out) const {
	cout << "Particles for planning:" << endl;
	double goal_count[ModelParams::N_PED_IN][10] = { { 0 } };
	double q_goal_count[ModelParams::N_PED_IN][10] = { { 0 } }; //without weight, it is q;

	double type_count[ModelParams::N_PED_IN][AGENT_DIS + 1] = { { 0 } };

	double q_single_weight;
	q_single_weight = 1.0 / particles.size();
	cout << "Current Belief with " << particles.size() << " particles" << endl;
	if (particles.size() == 0)
		return;
	const PomdpState *pomdp_state =
			static_cast<const PomdpState *>(particles.at(0));

	if (DESPOT::Debug_mode) {
		DESPOT::Debug_mode = false;
		PrintState(*pomdp_state);
		DESPOT::Debug_mode = true;
	}

	for (int i = 0; i < particles.size(); i++) {
		const PomdpState *pomdp_state =
				static_cast<const PomdpState *>(particles.at(i));
		for (int j = 0; j < pomdp_state->num; j++) {
			goal_count[j][pomdp_state->agents[j].intention] +=
					particles[i]->weight;
			q_goal_count[j][pomdp_state->agents[j].intention] +=
					q_single_weight;
			type_count[j][pomdp_state->agents[j].mode] += particles[i]->weight;
		}
	}

	cout << "agent 0 vel: " << pomdp_state->agents[0].speed << endl;

	for (int j = 0; j < 6; j++) {
		cout << "Ped " << pomdp_state->agents[j].id << " Belief is ";
		for (int i = 0;
				i < world_model.GetNumIntentions(pomdp_state->agents[j].id);
				i++) {
			cout << (goal_count[j][i] + 0.0) << " ";
		}
		cout << endl;
		for (int i = 0; i < AGENT_DIS + 1; i++) {
			cout << (type_count[j][i] + 0.0) << " ";
		}
		cout << endl;
	}

	logv << "<><><> q:" << endl;
	for (int j = 0; j < 6; j++) {
		logv << "Ped " << pomdp_state->agents[j].id << " Belief is ";
		for (int i = 0;
				i < world_model.GetNumIntentions(pomdp_state->agents[j].id);
				i++) {
			logv << (q_goal_count[j][i] + 0.0) << " ";
		}
		logv << endl;
	}
}

State *ContextPomdp::Allocate(int state_id, double weight) const {
	//num_active_particles ++;
	PomdpState *particle = memory_pool_.Allocate();
	particle->state_id = state_id;
	particle->weight = weight;
	return particle;
}

State *ContextPomdp::Copy(const State *particle) const {
	PomdpState *new_particle = memory_pool_.Allocate();
	*new_particle = *static_cast<const PomdpState *>(particle);

	new_particle->SetAllocated();
	return new_particle;
}

State *ContextPomdp::CopyForSearch(const State *particle) const {
	PomdpState *new_particle = memory_pool_.Allocate();
	const PomdpStateWorld *world_state =
			static_cast<const PomdpStateWorld *>(particle);

	new_particle->num = min(ModelParams::N_PED_IN, world_state->num);
	new_particle->car = world_state->car;
	for (int i = 0; i < new_particle->num; i++) {
		new_particle->agents[i] = world_state->agents[i];
	}
	new_particle->time_stamp = world_state->time_stamp;
	new_particle->SetAllocated();
	return new_particle;
}

void ContextPomdp::Free(State *particle) const {
	//num_active_particles --;
	memory_pool_.Free(static_cast<PomdpState *>(particle));
}

int ContextPomdp::NumActiveParticles() const {
	return memory_pool_.num_allocated();
}

double ContextPomdp::ImportanceScore(PomdpState *state,
		ACT_TYPE last_action) const {
	double score = 1.8; //0.3 * 6; 0.3 basic score for each pedestrian
	for (int i = 0; i < state->num; i++) {
		AgentStruct agent = state->agents[i];
		CarStruct car = state->car;
		COORD ped_pos = agent.pos;

		const COORD &goal = world_model.GetGoalPos(agent);
		double move_dw, move_dh;
		if (world_model.IsStopIntention(agent.intention, agent.id)) {
			move_dw = 0;
			move_dh = 0;
		} else {
			MyVector goal_vec(goal.x - ped_pos.x, goal.y - ped_pos.y);
			double a = goal_vec.GetAngle();
			MyVector move(a, agent.speed * 1.0, 0); //movement in unit time
			move_dw = move.dw;
			move_dh = move.dh;
		}

		int count = 0;

		for (int t = 1; t <= 5; t++) {
			ped_pos.x += move_dw;
			ped_pos.y += move_dh;

			Random random(double(state->scenario_id));
			int step = car.vel / ModelParams::CONTROL_FREQ;
			for (int i = 0; i < step; i++) {
				world_model.RobStep(state->car, GetLane(last_action), random);
				world_model.RobVelStep(state->car,
						GetAcceleration(last_action), random);
			}

			double d = COORD::EuclideanDistance(car.pos, ped_pos);

			if (d <= 1 && count < 3) {
				count++;
				score += 4;
			} else if (d <= 2 && count < 3) {
				count++;
				score += 2;
			} else if (d <= 3 && count < 3) {
				count++;
				score += 1;
			}
		}
	}

	return score;
}

std::vector<double> ContextPomdp::ImportanceWeight(std::vector<State *> particles,
		ACT_TYPE last_action) const {
	double total_weight = State::Weight(particles);
	double new_total_weight = 0;
	int particles_num = particles.size();

	std::vector<PomdpState *> pomdp_state_particles;
	std::vector<double> importance_weight;

	bool use_is_despot = true;
	if (use_is_despot == false) {
		for (int i = 0; i < particles_num; i++) {
			importance_weight.push_back(particles[i]->weight);
		}
		return importance_weight;
	}

	cout << "use importance sampling ***** " << endl;

	for (int i = 0; i < particles_num; i++) {
		pomdp_state_particles.push_back(
				static_cast<PomdpState *>(particles[i]));
	}

	for (int i = 0; i < particles_num; i++) {

		importance_weight.push_back(
				pomdp_state_particles[i]->weight
						* ImportanceScore(pomdp_state_particles[i],
								last_action));
		new_total_weight += importance_weight[i];
	}

	//normalize to total_weight
	for (int i = 0; i < particles_num; i++) {
		importance_weight[i] = importance_weight[i] * total_weight
				/ new_total_weight;
		assert(importance_weight[i] > 0);
	}

	return importance_weight;
}

int ContextPomdp::NumObservations() const {
	return std::numeric_limits<int>::max();
}
int ContextPomdp::ParallelismInStep() const {
	return ModelParams::N_PED_IN;
}
void ContextPomdp::ExportState(const State &state, std::ostream &out) const {
	PomdpState cardriveState = static_cast<const PomdpState &>(state);
	ios::fmtflags old_settings = out.flags();

	int Width = 7;
	int Prec = 3;
	out << cardriveState.scenario_id << " ";
	out << cardriveState.weight << " ";
	out << cardriveState.num << " ";
	out << cardriveState.car.heading_dir << " " << cardriveState.car.pos.x
			<< " " << cardriveState.car.pos.y << " " << cardriveState.car.vel
			<< " ";
	for (int i = 0; i < ModelParams::N_PED_IN; i++)
		out << cardriveState.agents[i].intention << " "
				<< cardriveState.agents[i].id << " "
				<< cardriveState.agents[i].pos.x << " "
				<< cardriveState.agents[i].pos.y << " "
				<< cardriveState.agents[i].speed << " ";

	out << endl;

	out.flags(old_settings);
}
State *ContextPomdp::ImportState(std::istream &in) const {
	PomdpState *cardriveState = memory_pool_.Allocate();

	if (in.good()) {
		string str;
		while (getline(in, str)) {
			if (!str.empty()) {
				istringstream ss(str);

				ss >> cardriveState->scenario_id;
				ss >> cardriveState->weight;
				ss >> cardriveState->num;
				ss >> cardriveState->car.heading_dir >> cardriveState->car.pos.x
						>> cardriveState->car.pos.y >> cardriveState->car.vel;
				for (int i = 0; i < ModelParams::N_PED_IN; i++)
					ss >> cardriveState->agents[i].intention
							>> cardriveState->agents[i].id
							>> cardriveState->agents[i].pos.x
							>> cardriveState->agents[i].pos.y
							>> cardriveState->agents[i].speed;
			}
		}
	}

	return cardriveState;
}

void ContextPomdp::ImportStateList(std::vector<State *> &particles,
		std::istream &in) const {
	if (in.good()) {
		int PID = 0;
		string str;
		getline(in, str);
		istringstream ss(str);
		int size;
		ss >> size;
		particles.resize(size);
		while (getline(in, str)) {
			if (!str.empty()) {
				if (PID >= particles.size())
					cout << "Import particles error: PID>=particles.size()!"
							<< endl;

				PomdpState *cardriveState = memory_pool_.Allocate();

				istringstream ss(str);

				ss >> cardriveState->scenario_id;
				ss >> cardriveState->weight;
				ss >> cardriveState->num;
				ss >> cardriveState->car.heading_dir >> cardriveState->car.pos.x
						>> cardriveState->car.pos.y >> cardriveState->car.vel;
				for (int i = 0; i < ModelParams::N_PED_IN; i++)
					ss >> cardriveState->agents[i].intention
							>> cardriveState->agents[i].id
							>> cardriveState->agents[i].pos.x
							>> cardriveState->agents[i].pos.y
							>> cardriveState->agents[i].speed;
				particles[PID] = cardriveState;
				PID++;

			}
		}
	}
}

bool ContextPomdp::ValidateState(const PomdpState &state, const char *msg) const {

	for (int i = 0; i < state.num; i++) {
		auto &agent = state.agents[i];

		if (agent.type >= AgentType::num_values) {
			ERR(string_sprintf("non-initialized type in state: %d", agent.type));
		}

		if (agent.speed == -1) {
			ERR("non-initialized speed in state");
		}
	}
}

OBS_TYPE ContextPomdp::StateToIndex(const State *state) const {
	std::hash<std::vector<int>> myhash;
	std::vector<int> obs_vec = ObserveVector(*state);
	OBS_TYPE obs = myhash(obs_vec);
	Obs_hash_table[obs] = obs_vec;

	if (obs <= (OBS_TYPE) 140737351976300) {
		cout << "empty obs: " << obs << endl;
		return obs;
	}

	return obs;
}

double ContextPomdp::GetAccelerationID(ACT_TYPE action, bool debug) const {
	return (action % int(ModelParams::NUM_ACC));
}

double ContextPomdp::GetAcceleration(ACT_TYPE action, bool debug) const {
	double acc_ID = (action % int(ModelParams::NUM_ACC));
	return GetAccfromAccID(acc_ID);
}

double ContextPomdp::GetAccelerationNoramlized(ACT_TYPE action, bool debug) const {
	double acc_ID = (action % int(ModelParams::NUM_ACC));
	return GetNormalizeAccfromAccID(acc_ID);
}

int ContextPomdp::GetLaneID(ACT_TYPE action, bool debug) {
	return FloorIntRobust(action / (ModelParams::NUM_ACC));
}

double ContextPomdp::GetLane(ACT_TYPE action, bool debug) {
	int lane_ID = FloorIntRobust(action / (ModelParams::NUM_ACC));
	int shifted_lane = lane_ID - 1;

	if (debug)
		cout << "[GetLane] (lane_ID, lane)=" << "(" << lane_ID << ","
				<< shifted_lane << ")" << endl;
	return shifted_lane;
}

ACT_TYPE ContextPomdp::GetActionID(double lane, double acc, bool debug) {
	if (debug) {
		cout << "[GetActionID] steer_ID=" << GetLaneIDfromLane(lane) << endl;
		cout << "[GetActionID] acc_ID=" << GetAccIDfromAcc(acc) << endl;
	}

	return (ACT_TYPE) (GetLaneIDfromLane(lane)
			* ClosestInt(ModelParams::NUM_ACC) + GetAccIDfromAcc(acc));
}

ACT_TYPE ContextPomdp::GetActionID(int lane_id, int acc_id) {
	return (ACT_TYPE) (lane_id * ClosestInt(ModelParams::NUM_ACC) + acc_id);
}

double ContextPomdp::GetAccfromAccID(int acc) {
	switch (acc) {
	case AccCode::MTN:
		return 0;
	case AccCode::ACC:
		return ModelParams::ACC_SPEED;
	case AccCode::DEC:
		return -ModelParams::ACC_SPEED;
	}
}

double ContextPomdp::GetNormalizeAccfromAccID(int acc) {
	switch (acc) {
	case AccCode::MTN:
		return 0;
	case AccCode::ACC:
		return 1.0;
	case AccCode::DEC:
		return -1.0;
	}
}

int ContextPomdp::GetAccIDfromAcc(float acc) {
	if (fabs(acc - 0) < 1e-5)
		return AccCode::MTN;
	else if (fabs(acc - ModelParams::ACC_SPEED) < 1e-5)
		return AccCode::ACC;
	else if (fabs(acc - (-ModelParams::ACC_SPEED)) < 1e-5)
		return AccCode::DEC;
}

LaneCode ContextPomdp::GetLaneIDfromLane(float lane) {
	switch (ClosestInt(lane)) {
	case -1:
		return LaneCode::LEFT;
	case 0:
		return LaneCode::KEEP;
	case 1:
		return LaneCode::RIGHT;
	};
}

void ContextPomdp::PrintStateIDs(const State& s) {
	const PomdpState& curr_state = static_cast<const PomdpState&>(s);

	cout << "Sampled peds: ";
	for (int i = 0; i < curr_state.num; i++)
		cout << curr_state.agents[i].id << " ";
	cout << endl;
}

void ContextPomdp::CheckPreCollision(const State* s) {
	const PomdpState* curr_state = static_cast<const PomdpState*>(s);

	int collision_peds_id;

	if (curr_state->car.vel > 0.001
			&& world_model.InCollision(*curr_state, collision_peds_id)) {
		cout << "--------------------------- pre-collision ----------------------------"
			<< endl;
		cout << "pre-col ped: " << collision_peds_id << endl;
	}
}

void ContextPomdp::SetPathID(State* state, despot::Shared_QNode* qnode) {
	PomdpState* pomdp_state = static_cast<PomdpState*>(state);
	pomdp_state->car.path_idx = qnode->path_id;
}
