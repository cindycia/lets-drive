/*
 * gamma_step.cpp
 *
 *  Created on: Jan 11, 2020
 *      Author: panpan
 */

#include "world_model.h"
#include "Vector2.h"
#include "threaded_print.h"

bool use_noise_in_rvo = false;
double GenerateGaussian(double rNum) {
	if (FIX_SCENARIO != 1 && !CPUDoPrint)
		rNum = QuickRandom::RandGeneration(rNum);
	double result = sqrt(-2 * log(rNum));
	if (FIX_SCENARIO != 1 && !CPUDoPrint)
		rNum = QuickRandom::RandGeneration(rNum);

	result *= cos(2 * M_PI * rNum);
	return result;
}

void WorldModel::GammaAgentStep(AgentStruct& agent, int intention_id) {
	int agent_id = agent.id;
	EnsureMeanDirExist(agent_id, intention_id);
	agent.pos = ped_mean_dirs[agent_id][intention_id] + agent.pos;
}

void WorldModel::GammaAgentStep(AgentStruct agents[], double& random,
		int num_agents, CarStruct car) {
	GammaSimulateAgents(agents, num_agents, car);

	for (int i = 0; i < num_agents; ++i) {
		auto& agent = agents[i];
		if (agent.mode == AGENT_ATT) {
			COORD rvo_vel = GetGammaVel(agent, i);
			AgentApplyGammaVel(agent, rvo_vel);
			if (use_noise_in_rvo) {
				double rNum = GenerateGaussian(random);
				agent.pos.x += rNum * ModelParams::NOISE_PED_POS / freq;
				rNum = GenerateGaussian(rNum);
				agent.pos.y += rNum * ModelParams::NOISE_PED_POS / freq;
			}
		}
	}
}


void WorldModel::AddEgoGammaAgent(int id_in_sim, const CarStruct& car) {
	int threadID = GetThreadID();

	double car_x, car_y, car_yaw;
	car_x = car.pos.x;
	car_y = car.pos.y;
	car_yaw = car.heading_dir;

	traffic_agent_sim_[threadID]->addAgent(default_car_, id_in_sim);
	traffic_agent_sim_[threadID]->setAgentPosition(id_in_sim,
			RVO::Vector2(car_x, car_y));
	RVO::Vector2 agt_heading(cos(car_yaw), sin(car_yaw));
	traffic_agent_sim_[threadID]->setAgentHeading(id_in_sim, agt_heading);
	traffic_agent_sim_[threadID]->setAgentVelocity(id_in_sim,
			car.vel * agt_heading);
	traffic_agent_sim_[threadID]->setAgentPrefVelocity(id_in_sim,
			car.vel * agt_heading); // assume that other agents do not know the ego-vehicle's intention and that they also don't infer the intention

	// set agent bounding box corners
	RVO::Vector2 sideward_vec = RVO::Vector2(-agt_heading.y(), agt_heading.x()); // rotate 90 degree counter-clockwise
	traffic_agent_sim_[threadID]->setAgentBoundingBoxCorners(id_in_sim,
			GetBoundingBoxCorners(agt_heading, sideward_vec,
					RVO::Vector2(car_x, car_y), ModelParams::CAR_FRONT, ModelParams::CAR_WIDTH/2.0));
}

void WorldModel::AddGammaAgent(const AgentStruct& agent, int id_in_sim) {
	int threadID = GetThreadID();

	double car_x, car_y, car_yaw, car_speed;
	car_x = agent.pos.x;
	car_y = agent.pos.y;
	car_yaw = agent.heading_dir;
	car_speed = agent.speed;

	if (agent.type == AgentType::car)
		traffic_agent_sim_[threadID]->addAgent(default_car_, id_in_sim);
	else if (agent.type == AgentType::ped)
		traffic_agent_sim_[threadID]->addAgent(default_ped_, id_in_sim);
	else
		traffic_agent_sim_[threadID]->addAgent(default_bike_, id_in_sim);

	traffic_agent_sim_[threadID]->setAgentPosition(id_in_sim,
			RVO::Vector2(car_x, car_y));
	RVO::Vector2 agt_heading(cos(car_yaw), sin(car_yaw));
	traffic_agent_sim_[threadID]->setAgentHeading(id_in_sim, agt_heading);
	traffic_agent_sim_[threadID]->setAgentVelocity(id_in_sim,
			car_speed * agt_heading);
	// assume that other agents do not know the vehicle's intention
	// and that they also don't infer the intention
	traffic_agent_sim_[threadID]->setAgentPrefVelocity(id_in_sim,
			car_speed * agt_heading);
	// rotate 90 degree counter-clockwise
	RVO::Vector2 sideward_vec = RVO::Vector2(-agt_heading.y(), agt_heading.x());
	double bb_x = agent.bb_extent_x;
	double bb_y = agent.bb_extent_y;
	assert(bb_x > 0);
	assert(bb_y > 0);

	traffic_agent_sim_[threadID]->setAgentBoundingBoxCorners(id_in_sim,
			GetBoundingBoxCorners(agt_heading, sideward_vec,
					RVO::Vector2(car_x, car_y), bb_y, bb_x));
}

void WorldModel::GammaSimulateAgents(AgentStruct agents[], int num_agents,
		CarStruct& car) {
	// tout << __FUNCTION__ << "_" << __LINE__ << endl;

	int threadID = GetThreadID();

	// Construct a new set of agents every time
	traffic_agent_sim_[threadID]->clearAllAgents();

	// adding pedestrians
	for (int i = 0; i < num_agents; i++) {
		bool frozen_agent = (agents[i].mode == AGENT_DIS);
		if (agents[i].type == AgentType::car) {
			traffic_agent_sim_[threadID]->addAgent(default_car_, i,
					frozen_agent);
		} else if (agents[i].type == AgentType::ped) {
			traffic_agent_sim_[threadID]->addAgent(default_ped_, i,
					frozen_agent);
		} else {
			traffic_agent_sim_[threadID]->addAgent(default_bike_, i,
					frozen_agent);
		}

		traffic_agent_sim_[threadID]->setAgentPosition(i,
				RVO::Vector2(agents[i].pos.x, agents[i].pos.y));
		RVO::Vector2 agt_heading(cos(agents[i].heading_dir),
				sin(agents[i].heading_dir));
		traffic_agent_sim_[threadID]->setAgentHeading(i, agt_heading);
		traffic_agent_sim_[threadID]->setAgentVelocity(i,
				RVO::Vector2(agents[i].vel.x, agents[i].vel.y));

		int intention_id = agents[i].intention;
		ValidateIntention(agents[i].id, intention_id, __FUNCTION__, __LINE__);

		auto goal_pos = GetGoalPos(agents[i], intention_id);
		RVO::Vector2 goal(goal_pos.x, goal_pos.y);
		if (RVO::abs(goal - traffic_agent_sim_[threadID]->getAgentPosition(i)) < 0.5) {
			// Agent is within 0.5 meter of its goal, set preferred velocity to zero
			traffic_agent_sim_[threadID]->setAgentPrefVelocity(i,
					RVO::Vector2(0.0f, 0.0f));
		} else {
			double pref_speed = 0.0;
			pref_speed = agents[i].speed;
			traffic_agent_sim_[threadID]->setAgentPrefVelocity(i,
					normalize(goal - traffic_agent_sim_[threadID]->getAgentPosition(i)) * pref_speed);
		}

		traffic_agent_sim_[threadID]->setAgentBoundingBoxCorners(i,
				GetBoundingBoxCorners(agents[i]));
	}

	// adding car as a "special" pedestrian
	AddEgoGammaAgent(num_agents, car);

	traffic_agent_sim_[threadID]->doStep();
}

COORD WorldModel::GetGammaVel(AgentStruct& agent, int i) {
	// tout << __FUNCTION__ << "_" << __LINE__ << endl;

	int threadID = GetThreadID();
	assert(agent.mode == AGENT_ATT);

	COORD new_pos;
	new_pos.x = traffic_agent_sim_[threadID]->getAgentPosition(i).x(); // + random.NextGaussian() * (traffic_agent_sim_[threadID]->getAgentPosition(agent).x() - agents[i].pos.x)/5.0; //random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
	new_pos.y = traffic_agent_sim_[threadID]->getAgentPosition(i).y(); // + random.NextGaussian() * (traffic_agent_sim_[threadID]->getAgentPosition(agent).y() - agents[i].pos.y)/5.0;//random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;

	return (new_pos - agent.pos) * freq;
}

void WorldModel::AgentApplyGammaVel(AgentStruct& agent, COORD& rvo_vel) {
	// tout << __FUNCTION__ << "_" << __LINE__ << endl;

	COORD old_pos = agent.pos;
	double rvo_speed = rvo_vel.Length();
	if (agent.type == AgentType::car) {
		rvo_vel.AdjustLength(PURSUIT_LEN);
		COORD pursuit_point = agent.pos + rvo_vel;
		double steering = PControlAngle<AgentStruct>(agent, pursuit_point);
		BicycleModel(agent, steering, rvo_speed);
	} else if (agent.type == AgentType::ped) {
		agent.pos = agent.pos + rvo_vel * (1.0 / freq);
	}

	if (!IsStopIntention(agent.intention, agent.id)
			&& !IsCurVelIntention(agent.intention, agent.id)) {
		auto& path = PathCandidates(agent.id)[agent.intention];
		agent.pos_along_path = path.Nearest(agent.pos);
	}
	agent.vel = (agent.pos - old_pos) * freq;
	agent.speed = agent.vel.Length();
}


