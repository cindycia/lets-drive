/*
 * ego_step.cpp
 *
 *  Created on: Jan 11, 2020
 *      Author: panpan
 */

#include "world_model.h"
#include "threaded_print.h"


void WorldModel::RobStep(CarStruct &car, double lane, Random& random) {
	// tout << __FUNCTION__ << "_" << __LINE__ << endl;
	double steering = GetSteerFromLane(car, lane);
	// tout << __FUNCTION__ << "_" << __LINE__ << endl;
	BicycleModel(car, steering, car.vel);
}

void WorldModel::RobStep(CarStruct &car, double lane, double& random) {
	// tout << __FUNCTION__ << "_" << __LINE__ << endl;
	double steering = GetSteerFromLane(car, lane);
	// tout << __FUNCTION__ << "_" << __LINE__ << endl;
	BicycleModel(car, steering, car.vel);
}

void WorldModel::RobStep(CarStruct &car, double& random, double acc,
		double lane) {
	// tout << __FUNCTION__ << "_" << __LINE__ << endl;
	double end_vel = car.vel + acc / freq;
	end_vel = max(min(end_vel, ModelParams::VEL_MAX), 0.0);
	double steering = GetSteerFromLane(car, lane);
	BicycleModel(car, steering, end_vel);
}

void WorldModel::RobStep(CarStruct &car, Random& random, double acc,
		double lane) {
	// tout << __FUNCTION__ << "_" << __LINE__ << endl;
	double end_vel = car.vel + acc / freq;
	end_vel = max(min(end_vel, ModelParams::VEL_MAX), 0.0);
	double steering = GetSteerFromLane(car, lane);
	BicycleModel(car, steering, end_vel);
}

void WorldModel::RobVelStep(CarStruct &car, double acc, Random& random) {
	const double N = ModelParams::NOISE_ROBVEL;
	// tout << __FUNCTION__ << "_" << __LINE__ << endl;
	if (N > 0) {
		double prob = random.NextDouble();
		if (prob > N) {
			car.vel += acc / freq;
		}
	} else {
		car.vel += acc / freq;
	}
	car.vel = max(min(car.vel, ModelParams::VEL_MAX), 0.0);
	return;
}

void WorldModel::RobVelStep(CarStruct &car, double acc, double& random) {
	// tout << __FUNCTION__ << "_" << __LINE__ << endl;
	const double N = ModelParams::NOISE_ROBVEL;
	if (N > 0) {
		if (FIX_SCENARIO != 1 && !CPUDoPrint)
			random = QuickRandom::RandGeneration(random);
		double prob = random;
		if (prob > N) {
			car.vel += acc / freq;
		}
	} else {
		car.vel += acc / freq;
	}
	car.vel = max(min(car.vel, ModelParams::VEL_MAX), 0.0);
	return;
}

void WorldModel::RobStepCurVel(CarStruct &car) {
	// tout << __FUNCTION__ << "_" << __LINE__ << endl;
	car.pos.x += (car.vel / freq) * cos(car.heading_dir);
	car.pos.y += (car.vel / freq) * sin(car.heading_dir);
}

void WorldModel::RobStepCurAction(CarStruct &car, double acc, double lane) {
	// tout << __FUNCTION__ << "_" << __LINE__ << endl;
	double det_prob = 1;
	RobStep(car, lane, det_prob);
	RobVelStep(car, acc, det_prob);
}


void WorldModel::BicycleModel(CarStruct &car, double steering, double end_vel) {
	// tout << __FUNCTION__ << "_" << __LINE__ << endl;
	if (abs(steering) > 1e-5) {
		double TurningRadius = ModelParams::CAR_WHEEL_DIST / tan(steering);
		assert(TurningRadius != 0);
		double beta = end_vel / freq / TurningRadius;

		COORD rear_pos;
		rear_pos.x = car.pos.x - ModelParams::CAR_REAR * cos(car.heading_dir);
		rear_pos.y = car.pos.y - ModelParams::CAR_REAR * sin(car.heading_dir);
		// move and rotate
		rear_pos.x += TurningRadius
				* (sin(car.heading_dir + beta) - sin(car.heading_dir));
		rear_pos.y += TurningRadius
				* (cos(car.heading_dir) - cos(car.heading_dir + beta));
		car.heading_dir = CapAngle(car.heading_dir + beta);
		car.pos.x = rear_pos.x + ModelParams::CAR_REAR * cos(car.heading_dir);
		car.pos.y = rear_pos.y + ModelParams::CAR_REAR * sin(car.heading_dir);
	} else {
		car.pos.x += (end_vel / freq) * cos(car.heading_dir);
		car.pos.y += (end_vel / freq) * sin(car.heading_dir);
	}
}

void WorldModel::BicycleModel(AgentStruct &agent, double steering,
		double end_vel) {
	// tout << __FUNCTION__ << "_" << __LINE__ << endl;
	if (abs(steering) > 1e-5) {
		assert(tan(steering) != 0);
		// assuming front-real length is 0.8 * total car length
		double TurningRadius = agent.bb_extent_y * 2 * 0.8 / tan(steering);
		assert(TurningRadius != 0);
		double beta = end_vel / freq / TurningRadius;

		COORD rear_pos;
		rear_pos.x = agent.pos.x
				- agent.bb_extent_y * 2 * 0.4 * cos(agent.heading_dir);
		rear_pos.y = agent.pos.y
				- agent.bb_extent_y * 2 * 0.4 * sin(agent.heading_dir);
		// move and rotate
		rear_pos.x += TurningRadius
				* (sin(agent.heading_dir + beta) - sin(agent.heading_dir));
		rear_pos.y += TurningRadius
				* (cos(agent.heading_dir) - cos(agent.heading_dir + beta));
		agent.heading_dir = CapAngle(agent.heading_dir + beta);
		agent.pos.x = rear_pos.x
				+ agent.bb_extent_y * 2 * 0.4 * cos(agent.heading_dir);
		agent.pos.y = rear_pos.y
				+ agent.bb_extent_y * 2 * 0.4 * sin(agent.heading_dir);
	} else {
		agent.pos.x += (end_vel / freq) * cos(agent.heading_dir);
		agent.pos.y += (end_vel / freq) * sin(agent.heading_dir);
	}
}


