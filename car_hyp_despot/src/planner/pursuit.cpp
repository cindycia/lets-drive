/*
 * pursuit.cpp
 *
 *  Created on: Jan 11, 2020
 *      Author: panpan
 */

#include "world_model.h"
#include "threaded_print.h"



double WorldModel::PurepursuitAngle(const CarStruct& car,
		COORD& pursuit_point) const {
	// tout << __FUNCTION__ << "_" << __LINE__ << endl;
	COORD rear_pos;
	rear_pos.x = car.pos.x - ModelParams::CAR_REAR * cos(car.heading_dir);
	rear_pos.y = car.pos.y - ModelParams::CAR_REAR * sin(car.heading_dir);

	double offset = (rear_pos - pursuit_point).Length();
	double target_angle = atan2(pursuit_point.y - rear_pos.y,
			pursuit_point.x - rear_pos.x);
	double angular_offset = CapAngle(target_angle - car.heading_dir);

	COORD relative_point(offset * cos(angular_offset),
			offset * sin(angular_offset));

	if (abs(relative_point.y) < 0.01)
		return 0;
	else {

		double turning_radius = relative_point.LengthSq()
				/ (2 * abs(relative_point.y)); // Intersecting chords theorem.
		if (abs(turning_radius) < 0.1)
			if (turning_radius > 0)
				return ModelParams::MAX_STEER_ANGLE;
			if (turning_radius < 0)
				return -ModelParams::MAX_STEER_ANGLE;

		double steering_angle = atan2(ModelParams::CAR_WHEEL_DIST,
				turning_radius);
		if (relative_point.y < 0)
			steering_angle *= -1;

		return max(min(steering_angle, ModelParams::MAX_STEER_ANGLE), -ModelParams::MAX_STEER_ANGLE);
	}

}

double WorldModel::PurepursuitAngle(const AgentStruct& agent,
		COORD& pursuit_point) const {
	// tout << __FUNCTION__ << "_" << __LINE__ << endl;
	COORD rear_pos;
	rear_pos.x = agent.pos.x
			- agent.bb_extent_y * 2 * 0.4 * cos(agent.heading_dir);
	rear_pos.y = agent.pos.y
			- agent.bb_extent_y * 2 * 0.4 * sin(agent.heading_dir);

	double offset = (rear_pos - pursuit_point).Length();
	double target_angle = atan2(pursuit_point.y - rear_pos.y,
			pursuit_point.x - rear_pos.x);
	double angular_offset = CapAngle(target_angle - agent.heading_dir);

	COORD relative_point(offset * cos(angular_offset),
			offset * sin(angular_offset));
	if (abs(relative_point.y) < 0.01)
		return 0;
	else {
		double turning_radius = relative_point.LengthSq()
				/ (2 * abs(relative_point.y)); // Intersecting chords theorem.
		if (abs(turning_radius) < 0.1)
			if (turning_radius > 0)
				return ModelParams::MAX_STEER_ANGLE;
			if (turning_radius < 0)
				return -ModelParams::MAX_STEER_ANGLE;

		double steering_angle = atan2(agent.bb_extent_y * 2 * 0.8,
				turning_radius);
		if (relative_point.y < 0)
			steering_angle *= -1;

		return max(min(steering_angle, ModelParams::MAX_STEER_ANGLE), -ModelParams::MAX_STEER_ANGLE);
	}
}
