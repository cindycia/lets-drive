/*
 * Agent.cpp
 * RVO2 Library
 *
 * Copyright 2008 University of North Carolina at Chapel Hill
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please send all bug reports to <geom@cs.unc.edu>.
 *
 * The authors may be contacted via:
 *
 * Jur van den Berg, Stephen J. Guy, Jamie Snape, Ming C. Lin, Dinesh Manocha
 * Dept. of Computer Science
 * 201 S. Columbia St.
 * Frederick P. Brooks, Jr. Computer Science Bldg.
 * Chapel Hill, N.C. 27599-3175
 * United States of America
 *
 * <http://gamma.cs.unc.edu/RVO2/>
 */


#include "Agent.h"

#include "KdTree.h"
#include "Obstacle.h"

#include <iostream>

#define USE_OLD_ORCA false
#define DENSE_AWARE false
#define COMPUTE_VEL_FOR_VEH true
#define ONE_VEHICLE true
#define USE_PATIENCE false

#define STRIDE_FACTOR 1.57f
#define STRIDE_BUFFER 0.5f

#define TIME_STEP_TAU 2.5f
#define PI 3.1415926
#define THETA_TOTAL 90.f
#define THETA_INTERVAL 15.f

namespace RVO {
	Agent::Agent(RVOSimulator *sim) : maxNeighbors_(0), maxSpeed_(0.0f), neighborDist_(0.0f), radius_(0.0f), sim_(sim), timeHorizon_(0.0f), timeHorizonObst_(0.0f), id_(0) { 
		use_new_pref_vel_ = false;
		change_dir_iter_ = 0;
		patience_ = 1.0;

		updated_ = true;

		setStrideParameters( STRIDE_FACTOR, STRIDE_BUFFER );
	}

	void Agent::computeNeighbors()
	{
		obstacleNeighbors_.clear();
		float rangeSq = sqr(timeHorizonObst_ * maxSpeed_ + radius_);
		sim_->kdTree_->computeObstacleNeighbors(this, rangeSq);

		agentNeighbors_.clear();

		if (maxNeighbors_ > 0) {
			rangeSq = sqr(neighborDist_);
			sim_->kdTree_->computeAgentNeighbors(this, rangeSq);
		}
	}

	size_t Agent::computeObstORCALines(){
		orcaLines_.clear();

		const float invTimeHorizonObst = 1.0f / timeHorizonObst_;

		/* Create obstacle ORCA lines. */
		for (size_t i = 0; i < obstacleNeighbors_.size(); ++i) {

			const Obstacle *obstacle1 = obstacleNeighbors_[i].second;
			const Obstacle *obstacle2 = obstacle1->nextObstacle_;

			const Vector2 relativePosition1 = obstacle1->point_ - position_;
			const Vector2 relativePosition2 = obstacle2->point_ - position_;

			/*
			 * Check if velocity obstacle of obstacle is already taken care of by
			 * previously constructed obstacle ORCA lines.
			 */
			bool alreadyCovered = false;

			for (size_t j = 0; j < orcaLines_.size(); ++j) {
				if (det(invTimeHorizonObst * relativePosition1 - orcaLines_[j].point, orcaLines_[j].direction) - invTimeHorizonObst * radius_ >= -RVO_EPSILON && det(invTimeHorizonObst * relativePosition2 - orcaLines_[j].point, orcaLines_[j].direction) - invTimeHorizonObst * radius_ >=  -RVO_EPSILON) {
					alreadyCovered = true;
					break;
				}
			}

			if (alreadyCovered) {
				continue;
			}

			/* Not yet covered. Check for collisions. */

			const float distSq1 = absSq(relativePosition1);
			const float distSq2 = absSq(relativePosition2);

			const float radiusSq = sqr(radius_);

			const Vector2 obstacleVector = obstacle2->point_ - obstacle1->point_;
			const float s = (-relativePosition1 * obstacleVector) / absSq(obstacleVector);
			const float distSqLine = absSq(-relativePosition1 - s * obstacleVector);

			Line line;

			if (s < 0.0f && distSq1 <= radiusSq) {
				/* Collision with left vertex. Ignore if non-convex. */
				if (obstacle1->isConvex_) {
					line.point = Vector2(0.0f, 0.0f);
					line.direction = normalize(Vector2(-relativePosition1.y(), relativePosition1.x()));
					orcaLines_.push_back(line);
				}

				continue;
			}
			else if (s > 1.0f && distSq2 <= radiusSq) {
				/* Collision with right vertex. Ignore if non-convex
				 * or if it will be taken care of by neighoring obstace */
				if (obstacle2->isConvex_ && det(relativePosition2, obstacle2->unitDir_) >= 0.0f) {
					line.point = Vector2(0.0f, 0.0f);
					line.direction = normalize(Vector2(-relativePosition2.y(), relativePosition2.x()));
					orcaLines_.push_back(line);
				}

				continue;
			}
			else if (s >= 0.0f && s < 1.0f && distSqLine <= radiusSq) {
				/* Collision with obstacle segment. */
				line.point = Vector2(0.0f, 0.0f);
				line.direction = -obstacle1->unitDir_;
				orcaLines_.push_back(line);
				continue;
			}

			/*
			 * No collision.
			 * Compute legs. When obliquely viewed, both legs can come from a single
			 * vertex. Legs extend cut-off line when nonconvex vertex.
			 */

			Vector2 leftLegDirection, rightLegDirection;

			if (s < 0.0f && distSqLine <= radiusSq) {
				/*
				 * Obstacle viewed obliquely so that left vertex
				 * defines velocity obstacle.
				 */
				if (!obstacle1->isConvex_) {
					/* Ignore obstacle. */
					continue;
				}

				obstacle2 = obstacle1;

				const float leg1 = std::sqrt(distSq1 - radiusSq);
				leftLegDirection = Vector2(relativePosition1.x() * leg1 - relativePosition1.y() * radius_, relativePosition1.x() * radius_ + relativePosition1.y() * leg1) / distSq1;
				rightLegDirection = Vector2(relativePosition1.x() * leg1 + relativePosition1.y() * radius_, -relativePosition1.x() * radius_ + relativePosition1.y() * leg1) / distSq1;
			}
			else if (s > 1.0f && distSqLine <= radiusSq) {
				/*
				 * Obstacle viewed obliquely so that
				 * right vertex defines velocity obstacle.
				 */
				if (!obstacle2->isConvex_) {
					/* Ignore obstacle. */
					continue;
				}

				obstacle1 = obstacle2;

				const float leg2 = std::sqrt(distSq2 - radiusSq);
				leftLegDirection = Vector2(relativePosition2.x() * leg2 - relativePosition2.y() * radius_, relativePosition2.x() * radius_ + relativePosition2.y() * leg2) / distSq2;
				rightLegDirection = Vector2(relativePosition2.x() * leg2 + relativePosition2.y() * radius_, -relativePosition2.x() * radius_ + relativePosition2.y() * leg2) / distSq2;
			}
			else {
				/* Usual situation. */
				if (obstacle1->isConvex_) {
					const float leg1 = std::sqrt(distSq1 - radiusSq);
					leftLegDirection = Vector2(relativePosition1.x() * leg1 - relativePosition1.y() * radius_, relativePosition1.x() * radius_ + relativePosition1.y() * leg1) / distSq1;
				}
				else {
					/* Left vertex non-convex; left leg extends cut-off line. */
					leftLegDirection = -obstacle1->unitDir_;
				}

				if (obstacle2->isConvex_) {
					const float leg2 = std::sqrt(distSq2 - radiusSq);
					rightLegDirection = Vector2(relativePosition2.x() * leg2 + relativePosition2.y() * radius_, -relativePosition2.x() * radius_ + relativePosition2.y() * leg2) / distSq2;
				}
				else {
					/* Right vertex non-convex; right leg extends cut-off line. */
					rightLegDirection = obstacle1->unitDir_;
				}
			}

			/*
			 * Legs can never point into neighboring edge when convex vertex,
			 * take cutoff-line of neighboring edge instead. If velocity projected on
			 * "foreign" leg, no constraint is added.
			 */

			const Obstacle *const leftNeighbor = obstacle1->prevObstacle_;

			bool isLeftLegForeign = false;
			bool isRightLegForeign = false;

			if (obstacle1->isConvex_ && det(leftLegDirection, -leftNeighbor->unitDir_) >= 0.0f) {
				/* Left leg points into obstacle. */
				leftLegDirection = -leftNeighbor->unitDir_;
				isLeftLegForeign = true;
			}

			if (obstacle2->isConvex_ && det(rightLegDirection, obstacle2->unitDir_) <= 0.0f) {
				/* Right leg points into obstacle. */
				rightLegDirection = obstacle2->unitDir_;
				isRightLegForeign = true;
			}

			/* Compute cut-off centers. */
			const Vector2 leftCutoff = invTimeHorizonObst * (obstacle1->point_ - position_);
			const Vector2 rightCutoff = invTimeHorizonObst * (obstacle2->point_ - position_);
			const Vector2 cutoffVec = rightCutoff - leftCutoff;

			/* Project current velocity on velocity obstacle. */

			/* Check if current velocity is projected on cutoff circles. */
			const float t = (obstacle1 == obstacle2 ? 0.5f : ((velocity_ - leftCutoff) * cutoffVec) / absSq(cutoffVec));
			const float tLeft = ((velocity_ - leftCutoff) * leftLegDirection);
			const float tRight = ((velocity_ - rightCutoff) * rightLegDirection);

			if ((t < 0.0f && tLeft < 0.0f) || (obstacle1 == obstacle2 && tLeft < 0.0f && tRight < 0.0f)) {
				/* Project on left cut-off circle. */
				const Vector2 unitW = normalize(velocity_ - leftCutoff);

				line.direction = Vector2(unitW.y(), -unitW.x());
				line.point = leftCutoff + radius_ * invTimeHorizonObst * unitW;
				orcaLines_.push_back(line);
				continue;
			}
			else if (t > 1.0f && tRight < 0.0f) {
				/* Project on right cut-off circle. */
				const Vector2 unitW = normalize(velocity_ - rightCutoff);

				line.direction = Vector2(unitW.y(), -unitW.x());
				line.point = rightCutoff + radius_ * invTimeHorizonObst * unitW;
				orcaLines_.push_back(line);
				continue;
			}

			/*
			 * Project on left leg, right leg, or cut-off line, whichever is closest
			 * to velocity.
			 */
			const float distSqCutoff = ((t < 0.0f || t > 1.0f || obstacle1 == obstacle2) ? std::numeric_limits<float>::infinity() : absSq(velocity_ - (leftCutoff + t * cutoffVec)));
			const float distSqLeft = ((tLeft < 0.0f) ? std::numeric_limits<float>::infinity() : absSq(velocity_ - (leftCutoff + tLeft * leftLegDirection)));
			const float distSqRight = ((tRight < 0.0f) ? std::numeric_limits<float>::infinity() : absSq(velocity_ - (rightCutoff + tRight * rightLegDirection)));

			if (distSqCutoff <= distSqLeft && distSqCutoff <= distSqRight) {
				/* Project on cut-off line. */
				line.direction = -obstacle1->unitDir_;
				line.point = leftCutoff + radius_ * invTimeHorizonObst * Vector2(-line.direction.y(), line.direction.x());
				orcaLines_.push_back(line);
				continue;
			}
			else if (distSqLeft <= distSqRight) {
				/* Project on left leg. */
				if (isLeftLegForeign) {
					continue;
				}

				line.direction = leftLegDirection;
				line.point = leftCutoff + radius_ * invTimeHorizonObst * Vector2(-line.direction.y(), line.direction.x());
				orcaLines_.push_back(line);
				continue;
			}
			else {
				/* Project on right leg. */
				if (isRightLegForeign) {
					continue;
				}

				line.direction = -rightLegDirection;
				line.point = rightCutoff + radius_ * invTimeHorizonObst * Vector2(-line.direction.y(), line.direction.x());
				orcaLines_.push_back(line);
				continue;
			}
		}

		return orcaLines_.size();

	}

	void Agent::computeAgentORCALines(){

		//const float invTimeHorizon = 1.0f / timeHorizon_;
		float invTimeHorizon = 1.0f / timeHorizon_;

		veh_in_neighbor_ = false;

		/* Create agent ORCA lines. */
		for (size_t i = 0; i < agentNeighbors_.size(); ++i) {
			const Agent *const other = agentNeighbors_[i].second;
			if(other->tag_ == "vehicle") {
				if(tag_ == "vehicle" && ONE_VEHICLE) continue; // there is only one vehicle; the other 'vehicle' is just part of the vehicle
				if(USE_OLD_ORCA){
					invTimeHorizon = 1.0f/(timeHorizon_);
				} else{
					invTimeHorizon = 1.0f/(timeHorizon_ *10);
				}
				
				veh_in_neighbor_ = true;
				veh_vel_avoiding_ = other->prefVelocity_;
				veh_pos_ = other->position_;
			}

			const Vector2 relativePosition = other->position_ - position_;
			const Vector2 relativeVelocity = velocity_ - other->velocity_;
			const float distSq = absSq(relativePosition);
			const float combinedRadius = radius_ + other->radius_;
			const float combinedRadiusSq = sqr(combinedRadius);

			Line line;
			Vector2 u;

			if (distSq > combinedRadiusSq) {
				/* No collision. */
				const Vector2 w = relativeVelocity - invTimeHorizon * relativePosition;
				/* Vector from cutoff center to relative velocity. */
				const float wLengthSq = absSq(w);

				const float dotProduct1 = w * relativePosition;

				/// sqr(dotProduct1) > combinedRadiusSq * wLengthSq ==> |relativePosition| * cos(alpha) > |combinedRadius| ==> |invTimeHorizon*relativePosition| * cos(alpha) > |invTimeHorizon*combinedRadius|
				/// note that invTimeHorizon*relativePosition is the vector from origin to the position (PA-PB)*invTimeHorizon (i.e. (PA-PB)/tau, i.e. the center of the smaller circle), 
				/// and invTimeHorizon*combinedRadius is the radius of the smaller circle. Draw the figure and it is easy to find out that the second condition guarantees that the shortest distance point is on the circle edge instead fof the leg
				if (dotProduct1 < 0.0f && sqr(dotProduct1) > combinedRadiusSq * wLengthSq) { 
					/* Project on cut-off circle. */
					const float wLength = std::sqrt(wLengthSq);
					const Vector2 unitW = w / wLength;

					/// direction is the vector that unityW rotate 90 in clockwise. Hence its left side is the feasible region of this line
					line.direction = Vector2(unitW.y(), -unitW.x());
					u = (combinedRadius * invTimeHorizon - wLength) * unitW;
				}
				else {
					/* Project on legs. */
					const float leg = std::sqrt(distSq - combinedRadiusSq);

					if (det(relativePosition, w) > 0.0f) {
						/* Project on left leg. */
						line.direction = Vector2(relativePosition.x() * leg - relativePosition.y() * combinedRadius, relativePosition.x() * combinedRadius + relativePosition.y() * leg) / distSq;
					}
					else {
						/* Project on right leg. */
						line.direction = -Vector2(relativePosition.x() * leg + relativePosition.y() * combinedRadius, -relativePosition.x() * combinedRadius + relativePosition.y() * leg) / distSq;
					}

					const float dotProduct2 = relativeVelocity * line.direction;

					u = dotProduct2 * line.direction - relativeVelocity;
				}
			}
			else {
				/* Collision. Project on cut-off circle of time timeStep. */
				const float invTimeStep = 1.0f / sim_->timeStep_;

				/* Vector from cutoff center to relative velocity. */
				const Vector2 w = relativeVelocity - invTimeStep * relativePosition;

				const float wLength = abs(w);
				const Vector2 unitW = w / wLength;

				line.direction = Vector2(unitW.y(), -unitW.x());
				u = (combinedRadius * invTimeStep - wLength) * unitW;
			}

			if(!USE_OLD_ORCA){
				if(tag_ == "vehicle"){
					boundTrackedVelocity(max_tracking_angle_);
					if(other->tag_ == "vehicle"){
						line.point = velocity_ + 0.5f * u;
					}
					else{
						float res_changing_dist = 10.0;
						float ped_responsibility;
						float dist_to_collision = std::sqrt(distSq) - combinedRadius;
						if(dist_to_collision < 0) {//collision.
							ped_responsibility = 0.95;
						}
						else if(dist_to_collision > res_changing_dist){
							ped_responsibility = 0.05;
						}
						else{
							ped_responsibility = 0.05 + (res_changing_dist - dist_to_collision)/res_changing_dist * 0.9;
						}
						
						line.point = velocity_ + (1.0f - ped_responsibility) * u;
					}
				}
				else{
					if(other->tag_ == "vehicle"){
						float res_changing_dist = 10.0;
						float ped_responsibility;
						float dist_to_collision = std::sqrt(distSq) - combinedRadius;
						if(dist_to_collision < 0) {//collision.
							ped_responsibility = 0.95;
						}
						else if(dist_to_collision > res_changing_dist){
							ped_responsibility = 0.05;
						}
						else{
							ped_responsibility = 0.05 + (res_changing_dist - dist_to_collision)/res_changing_dist * 0.9;
						}
						
						line.point = velocity_ + ped_responsibility * u;
					}
					else{
						line.point = velocity_ + 0.5f * u;
					}
				}
			} else{
				line.point = velocity_ + 0.5f * u;
			}

			orcaLines_.push_back(line);
		}
	}

	void Agent::adaptPatience(){
		if(!USE_OLD_ORCA){
			if((abs(prefVelocity_) <= 0.05 || abs(velocity_) >= 0.35 * abs(prefVelocity_))) {
				patience_ = 1.0; //reset agent's patience to 1
			}
			else{
				patience_ *= 0.9; //0.5;
			}

			if(abs(velocity_)<0.11){
				//std::cout<<"agent: "<<id_<<" p="<<patience_<<" v="<<abs(velocity_)<<" pv="<<abs(prefVelocity_)<<std::endl;
			}
			if(patience_ > 0.3){
				Line line1;
				line1.point = Vector2(0.0f, 0.0f);
				//line1.direction = normalize(velocity_.rotate(-90.0));
				line1.direction = normalize(prefVelocity_.rotate(-90.0));
				orcaLines_.push_back(line1);
			}
		}
	}

	void Agent::boundTrackedVelocity(float _max_tracking_bound){
		if(!USE_OLD_ORCA){
			if(COMPUTE_VEL_FOR_VEH){
			//if(false){
				if(tag_ != "velocity") {
					return;
				}
				else {
					Line line1;
					line1.point = Vector2(0.0f, 0.0f);
					// the feasible space is on the left side of a vector
					line1.direction = normalize(heading_.rotate(-_max_tracking_bound)); // rotate clockwise by |_max_tracking_bound| angle
					orcaLines_.push_back(line1);
					line1.direction = normalize(heading_.rotate(-(180.0-_max_tracking_bound)));
					orcaLines_.push_back(line1);
				}
			}
		}
	}

	float Agent::getSpeed(Vector2 prefDir, float prefSpeed){
		
		float availSpace = 1e6f;	

		// Not the speed-dependent stride length, but rather the mid-point of the
		float strideLen = 1.f;	
		// elliptical personal space.
		Vector2 critPt = position_ + strideLen * prefDir;
		float density = 0.f;
		// For now, assume some constants
		const float area = 1.5f;
		const float areaSq2Inv = 1.f / ( 2 * area * area );
		const float sqrt2Pi = sqrtf( 6.283182 ); // 6.283185 = 2*Pi
		const float norm = 1.f / ( area * sqrt2Pi );

		// AGENTS
		for ( size_t i = 0; i < agentNeighbors_.size(); ++i ) {
			const Agent* const other = agentNeighbors_[i].second;
			Vector2 critDisp = other->position_ - critPt;
			// dot project gets projection, in the preferred direction
			Vector2 yComp = ( critDisp * prefDir ) * prefDir;
			// penalize displacement perpindicular to the preferred direction
			Vector2 xComp = ( critDisp - yComp ) * 2.5f;		
			critDisp.set( xComp + yComp );
			float distSq = absSq( critDisp );
			density += norm * expf( -distSq * areaSq2Inv );	
		}
		//// OBSTACLES
		const float OBST_AREA = 0.75f;
		const float OBST_AREA_SQ_INV = 1.f / ( 2 * OBST_AREA * OBST_AREA );
		const float OBST_SCALE = norm;// * 6.25f;	// what is the "density" of an obstacle?
		for ( size_t i = 0; i < obstacleNeighbors_.size(); ++i ) {
			const Obstacle* const obst = obstacleNeighbors_[i].second;
			Vector2 nearPt;
			float distSq;	// set by distanceSqToPoint
			if ( obst->distanceSqToPoint( critPt, nearPt, distSq ) == 
				 RVO::Obstacle::LAST ) continue;

			if ( ( nearPt - position_ ) * prefDir < 0.f ) continue; //// the obstacle is behind
			density += OBST_SCALE * expf( -distSq * OBST_AREA_SQ_INV );
		}

		const float	AGENT_WIDTH = 0.48f;
		if ( density < 0.001f ) {
			availSpace = 100.f;
		} else {
			availSpace = AGENT_WIDTH / density ;
		}

		// Compute the maximum speed I could take for the available space
		/*float maxSpeed = _speedConst * availSpace * availSpace;
		if ( maxSpeed < prefSpeed ) prefVelocity_.set( prefDir * maxSpeed );*/

		float maxSpeed = _speedConst * availSpace * availSpace;
		if ( maxSpeed < prefSpeed ) return maxSpeed;
		else return prefSpeed;
	}

	Vector2 Agent::getTarget(){
		return position_ + prefVelocity_ * 5.f;
	}

	float Agent::costToGoal(Vector2 velocity, Vector2 goal){
		return abs((position_ + (velocity * TIME_STEP_TAU) ) - goal);
	}

	void Agent::adaptPreferredVelocity() {
		if ( DENSE_AWARE ) {
			
			float prefSpeed = abs(prefVelocity_);
			if(prefSpeed < 0.01) return;

			Vector2 goal = getTarget();

			Vector2 prefDir( normalize(prefVelocity_) );
			float maxFDSpeed = getSpeed(prefDir, prefSpeed);
			Vector2 bestVelocity = prefDir * maxFDSpeed;

			float bestCost = costToGoal(bestVelocity, goal);

			for (float angle = THETA_INTERVAL; angle < THETA_TOTAL; angle += THETA_INTERVAL){
				Vector2 currDir = prefDir.rotate(angle);
				float currFDSpeed = getSpeed(currDir, prefSpeed);
				Vector2 currVel = currDir * currFDSpeed;
				float currCost = costToGoal(currVel, goal);
				if(currCost < bestCost){
					bestCost = currCost;
					bestVelocity = currVel;
				}

				currDir = prefDir.rotate(-angle);
				currFDSpeed = getSpeed(currDir, prefSpeed);
				currVel = currDir * currFDSpeed;
				currCost = costToGoal(currVel, goal);
				if(currCost < bestCost){
					bestCost = currCost;
					bestVelocity = currVel;
				}
			}

			prefVelocity_ = bestVelocity;		
		}
	}

	void Agent::setStrideParameters( float factor, float buffer ) {
				_strideConst = 0.5f * ( 1.f + buffer ) / factor ;
				_speedConst = 1.f / ( _strideConst * _strideConst );
	}

	/* Search for the best new velocity. */
	void Agent::computeNewVelocity()
	{
		if(!COMPUTE_VEL_FOR_VEH){
			if(tag_ == "vehicle") {
				newVelocity_ = prefVelocity_;
				return;
			}
		}


		adaptPreferredVelocity();

		const size_t numObstLines = computeObstORCALines();

		if(USE_PATIENCE) adaptPatience();

		computeAgentORCALines();
				
		if(!USE_OLD_ORCA){
			if(veh_in_neighbor_ && abs(prefVelocity_) <= 0.1){
				float dist_to_veh = abs(position_-veh_pos_);
				float no_collision_dist = abs(veh_vel_avoiding_) * 2.0f + 0.8 + 0.3; // no collision within 4 seconds, 0.3 second is for delay; 0.8 is the distance from car center to car front; 0.3 is safety margin
				if(dist_to_veh < no_collision_dist){
					if(distPointLine(Vector2(0.0f, 0.0f), veh_vel_avoiding_, position_-veh_pos_) < /*0.95*/1.05f) {// the distance of the ped to the center line of the vehicle
						
						if(leftOf(Vector2(0.0f, 0.0f), veh_vel_avoiding_, position_-veh_pos_)>0){ // agent at the left side of the vehicle; rotate counter-colckwise
							prefVelocity_ = (normalize(veh_vel_avoiding_) * 1.5).rotate(90.0);
						} else{
							prefVelocity_ = (normalize(veh_vel_avoiding_) * 1.5).rotate(-90.0);
						}
					}
				}
			}

			size_t lineFail = linearProgram2(orcaLines_, maxSpeed_, prefVelocity_, false, newVelocity_, patience_);

			if (lineFail < orcaLines_.size()) {
				linearProgram3(orcaLines_, numObstLines, lineFail, maxSpeed_, newVelocity_);
				static int num_of_line_fail = 0;
				if(tag_=="vehicle"){
					num_of_line_fail ++;
					std::cout<<"fail: "<<num_of_line_fail<<std::endl;
				}
				
			}
		} else{
			size_t lineFail = linearProgram2(orcaLines_, maxSpeed_, prefVelocity_, false, newVelocity_);

			if (lineFail < orcaLines_.size()) {
				linearProgram3(orcaLines_, numObstLines, lineFail, maxSpeed_, newVelocity_);
			}
		}
	}

	void Agent::insertAgentNeighbor(const Agent *agent, float &rangeSq)
	{	

		if (this != agent) {
			///const float distSq = absSq(position_ - agent->position_);
			float distSq = abs(position_ - agent->position_) - agent->radius_ < 0 ? 0:sqr(abs(position_ - agent->position_) - agent->radius_);
			
			if (distSq < rangeSq) {
				if (agentNeighbors_.size() < maxNeighbors_) {
					agentNeighbors_.push_back(std::make_pair(distSq, agent));
				}

				size_t i = agentNeighbors_.size() - 1;

				while (i != 0 && distSq < agentNeighbors_[i - 1].first) {
					agentNeighbors_[i] = agentNeighbors_[i - 1];
					--i;
				}

				agentNeighbors_[i] = std::make_pair(distSq, agent);

				if (agentNeighbors_.size() == maxNeighbors_) {
					rangeSq = agentNeighbors_.back().first;
				}
			}
		}
	}

	void Agent::insertObstacleNeighbor(const Obstacle *obstacle, float rangeSq)
	{
		const Obstacle *const nextObstacle = obstacle->nextObstacle_;

		const float distSq = distSqPointLineSegment(obstacle->point_, nextObstacle->point_, position_);

		if (distSq < rangeSq) {
			obstacleNeighbors_.push_back(std::make_pair(distSq, obstacle));

			size_t i = obstacleNeighbors_.size() - 1;

			while (i != 0 && distSq < obstacleNeighbors_[i - 1].first) {
				obstacleNeighbors_[i] = obstacleNeighbors_[i - 1];
				--i;
			}

			obstacleNeighbors_[i] = std::make_pair(distSq, obstacle);
		}
	}

	void Agent::update()
	{
		velocity_ = newVelocity_;
		position_ += velocity_ * sim_->timeStep_;
	}

	float objFuncValue(Vector2 v, Vector2 optVelocity){
		float x = v.x();
		float y = v.y();
		float x_pref = optVelocity.x();
		float y_pref = optVelocity.y();
		return (x - x_pref)*(x - x_pref) + (y-y_pref)*(y-y_pref) + std::fabs(x*x + y*y - x_pref*x_pref-y_pref*y_pref);
	}
	float objFuncValue(Vector2 v, Vector2 optVelocity, double w){
		float x = v.x();
		float y = v.y();
		float x_pref = optVelocity.x();
		float y_pref = optVelocity.y();
		return (x - x_pref)*(x - x_pref) + (y-y_pref)*(y-y_pref) + w * std::fabs(x*x + y*y - x_pref*x_pref-y_pref*y_pref);
	}

	bool linearProgram1(const std::vector<Line> &lines, size_t lineNo, float radius, const Vector2 &optVelocity, bool directionOpt, Vector2 &result)
	{
		const float dotProduct = lines[lineNo].point * lines[lineNo].direction;

		/// note that sqr means square rather than square root. 
		/// rewrite discriminant as sqr(radius) - (absSq(lines[lineNo].point) - sqr(dotProduct))
		/// absSq(lines[lineNo].point) - sqr(dotProduct) is the square of the minimum speed that to avoid collision
		/// radius is the square of the maximum speed
		const float discriminant = sqr(dotProduct) + sqr(radius) - absSq(lines[lineNo].point);

		if (discriminant < 0.0f) { /// the minmum speed requred to avoid collision is larger than the max speed allowed
			/* Max speed circle fully invalidates line lineNo. */
			return false;
		}

		const float sqrtDiscriminant = std::sqrt(discriminant);

		/// thoese orca lines might or might not(when parellel) intersect with lineNo.
		/// tLeft is the intersection point on the left of lines[lineNo].point that is closest to lines[lineNo].point; the radius of max_speed also
		/// intersects with lines[lineNo] and have two intersection points; we also add the left point into the intersection points to be compared.
		/// tRight is similar.
		float tLeft = -dotProduct - sqrtDiscriminant; /// initialized to the distance to max_speed intersection point
		float tRight = -dotProduct + sqrtDiscriminant;

		for (size_t i = 0; i < lineNo; ++i) {
			const float denominator = det(lines[lineNo].direction, lines[i].direction);
			const float numerator = det(lines[i].direction, lines[lineNo].point - lines[i].point);

			if (std::fabs(denominator) <= RVO_EPSILON) {
				/* Lines lineNo and i are (almost) parallel. */
				if (numerator < 0.0f) {
					return false;
				}
				else {
					continue;
				}
			}

			/// t is the distance from lines[lineNo].point to the intersection point of lines[i] and lines[lineNo] (with direction)
			const float t = numerator / denominator;

			/// (it is defined that the left side of the orca line is feasible)
			/// Line i will futher devides the left part of line lineNo into two parts: the left part and the right part
			/// when denominator >= 0, the right part of the left part of line lineNo is feasible
			if (denominator >= 0.0f) {
				/* Line i bounds line lineNo on the right. */
				tRight = std::min(tRight, t); 
				/// 
				/// if there is no dynamic constraint (i.e., v<=max_speed), tRight = t. Previously we already set tRigth
				/// to max_speed. this min is to choose which point it should use as the boundary, the max_pseed point or the intersection point
			}
			else {
				/* Line i bounds line lineNo on the left. */
				tLeft = std::max(tLeft, t);
			}

			if (tLeft > tRight) {
				return false;
			}
		}

		if(USE_OLD_ORCA){
			if (directionOpt) {
				/* Optimize direction. */
				if (optVelocity * lines[lineNo].direction > 0.0f) {
					/* Take right extreme. */
					result = lines[lineNo].point + tRight * lines[lineNo].direction;
				}
				else {
					/* Take left extreme. */
					result = lines[lineNo].point + tLeft * lines[lineNo].direction;
				}
			}
			else {
				/// Optimize closest point.
				const float t = lines[lineNo].direction * (optVelocity - lines[lineNo].point);
				/// check this optimal point is on the line segment or not
				if (t < tLeft) {
					result = lines[lineNo].point + tLeft * lines[lineNo].direction;
				}
				else if (t > tRight) {
					result = lines[lineNo].point + tRight * lines[lineNo].direction;
				}
				else {
					result = lines[lineNo].point + t * lines[lineNo].direction;
				}
			}
		} else{
			if (directionOpt) {
				/* Optimize direction. */
				if (optVelocity * lines[lineNo].direction > 0.0f) {
					/* Take right extreme. */
					result = lines[lineNo].point + tRight * lines[lineNo].direction;
				}
				else {
					/* Take left extreme. */
					result = lines[lineNo].point + tLeft * lines[lineNo].direction;
				}
			}else {
				float dist_to_line_seg = std::sqrt( distSqPointLineSegment(lines[lineNo].point + tLeft * lines[lineNo].direction, 
					lines[lineNo].point + tRight * lines[lineNo].direction, Vector2(0.0, 0.0)) );
				float v_pref_len = abs(optVelocity);
				if(dist_to_line_seg > v_pref_len){
					const float t = ( lines[lineNo].direction * (optVelocity - 2.0 * lines[lineNo].point)) / 2.0;
					/// check this optimal point is on the line segment or not
					if (t < tLeft) {
						result = lines[lineNo].point + tLeft * lines[lineNo].direction;
					}
					else if (t > tRight) {
						result = lines[lineNo].point + tRight * lines[lineNo].direction;
					}
					else {
						result = lines[lineNo].point + t * lines[lineNo].direction;
					}
				} else{
					float ab = lines[lineNo].direction * lines[lineNo].point;
					float sqrt_delta = std::sqrt(ab*ab - lines[lineNo].point * lines[lineNo].point + v_pref_len * v_pref_len);
					
					float t_left = -ab - sqrt_delta; //left intersection point of the line and the circle
					float t_right = -ab + sqrt_delta; //right intersection point of the line and the circle

					if(t_left < tLeft){
						if(t_right > tRight){
							if(objFuncValue(lines[lineNo].point + tLeft * lines[lineNo].direction, optVelocity) 
								< objFuncValue(lines[lineNo].point + tRight * lines[lineNo].direction, optVelocity)){
								result = lines[lineNo].point + tLeft * lines[lineNo].direction;
							} else{
								result = lines[lineNo].point + tRight * lines[lineNo].direction;
							}
						}else{
							if(objFuncValue(lines[lineNo].point + tLeft * lines[lineNo].direction, optVelocity) 
								< objFuncValue(lines[lineNo].point + t_right * lines[lineNo].direction, optVelocity)){
								result = lines[lineNo].point + tLeft * lines[lineNo].direction;
							} else{
								result = lines[lineNo].point + t_right * lines[lineNo].direction;
							}
						}
					}else{
						if(t_right > tRight){
							if(objFuncValue(lines[lineNo].point + t_left * lines[lineNo].direction, optVelocity) 
								< objFuncValue(lines[lineNo].point + tRight * lines[lineNo].direction, optVelocity)){
								result = lines[lineNo].point + t_left * lines[lineNo].direction;
							} else{
								result = lines[lineNo].point + tRight * lines[lineNo].direction;
							}
						}else{
							if(objFuncValue(lines[lineNo].point + t_left * lines[lineNo].direction, optVelocity) 
								< objFuncValue(lines[lineNo].point + t_right * lines[lineNo].direction, optVelocity)){
								result = lines[lineNo].point + t_left * lines[lineNo].direction;
							} else{
								result = lines[lineNo].point + t_right * lines[lineNo].direction;
							}
						}
					}

				}
			}
		}
		 

		return true;
	}


	bool linearProgram1(const std::vector<Line> &lines, size_t lineNo, float radius, const Vector2 &optVelocity, bool directionOpt, double w, std::vector<CostVelPair> & opt_vel_candidates)
	{
		const float dotProduct = lines[lineNo].point * lines[lineNo].direction;

		/// note that sqr means square rather than square root. 
		/// rewrite discriminant as sqr(radius) - (absSq(lines[lineNo].point) - sqr(dotProduct))
		/// absSq(lines[lineNo].point) - sqr(dotProduct) is the square of the minimum speed that to avoid collision
		/// radius is the square of the maximum speed
		const float discriminant = sqr(dotProduct) + sqr(radius) - absSq(lines[lineNo].point);

		if (discriminant < 0.0f) { /// the minmum speed requred to avoid collision is larger than the max speed allowed
			/* Max speed circle fully invalidates line lineNo. */
			return false;
		}

		const float sqrtDiscriminant = std::sqrt(discriminant);

		/// thoese orca lines might or might not(when parellel) intersect with lineNo.
		/// tLeft is the intersection point on the left of lines[lineNo].point that is closest to lines[lineNo].point; the radius of max_speed also
		/// intersects with lines[lineNo] and have two intersection points; we also add the left point into the intersection points to be compared.
		/// tRight is similar.
		float tLeft = -dotProduct - sqrtDiscriminant; /// initialized to the distance to max_speed intersection point
		float tRight = -dotProduct + sqrtDiscriminant;

		for (size_t i = 0; i < lineNo; ++i) {
			const float denominator = det(lines[lineNo].direction, lines[i].direction);
			const float numerator = det(lines[i].direction, lines[lineNo].point - lines[i].point);

			if (std::fabs(denominator) <= RVO_EPSILON) {
				/* Lines lineNo and i are (almost) parallel. */
				if (numerator < 0.0f) {
					return false;
				}
				else {
					continue;
				}
			}

			/// t is the distance from lines[lineNo].point to the intersection point of lines[i] and lines[lineNo] (with direction)
			const float t = numerator / denominator;

			/// (it is defined that the left side of the orca line is feasible)
			/// Line i will futher devides the left part of line lineNo into two parts: the left part and the right part
			/// when denominator >= 0, the right part of the left part of line lineNo is feasible
			if (denominator >= 0.0f) {
				/* Line i bounds line lineNo on the right. */
				tRight = std::min(tRight, t); 
				/// 
				/// if there is no dynamic constraint (i.e., v<=max_speed), tRight = t. Previously we already set tRigth
				/// to max_speed. this min is to choose which point it should use as the boundary, the max_pseed point or the intersection point
			}
			else {
				/* Line i bounds line lineNo on the left. */
				tLeft = std::max(tLeft, t);
			}

			if (tLeft > tRight) {
				return false;
			}
		}

		if (directionOpt) {
			/* Optimize direction. */
			if (optVelocity * lines[lineNo].direction > 0.0f) {
				/* Take right extreme. */
				Vector2 result = lines[lineNo].point + tRight * lines[lineNo].direction;
				opt_vel_candidates.push_back(CostVelPair(objFuncValue(result, optVelocity, w), result));
			}
			else {
				/* Take left extreme. */
				Vector2 result = lines[lineNo].point + tLeft * lines[lineNo].direction;
				opt_vel_candidates.push_back(CostVelPair(objFuncValue(result, optVelocity, w), result));
			}
		}		
		else {
			float dist_to_line_seg = std::sqrt( distSqPointLineSegment(lines[lineNo].point + tLeft * lines[lineNo].direction, 
				lines[lineNo].point + tRight * lines[lineNo].direction, Vector2(0.0, 0.0)) );
			float v_pref_len = abs(optVelocity);
			if(dist_to_line_seg > v_pref_len){
				const float t = ( lines[lineNo].direction * (optVelocity - (1+w) * lines[lineNo].point)) / (1+w);
				/// check this optimal point is on the line segment or not
				if (t < tLeft) {
					Vector2 result = lines[lineNo].point + tLeft * lines[lineNo].direction;
					opt_vel_candidates.push_back(CostVelPair(objFuncValue(result, optVelocity, w), result));
				}
				else if (t > tRight) {
					Vector2 result = lines[lineNo].point + tRight * lines[lineNo].direction;
					opt_vel_candidates.push_back(CostVelPair(objFuncValue(result, optVelocity, w), result));
				}
				else {
					Vector2 result = lines[lineNo].point + t * lines[lineNo].direction;
					opt_vel_candidates.push_back(CostVelPair(objFuncValue(result, optVelocity, w), result));
				}
			} else{
				float ab = lines[lineNo].direction * lines[lineNo].point;

				float delta = ab*ab - lines[lineNo].point * lines[lineNo].point + v_pref_len * v_pref_len;
				if(delta<=1e-6) delta = 0;
				float sqrt_delta = std::sqrt(delta);
				//float sqrt_delta = std::sqrt(ab*ab - lines[lineNo].point * lines[lineNo].point + v_pref_len * v_pref_len);
				
				float t_left = -ab - sqrt_delta; //left intersection point of the line and the circle
				float t_right = -ab + sqrt_delta; //right intersection point of the line and the circle

				if(t_left < tLeft){
					if(t_right > tRight){
						// if(objFuncValue(lines[lineNo].point + tLeft * lines[lineNo].direction, optVelocity) 
						// 	< objFuncValue(lines[lineNo].point + tRight * lines[lineNo].direction, optVelocity)){
						// 	result = lines[lineNo].point + tLeft * lines[lineNo].direction;
						// } else{
						// 	result = lines[lineNo].point + tRight * lines[lineNo].direction;
						// }
						Vector2 result = lines[lineNo].point + tLeft * lines[lineNo].direction;
						opt_vel_candidates.push_back(CostVelPair(objFuncValue(result, optVelocity, w), result));
						result = lines[lineNo].point + tRight * lines[lineNo].direction;
						opt_vel_candidates.push_back(CostVelPair(objFuncValue(result, optVelocity, w), result));

					}else{
						// if(objFuncValue(lines[lineNo].point + tLeft * lines[lineNo].direction, optVelocity) 
						// 	< objFuncValue(lines[lineNo].point + t_right * lines[lineNo].direction, optVelocity)){
						// 	result = lines[lineNo].point + tLeft * lines[lineNo].direction;
						// } else{
						// 	result = lines[lineNo].point + t_right * lines[lineNo].direction;
						// }
						Vector2 result = lines[lineNo].point + tLeft * lines[lineNo].direction;
						opt_vel_candidates.push_back(CostVelPair(objFuncValue(result, optVelocity, w), result));
						result = lines[lineNo].point + t_right * lines[lineNo].direction;
						opt_vel_candidates.push_back(CostVelPair(objFuncValue(result, optVelocity, w), result));
					}
				}else{
					if(t_right > tRight){
						// if(objFuncValue(lines[lineNo].point + t_left * lines[lineNo].direction, optVelocity) 
						// 	< objFuncValue(lines[lineNo].point + tRight * lines[lineNo].direction, optVelocity)){
						// 	result = lines[lineNo].point + t_left * lines[lineNo].direction;
						// } else{
						// 	result = lines[lineNo].point + tRight * lines[lineNo].direction;
						// }
						Vector2 result = lines[lineNo].point + t_left * lines[lineNo].direction;
						opt_vel_candidates.push_back(CostVelPair(objFuncValue(result, optVelocity, w), result));
						result = lines[lineNo].point + tRight * lines[lineNo].direction;
						opt_vel_candidates.push_back(CostVelPair(objFuncValue(result, optVelocity, w), result));
					}else{
						// if(objFuncValue(lines[lineNo].point + t_left * lines[lineNo].direction, optVelocity) 
						// 	< objFuncValue(lines[lineNo].point + t_right * lines[lineNo].direction, optVelocity)){
						// 	result = lines[lineNo].point + t_left * lines[lineNo].direction;
						// } else{
						// 	result = lines[lineNo].point + t_right * lines[lineNo].direction;
						// }
						Vector2 result = lines[lineNo].point + t_left * lines[lineNo].direction;
						opt_vel_candidates.push_back(CostVelPair(objFuncValue(result, optVelocity, w), result));
						result = lines[lineNo].point + t_right * lines[lineNo].direction;
						opt_vel_candidates.push_back(CostVelPair(objFuncValue(result, optVelocity, w), result));
					}
				}

			}
		} 

		return true;
	}

	size_t linearProgram2(const std::vector<Line> &lines, float radius, const Vector2 &optVelocity, bool directionOpt, Vector2 &result)
	{
		if (directionOpt) {
			/*
			 * Optimize direction. Note that the optimization velocity is of unit
			 * length in this case.
			 */
			result = optVelocity * radius;
		}
		else if (absSq(optVelocity) > sqr(radius)) {
			/* Optimize closest point and outside circle. radius is the max_speed (see the place linearProgram2 is called), hence this step is just 
			normalize optVelocity when it exceeds max speed*/
			result = normalize(optVelocity) * radius;
		}
		else {
			/* Optimize closest point and inside circle. */
			result = optVelocity;
		}
		/// initially result = optVelocity; adding constraints one by one will narrow down the feasible region; the result with fewer 
		/// constraints must be better than that with more constraints. Hence the following for_loop will look for a best opt_vel under current constraints
		for (size_t i = 0; i < lines.size(); ++i) {
			if (det(lines[i].direction, lines[i].point - result) > 0.0f) {/// equivalent to det(lines[i].point - result, lines[i].direction) < 0.0f, which means the result (current opt vel is in the right side of the orca line (which means not satisified the constraints))
				// Result does not satisfy constraint i. Compute new optimal result.
				const Vector2 tempResult = result;

				if (!linearProgram1(lines, i, radius, optVelocity, directionOpt, result)) {
					result = tempResult;
					return i;
				}
			}
		}

		if(!USE_OLD_ORCA){
			double pre_vel_len = abs(optVelocity);

			if(pre_vel_len <= 0.05 || abs(result) >= 0.2 * pre_vel_len ) {
				return lines.size(); // goal is not_moving or computed vel is not very small
			}
			else{// recompute vel using the objective function with increased weight w
				double w = 4;//weight for the absolute difference between vel and pref_vel

				if (directionOpt) {
					 //Optimize direction. Note that the optimization velocity is of unit
					 //length in this case.
					result = optVelocity * radius;
				}
				else if (absSq(optVelocity) > sqr(radius)) {
					//Optimize closest point and outside circle.
					result = normalize(optVelocity) * radius;
				}
				else {
					//Optimize closest point and inside circle.
					result = optVelocity;
				}

				std::vector<CostVelPair> opt_vel_candidates;
				opt_vel_candidates.push_back(CostVelPair(0, result));

				for (size_t i = 0; i < lines.size(); ++i) {
					const Vector2 tempResult = opt_vel_candidates.back().second;

					//clear all candidates that does not satisfy the new constraint
					std::vector<CostVelPair> new_opt_vel_candidates;
					for(int j = 0; j<opt_vel_candidates.size(); j++){
						if (det(lines[i].direction, lines[i].point - opt_vel_candidates[j].second) <= 0.0f)
							new_opt_vel_candidates.push_back(opt_vel_candidates[j]);
					}
					opt_vel_candidates.clear();
					opt_vel_candidates = new_opt_vel_candidates;
					new_opt_vel_candidates.clear();


					if (!linearProgram1(lines, i, radius, optVelocity, directionOpt, w, opt_vel_candidates)) {
						result = tempResult;
						return i;
					}
					
				}

				float min_cost = opt_vel_candidates[0].first;
				int opt_vel_index = 0;
				for(int j=1; j<opt_vel_candidates.size(); j++){
					if(opt_vel_candidates[j].first < min_cost) {
						opt_vel_index = j;
					}
				}
				result = opt_vel_candidates[opt_vel_index].second;
			}
		}
		


		return lines.size();
	}


	size_t linearProgram2(const std::vector<Line> &lines, float radius, const Vector2 &optVelocity, bool directionOpt, Vector2 &result, double & patience)
	{
		if (directionOpt) {
			/*
			 * Optimize direction. Note that the optimization velocity is of unit
			 * length in this case.
			 */
			result = optVelocity * radius;
		}
		else if (absSq(optVelocity) > sqr(radius)) {
			/* Optimize closest point and outside circle. radius is the max_speed (see the place linearProgram2 is called), hence this step is just 
			normalize optVelocity when it exceeds max speed*/
			result = normalize(optVelocity) * radius;
		}
		else {
			/* Optimize closest point and inside circle. */
			result = optVelocity;
		}
		/// initially result = optVelocity; adding constraints one by one will narrow down the feasible region; the result with fewer 
		/// constraints must be better than that with more constraints. Hence the following for_loop will look for a best opt_vel under current constraints
		for (size_t i = 0; i < lines.size(); ++i) {
			if (det(lines[i].direction, lines[i].point - result) > 0.0f) {/// equivalent to det(lines[i].point - result, lines[i].direction) < 0.0f, which means the result (current opt vel is in the right side of the orca line (which means not satisified the constraints))
				// Result does not satisfy constraint i. Compute new optimal result.
				const Vector2 tempResult = result;

				if (!linearProgram1(lines, i, radius, optVelocity, directionOpt, result)) {
					result = tempResult;
					return i;
				}
			}
		}

		if(!USE_OLD_ORCA && USE_PATIENCE){
			double pre_vel_len = abs(optVelocity);

			if(pre_vel_len <= 0.05 || abs(result) >= 0.2 * pre_vel_len) {
				//patience = 1.0; //reset agent's patience to 1
				return lines.size(); // goal is not_moving or computed vel is not very small
			}
			else{// recompute vel using the objective function with increased weight w
				//patience *= 0.5;
				/*if(patience <= 0.2)
					patience = 0.2;*/


				double w;//weight for the absolute difference between vel and pref_vel
				w = 1.0/patience;

							//w = 1.0;
				
				if (directionOpt) {
					 //Optimize direction. Note that the optimization velocity is of unit
					 //length in this case.
					result = optVelocity * radius;
				}
				else if (absSq(optVelocity) > sqr(radius)) {
					//Optimize closest point and outside circle.
					result = normalize(optVelocity) * radius;
				}
				else {
					//Optimize closest point and inside circle.
					result = optVelocity;
				}

				std::vector<CostVelPair> opt_vel_candidates;
				opt_vel_candidates.push_back(CostVelPair(0, result));

				for (size_t i = 0; i < lines.size(); ++i) {
					const Vector2 tempResult = opt_vel_candidates.back().second;

					//clear all candidates that does not satisfy the new constraint
					std::vector<CostVelPair> new_opt_vel_candidates;
					for(int j = 0; j<opt_vel_candidates.size(); j++){
						if (det(lines[i].direction, lines[i].point - opt_vel_candidates[j].second) <= 0.0f)
							new_opt_vel_candidates.push_back(opt_vel_candidates[j]);
					}
					opt_vel_candidates.clear();
					opt_vel_candidates = new_opt_vel_candidates;
					new_opt_vel_candidates.clear();


					if (!linearProgram1(lines, i, radius, optVelocity, directionOpt, w, opt_vel_candidates)) {
						result = tempResult;
						return i;
					}
					
				}

				float min_cost = opt_vel_candidates[0].first;
				int opt_vel_index = 0;
				for(int j=1; j<opt_vel_candidates.size(); j++){
					if(opt_vel_candidates[j].first < min_cost) {
						opt_vel_index = j;
					}
				}
				result = opt_vel_candidates[opt_vel_index].second;
			}
		}


		return lines.size();
	}

	void linearProgram3(const std::vector<Line> &lines, size_t numObstLines, size_t beginLine, float radius, Vector2 &result)
	{
		float distance = 0.0f;

		for (size_t i = beginLine; i < lines.size(); ++i) {
			if (det(lines[i].direction, lines[i].point - result) > distance) {
				/* Result does not satisfy constraint of line i. */
				std::vector<Line> projLines(lines.begin(), lines.begin() + static_cast<ptrdiff_t>(numObstLines));

				for (size_t j = numObstLines; j < i; ++j) {
					Line line;

					float determinant = det(lines[i].direction, lines[j].direction);

					if (std::fabs(determinant) <= RVO_EPSILON) {
						/* Line i and line j are parallel. */
						if (lines[i].direction * lines[j].direction > 0.0f) {
							/* Line i and line j point in the same direction. */
							continue;
						}
						else {
							/* Line i and line j point in opposite direction. */
							line.point = 0.5f * (lines[i].point + lines[j].point);
						}
					}
					else {
						line.point = lines[i].point + (det(lines[j].direction, lines[i].point - lines[j].point) / determinant) * lines[i].direction;
					}

					line.direction = normalize(lines[j].direction - lines[i].direction);
					projLines.push_back(line);
				}

				const Vector2 tempResult = result;

				if (linearProgram2(projLines, radius, Vector2(-lines[i].direction.y(), lines[i].direction.x()) /*rotate counterclockwisely by 90*/, true, result) < projLines.size()) {
					/* This should in principle not happen.  The result is by definition
					 * already in the feasible region of this linear program. If it fails,
					 * it is due to small floating point error, and the current result is
					 * kept.
					 */
					result = tempResult;
				}

				distance = det(lines[i].direction, lines[i].point - result);
			}
		}
	}
}
