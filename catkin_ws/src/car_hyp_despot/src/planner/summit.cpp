/*
 * summit.cpp
 *
 *  Created on: Jan 9, 2020
 *      Author: panpan
 */

#include "path.h"
#include "world_model.h"
#include "context_pomdp.h"
#include "carla/client/Client.h"
#include "carla/geom/Vector2D.h"
#include "carla/sumonetwork/SumoNetwork.h"
#include "carla/occupancy/OccupancyMap.h"
#include "carla/segments/SegmentMap.h"
#include "threaded_print.h"
std::mutex ThreadStream::_mutex_threadstream{};


namespace cc = carla::client;
namespace cg = carla::geom;
namespace sumo = carla::sumonetwork;
namespace occu = carla::occupancy;
namespace segm = carla::segments;
namespace side = carla::sidewalk;

static vector<sumo::SumoNetwork> networks_;
static occu::OccupancyMap network_occupancy_map_;
static side::Sidewalk sidewalk_network_map_ =
		side::Sidewalk(std::vector<std::vector<cg::Vector2D>>());

PathTree* WorldModel::path_tree_;

void WorldModel::ConnectCarla() {
	// Connect with SUMMOT server
	logi << "Loading summit map " << map_location << endl;
	std::string homedir = getenv("HOME");
	auto summit_root = homedir + "/summit/";

	for (int i=0; i< Globals::config.NUM_THREADS;i++)
		networks_.push_back(sumo::SumoNetwork::Load(
				summit_root + "Data/" + map_location + ".net.xml"));

	network_occupancy_map_ = occu::OccupancyMap::Load(
			summit_root + "Data/" + map_location + ".network.wkt");

	sidewalk_network_map_ = network_occupancy_map_.CreateSidewalk(1.5);

	logi << "[WorldModel] Summit connected" << endl;
}

double WorldModel::GetSteerFromLane(CarStruct &car, double lane) {
	logd << "[GetSteerFromLane] lane " << lane << ", " << car.Text() << endl;

    double steer_to_path = GetSteerToPath(car);
    logd << "[GetSteerFromLane] steering angle " << steer_to_path << endl;
	return steer_to_path;
}

const Path* WorldModel::GetPath(int path_idx) const {
	if (path_tree_->cur_idx() >= path_idx)
		return WorldModel::path_tree_->at(path_idx);
	else
		return &path;
}

double WorldModel::GetSteerToPath(const CarStruct& car) const {
	COORD car_goal;

	if (car.path_idx == -1) {
		int pos_along_path = path.Nearest(car.pos);
		int next_pos = path.Forward(pos_along_path, PURSUIT_LEN);
		car_goal = path[next_pos]; // target at 3 meters ahead along the path
	} else
		car_goal = WorldModel::path_tree_->pursuit_point(car.pos, car.path_idx);

	// return PControlAngle<CarStruct>(car, car_goal);
	return PurepursuitAngle(car, car_goal);
}

PathTree* WorldModel::ResetPathTree() {
	logd << "Reseting path tree" << endl;

	if (WorldModel::path_tree_)
		delete WorldModel::path_tree_;
	WorldModel::path_tree_ = new PathTree();
	return WorldModel::path_tree_;
}

std::vector<int> WorldModel::SetRootPathIDs(Shared_VNode* root) {
	if (path.size()==0)
		ERR("No current path available!!");

	std::vector<int> legal_lanes;
	for (int laneID=0; laneID < ModelParams::NumLaneDecisions; laneID++) {
		if (laneID != last_decision_lane && laneID + last_decision_lane == 2) {
			tout << "skipping root lane " << GetLaneText(LaneCode(laneID)) << endl;
			;// Excluding oscillating behaviors.
		} else {
			PomdpState* state = static_cast<PomdpState*>(root->particles()[0]);
			Path* lc_path_prt = ParseLanePath(state->car.pos, state->car.heading_dir, laneID);
			if (lc_path_prt) {
				int new_idx = WorldModel::path_tree_->insert(lc_path_prt);
				root->path_id(laneID, new_idx);
				root_path_id_map.insert(std::make_pair(laneID, new_idx));

				logi << "path parsed " << new_idx << " for root lane " <<
									GetLaneText(LaneCode(laneID)) << endl;
				legal_lanes.push_back(laneID);
			}
		}

	}

	for (State* particle: root->particles()) {
		PomdpState* state = static_cast<PomdpState*>(particle);
		state->car.path_idx = root->path_id(LaneCode::KEEP);
		if(state->car.path_idx == -1)
			ERR("path idx at root should not be -1!");
	}

	return legal_lanes;
}

double Dist(COORD a, cg::Vector2D b) {
	return sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y));
}

bool WorldModel::IsInMap(const CarStruct& car, bool insearch) {
	double l = ModelParams::CAR_LENGTH / 4.0;
	double w = ModelParams::CAR_WIDTH / 4.0;
	if (insearch) {
		l = ModelParams::CAR_LENGTH / 2.0;
		w = ModelParams::CAR_WIDTH / 2.0;
	}
	COORD heading_dir(cos(car.heading_dir)*l, sin(car.heading_dir)*l);
	COORD tan_dir(-sin(car.heading_dir)*w, cos(car.heading_dir)*w);

	cg::Vector2D ego_veh_pos = cg::Vector2D(car.pos.x + heading_dir.x + tan_dir.x,
			car.pos.y + heading_dir.y + tan_dir.y);
	if(!network_occupancy_map_.Contains(ego_veh_pos))
		return false;
	ego_veh_pos = cg::Vector2D(car.pos.x - heading_dir.x + tan_dir.x,
			car.pos.y - heading_dir.y + tan_dir.y);
	if(!network_occupancy_map_.Contains(ego_veh_pos))
		return false;
	ego_veh_pos = cg::Vector2D(car.pos.x + heading_dir.x - tan_dir.x,
			car.pos.y + heading_dir.y - tan_dir.y);
	if(!network_occupancy_map_.Contains(ego_veh_pos))
		return false;
	ego_veh_pos = cg::Vector2D(car.pos.x - heading_dir.x - tan_dir.x,
			car.pos.y - heading_dir.y - tan_dir.y);
	if(!network_occupancy_map_.Contains(ego_veh_pos))
		return false;

	return true;
}

bool WorldModel::ValidForLaneChange(const CarStruct &car) {

	if (car.path_idx == -1)
		return false;

	double trav_dist = WorldModel::path_tree_->trav_dist(car.pos, car.path_idx);

	logd << "[ValidForLaneChange] " << car.Text() << endl;

	if (LaneChangedAfterRoot(car.path_idx) &&
			trav_dist < 5.0){
	    logd << "[ValidForLaneChange] lane status: lane change ongoing " << endl;
		return false;
	}
	else {
		logd << "[ValidForLaneChange] lane status: valid for lane change " << endl;
		return true;
	}
}

void WorldModel::ExtendPath(const COORD car_pos, Path& path, double desired_length) {
	int pos_along = path.Nearest(car_pos);
	double left_length = path.GetLength(pos_along);

	if(left_length < desired_length - 1.0) {
		logd << "[WorldModel::ExtendCurLane] Extending cur path "<< endl;
		int tid = MapThread(this_thread::get_id());
		cg::Vector2D ego_veh_pos = cg::Vector2D(path.back().x, path.back().y);
		auto cur_route_point = networks_[tid].GetNearestRoutePoint(ego_veh_pos);

		std::vector<std::vector<sumo::RoutePoint>> new_path_candidates =
					networks_[tid].GetNextRoutePaths(cur_route_point,
							desired_length - left_length, 1.0);

		if (new_path_candidates.size()>0) {
			int rand_path = Random::RANDOM.NextInt(new_path_candidates.size());
			auto& route_path = new_path_candidates[rand_path];

			logd << "parsed " << route_path.size() << " route points" << endl;
			for (sumo::RoutePoint& route_point : route_path) {
				cg::Vector2D coord = networks_[tid].GetRoutePointPosition(route_point);
				path.AppendInterpolatedLine(COORD(coord.x, coord.y));
			}

			logd << "cur path extended to length " << path.GetLength() << endl;
		}
	}
}

bool WorldModel::ExtendCurLane(const PomdpState* state) const {
	auto car = state->car;
	const COORD& pos = car.pos;

	Path cur_path = GetPath(car.path_idx)[0];

	int pos_along = cur_path.Nearest(pos);
	double left_length = cur_path.GetLength(pos_along);

	if(left_length < 10) {
		logd << "[WorldModel::ExtendCurLane] Extending cur path "<< endl;
		int tid = MapThread(this_thread::get_id());
		cg::Vector2D ego_veh_pos = cg::Vector2D(pos.x, pos.y);
		auto cur_route_point = networks_[tid].GetNearestRoutePoint(ego_veh_pos);

		std::vector<std::vector<sumo::RoutePoint>> new_path_candidates =
					networks_[tid].GetNextRoutePaths(cur_route_point, 10.0, 1.0);

		if (new_path_candidates.size()>0) {
			int rand_path = Random::RANDOM.NextInt(new_path_candidates.size());
			auto& route_path = new_path_candidates[rand_path];

			Path* path_ptr = new Path();
			path_ptr[0] = cur_path;

			logd << "parsed " << route_path.size() << " route points" << endl;
			for (sumo::RoutePoint& route_point : route_path) {
				cg::Vector2D coord = networks_[tid].GetRoutePointPosition(
						route_point);
				path_ptr[0].AppendInterpolatedLine(COORD(coord.x, coord.y));
			}

			WorldModel::path_tree_->replace(car.path_idx, path_ptr);

			logd << "cur path extended to length " << path_ptr->GetLength() << endl;
		}
	}
	return true;
}

bool WorldModel::LaneExist(COORD pos, double yaw, int laneID) const {
	int tid = MapThread(this_thread::get_id());

	cg::Vector2D ego_veh_pos = cg::Vector2D(pos.x, pos.y);
	cg::Vector2D forward_vec;
	auto cur_route_point = networks_[tid].GetNearestRoutePoint(ego_veh_pos);
	auto rp_list = networks_[tid].GetNextRoutePoints(cur_route_point,1.0);
	if (rp_list.size() == 1) {
		auto next_rp_along_lane = rp_list[0];
		forward_vec = networks_[tid].GetRoutePointPosition(next_rp_along_lane) -
			networks_[tid].GetRoutePointPosition(cur_route_point);
	} else {
		forward_vec = cg::Vector2D(cosf(yaw), sinf(yaw));
	}
	cg::Vector2D sidewalk_vec = forward_vec.Rotate(M_PI / 2.0); // rotate clockwise by 90 degree

	cg::Vector2D ego_veh_pos_in_new_lane;
	if (laneID == LaneCode::LEFT)
		ego_veh_pos_in_new_lane = ego_veh_pos - 4.0 * sidewalk_vec;
	else if (laneID == LaneCode::RIGHT)
		ego_veh_pos_in_new_lane = ego_veh_pos + 4.0 * sidewalk_vec;
	else
		ego_veh_pos_in_new_lane = ego_veh_pos;

	if (!network_occupancy_map_.Contains(ego_veh_pos_in_new_lane)) {
		return false;
	}

	auto new_route_point = networks_[tid].GetNearestRoutePoint(
			ego_veh_pos_in_new_lane);

	bool has_lane;
	if (laneID == LaneCode::KEEP)
		has_lane = true;
	else
		has_lane = (new_route_point.edge == cur_route_point.edge
				&& new_route_point.lane != cur_route_point.lane);

	return has_lane;
}

std::vector<COORD> shifts = {
		COORD(0, 0),
		COORD(1.0, 0),
		COORD(-1.0, 0),
		COORD(0, 1.0),
		COORD(0, -1.0)};

bool ParseAlignedRoutePoint(int tid, cg::Vector2D ego_veh_pos, double yaw,
		sumo::RoutePoint& cur_route_point, cg::Vector2D& sidewalk_vec) {
	int shift_pos = 0;
	bool found = false;
	cg::Vector2D forward_vec = cg::Vector2D(cosf(yaw), sinf(yaw));
	sidewalk_vec = forward_vec.Rotate(M_PI / 2.0); // rotate clockwise by 90 degree

	while (!found && shift_pos < shifts.size()) {
		cg::Vector2D ego_check_pos = ego_veh_pos +
				shifts[shift_pos].x * forward_vec +
				shifts[shift_pos].y * sidewalk_vec;
		cur_route_point = networks_[tid].GetNearestRoutePoint(ego_check_pos);
		auto rp_list = networks_[tid].GetNextRoutePoints(cur_route_point, 0.5);
		if (rp_list.size() > 0) {
			auto next_rp_along_lane = rp_list[0];
			forward_vec = networks_[tid].GetRoutePointPosition(next_rp_along_lane) -
				networks_[tid].GetRoutePointPosition(cur_route_point);
			forward_vec = forward_vec.MakeUnitVector();
		} else {
			forward_vec = cg::Vector2D(0.0, 0.0);
		}

		float alignment = cg::Vector2D::DotProduct(forward_vec, cg::Vector2D(cosf(yaw), sinf(yaw)));
		if (alignment > 0.2) {
			found = true;
			sidewalk_vec = forward_vec.Rotate(M_PI / 2.0);
			break;
		}

		forward_vec = cg::Vector2D(cosf(yaw), sinf(yaw));
		sidewalk_vec = forward_vec.Rotate(M_PI / 2.0);
		shift_pos ++;
	}

	return found;
}

Path* WorldModel::ParseLanePath(COORD pos, double yaw, int laneID) const {
	int tid = MapThread(this_thread::get_id());

	cg::Vector2D ego_veh_pos = cg::Vector2D(pos.x, pos.y);

	if (!network_occupancy_map_.Contains(ego_veh_pos)) {
		DEBUG("Vehicle out of map");
		return NULL;
	}

	cg::Vector2D sidewalk_vec;
	sumo::RoutePoint cur_route_point;

	bool found = ParseAlignedRoutePoint(tid, ego_veh_pos, yaw, cur_route_point, sidewalk_vec);
	if(!found) {
		DEBUG("No current route point found");
		return NULL;
	}

	cg::Vector2D ego_veh_pos_in_new_lane;
	if (laneID == LaneCode::LEFT)
		ego_veh_pos_in_new_lane = ego_veh_pos - 4.0 * sidewalk_vec;
	else if (laneID == LaneCode::RIGHT)
		ego_veh_pos_in_new_lane = ego_veh_pos + 4.0 * sidewalk_vec;
	else
		ego_veh_pos_in_new_lane = ego_veh_pos;

	if (!network_occupancy_map_.Contains(ego_veh_pos_in_new_lane)) {
//		DEBUG("Next lane out of map");
		return NULL;
	}

	Path* path_ptr = new Path();
	Path& path = *path_ptr;
	path.push_back(pos);

	sumo::RoutePoint new_route_point;
	bool has_lane;
	if (laneID == LaneCode::KEEP) {
		has_lane = true;
		new_route_point = cur_route_point;
	} else {
		new_route_point = networks_[tid].GetNearestRoutePoint(
				ego_veh_pos_in_new_lane);
		has_lane = (new_route_point.edge == cur_route_point.edge
				&& new_route_point.lane != cur_route_point.lane);
	}

	if (has_lane) {
		std::vector<std::vector<sumo::RoutePoint>> new_path_candidates =
				networks_[tid].GetNextRoutePaths(new_route_point, 10.0, 1.0);

		if (new_path_candidates.size()>0) {
			int rand_path = Random::RANDOM.NextInt(new_path_candidates.size());
			auto& route_path = new_path_candidates[rand_path];
			for (sumo::RoutePoint& route_point : route_path) {
				cg::Vector2D coord = networks_[tid].GetRoutePointPosition(
						route_point);
				path.AppendInterpolatedLine(COORD(coord.x, coord.y));
			}
		}

		return path_ptr;
	}
	else {
		logd << "[ParseLanePaths] no " << GetLaneText(LaneCode(laneID)) << endl;
		return NULL;
	}
}

bool WorldModel::ParseLanePath(despot::Shared_VNode* vnode, const PomdpState* state, int laneID) const {
	if(vnode->path_id(laneID) != -1){
		logd << "[ParseLanePaths] existing path " <<  vnode->path_id(laneID) <<
				" for " << GetLaneText(LaneCode(laneID)) << endl;
		return true;
	}

	Path* path_ptr = ParseLanePath(state->car.pos, state->car.heading_dir, laneID);

	if (path_ptr == NULL)
		return false;

	if (path_ptr[0].size() == 1) {
		logd << "[ParseLanePaths] no path for " << GetLaneText(LaneCode(laneID)) << endl;
		return false;
	}
	else {
		int new_idx = WorldModel::path_tree_->insert(path_ptr);
		vnode->path_id(laneID, new_idx);
		logd << "[ParseLanePaths] new path " <<  vnode->path_id(laneID) <<
							" of length " << path.GetLength() <<
							" for " << GetLaneText(LaneCode(laneID)) <<
							" at level " << vnode->depth() << endl;
		return true;
	}
}

std::vector<Path> WorldModel::ParsePathCandidates(std::string edge, int lane, int segment,
		float offset, std::string type) {
	int tid = MapThread(this_thread::get_id());
	std::vector<Path> path_list = std::vector<Path>();

	if (type != "ped"){
		sumo::RoutePoint start_rp = sumo::RoutePoint();
		start_rp.edge = edge;
		start_rp.lane = lane;
		start_rp.segment = segment;
		start_rp.offset = offset;

		std::vector<std::vector<sumo::RoutePoint>> path_candidates =
						networks_[tid].GetNextRoutePaths(start_rp, 50.0, 1.0);
		for (std::vector<sumo::RoutePoint>& route_path:path_candidates) {
			Path path;
			cg::Vector2D coord = networks_[tid].GetRoutePointPosition(start_rp);
			path.push_back(COORD(coord.x, coord.y));
			for (sumo::RoutePoint& route_point : route_path) {
				cg::Vector2D coord = networks_[tid].GetRoutePointPosition(
						route_point);
				path.AppendInterpolatedLine(COORD(coord.x, coord.y));
			}
			path_list.push_back(path);
		}
	} else {
		side::SidewalkRoutePoint start_rp = side::SidewalkRoutePoint();
		start_rp.polygon_id = lane;
		start_rp.segment_id = segment;
		start_rp.offset = offset;

		Path path;
		cg::Vector2D coord = sidewalk_network_map_.GetRoutePointPosition(start_rp);
		path.push_back(COORD(coord.x, coord.y));
		auto next_rp = start_rp;
		for (int i = 0; i < 50; i++) {
			next_rp = sidewalk_network_map_.GetNextRoutePoint(next_rp, 1.0);
			cg::Vector2D coord = sidewalk_network_map_.GetRoutePointPosition(next_rp);
			path.AppendInterpolatedLine(COORD(coord.x, coord.y));
		}
		path_list.push_back(path);
	}

	return path_list;
}
