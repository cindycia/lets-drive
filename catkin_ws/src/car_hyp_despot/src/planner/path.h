#pragma once
#include<vector>
#include <stdio.h>
#include <iostream>
#include "coord.h"
#include "param.h"
#include "debug_util.h"
#include <mutex>

#define PURSUIT_LEN 5.0

struct Path : std::vector<COORD> {
    int Nearest(const COORD pos) const;
    double MinDist(COORD pos);
    int Forward(double i, double len) const;
	double GetYaw(int i) const;
	void AppendInterpolatedLine(COORD end);
	Path Interpolate(double max_len = 10000.0) const;
	void CutJoin(const Path& p);

	double GetLength(int start=0) const;
	double GetTravelledLength(const COORD& car_pos) const;
	double GetCurDir(int pos_along = 0);
	COORD GetCrossDir(int, bool);

	void Text();
	void ShortText();

	void CopyTo(Path& des) const{
		des.assign(begin(),end());
	}

	Path CopyWithoutTravelledPoints(double dist_to_remove);
	Path TruncateHead(const COORD& car_pos) const;
};

class PathTree{
	std::mutex path_tree_mutex;

	std::vector<const Path*> path_list_;

public:

	int insert(const Path* path){
		std::lock_guard<std::mutex> lck(path_tree_mutex);

		int path_idx = path_list_.size();
		path_list_.push_back(path);

		return path_idx;
	}

	~PathTree() {
		for (const Path* path: path_list_)
			delete path;
		path_list_.resize(0);
	}

	const Path* at(int idx) {
		std::lock_guard<std::mutex> lck(path_tree_mutex);

		if (idx < 0)
			ERR("Invalid path id < 0!");
		if (idx >= path_list_.size())
			ERR("Overflow in path list!");
		return path_list_[idx];
	}

	void replace(int idx, const Path* src) {
		std::lock_guard<std::mutex> lck(path_tree_mutex);

		if (idx < 0)
			ERR("Invalid path id < 0!");
		if (idx >= path_list_.size())
			ERR("Overflow in path list!");
		delete path_list_[idx];
		path_list_[idx] = src;
	}

	int cur_idx(){
		std::lock_guard<std::mutex> lck(path_tree_mutex);

		return path_list_.size() - 1;
	}

	double length_left(COORD pos, int path_idx) {
		std::lock_guard<std::mutex> lck(path_tree_mutex);

		const Path& path = path_list_[path_idx][0];
		int pos_along = path.Nearest(pos);
		return path.GetLength(pos_along);
	}

	double trav_dist(COORD pos, int path_idx) {
		std::lock_guard<std::mutex> lck(path_tree_mutex);

		return path_list_[path_idx]->GetTravelledLength(pos);
	}

	COORD pursuit_point(COORD pos, int path_idx) {
		std::lock_guard<std::mutex> lck(path_tree_mutex);

		const Path& path = path_list_[path_idx][0];
		int pos_along_path = path.Nearest(pos);
		int next_pos = path.Forward(pos_along_path, PURSUIT_LEN);
		return path[next_pos]; // target at
	}
};

double CapAngle(double x);
