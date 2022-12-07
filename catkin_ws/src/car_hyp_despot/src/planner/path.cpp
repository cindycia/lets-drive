//#include "threaded_print.h"

#include "path.h"
#include<iostream>
#include "math_utils.h"
#include <fstream>

using namespace std;

int Path::Nearest(const COORD pos) const {
    auto& path = *this;
    double dmin = COORD::EuclideanDistance(pos, path[0]);
    int imin = 0;
    for(int i=0; i<path.size(); i++) {
        double d = COORD::EuclideanDistance(pos, path[i]);
        if(dmin > d) {
            dmin = d;
            imin = i;
        }
    }
    return imin;
}

double Path::MinDist(COORD pos) {
    COORD mc = at(Nearest(pos));
    double d = COORD::EuclideanDistance(mc, pos);
    return d;
}

int Path::Forward(double i, double len) const {
    auto& path = *this;
    float step=(len / ModelParams::PATH_STEP);

    if(step-(int)step>1.0-1e-5)
    {
    	step++;
    }
    i += (int)(step);

    if(i > path.size()-1) {
        i = path.size()-1;
    }
    return i;
}

double Path::GetYaw(int i) const {
    auto& path = *this;
	
	int j = Forward(i, 1.0);
	if(i==j) { i = max(0, i-3);}

	const COORD& pos = path[i];
	const COORD& forward_pos = path[j];
	MyVector vec(forward_pos.x - pos.x, forward_pos.y - pos.y);
    double a = vec.GetAngle();
	return a;
}

void Path::AppendInterpolatedLine(COORD end) {
	Path& path = this[0];
	COORD start = path.back();

	Path tail_path;

	const double step = ModelParams::PATH_STEP;
    double d = COORD::EuclideanDistance(start, end);
	double dx = (end.x-start.x) / d;
	double dy = (end.y-start.y) / d;

	double s=0;
	while(s < d) {
		double px = start.x + dx*s;
		double py = start.y + dy*s;

		if (tail_path.size()==0 || px != tail_path.back().x || py != tail_path.back().y)
			tail_path.push_back(COORD(px, py));

		s += step;
	}
	path.insert( path.end(), tail_path.begin(), tail_path.end());
}

Path Path::Interpolate(double max_len) const {
    auto& path = *this;
	Path p;

	const double step = ModelParams::PATH_STEP;
	double t=0, ti=0;
	for(int i=0; i<path.size()-1; i++) {
        double d = COORD::EuclideanDistance(path[i], path[i+1]);
        // assert(d < 1001.0);
		double dx = (path[i+1].x-path[i].x) / d;
		double dy = (path[i+1].y-path[i].y) / d;
		double sx = path[i].x;
		double sy = path[i].y;
		while(t < ti+d) {
			double u = t - ti;
			double nx = sx + dx*u;
			double ny = sy + dy*u;

			if (p.size()==0 || nx != p.back().x || ny != p.back().y)
				p.push_back(COORD(nx, ny));

			t += step;

			if (p.size()* ModelParams::PATH_STEP >= max_len)
				break;
		}

		ti += d;
	}
	p.push_back(path[path.size()-1]);
	return p;
}

void Path::CutJoin(const Path& p) {
	int i = max(0, Nearest(p[0])-1);
	erase(begin()+i, end());
	insert(end(), p.begin()/*+1*/, p.end());
}

double Path::GetLength(int start) const{
    auto& path = *this;

	double path_len = 0;
	if (path.size() ==0)
		return 0.0;

	for(int i=start; i<path.size()-1; i++) {
        double d = COORD::EuclideanDistance(path[i], path[i+1]);
        path_len += d;
	}
	return path_len;
}

double Path::GetTravelledLength(const COORD& car_pos) const{
    auto& path = *this;
    if (path.size() == 0)
    	return 0.0;
    int idx = path.Nearest(car_pos);
	double path_len = 0;
	for(int i=0; i<idx; i++) {
        double d = COORD::EuclideanDistance(path[i], path[i+1]);
        path_len += d;
	}
	return path_len;
}

Path Path::CopyWithoutTravelledPoints(double dist_to_remove){
	float d = 0;
	auto iter = begin();
	while (d< dist_to_remove && size() > 0){
		iter = iter + 1;
		d += ModelParams::PATH_STEP;
	}

	Path new_path;
	new_path.assign(iter, end());
	return new_path;
}

double Path::GetCurDir(int pos_along){
    auto& path = *this;

    int end_pos = min(pos_along + 150, int(path.size())-1);

	return COORD::SlopAngle(path[pos_along], path[end_pos]);
}

double CapAngle(double x){
    x = fmod(x,2*M_PI);
    if (x < 0)
        x += 2*M_PI;
    return x;
}

COORD Path::GetCrossDir(int pos_along, bool dir){
	auto& path = *this;
	COORD& pos = path[pos_along];
	double cur_dir = GetCurDir(pos_along);
	if (dir) // left, ccw
		cur_dir = cur_dir + M_PI/2;
	else // right.cw
		cur_dir = cur_dir - M_PI/2; 

	cur_dir = CapAngle(cur_dir);

	return COORD(cos(cur_dir), sin(cur_dir));
}

void Path::Text(){
    auto& path = *this;
	cout << "Path: ";
    for (auto point: path){
    	cout << point.x << " " << point.y << " ";
    }
    cout << endl;
}

Path Path::TruncateHead(const COORD& car_pos) const {
	auto& path = *this;
	Path new_path;

	if (path.size() == 0) {
		new_path.assign(begin(), end());
		return new_path;
	} else {
		int idx = path.Nearest(car_pos);
		new_path.assign(begin() + idx, end());
		return new_path;
	}
}

//void Path::ShortText(){
//    auto& path = *this;
//    tout << "path: size=" << path.size() << " length=" << path.GetLength() << endl;
//}
