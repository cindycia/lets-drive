#ifndef GLOBALS_H
#define GLOBALS_H

#include <typeinfo>
#include <memory>
#include <set>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <list>
#include <algorithm>
#include <ctime>
#include <vector>
#include <queue>
#include <cmath>
#include <cassert>
#include <limits>
#include <sstream>
#include <map>
#include <inttypes.h>
#include <despot/config.h>

#include <despot/util/exec_tracker.h>
#include <despot/util/logging.h>

namespace despot {

typedef uint64_t OBS_TYPE;
typedef int ACT_TYPE;

namespace Globals {
extern const double NEG_INFTY;
extern const double POS_INFTY;
extern const double INF;
extern const double TINY;

extern Config config;
extern ExecTracker tracker;

inline bool Fequals(double a, double b) {
	return std::fabs(a - b) < TINY;
}

inline double Discount() {
	return config.discount;
}

inline double Discount(int d) {
	return std::pow(config.discount, d);
}

inline void Track(std::string addr, std::string loc) {
	tracker.Track(addr, loc);
}

inline void Untrack(std::string addr) {
	tracker.Untrack(addr);
}

inline void PrintLocs() {
	tracker.PrintLocs();
}
} // namespace

/*================ For Hyp-despot ===============*/
enum MEMORY_MODE { INIT, ALLOC_ROOT, ALLOC, RESET, DESTROY};

enum DRIVING_MODE {NO,
	IMITATION,
	LETS_DRIVE,
	LETS_DRIVE_ZERO,
	JOINT_POMDP,
	ROLL_OUT,
	IMITATION_EXPLORE,
	LETS_DRIVE_LABELLER,
	JOINT_POMDP_LABELLER};

enum OBS_PARAM
{
	OBS_LONG64,
	OBS_INT_ARRAY,
};

enum {
		RWD_TOTAL,
		RWD_NCOL,
		RWD_COL,
		RWD_NUM_FACTORS
	};


extern OBS_PARAM Obs_type;

} // namespace despot


#endif
