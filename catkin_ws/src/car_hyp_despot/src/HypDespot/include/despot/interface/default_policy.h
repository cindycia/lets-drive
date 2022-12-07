#ifndef POLICY_H
#define POLICY_H

#include <vector>

#include <despot/random_streams.h>
#include <despot/interface/lower_bound.h>
#include <despot/util/random.h>
#include <despot/core/history.h>

#include <string.h>
#include <queue>
#include <vector>
#include <stdlib.h>
#include <despot/core/globals.h>
#include <despot/interface/pomdp.h>

namespace despot {

class State;
class StateIndexer;
class StatePolicy;
class DSPOMDP;
class MMAPInferencer;

/* =============================================================================
 * ValuedAction struct
 * =============================================================================*/

struct ValuedAction {
	ACT_TYPE action;
	double value;

	ValuedAction();
	ValuedAction(ACT_TYPE _action, double _value);

	friend std::ostream& operator<<(std::ostream& os, const ValuedAction& va);
};


struct FactoredValuedAction {
	ACT_TYPE action;
	double value[RWD_NUM_FACTORS];

	FactoredValuedAction();
	FactoredValuedAction(ACT_TYPE _action, double _value[]);
	FactoredValuedAction(ACT_TYPE _action, std::initializer_list<double> _value);

	friend std::ostream& operator<<(std::ostream& os, const FactoredValuedAction& va);
};

/* =============================================================================
 * DefaultPolicy class
 * =============================================================================*/
/**
 * [Optional]
 * Design your custom default policy by inheriting this class
 */
class DefaultPolicy: public ScenarioLowerBound {
private:
	mutable int initial_depth_;
	ParticleLowerBound* particle_lower_bound_;

	ValuedAction RecursiveValue(const std::vector<State*>& particles,
		RandomStreams& streams, History& history) const;

	FactoredValuedAction RecursiveFactoredValue(const std::vector<State*>& particles,
			RandomStreams& streams, History& history) const;
public:
	DefaultPolicy(const DSPOMDP* model, ParticleLowerBound* particle_lower_bound);
	virtual ~DefaultPolicy();

	void Reset();

	/**
	 * Returns an action based on the weighted scenarios and the history
	 *
	 * @param particles States in the head of the scenarios
	 * @param streams   Random streams attached to the scenarios
	 * @param history   The current action-observation history
	 */
	virtual ACT_TYPE Action(const std::vector<State*>& particles, RandomStreams& streams,
		History& history) const = 0;

	ParticleLowerBound* particle_lower_bound() const;

	ValuedAction Value(const std::vector<State*>& particles, RandomStreams& streams,
		History& history) const;

	FactoredValuedAction FactoredValue(const std::vector<State*>& particles, RandomStreams& streams,
			History& history) const;
};

} // namespace despot

#endif
