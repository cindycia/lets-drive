#include <despot/interface/lower_bound.h>
#include <despot/interface/pomdp.h>
#include <despot/core/node.h>
#include <despot/solver/pomcp.h>

using namespace std;

namespace despot {

/* =============================================================================
 * ValuedAction class
 * =============================================================================*/

ValuedAction::ValuedAction() :
	action(-1),
	value(0) {
}

ValuedAction::ValuedAction(ACT_TYPE _action, double _value) :
	action(_action),
	value(_value) {
}

ostream& operator<<(ostream& os, const ValuedAction& va) {
	os << "(" << va.action << ", " << va.value << ")";
	return os;
}


/* =============================================================================
 * FactoredValuedAction class
 * =============================================================================*/

FactoredValuedAction::FactoredValuedAction() :
	action(-1) {
	for (int i=0;i<RWD_NUM_FACTORS;i++)
		value[i] = 0.0;
}

FactoredValuedAction::FactoredValuedAction(ACT_TYPE _action, double _value[]) :
	action(_action) {
	for (int i=0;i<RWD_NUM_FACTORS;i++)
		value[i] = _value[i];
}

FactoredValuedAction::FactoredValuedAction(ACT_TYPE _action, std::initializer_list<double> _value) :
	action(_action) {
	std::copy(_value.begin(), _value.end(), value);
}

ostream& operator<<(ostream& os, const FactoredValuedAction& va) {
	os << "(" << va.action << ", " << va.value << ")";
	return os;
}

/* =============================================================================
 * ScenarioLowerBound class
 * =============================================================================*/

ScenarioLowerBound::ScenarioLowerBound(const DSPOMDP* model) :
	model_(model){
}

void ScenarioLowerBound::Init(const RandomStreams& streams) {
}

void ScenarioLowerBound::Reset() {
}

void ScenarioLowerBound::Learn(VNode* tree) {
}

FactoredValuedAction ScenarioLowerBound::FactoredValue(
		const std::vector<State*>& particles, RandomStreams& streams,
		History& history) const {
	throw "this function should not be reached";
}
/* =============================================================================
 * ParticleLowerBound class
 * =============================================================================*/

ParticleLowerBound::ParticleLowerBound(const DSPOMDP* model) :
	ScenarioLowerBound(model) {
}

ValuedAction ParticleLowerBound::Value(const vector<State*>& particles,
	RandomStreams& streams, History& history) const {
	return Value(particles);
}

FactoredValuedAction ParticleLowerBound::FactoredValue(
		const std::vector<State*>& particles) const {
	throw "this function should not be reached";
}

/* =============================================================================
 * BeliefLowerBound class
 * =============================================================================*/

BeliefLowerBound::BeliefLowerBound(const DSPOMDP* model) :
	model_(model) {
}

void BeliefLowerBound::Learn(VNode* tree) {
}

} // namespace despot
