#include <despot/interface/default_policy.h>
#include <despot/interface/pomdp.h>
#include <unistd.h>
#include <despot/GPUcore/thread_globals.h>

using namespace std;

using namespace despot;
#include "threaded_print.h"

namespace despot {

/* =============================================================================
 * DefaultPolicy class
 * =============================================================================*/

DefaultPolicy::DefaultPolicy(const DSPOMDP* model, ParticleLowerBound* particle_lower_bound) :
	ScenarioLowerBound(model),
	particle_lower_bound_(particle_lower_bound) {
	assert(particle_lower_bound_ != NULL);
}

DefaultPolicy::~DefaultPolicy() {
}

ValuedAction DefaultPolicy::Value(const vector<State*>& particles,
	RandomStreams& streams, History& history) const {

	logd << "[DefaultPolicy::Value] start rollout at depth " << history.Size() << endl;

	vector<State*> copy;
	for (int i = 0; i < particles.size(); i++)
		copy.push_back(model_->Copy(particles[i]));

	initial_depth_ = history.Size();
	ValuedAction va = RecursiveValue(copy, streams, history);

	for (int i = 0; i < copy.size(); i++)
		model_->Free(copy[i]);

	logd << "[DefaultPolicy::Value] rollout value " << va.value << endl;

	return va;
}


FactoredValuedAction DefaultPolicy::FactoredValue(const vector<State*>& particles,
	RandomStreams& streams, History& history) const {

	logd << "[DefaultPolicy::FactoredValue] start rollout at depth " << history.Size() << endl;

	vector<State*> copy;
	for (int i = 0; i < particles.size(); i++)
		copy.push_back(model_->Copy(particles[i]));

	initial_depth_ = history.Size();
	FactoredValuedAction va = RecursiveFactoredValue(copy, streams, history);

	for (int i = 0; i < copy.size(); i++)
		model_->Free(copy[i]);

	logd << "[DefaultPolicy::FactoredValue] rollout value "
			<< va.value[RWD_TOTAL] << " "
			<< va.value[RWD_NCOL] << " "
			<< va.value[RWD_COL] << " "
			" num_pariticles " << particles.size() << endl;

	return va;
}

ValuedAction DefaultPolicy::RecursiveValue(const vector<State*>& particles,
	RandomStreams& streams, History& history) const {
	if (streams.Exhausted()
		|| (history.Size() - initial_depth_
			>= Globals::config.max_policy_sim_len)) {

		auto blb_value = particle_lower_bound_->Value(particles);

		logd << "[DefaultPolicy::RecursiveValue] rollout blb value: " << blb_value.value << endl;

		return blb_value;
	} else {
		ACT_TYPE action = Action(particles, streams, history);
    
    // if (history.Size() == initial_depth_) 
      // cout << "roll out default action at initial depth " << action << endl;

		double value = 0;

		map<OBS_TYPE, vector<State*> > partitions;
		OBS_TYPE obs;
		double reward;
		for (int i = 0; i < particles.size(); i++) {
			State* particle = particles[i];
			bool terminal = model_->Step(*particle,
				streams.Entry(particle->scenario_id), action, reward, obs);

			if(terminal){
				logd << "rollout terminal reward " << reward << endl;
			}

			value += reward * particle->weight;

			if (!terminal) {
				partitions[obs].push_back(particle);
			}
		}

	    if(DoPrintCPU) printf("action, ave_reward= %d %f\n",action,value);


		for (map<OBS_TYPE, vector<State*> >::iterator it = partitions.begin();
			it != partitions.end(); it++) {
			OBS_TYPE obs = it->first;
			history.Add(action, obs);
			streams.Advance();
			ValuedAction va = RecursiveValue(it->second, streams, history);
			value += Globals::Discount() * va.value;
			streams.Back();
			history.RemoveLast();
		}

		return ValuedAction(action, value);
	}
}


FactoredValuedAction DefaultPolicy::RecursiveFactoredValue(const vector<State*>& particles,
	RandomStreams& streams, History& history) const {
	if (streams.Exhausted()
		|| (history.Size() - initial_depth_
			>= Globals::config.max_policy_sim_len)) {

		FactoredValuedAction blb_value = particle_lower_bound_->FactoredValue(particles);

		logd << "[DefaultPolicy::RecursiveFactoredValue] rollout blb value: " << blb_value.value << endl;

		return blb_value;
	} else {
		ACT_TYPE action = Action(particles, streams, history);

    // if (history.Size() == initial_depth_)
      // cout << "roll out default action at initial depth " << action << endl;

		double value[3] = {0,0,0};

		map<OBS_TYPE, vector<State*> > partitions;
		OBS_TYPE obs;
		double reward[3];
		for (int i = 0; i < particles.size(); i++) {
			State* particle = particles[i];
			bool terminal = model_->Step(*particle,
				streams.Entry(particle->scenario_id), action, reward, obs);

			if(terminal){
				logd << "rollout terminal reward " << reward << endl;
			}

			for (int i=0; i<model_->NumRewardFactors(); i++)
				value[i] += reward[i] * particle->weight;

			if (!terminal) {
				partitions[obs].push_back(particle);
			}
		}

	    if(DoPrintCPU) printf("action, ave_reward= %d %f\n",action,value[RWD_TOTAL]);

		for (map<OBS_TYPE, vector<State*> >::iterator it = partitions.begin();
			it != partitions.end(); it++) {
			OBS_TYPE obs = it->first;
			history.Add(action, obs);
			streams.Advance();
			FactoredValuedAction va = RecursiveFactoredValue(it->second, streams, history);
			for (int i=0; i<model_->NumRewardFactors(); i++)
				value[i] += Globals::Discount() * va.value[i];
			streams.Back();
			history.RemoveLast();
		}

		return FactoredValuedAction(action, value);
	}
}

void DefaultPolicy::Reset() {
}

ParticleLowerBound* DefaultPolicy::particle_lower_bound() const {
	return particle_lower_bound_;
}

} // namespace despot
