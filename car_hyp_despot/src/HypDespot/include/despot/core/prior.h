#ifndef PRIOR_H
#define PRIOR_H

#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
using namespace std;

using namespace despot;
#include <despot/planner.h>

/* =============================================================================
 * SolverPrior class
 * =============================================================================*/

class SolverPrior {
protected:
	const DSPOMDP* model_;
	ActionStateHistory as_history_;
	VariableActionStateHistory as_history_in_search_;
	std::vector<double> action_probs_;

	std::vector<double> root_action_probs_;
	double root_value_;

	int prior_id_;

public:
	static std::vector<SolverPrior*> nn_priors;
	static double prior_discount_optact;
	static bool prior_force_steer;
	static bool prior_force_acc;
	static bool disable_policy_net;
	static bool disable_value;
	static bool use_factored_value;
	static double despot_value_col_factor;
	static double despot_value_ncol_factor;
	static int prior_min_depth;
	static int belief_record_min_depth;
	std::vector<State*> unlabelled_belief_;
	std::vector<torch::Tensor> unlabelled_hist_images_;
	std::vector<float> unlabelled_semantic_;
	int unlabelled_belief_depth_;

public:
	ACT_TYPE searched_action;
	ACT_TYPE default_action;

public:
	SolverPrior(const DSPOMDP* model) :
			model_(model), searched_action(-1), default_action(-1), prior_id_(
					-1) {
	}
	virtual ~SolverPrior() {
		;
	}

	const std::vector<double>& action_probs() const;

	inline virtual const ActionStateHistory& history() const {
		return as_history_;
	}

	inline virtual VariableActionStateHistory& history_in_search() {
		return as_history_in_search_;
	}

	inline virtual void history_in_search(VariableActionStateHistory h) {
		as_history_in_search_ = h;
	}

	inline virtual void history(ActionStateHistory h) {
		as_history_ = h;
	}

	inline const std::vector<const State*>& history_states() {
		return as_history_.states();
	}

	inline std::vector<State*>& history_states_for_search() {
		return as_history_in_search_.states();
	}

	inline void prior_id(int id) {
		prior_id_ = id;
	}

	inline int prior_id() {
		return prior_id_;
	}

	inline virtual int SmartCount(ACT_TYPE action) const {
		return 10;
	}

	inline virtual double SmartValue(ACT_TYPE action) const {
		return 1;
	}

	inline virtual void Add(ACT_TYPE action, const State* state) {
		as_history_.Add(action, state);
	}
	inline virtual void Add_in_search(ACT_TYPE action, State* state) {
		as_history_in_search_.Add(action, state);
	}

	inline virtual void PopLast(bool insearch) {
		(insearch) ?
				as_history_in_search_.RemoveLast() : as_history_.RemoveLast();
	}

	inline virtual void PopAll(bool insearch) {
		(insearch) ?
				as_history_in_search_.Truncate(0) : as_history_.Truncate(0);
	}

	inline void Truncate(int d, bool insearch) {
		(insearch) ?
				as_history_in_search_.Truncate(d) : as_history_.Truncate(d);
	}

	inline size_t Size(bool insearch) const {
		size_t s =
				(insearch) ? as_history_in_search_.Size() : as_history_.Size();
		return s;
	}


public:
	virtual std::vector<ACT_TYPE> ComputeLegalActions(Shared_VNode* vnode, const State* state,
			const DSPOMDP* model) = 0;
	virtual void DebugHistory(string msg) = 0;
	virtual void RecordCurHistory()=0;
	virtual void CompareHistoryWithRecorded()=0;
public:
	virtual void root_car_pos(double x, double y) = 0;
	virtual void Record_hist_len() = 0;
	virtual void print_prior_actions(ACT_TYPE) = 0;
	virtual void Clear_hist_timestamps() = 0;
	
	virtual std::vector<torch::Tensor> Process_node_input(despot::VNode* cur_node, bool record=false) = 0;
	virtual std::vector<torch::Tensor> Process_semantic_input(std::vector<despot::VNode*>& cur_node,
			bool record = false, int child_id = -1) = 0;
	virtual std::vector<torch::Tensor> Process_nodes_input(const std::vector<despot::VNode*>& vnodes,
			const std::vector<State*>& vnode_states, bool record = false, int child_id = -1) = 0;

	virtual bool Compute(vector<torch::Tensor>& images, vector<torch::Tensor>& semantic, vector<despot::VNode*>& vnode) =0;
	virtual void ComputePreference(vector<torch::Tensor>& images, vector<torch::Tensor>& semantic, vector<despot::VNode*>& vnode) =0;
	virtual void ComputeValue(vector<torch::Tensor>& images, vector<torch::Tensor>& semantic, vector<despot::VNode*>& vnode) =0;
	
	virtual double GetValue() = 0;
	virtual std::vector<double> GetPolicyProbs() = 0;
	virtual bool policy_ready() = 0;
	virtual int ConvertToNNID(int) = 0;

	virtual at::Tensor Process_track_state_to_map_tensor(const State* s) = 0;
	virtual at::Tensor Process_tracked_state_to_car_tensor(const State* s) = 0;

	virtual at::Tensor last_car_tensor() = 0;
	virtual void add_car_tensor(at::Tensor) = 0;

	virtual double last_semantic() = 0;
	virtual void add_semantic(double) = 0;

	virtual at::Tensor last_map_tensor() = 0;
	virtual void add_map_tensor(at::Tensor) = 0;

	virtual void Add_tensor_hist(const State* s) = 0;
	virtual void Trunc_tensor_hist(int size) = 0;

	virtual void Set_tensor_hist(std::vector<torch::Tensor>&) = 0;
	virtual void Set_semantic_hist(std::vector<float>&) = 0;

	virtual int Tensor_hist_size() = 0;

	virtual void root_input(at::Tensor& images) = 0;
	virtual void root_semantic_input(at::Tensor& semantic) = 0;

	virtual at::Tensor root_semantic_input() = 0;
	virtual at::Tensor root_input() = 0;

	virtual void SetDefaultPriorPolicy(std::vector<despot::VNode*>& vnodes) = 0;
	virtual void SetDefaultPriorValue(std::vector<despot::VNode*>& vnodes) = 0;

	virtual ACT_TYPE SamplePriorAction(despot::VNode*) = 0;

	virtual void CleanUnlabelledBelief() = 0;
};

void Debug_state(State* state, std::string msg, const DSPOMDP* model);
void Record_debug_state(State* state);
int encode_vel(double cur_vel);
bool detectNAN(at::Tensor tensor);


double noncol_value_transform(double raw_value);
double col_value_transform(double raw_value);

double noncol_value_inv_transform(double value);
double col_value_inv_transform(double value);


#define SRV_DATA_TYPE float
#define TORCH_DATA_TYPE at::kFloat
//#define CV_DATA_TYPE CV_8UC1
#define CV_DATA_TYPE CV_32FC1
#define MAP_INTENSITY 255
#define NUM_CHANNELS 5
#define NUM_HIST_CHANNELS 4
#define IMSIZE 64
#define NUM_DOWNSAMPLE 4

#endif
