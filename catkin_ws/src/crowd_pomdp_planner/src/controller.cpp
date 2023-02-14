#include <csignal>
#include <time.h>
#include <boost/bind.hpp>
#include <unistd.h>

#include <core/node.h>
#include <core/solver.h>
#include <core/globals.h>
#include <despot/util/logging.h>

#include "world_simulator.h"
#include "crowd_belief.h"
#include "neural_prior.h"
#include "threaded_print.h"

#include "controller.h"

using namespace std;
using namespace despot;

int Controller::b_drive_mode = 0;
int Controller::gpu_id = 0;
int Controller::summit_port = 2000;
float Controller::time_scale = 1.0;

std::string Controller::model_file_ = "";
std::string Controller::value_model_file_ = "";
std::string Controller::map_location = "";
bool path_missing = true;

static ACT_TYPE action = (ACT_TYPE) (-1);
static OBS_TYPE obs = (OBS_TYPE) (-1);

bool predict_peds = true;

struct my_sig_action {
	typedef void (*handler_type)(int, siginfo_t*, void*);

	explicit my_sig_action(handler_type handler) {
		memset(&_sa, 0, sizeof(struct sigaction));
		_sa.sa_sigaction = handler;
		_sa.sa_flags = SA_SIGINFO;
	}

	operator struct sigaction const*() const {
		return &_sa;
	}
protected:
	struct sigaction _sa;
};

struct div_0_exception {
};

void handle_div_0(int sig, siginfo_t* info, void*) {
	switch (info->si_code) {
	case FPE_INTDIV:
		cout << "Integer divide by zero." << endl;
		break;
	case FPE_INTOVF:
		cout << "Integer overflow. " << endl;
		break;
	case FPE_FLTUND:
		cout << "Floating-point underflow. " << endl;
		break;
	case FPE_FLTRES:
		cout << "Floating-point inexact result. " << endl;
		break;
	case FPE_FLTINV:
		cout << "Floating-point invalid operation. " << endl;
		break;
	case FPE_FLTSUB:
		cout << "Subscript out of range. " << endl;
		break;
	case FPE_FLTDIV:
		cout << "Floating-point divide by zero. " << endl;
		break;
	case FPE_FLTOVF:
		cout << "Floating-point overflow. " << endl;
		break;
	};
	exit(-1);
}


bool file_exists (const std::string& name) {
    return ( access( name.c_str(), F_OK ) != -1 );
}


Controller::Controller(ros::NodeHandle& _nh, bool fixed_path) :
		nh_(_nh), last_action_(-1), last_obs_(-1), model_(NULL), prior_(NULL), ped_belief_(
				NULL), context_pomdp_(NULL), summit_driving_simulator_(NULL) {
	my_sig_action sa(handle_div_0);
	if (0 != sigaction(SIGFPE, sa, NULL)) {
		std::cerr << "!!!!!!!! fail to setup segfault handler !!!!!!!!"
				<< std::endl;
	}

	control_freq_ = ModelParams::CONTROL_FREQ;

	logi << " Controller constructed at the " << Globals::ElapsedTime()
			<< "th second" << endl;
}


void Controller::ModelFileCallback(const std_msgs::String::ConstPtr& file){
	 model_file_ = std::string(file->data.c_str());
	 cout << "Get model path: " << model_file_ << endl;
}

void Controller::ValModelFileCallback(const std_msgs::String::ConstPtr& file){
	value_model_file_ = std::string(file->data.c_str());
	 cout << "Get value model path: " << value_model_file_ << endl;
}


DSPOMDP* Controller::InitializeModel(option::Option* options) {
	cerr << "DEBUG: Initializing model" << endl;
	DSPOMDP* model = new ContextPomdp();
	model_ = model;
	context_pomdp_ = static_cast<ContextPomdp*>(model);

	return model;
}

void Controller::CreateNNPriors(DSPOMDP* model) {
	logv << "DEBUG: Creating solver prior " << endl;

	if (Globals::config.use_multi_thread_) {
		SolverPrior::nn_priors.resize(Globals::config.NUM_THREADS);
	} else
		SolverPrior::nn_priors.resize(1);

	for (int i = 0; i < SolverPrior::nn_priors.size(); i++) {
		logv << "DEBUG: Creating prior " << i << endl;
		SolverPrior::nn_priors[i] =
				static_cast<ContextPomdp*>(model)->CreateSolverPrior(
						summit_driving_simulator_, "NEURAL", false);
		SolverPrior::nn_priors[i]->prior_id(i);
		if (Globals::config.use_prior) {
			if(value_model_file_.find(".pt") == std::string::npos) {
				cerr << "value_model_file_=" << value_model_file_<< endl;
				ERR("Invalid value model file!");
			}
			if(model_file_.find(".pt") == std::string::npos) {
				cerr << "model_file_=" << model_file_<< endl;
				ERR("Invalid model file!");
			}

			if (!SolverPrior::disable_value)
				static_cast<PedNeuralSolverPrior*>(SolverPrior::nn_priors[i])->Load_value_model(
						value_model_file_);
			static_cast<PedNeuralSolverPrior*>(SolverPrior::nn_priors[i])->Load_model(
					model_file_);

			if (Globals::config.use_prior){
				logd << "[" << __FUNCTION__<< "] Testing model start" << endl;
				static_cast<PedNeuralSolverPrior*>(SolverPrior::nn_priors[i])->Test_model("");
				logd << "[" << __FUNCTION__<< "] Testing model end" << endl;
			}
		}	
	}

	prior_ = SolverPrior::nn_priors[0];
	logv << "DEBUG: Created solver prior " << typeid(*prior_).name() << "at ts "
			<< Globals::ElapsedTime() << endl;
}

World* Controller::InitializeWorld(std::string& world_type, DSPOMDP* model,
		option::Option* options) {
	cerr << "DEBUG: Initializing world" << endl;

	World* world = new WorldSimulator(nh_, static_cast<DSPOMDP*>(model),
			Globals::config.root_seed, map_location, summit_port);
	logi << "WorldSimulator constructed at the " << Globals::ElapsedTime()
			<< "th second" << endl;

	if (Globals::config.useGPU) {
		model->InitGPUModel();
		logi << "InitGPUModel finished at the " << Globals::ElapsedTime()
				<< "th second" << endl;
	}

	summit_driving_simulator_ = static_cast<WorldSimulator*>(world);
	summit_driving_simulator_->time_scale = time_scale;

	world->Connect();
	logi << "Connect finished at the " << Globals::ElapsedTime() << "th second"
			<< endl;

	CreateNNPriors(model);
	logi << "CreateNNPriors finished at the " << Globals::ElapsedTime()
			<< "th second" << endl;

	world->Initialize();
	logi << "Initialize finished at the " << Globals::ElapsedTime()
			<< "th second" << endl;

	return world;
}


void Controller::InitializeDefaultParameters() {
	cerr << "DEBUG: Initializing parameters" << endl;
	Globals::config.root_seed = time(NULL);
	Globals::config.time_per_move = (1.0 / ModelParams::CONTROL_FREQ) * 0.9
			/ time_scale;
	Globals::config.time_scale = time_scale;
	Globals::config.sim_len = 600;
	Globals::config.xi = 0.97;
	Globals::config.GPUid = gpu_id;
	Globals::config.use_multi_thread_ = true;
	Globals::config.exploration_mode = UCT;
	Globals::config.exploration_constant_o = 1.0;
	Globals::config.experiment_mode = true;
	Globals::config.state_source = STATE_FROM_SIM;

	Obs_type = OBS_INT_ARRAY;
	DESPOT::num_Obs_element_in_GPU = 1 + ModelParams::N_PED_IN * 2 + 3;
	Globals::config.useGPU = false;

	//if (!file_exists(Controller::model_file_) && b_drive_mode != JOINT_POMDP_LABELLER)
	//	b_drive_mode = JOINT_POMDP;

	if (!file_exists(Controller::model_file_) && b_drive_mode != JOINT_POMDP_LABELLER && b_drive_mode != JOINT_POMDP)
		ERR("Policy model file does not exist for non-POMDP mode!!");

	if (b_drive_mode == LETS_DRIVE || b_drive_mode == LETS_DRIVE_LABELLER) {
		Globals::config.use_prior = true;
		Globals::config.close_loop_prior = false;
	} else if (b_drive_mode == LETS_DRIVE_ZERO) {
		Globals::config.use_prior = true;
		Globals::config.close_loop_prior = true;
	}
	else
		Globals::config.use_prior = false;

	int search_depth_with_prior = 5;

	if (b_drive_mode == JOINT_POMDP || b_drive_mode == ROLL_OUT) {
		Globals::config.num_scenarios = 5;
		Globals::config.NUM_THREADS = 10;
		Globals::config.discount = 0.95;
		Globals::config.search_depth = 11;
		Globals::config.max_policy_sim_len = 11;
		if (b_drive_mode == JOINT_POMDP)
			Globals::config.pruning_constant = 0.001;
		else if (b_drive_mode == ROLL_OUT)
			Globals::config.pruning_constant = 100000000.0;
		Globals::config.exploration_constant = 0.1;
		Globals::config.silence = true;
	}
	else if (b_drive_mode == JOINT_POMDP_LABELLER) {
		Globals::config.num_scenarios = 5;
		Globals::config.NUM_THREADS = 10;
		Globals::config.discount = 0.95;
		Globals::config.search_depth = 11;
		Globals::config.max_policy_sim_len = 11;
		if (b_drive_mode == JOINT_POMDP)
			Globals::config.pruning_constant = 0.001;
		else if (b_drive_mode == ROLL_OUT)
			Globals::config.pruning_constant = 100000000.0;
		Globals::config.exploration_constant = 0.1;
		Globals::config.silence = true;
		Globals::config.state_source = STATE_FROM_TOPIC;
	}
	else if (b_drive_mode == LETS_DRIVE) {
		Globals::config.num_scenarios = 5;
		Globals::config.NUM_THREADS = 10;
		Globals::config.discount = 0.95;
		Globals::config.search_depth = search_depth_with_prior;
		Globals::config.max_policy_sim_len = 11;
		Globals::config.pruning_constant = 0.001;
		Globals::config.exploration_constant = 2 * ModelParams::REWARD_FACTOR_VEL;
		Globals::config.silence = true;
		SolverPrior::disable_value = false;
		SolverPrior::disable_policy_net = false;
		SolverPrior::prior_min_depth = search_depth_with_prior;

	} else if (b_drive_mode == LETS_DRIVE_ZERO) {
		Globals::config.num_scenarios = 5;
		Globals::config.NUM_THREADS = 10;
		Globals::config.discount = 0.95;
		Globals::config.search_depth = search_depth_with_prior;
		Globals::config.max_policy_sim_len = 11;
		Globals::config.pruning_constant = 0.001;
		Globals::config.exploration_constant = 2 * ModelParams::REWARD_FACTOR_VEL;
		Globals::config.silence = true;
		SolverPrior::disable_value = false;
		SolverPrior::disable_policy_net = false;
		SolverPrior::prior_min_depth = search_depth_with_prior;

	} else if (b_drive_mode == LETS_DRIVE_LABELLER) {
		Globals::config.num_scenarios = 5;
		Globals::config.NUM_THREADS = 10;
		Globals::config.discount = 0.95;
		Globals::config.search_depth = search_depth_with_prior;
		Globals::config.max_policy_sim_len = 11;
		Globals::config.pruning_constant = 0.001;
		Globals::config.exploration_constant = 2 * ModelParams::REWARD_FACTOR_VEL;
		Globals::config.silence = true;
		Globals::config.state_source = STATE_FROM_TOPIC;
		SolverPrior::disable_value = false;
		SolverPrior::disable_policy_net = false;
		SolverPrior::prior_min_depth = search_depth_with_prior;

	} else if (b_drive_mode == IMITATION || b_drive_mode == IMITATION_EXPLORE) {

		Globals::config.use_prior = true;
		Globals::config.close_loop_prior = false;
		Globals::config.num_scenarios = 1;
		Globals::config.NUM_THREADS = 1;
		Globals::config.search_depth = 0; // no search
		SolverPrior::disable_value = false;
		SolverPrior::disable_policy_net = false;
	}

	if (!file_exists(Controller::value_model_file_))
		SolverPrior::disable_value = true;

	cout << "Using value network? ";
	if (SolverPrior::disable_value)
		cout << "No." << endl;
	else
		cout << "Yes." << endl;

	logging::level(3);

	logi << "Planner default parameters:" << endl;
	Globals::config.text();
}

std::string Controller::ChooseSolver() {
	return "DESPOT";
}

Controller::~Controller() {

}

bool Controller::RunPreStep(Solver* solver, World* world, Logger* logger) {
	cerr << "DEBUG: Running pre step" << endl;

	logger->CheckTargetTime();

	cerr << "DEBUG: Pre-updating world state" << endl;

	auto start_t = Time::now();

	State* cur_state = world->GetCurrentState();
	if (!cur_state)
		ERR(string_sprintf("cur state NULL"));

	cerr << "DEBUG: Pre-updating belief" << endl;

	ped_belief_->Update(last_action_, cur_state);
	ped_belief_->Text(cout);

	UpdatePriors(cur_state, context_pomdp_->CopyForSearch(cur_state));

	logi << "[RunStep] Time spent in Update(): "
			<< Globals::ElapsedTime(start_t) << endl;

	return true;
}

void Controller::TrackStateInNNPriors(State* state) {
	SolverPrior::nn_priors[0]->Add_tensor_hist(state);
	for (int i = 0; i < SolverPrior::nn_priors.size(); i++) {
		if (i > 0) {
			SolverPrior::nn_priors[i]->add_map_tensor(
					SolverPrior::nn_priors[0]->last_map_tensor());
			SolverPrior::nn_priors[i]->add_semantic(
					SolverPrior::nn_priors[0]->last_semantic());
		}
	}
}

void Controller::LoadNNPriorsFromTopic() {
	for (int i = 0; i < SolverPrior::nn_priors.size(); i++) {
		SolverPrior::nn_priors[i]->Set_tensor_hist(
				SolverPrior::nn_priors[0]->unlabelled_hist_images_);
		SolverPrior::nn_priors[i]->Set_semantic_hist(
				SolverPrior::nn_priors[0]->unlabelled_semantic_);
	}
}

void Controller::UpdatePriors(const State* cur_state, State* search_state) {
	logd << "[UpdatePriors]" << endl;
	TrackStateInNNPriors(search_state);

	for (int i = 0; i < SolverPrior::nn_priors.size(); i++) {
		SolverPrior::nn_priors[i]->CompareHistoryWithRecorded();
		SolverPrior::nn_priors[i]->Add(last_action_, cur_state);
		SolverPrior::nn_priors[i]->Add_in_search(-1, search_state);

		logv << __FUNCTION__ << " add history search state of ts "
				<< static_cast<PomdpState*>(search_state)->time_stamp
				<< " hist len " << SolverPrior::nn_priors[i]->Size(true)
				<< endl;

		if (SolverPrior::nn_priors[i]->Size(true) == 10)
			Record_debug_state(search_state);

		int cur_search_hist_len = SolverPrior::nn_priors[i]->Size(true);
		int cur_tensor_hist_len = SolverPrior::nn_priors[i]->Tensor_hist_size();

		if (Globals::config.use_prior
				&& cur_search_hist_len != cur_tensor_hist_len)
			ERR(string_sprintf("Prior %d state hist length %d mismatch with tensor hist length %d",
					i, cur_search_hist_len, cur_tensor_hist_len));

		SolverPrior::nn_priors[i]->RecordCurHistory();
	}
	logi << "history len = " << SolverPrior::nn_priors[0]->Size(false) << endl;
	logi << "history_in_search len = " << SolverPrior::nn_priors[0]->Size(true)
			<< endl;
	if (Globals::config.use_prior)
		logi << "tensor len = " << SolverPrior::nn_priors[0]->Tensor_hist_size() << endl;
}

static int sample(std::vector<float> probs) {
	double SMOOTH = 0.0;
	double r = Random::RANDOM.NextDouble() + SMOOTH * probs.size();
	double accum_prob = 0.0;
	for (int idx=0; idx < probs.size(); idx++) {
		accum_prob += probs[idx] + SMOOTH;
		if (accum_prob >= r - 1e-5)
			return idx;
	}
	ERR("Sampling error");
}

void Controller::PreprocessParticles(std::vector<State*>& particles) {
	if (logging::level() >= logging::INFO) {
		logi << "Planning for POMDP state:" << endl;
		static_cast<ContextPomdp*>(model_)->PrintWorldState(
				*static_cast<const PomdpStateWorld*>(particles[0]));
	}
	// print future predictions for visualization purpose
	static_cast<const ContextPomdp*>(model_)->ForwardAndVisualize(particles[0],
			10);
	// predict particles and dump into belief for search
	for (int i = 0; i < particles.size(); i++) {
		particles[i] = static_cast<const ContextPomdp*>(model_)->PredictAgents(
				static_cast<const PomdpState*>(particles[i]), 0);
	}
}

bool Controller::RunStep(despot::Solver* solver, World* world, Logger* logger) {
	double step_start_t = get_time_second();
	tout << "DEBUG: Running step" << endl;

	logger->CheckTargetTime();
	std::vector<State*> particles;
	int cur_search_hist_len;
	auto start_t = Time::now();

	if (Globals::config.state_source == STATE_FROM_SIM) {

		cerr << "DEBUG: Getting world state" << endl;
		const State* cur_state = world->GetCurrentState();
		if (cur_state == NULL)
			ERR("cur_state is NULL");

		UpdatePriors(cur_state, context_pomdp_->CopyForSearch(cur_state));
		cur_search_hist_len = SolverPrior::nn_priors[0]->Size(true);

		cerr << "DEBUG: Updating belief" << endl;
		ped_belief_->Update(last_action_, cur_state);
		ped_belief_->Text(cout);

		particles = ped_belief_->Sample(Globals::config.num_scenarios * 2);
		PreprocessParticles(particles);
		
		cerr << "Number of planner agents: " << static_cast<PomdpState*>(particles[0])->num << endl;

	} else if (Globals::config.state_source == STATE_FROM_TOPIC) {
		const State* cur_state = world->GetCurrentState(); // just to initialize path

		ros::spinOnce();
		LoadNNPriorsFromTopic();

		particles = SolverPrior::nn_priors[0]->unlabelled_belief_;
		Globals::config.num_scenarios = particles.size();

		int i = 0;
		for (State* s: particles) {
			s->weight = 1.0/ Globals::config.num_scenarios;
			s->scenario_id = i;
			i++;
		}
		PreprocessParticles(particles);
	}

	ParticleBelief particle_belief(particles, model_);
	solver->belief(&particle_belief);
	assert(solver->belief());
	logi << "[RunStep] Time spent in Update(): "
			<< Globals::ElapsedTime(start_t) << endl;

	summit_driving_simulator_->ResetWorldModel();

	cerr << "DEBUG: Searching for action" << endl;
	start_t = Time::now();
	ACT_TYPE action =
			context_pomdp_->GetActionID(0.0, 0.0);
	double step_reward;

	if (b_drive_mode == NO || b_drive_mode == JOINT_POMDP  || b_drive_mode == JOINT_POMDP_LABELLER
			|| b_drive_mode == ROLL_OUT) {

		const State& sample = *particles[0];

		context_pomdp_->PrintStateIDs(sample);
		context_pomdp_->CheckPreCollision(&sample);

		action = solver->Search().action;
	} else if (b_drive_mode == LETS_DRIVE || b_drive_mode == LETS_DRIVE_ZERO ||
			b_drive_mode == LETS_DRIVE_LABELLER) {
		cerr << "DEBUG: Launch search with NN prior" << endl;
		action = solver->Search().action;

		cout << "recording SolverPrior::nn_priors[0]->searched_action" << endl;
		SolverPrior::nn_priors[0]->searched_action = action;
	} else if (b_drive_mode == IMITATION) {
		solver->Search().action; // just to query NNs at root

		std::vector<double> nn_action_probs = SolverPrior::nn_priors[0]->GetPolicyProbs();

		auto minmax = std::minmax_element(nn_action_probs.begin(), nn_action_probs.end());

		action = minmax.second - nn_action_probs.begin();
		logi << "sampled max-likelihood action " << action << " from root policy " << nn_action_probs << endl;

	} else if (b_drive_mode == IMITATION_EXPLORE) {
		solver->Search().action; // just to query NNs at root
		std::vector<double> nn_action_probs = SolverPrior::nn_priors[0]->GetPolicyProbs();

		double rand = Random::RANDOM.NextDouble();
		double accum_prob = 0.0;
		for (int act=0; act < model_->NumActions(); act++) {
			int actionID_intensor = SolverPrior::nn_priors[0]->ConvertToNNID(act);
			accum_prob += nn_action_probs[actionID_intensor];

//			tout << "accum_prob=" << accum_prob <<", act="<<act<< endl;
			if (accum_prob > rand - 1e-5) {
				action = act;
				break;
			}
		}

		tout << "sampled random action " << action << " from root policy " << nn_action_probs << endl;

	} else
		ERR("drive mode not supported!");
	logi << "[RunStep] Time spent in " << typeid(*solver).name()
			<< "::Search(): " << Globals::ElapsedTime(start_t) << endl;

	bool terminal;
	OBS_TYPE obs;

	if (Globals::config.state_source == STATE_FROM_SIM) {
		TruncPriors(cur_search_hist_len);

		cerr << "DEBUG: Executing action" << endl;
		start_t = Time::now();
		terminal = world->ExecuteAction(action, obs);
		last_action_ = action;
		last_obs_ = obs;
		logi << "[RunStep] Time spent in ExecuteAction(): "
				<< Globals::ElapsedTime(start_t) << endl;
	} else if (Globals::config.state_source == STATE_FROM_TOPIC) {
		start_t = Time::now();
		terminal = world->ExecuteAction(action, obs);
		last_action_ = action;
		last_obs_ = obs;
		logi << "[RunStep] Time spent in ExecuteAction(): "
				<< Globals::ElapsedTime(start_t) << endl;
	}

	tout << "DEBUG: Ending step" << endl;
	return logger->SummarizeStep(step_++, round_, terminal, action, obs,
			step_start_t);
}

void Controller::TruncPriors(int cur_tensor_hist_len) {
	logd << "[TruncPriors]" << endl;
	for (int i = 0; i < SolverPrior::nn_priors.size(); i++) {
		SolverPrior::nn_priors[i]->Truncate(cur_tensor_hist_len, true);
		logd << __FUNCTION__ << " truncating search history length to "
				<< cur_tensor_hist_len << endl;
		SolverPrior::nn_priors[i]->CompareHistoryWithRecorded();
	}

	for (int i = 0; i < SolverPrior::nn_priors.size(); i++) {
		SolverPrior::nn_priors[i]->Trunc_tensor_hist(cur_tensor_hist_len);
	}
}

static int wait_count = 0;

void Controller::PlanningLoop(despot::Solver*& solver, World* world,
		Logger* logger) {

	logi << "Planning loop started at the " << Globals::ElapsedTime()
			<< "th second" << endl;

	if (Globals::config.state_source == STATE_FROM_SIM) {

		ros::spinOnce();

		logi << "First ROS spin finished at the " << Globals::ElapsedTime()
				<< "th second" << endl;

		int pre_step_count = 0;
		if (Globals::config.use_prior)
			pre_step_count = 4;
		else
			pre_step_count = 4;

		while (SolverPrior::nn_priors[0]->Size(true) < pre_step_count) {
			logi << "Executing pre-step" << endl;
			RunPreStep(solver, world, logger);
			ros::spinOnce();

			logi << "sleeping for " << 1.0 / control_freq_ / time_scale << "s"
					<< endl;
			Globals::sleep_ms(1000.0 / control_freq_ / time_scale);

			logi << "Pre-step sleep end" << endl;
		}

		logi << "Pre-steps end at the " << Globals::ElapsedTime()
				<< "th second" << endl;

		logi << "Executing first step" << endl;

		RunStep(solver, world, logger);
		logi << "First step end at the " << Globals::ElapsedTime() << "th second"
				<< endl;

		cerr << "DEBUG: before entering controlloop" << endl;
		while (true) {
			RunStep(solver, world, logger);
		}
	} else if (Globals::config.state_source == STATE_FROM_TOPIC) {
		double wait_time = 0.0;
		bool started = false;
		while (true) {
			ros::spinOnce();
			if (summit_driving_simulator_->topic_state_ready && summit_driving_simulator_->car_data_ready) {
				RunStep(solver, world, logger);
				summit_driving_simulator_->topic_state_ready = false;
				summit_driving_simulator_->car_data_ready = false;
				wait_time = 0.0;
				started = true;
			}
			else {
				auto start = Time::now();
				summit_driving_simulator_->TriggerUnlabelledBeliefTopic();
			    boost::this_thread::sleep( boost::posix_time::milliseconds(10) );
			    wait_time += Globals::ElapsedTime(start);;
			}
			if (Globals::ElapsedTime() > 100.0)
				ERR("Reach 100s max time");
			if (wait_time > 20.0 && started)
				ERR("Waited for more than 20s");
		}
	}
//	timer_ = nh_.createTimer(ros::Duration(1.0 / control_freq_ / time_scale),
//			(boost::bind(&Controller::RunStep, this, solver, world, logger)));
}

void Controller::ChooseNNModels() {
	if (false && Globals::config.close_loop_prior) {
		auto file = ros::topic::waitForMessage<std_msgs::String>(
				"policy_net_save_path", ros::Duration(15));
		if (file) {
			model_file_ = file->data;
			logi << "policy_net_save_path get at the " << Globals::ElapsedTime()
					<< "th second" << endl;
		} else
			ERR("policy_net_save_path not received after 15 seconds");

		file = ros::topic::waitForMessage<std_msgs::String>(
				"value_net_save_path", ros::Duration(15));
		if (file) {
			value_model_file_ = file->data;
			logi << "value_net_save_path get at the " << Globals::ElapsedTime()
					<< "th second" << endl;
		} else
			ERR("value_net_save_path not received after 15 seconds");

		model_name_Sub_ = nh_.subscribe("policy_net_save_path", 1,
				&Controller::ModelFileCallback, this);
		val_model_name_Sub_ = nh_.subscribe("value_net_save_path", 1,
				&Controller::ValModelFileCallback, this);
	}
}

int Controller::RunPlanning(int argc, char *argv[]) {
	cerr << "DEBUG: Starting planning" << endl;

	/* =========================
	 * initialize parameters
	 * =========================*/
	string solver_type = "DESPOT";
	bool search_solver;
	int num_runs = 1;
	string world_type = "pomdp";
	string belief_type = "DEFAULT";
	int time_limit = -1;

	option::Option *options = InitializeParamers(argc, argv, solver_type,
			search_solver, num_runs, world_type, belief_type, time_limit);
	if (options == NULL)
		return 0;
	logi << "InitializeParamers finished at the " << Globals::ElapsedTime()
			<< "th second" << endl;

	ChooseNNModels();

	if (Globals::config.useGPU)
		PrepareGPU();

	clock_t main_clock_start = clock();

	/* =========================
	 * initialize model
	 * =========================*/
	DSPOMDP *model = InitializeModel(options);
	assert(model != NULL);
	logi << "InitializeModel finished at the " << Globals::ElapsedTime()
			<< "th second" << endl;

	/* =========================
	 * initialize world
	 * =========================*/
	World *world = InitializeWorld(world_type, model, options);

	cerr << "DEBUG: End initializing world" << endl;
	assert(world != NULL);
	logi << "InitializeWorld finished at the " << Globals::ElapsedTime()
			<< "th second" << endl;

	/* =========================
	 * initialize belief
	 * =========================*/

	cerr << "DEBUG: Initializing belief" << endl;
	Belief* belief = model->InitialBelief(world->GetCurrentState(),
			belief_type);
	assert(belief != NULL);
	ped_belief_ = static_cast<CrowdBelief*>(belief);
	logi << "InitialBelief finished at the " << Globals::ElapsedTime()
			<< "th second" << endl;

	/* =========================
	 * initialize solver
	 * =========================*/
	cerr << "DEBUG: Initializing solver" << endl;

	solver_type = ChooseSolver();
	Solver *solver = InitializeSolver(model, NULL, solver_type,
			options);
	logi << "InitializeSolver finished at the " << Globals::ElapsedTime()
			<< "th second" << endl;

	/* =========================
	 * initialize logger
	 * =========================*/
	Logger *logger = NULL;
	InitializeLogger(logger, options, model, belief, solver, num_runs,
			main_clock_start, world, world_type, time_limit, solver_type);
	//world->world_seed(world_seed);

	/* =========================
	 * Display parameters
	 * =========================*/
	DisplayParameters(options, model);

	/* =========================
	 * run planning
	 * =========================*/
	cerr << "DEBUG: Starting rounds" << endl;
	logger->InitRound(world->GetCurrentState());
	round_ = 0;
	step_ = 0;
	logi << "InitRound finished at the " << Globals::ElapsedTime()
			<< "th second" << endl;

	PlanningLoop(solver, world, logger);
//	ros::spin();

	logger->EndRound();

	PrintResult(1, logger, main_clock_start);

	return 0;
}
