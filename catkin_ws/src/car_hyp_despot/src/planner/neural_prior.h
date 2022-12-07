/*
 * neural_prior.h
 *
 *  Created on: Dec 11, 2018
 *      Author: panpan
 */

#ifndef NEURAL_PRIOR_H_
#define NEURAL_PRIOR_H_

#include <despot/core/prior.h>

#include "opencv4/opencv2/imgproc.hpp"
#include "opencv4/opencv2/imgcodecs.hpp"
#include "opencv4/opencv2/highgui.hpp"


#include "disabled_util.h"
#include "despot/interface/pomdp.h"
#include <despot/core/mdp.h>
#include "despot/core/globals.h"
#include "despot/util/coord.h"

#include "param.h"
#include "state.h"
#include "world_model.h"
#include <cmath>
#include <utility>
#include <string>
#include "math_utils.h"
#include <nav_msgs/OccupancyGrid.h>
#include <vector>
#include "despot/core/node.h"

#include <ros/ros.h>
#include <msg_builder/TensorData.h>
#include <msg_builder/TensorDataHybrid.h>

using namespace cv;

class ContextPomdp;

//#ifndef __CUDACC__

struct CoordFrame {
	COORD center;
	COORD origin;
	COORD x_axis;
	COORD y_axis;
};

class PedNeuralSolverPrior:public SolverPrior{

	WorldModel& world_model;
//	cuda::GpuMat gImage;

	COORD point_to_indices(COORD pos, const CoordFrame& coord_frame, double resolution, int dim) const;
	COORD point_to_indices_unbounded(COORD pos, const CoordFrame& coord_frame, double resolution) const;
	void add_in_map(cv::Mat map_tensor, COORD indices, double map_intensity, double map_intensity_scale);

	CoordFrame select_coord_frame(const CarStruct& car);

	std::vector<COORD> get_image_space_car(const CarStruct& car, CoordFrame& frame, double resolution);
	std::vector<COORD> get_image_space_car_state(const CarStruct& car, CoordFrame& frame,
			double resolution, double dim);
	std::vector<COORD> get_image_space_agent(const AgentStruct agent, CoordFrame& frame, double resolution, double dim);

	void Process_states(std::vector<despot::VNode*> nodes, const std::vector<PomdpState*>& hist_states, const std::vector<int> hist_ids);

	void Process_image_to_tensor(cv::Mat& src_image, at::Tensor& des_tensor, std::string flag);
	void Process_exo_agent_images(const std::vector<PomdpState*>& hist_states,
			const std::vector<int>& hist_ids);
	void Process_ego_car_images(const std::vector<PomdpState*>& hist_states,
			const std::vector<int>& hist_ids);
	void Compute_pref(torch::Tensor input_tensor, torch::Tensor semantic_tensor, const ContextPomdp* ped_model, std::vector<despot::VNode*>& vnodes);
	void Compute_pref_hybrid(torch::Tensor input_tensor, torch::Tensor semantic_tensor, const ContextPomdp* ped_model, std::vector<despot::VNode*>& vnodes);
	bool Compute_val(torch::Tensor input_tensor, torch::Tensor semantic_tensor, const ContextPomdp* ped_model, std::vector<despot::VNode*>& vnodes);
	bool Compute_val_refracted(torch::Tensor input_tensor, torch::Tensor semantic_tensor, const ContextPomdp* ped_model, std::vector<despot::VNode*>& vnodes);
	void Compute_pref_libtorch(torch::Tensor input_tensor, torch::Tensor semantic_tensor, const ContextPomdp* ped_model, std::vector<despot::VNode*>& vnodes);
	int ConvertToNNID(int accID);
	void RecordUnlabelledBelief(despot::VNode* cur_node);
	void CleanUnlabelledBelief();

	void RecordUnlabelledHistImages();

	enum UPDATE_MODES { FULL=0, PARTIAL=1 };

	enum {
		VALUE, ACC_PI, ACC_MU, ACC_SIGMA, ANG, VEL_PI, VEL_MU, VEL_SIGMA,
		                   CAR_VALUE_0, RES_IMAGE_0
	};

	struct map_properties{

		double resolution;
		COORD origin;
		int dim;
		double map_intensity;
		int new_dim; // down sampled resolution

		double map_intensity_scale;
		double downsample_ratio;
	};

private:
	at::Tensor root_input_;
	at::Tensor root_semantic_input_;

	at::Tensor empty_map_tensor_;
	at::Tensor map_tensor_;
	std::vector<at::Tensor> map_hist_tensor_;
	std::vector<at::Tensor> car_hist_tensor_;
	at::Tensor path_tensor;
	at::Tensor lane_tensor;

	std::vector<const despot::VNode*> map_hist_links;
	std::vector<const despot::VNode*> car_hist_links;
	std::vector<double> hist_time_stamps;

	const despot::VNode* goal_link, *lane_link;

	cv::Mat map_image_;
	cv::Mat rescaled_map_;
	std::vector<cv::Mat> map_hist_images_;
	std::vector<cv::Mat> car_hist_images_;
	cv::Mat path_image_;
	cv::Mat lane_image_;

	std::vector<cv::Point3f> car_shape;

	COORD root_car_pos_;

	std::vector<at::Tensor> tracked_map_hist_;
	std::vector<double> tracked_semantic_hist_;
	std::vector<at::Tensor> car_hist_;

	cv::Mat map_hist_image_;
	cv::Mat car_hist_image_;
public:

	void root_input(at::Tensor& images) {
		root_input_ = images;
	}

	at::Tensor root_input() {
		return root_input_;
	}

	void root_semantic_input(at::Tensor& images) {
		root_semantic_input_ = images;
	}

	at::Tensor root_semantic_input() {
		return root_semantic_input_;
	}


	COORD root_car_pos(){return root_car_pos_;}

	void root_car_pos(double x, double y){
		root_car_pos_.x = x;
		root_car_pos_.y = y;
	}

	void Process_path_image(const PomdpState*);
	void Process_lane_image(const PomdpState*);

	at::Tensor Process_track_state_to_map_tensor(const State* s);
	at::Tensor Process_tracked_state_to_car_tensor(const State* s);
	at::Tensor Process_tracked_state_to_lane_tensor(const State* s);

	at::Tensor Process_lane_tensor(const State* s);
	at::Tensor Process_path_tensor(const State* s);

	at::Tensor last_car_tensor(){
		return car_hist_.back();
	}

	void add_car_tensor(at::Tensor t){
		car_hist_.push_back(t);
	}

	at::Tensor last_map_tensor(){
		return tracked_map_hist_.back();
	}

	void add_map_tensor(at::Tensor t){
		tracked_map_hist_.push_back(t);
	}

	double last_semantic() {
		return tracked_semantic_hist_.back();
	}

	void add_semantic(double vel) {
		tracked_semantic_hist_.push_back(vel);
	}

	void Add_tensor_hist(const State* s);
	void Trunc_tensor_hist(int size);

	void Set_tensor_hist(std::vector<torch::Tensor>&);
	void Set_semantic_hist(std::vector<float>&);

	int Tensor_hist_size();

public:

	PedNeuralSolverPrior(const DSPOMDP* model, WorldModel& world);
//	virtual const std::vector<double>& ComputePreference();

	void SetDefaultPriorPolicy(std::vector<despot::VNode*>& vnodes);
	void SetDefaultPriorValue(std::vector<despot::VNode*>& vnodes);

	ACT_TYPE SamplePriorAction(despot::VNode*);

	bool Compute(std::vector<torch::Tensor>& images, std::vector<torch::Tensor>& semantic, std::vector<despot::VNode*>& vnode);
	bool ComputeMiniBatch(std::vector<torch::Tensor>& images, std::vector<torch::Tensor>& semantic, std::vector<despot::VNode*>& vnode);

	void ComputeValue(std::vector<torch::Tensor>& images, std::vector<torch::Tensor>& semantic, std::vector<despot::VNode*>& vnode);
	void ComputeMiniBatchValue(std::vector<torch::Tensor>& images, std::vector<torch::Tensor>& semantic, std::vector<despot::VNode*>& vnode);

	void ComputePreference(std::vector<torch::Tensor>& images, std::vector<torch::Tensor>& semantic, std::vector<despot::VNode*>& vnode);
	void ComputeMiniBatchPref(std::vector<torch::Tensor>& images, std::vector<torch::Tensor>& semantic, std::vector<despot::VNode*>& vnode);

	void Update_prior_probs(at::Tensor& acc_probs_Tensor, at::Tensor& steer_probs_Tensor, despot::VNode* vnode);
	void Update_prior_probs(std::vector<float>& action_probs_Tensor, despot::VNode* vnode);

	std::vector<ACT_TYPE> ComputeLegalActions(despot::Shared_VNode* vnode, const State* state, const DSPOMDP* model);

	std::vector<torch::Tensor> Process_node_input(despot::VNode* cur_node, bool record);
	std::vector<torch::Tensor> Process_semantic_input(std::vector<despot::VNode*>& nodes,
			bool record_unlabelled=false, int record_child_id=-1);

	void Process_state(despot::VNode* cur_node);
	std::vector<torch::Tensor> Process_nodes_input(const std::vector<despot::VNode*>& vnodes,
			const std::vector<State*>& vnode_states, bool record_unlabelled=false, int record_child_id=-1);
//	at::Tensor Combine_images(const at::Tensor& node_image, const at::Tensor& hist_images){return torch::zeros({1,1,1});}
	torch::Tensor Combine_images(despot::VNode* cur_node);


	void get_history_settings(despot::VNode* cur_node, int mode, int &num_history, int &start_channel);
	void get_history_map_tensors(int mode, despot::VNode* cur_node);
	void Reuse_history(int new_channel, int start_channel, int mode);
	void get_history(int mode, despot::VNode* cur_node, std::vector<despot::VNode*>& parents_to_fix_images,
			std::vector<PomdpState*>& hist_states, std::vector<int>& hist_ids);
	std::vector<float> get_semantic_history(despot::VNode* cur_node);

	void Record_hist_len();

	void print_prior_actions(ACT_TYPE);

public:
	void Load_model(std::string);
	void Load_value_model(std::string);

	void Init();
	void Clear_hist_timestamps();

	void Test_model(std::string);
	void Test_all_srv(int batchsize, int num_acc_bins, int num_steer_bins);
	void Test_all_srv_hybrid(int batchsize, int num_guassian_modes, int num_steer_bins);
	void Test_all_libtorch(int batchsize, int num_guassian_modes, int num_steer_bins);
	void Test_val_libtorch(int batchsize, int num_guassian_modes, int num_steer_bins);
	void Test_val_libtorch_refracted(int batchsize, int num_guassian_modes, int num_steer_bins);
	void export_images(std::vector<despot::VNode*>& vnodes);

	bool query_srv(std::vector<despot::VNode*>& vnodes, at::Tensor images, at::Tensor semantic,
			at::Tensor& t_value, at::Tensor& t_acc, at::Tensor& t_ang);
	bool query_srv_hybrid(int batchsize, at::Tensor images, at::Tensor semantic, at::Tensor& t_value, at::Tensor& t_acc_pi,
			at::Tensor& t_acc_mu, at::Tensor& t_acc_sigma, at::Tensor& t_ang);

public:

	void DebugHistory(string msg);
	VariableActionStateHistory as_history_in_search_recorded;

	void RecordCurHistory();
	void CompareHistoryWithRecorded();

public:
	void update_ego_car_shape(std::vector<geometry_msgs::Point32> points);

public:

	bool policy_ready_;

	double GetValue() {return root_value_;}
	std::vector<double> GetPolicyProbs() {return root_action_probs_;}

	bool policy_ready() {return policy_ready_;}

public:
	int num_hist_channels;
	int num_peds_in_NN;
	nav_msgs::OccupancyGrid raw_map_;
	bool map_received;

	map_properties map_prop_;
	std::string model_file;
	std::string value_model_file;

	std::shared_ptr<torch::jit::script::Module> drive_net;
	std::shared_ptr<torch::jit::script::Module> drive_net_value;

	static ros::ServiceClient nn_client_;
	static ros::ServiceClient nn_client_val_;

};

//#endif



#endif /* NEURAL_PRIOR_H_ */
