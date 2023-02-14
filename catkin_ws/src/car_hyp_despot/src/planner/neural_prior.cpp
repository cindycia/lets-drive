#define ONEOVERSQRT2PI 1.0 / sqrt(2.0 * M_PI)

#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#include "context_pomdp.h"
#include "neural_prior.h"
#include "threaded_print.h"

#undef LOG
#define LOG(lv) \
if (despot::logging::level() < despot::logging::ERROR || despot::logging::level() < lv) ; \
else despot::logging::stream(lv)
#include <despot/util/logging.h>
#include <cmath>

double value_normalizer = 1.0;
const char* window_name = "images";

torch::Device device(torch::kCPU);
bool use_gpu_for_nn = true;

bool do_print = false;

int init_hist_len = 0;
int max_retry_count = 3;

int export_image_level = 4;

ros::ServiceClient PedNeuralSolverPrior::nn_client_;

bool detectNAN(double v) {
	if (isnan(v))
		return true;
	else if (v != v) {
		return true;
	}

	return false;
}

bool detectNAN(at::Tensor tensor) {
	auto nan_sum = torch::isnan(tensor).sum();
	std::vector<long> v(
				nan_sum.data<long>(), nan_sum.data<long>() + nan_sum.numel());
	if (v[0] > 0) {
		tout << "NAN indices " << torch::isnan(tensor).nonzero();
		ERR("NAN in tensor");
		return true;
	}
	return false;
}

bool detectNAN(cv::Mat matrix) {

	cv::Mat mask = cv::Mat(matrix != matrix);
//	double min, max;
//	cv::minMaxLoc(mask, &min, &max);
	double sum = cv::sum(mask)[0];

	if (sum > 0)
		return true;
	else
		return false;
}

cv::Mat rescale_image(const cv::Mat& image) {
	logd << "[rescale_image]" << endl;
	auto start = Time::now();
	Mat result = image;
	Mat dst;
	for (int i = 0; i < NUM_DOWNSAMPLE; i++) {
		pyrDown(result, dst, Size(result.cols / 2, result.rows / 2));
		result = dst;
	}

	logd << __FUNCTION__ << " rescale " << Globals::ElapsedTime(start) << " s"
			<< endl;
	return result;
}

float radians(float degrees) {
	return (degrees * M_PI) / 180.0;
}

void fill_polygon_edges(Mat& image, std::vector<COORD>& points) {
	float default_intensity = 0.5;
	if (points.size() == 2)
		default_intensity = 1.0;

	logv << "image size " << image.size[0] << "," << image.size[0] << endl;

	for (int i = 0; i < points.size(); i++) {
		int r0, c0, r1, c1;
		r0 = round(points[i].x);
		c0 = round(points[i].y);
		if (i + 1 < points.size()) {
			r1 = round(points[i + 1].x);
			c1 = round(points[i + 1].y);
		} else {
			r1 = round(points[0].x);
			c1 = round(points[0].y);
		}

		logv << "drawing line from " << r0 << "," << c0 << " to " << r1 << ","
				<< c1 << endl;
		cv::line(image, Point(r0, c0), Point(r1, c1), default_intensity);
	}
}

int img_counter = 0;
std::string img_folder = "";

void mkdir_safe(std::string dir) {
	if (mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
		if ( errno == EEXIST) {
			// alredy exists
		} else {
			// something else
			std::cerr << "cannot create folder " << dir << " error:"
					<< strerror(errno) << std::endl;
			ERR("");
		}
	}
}

bool dir_exist(string pathname) {
	struct stat info;

	if (stat(pathname.c_str(), &info) != 0) {
		printf("cannot access %s\n", pathname);
		return false;
	} else if (info.st_mode & S_IFDIR) { // S_ISDIR() doesn't exist on my windows
//	    printf( "%s is a directory\n", pathname );
		return true;
	} else {
		printf("%s is no directory\n", pathname);
		return false;
	}
}

void rm_files_in_folder(string folder) {
	// These are data types defined in the "dirent" header
	DIR *theFolder = opendir(folder.c_str());
	struct dirent *next_file;
	char filepath[256];

	while ((next_file = readdir(theFolder)) != NULL) {
		// build the path for each file in the folder
		sprintf(filepath, "%s/%s", folder.c_str(), next_file->d_name);
		cout << "Removing file " << filepath << endl;
		remove(filepath);
	}
	closedir(theFolder);
}

void clear_image_folder() {
	// std::string homedir = getenv("HOME");
	// img_folder = homedir + "/catkin_ws/visualize";
	// mkdir_safe(img_folder);
	// rm_files_in_folder(img_folder);
}

void export_image(Mat& image, string flag) {
	// logi << "[export_image] start" << endl;
	// std::ostringstream stringStream;
	// stringStream << img_folder << "/" << img_counter << "_" << flag << ".jpg";
	// std::string img_name = stringStream.str();

	// Mat tmp = image.clone();

	// double image_min, image_max;
	// cv::minMaxLoc(tmp, &image_min, &image_max);
	// logi << "saving image " << img_name << " with min-max values: " << image_min
	// 		<< ", " << image_max << endl;

	// cv::Mat for_save;
	// tmp.convertTo(for_save, CV_8UC3, 255.0);
	// imwrite(img_name, for_save);

//	imshow( img_name, image );
//
//	char c = (char)waitKey(0);
//
//	cvDestroyWindow((img_name).c_str());
	// logi << "[export_image] end" << endl;
}

void inc_counter() {
	img_counter++;
}

void reset_counter() {
	img_counter = 0;
}

void normalize(cv::Mat& image) {
//	auto start = Time::now();
	logd << "[normalize_image]" << endl;

	double image_min, image_max;
	cv::minMaxLoc(image, &image_min, &image_max);
	if (image_max > 0)
		image = image / image_max * MAP_INTENSITY;

//    logd << __FUNCTION__ << " " << Globals::ElapsedTime(start) << " s" << endl;

}

void merge_images(cv::Mat& image_src, cv::Mat& image_des) {
//	auto start = Time::now();
	logd << "[merge_images]" << endl;

	image_des = cv::max(image_src, image_des);

//    logd << __FUNCTION__ << " " << Globals::ElapsedTime(start) << " s" << endl;
}

void copy_to_tensor(cv::Mat& image, at::Tensor& tensor) {
//	auto start = Time::now();
	logd << "[copy_to_tensor] copying to tensor" << endl;

	tensor = torch::from_blob(image.data, { image.rows, image.cols },
			TORCH_DATA_TYPE).clone();

	//    for(int i=0; i<image.rows; i++)
//        for(int j=0; j<image.cols; j++)
//        	tensor[i][j] = image.at<float>(i,j);

//    logd << __FUNCTION__ << " " << Globals::ElapsedTime(start) << " s" << endl;

}

void print_full(at::Tensor& tensor, std::string msg) {

	Globals::lock_process();
//	logi << "Tensor " << msg << endl;
	auto tensor_double = tensor.accessor<float, 2>();
	for (int i = 0; i < tensor.size(0); i++) {
		logi << msg << " ";

		for (int j = 0; j < tensor.size(1); j++) {
			logi << std::setprecision(2) << tensor_double[i][j] << " ";
		}
		logi << endl;
	}
	Globals::unlock_process();
}

bool is_file_exist(string fileName) {
	std::ifstream infile(fileName);
	return infile.good();
}

double value_transform_inverse(double value) {
	value = value * value_normalizer;
	return value;
}

int encode_vel(double cur_vel) {
	double vel = max(0.0, min(cur_vel, ModelParams::VEL_MAX_NET - 0.0001));
	vel = vel / ModelParams::VEL_MAX_NET;

	double onehot_resolution = 1.0 / ModelParams::NUM_VEL_BINS;
	int bin_idx = int(floor((vel / onehot_resolution)));
	return bin_idx;
}

at::Tensor gaussian_probability(at::Tensor &sigma, at::Tensor &mu,
		at::Tensor &data) {
	// data = data.toType(at::kDouble);
	// sigma = sigma.toType(at::kDouble);
	// mu = mu.toType(at::kDouble);
	// data = data.toType(at::kDouble);
	data = data.unsqueeze(1).expand_as(sigma);
//    std::logd << "data=" << data << std::endl;
//    std::logd << "mu=" << mu  << std::endl;
//    std::logd << "sigma=" << sigma  << std::endl;
//    std::logd << "data - mu=" << data - mu  << std::endl;

	auto exponent = -0.5 * at::pow((data - mu) / sigma, at::Scalar(2));
//    std::logd << "exponent=" << exponent << std::endl;
	auto ret = ONEOVERSQRT2PI * (exponent.exp() / sigma);
//    std::logd << "ret=" << ret << std::endl;
	return at::prod(ret, 2);
}

at::Tensor gm_pdf(at::Tensor &pi, at::Tensor &sigma, at::Tensor &mu,
		at::Tensor &target) {
//    std::logd << "pi=" << pi << std::endl;
//    std::logd << "sigma=" << sigma << std::endl;
//    std::logd << "mu=" << mu << std::endl;
//    std::logd << "target=" << target << std::endl;

	auto prob_double = pi * gaussian_probability(sigma, mu, target);
	auto prob_float = prob_double.toType(at::kFloat);
//    std::logd << "prob_float=" << prob_float << std::endl;
	auto safe_sum = at::add(at::sum(prob_float, at::IntList(1)),
			at::Scalar(0.000001));
	return safe_sum;
}

void Show_params(std::shared_ptr<torch::jit::script::Module> drive_net) {
	/*const auto& model_params = drive_net->get_modules();

	 int iter = 0;

	 ofstream fout;
	 string model_param_file = "/home/panpan/NN_params.txt";
	 fout.open(model_param_file, std::ios::trunc);
	 assert(fout.is_open());

	 for(const auto& module: model_params) {
	 // fout << module.name() << ": " << std::endl;

	 const auto& module_params = module.get_parameters();
	 fout << module_params.size() << " params found:" << std::endl;
	 for (const auto& param: module_params){
	 fout<< param.value() <<std::endl;
	 }

	 //		if (module_name == "ang_head" ){
	 fout << "module_name sub_modules: " << std::endl;

	 const auto& sub_modules = module.get_modules();
	 for(const auto& sub_module: sub_modules) {
	 // fout << "sub-module found " << sub_module.name() << ": " << std::endl;

	 const auto& sub_module_params = sub_module.get_parameters();
	 fout << sub_module_params.size() << " params found:" << std::endl;
	 for (const auto& param: sub_module_params){
	 fout<< param.value() <<std::endl;
	 }
	 }
	 //		}
	 iter ++;
	 //		if (iter == 20) break;
	 }

	 fout.close();*/
}

void PedNeuralSolverPrior::Init() {
	// DONE: The environment map will be received via ROS topic as the OccupancyGrid type
	//		 Data will be stored in raw_map_ (class member)
	//       In the current stage, just use a randomized raw_map_ to develop your code.
	//		 Init raw_map_ including its properties here
	//	     Refer to python codes: bag_2_hdf5.parse_map_data_from_dict
	//		 (map_dict_entry is the raw OccupancyGrid data)

//	if (Globals::config.use_prior) {
		cerr << "DEBUG: Initializing Map" << endl;

		map_prop_.downsample_ratio = 1.0 / pow(2.0, NUM_DOWNSAMPLE);
		map_prop_.resolution = 0.0390625;
		map_prop_.origin = COORD(-20.0,-20.0);
		map_prop_.dim = 1024;
		map_prop_.new_dim = (int) (map_prop_.dim * map_prop_.downsample_ratio);
		map_prop_.map_intensity_scale = 1500.0;

		cerr << "DEBUG: Initializing Map image" << endl;

		map_image_ = cv::Mat(map_prop_.dim, map_prop_.dim, CV_DATA_TYPE,
				cv::Scalar(0.0));
		map_prop_.map_intensity = MAP_INTENSITY;

		logi << "Neural prior constants:" << endl;
		logi << "=> IMSIZE = " << IMSIZE << endl;
		const SRV_DATA_TYPE dummy = 1;
		logi << "=> SRV_DATA_TYPE = " << typeid(dummy).name() << endl;
		logi << "=> NUM_DOWNSAMPLE = " << NUM_DOWNSAMPLE << endl;
		logi << "=> MAP_INTENSITY = " << MAP_INTENSITY << endl;
		logi << "=> NUM_CHANNELS = " << NUM_CHANNELS << endl;

		logd << "Map properties: " << endl;
		logd << "-dim " << map_prop_.dim << endl;
		logd << "-new_dim " << map_prop_.new_dim << endl;
		logd << "-downsample_ratio " << map_prop_.downsample_ratio << endl;
		logd << "-map_intensity " << map_prop_.map_intensity << endl;
		logd << "-map_intensity_scale " << map_prop_.map_intensity_scale
				<< endl;
		logd << "-resolution " << map_prop_.resolution << endl;

		cerr << "DEBUG: Scaling map" << endl;

		rescaled_map_ = cv::Mat(IMSIZE, IMSIZE, CV_DATA_TYPE, cv::Scalar(0.0));

		cerr << "DEBUG: Initializing other images" << endl;

		path_image_ = cv::Mat(map_prop_.dim, map_prop_.dim, CV_DATA_TYPE);
		lane_image_ = cv::Mat(map_prop_.dim, map_prop_.dim, CV_DATA_TYPE);

		for (int i = 0; i < num_hist_channels; i++) {
			map_hist_images_.push_back(
					cv::Mat(map_prop_.dim, map_prop_.dim, CV_DATA_TYPE));
		}

		map_hist_image_ = cv::Mat(map_prop_.dim, map_prop_.dim, CV_DATA_TYPE);

		cerr << "DEBUG: Initializing tensors" << endl;

		empty_map_tensor_ = at::zeros( { map_prop_.new_dim, map_prop_.new_dim },
				TORCH_DATA_TYPE);
		map_tensor_ = at::zeros( { map_prop_.new_dim, map_prop_.new_dim },
				TORCH_DATA_TYPE);
		path_tensor = torch::zeros( { map_prop_.new_dim, map_prop_.new_dim },
				TORCH_DATA_TYPE);
		lane_tensor = torch::zeros( { map_prop_.new_dim, map_prop_.new_dim },
				TORCH_DATA_TYPE);

		logd << "[PedNeuralSolverPrior::Init] create tensors of size "
				<< map_prop_.new_dim << "," << map_prop_.new_dim << endl;
		for (int i = 0; i < num_hist_channels; i++) {

			map_hist_tensor_.push_back(torch::zeros( { map_prop_.new_dim,
					map_prop_.new_dim }, TORCH_DATA_TYPE));
			map_hist_links.push_back(NULL);
			car_hist_links.push_back(NULL);
			hist_time_stamps.push_back(-1);
		}

		goal_link = NULL;
		lane_link = NULL;

		policy_ready_ = false;

		clear_image_folder();
		logd << "[PedNeuralSolverPrior::Init] end " << endl;
//	}
}

void PedNeuralSolverPrior::Clear_hist_timestamps() {
	for (int i = 0; i < num_hist_channels; i++) {
		hist_time_stamps[i] = -1;
	}
}
/*
 * Initialize the prior class and the neural networks
 */
PedNeuralSolverPrior::PedNeuralSolverPrior(const DSPOMDP* model,
		WorldModel& world) :
		SolverPrior(model), world_model(world) {

	prior_id_ = 0;
	logd << "DEBUG: Initializing PedNeuralSolverPrior" << endl;

	action_probs_.resize(model->NumActions());

	// TODO: get num_peds_in_NN from ROS param
	num_peds_in_NN = 20;
	// TODO: get num_hist_channels from ROS param
	num_hist_channels = 4;

	// DONE Declare the neural network as a class member, and load it here

	logd << "DEBUG: Initializing car shape" << endl;

	// Car geometry
	car_shape = std::vector<cv::Point3f>(
			{ Point3f(3.6, 0.95, 1), Point3f(-0.8, 0.95, 1), Point3f(-0.8,
					-0.95, 1), Point3f(3.6, -0.95, 1) });

	map_received = false;
	drive_net = NULL;

}

void PedNeuralSolverPrior::Load_model(std::string path) {
	if (model_file.find("pth") != std::string::npos) {
		cerr << "no pt policy model to load" << endl;
		return;
	}

	auto start = Time::now();

	torch::DeviceType device_type;

	if (torch::cuda::is_available()) {
		std::cerr << "CUDA available! NN on GPU" << std::endl;
		device_type = torch::kCUDA;
	} else {
		std::cerr << "NN on CPU" << std::endl;
		device_type = torch::kCPU;
	}
	device = torch::Device(device_type);

	model_file = path;

	cerr << "DEBUG: Loading model " << model_file << endl;

	// DONE: Pass the model name through ROS params
	int trial = 0;
	while (drive_net == NULL && trial < 3) {
		try {
			drive_net = std::make_shared<torch::jit::script::Module>(
				torch::jit::load(model_file));
			trial++;
		}
		catch(...) {
			drive_net = NULL;
		}
	}

	if (drive_net == NULL)
		ERR("");

	if (use_gpu_for_nn)
		drive_net->to(at::kCUDA);

	cerr << "DEBUG: Loaded model " << model_file << endl;

	logd << __FUNCTION__ << " " << Globals::ElapsedTime(start) << " s" << endl;
}

void PedNeuralSolverPrior::Load_value_model(std::string path) {
	auto start = Time::now();

	torch::DeviceType device_type;

	if (torch::cuda::is_available()) {
		std::cerr << "CUDA available! NN on GPU" << std::endl;
		device_type = torch::kCUDA;
	} else {
		std::cerr << "NN on CPU" << std::endl;
		device_type = torch::kCPU;
	}
	device = torch::Device(device_type);

	value_model_file = path;

	cerr << "DEBUG: Loading value model, file name: " << value_model_file
			<< endl;

	// DONE: Pass the model name through ROS params
	drive_net_value = std::make_shared<torch::jit::script::Module>(
			torch::jit::load(value_model_file));

	if (drive_net_value == NULL)
		ERR("");

	if (use_gpu_for_nn)
		drive_net_value->to(at::kCUDA);

	cerr << "DEBUG: Loaded model " << value_model_file << endl;

	logd << __FUNCTION__ << " " << Globals::ElapsedTime(start) << " s" << endl;
}

COORD PedNeuralSolverPrior::point_to_indices(COORD pos,
		const CoordFrame& coord_frame, double resolution, int dim) const {
	COORD dir = pos - coord_frame.center;
	double x_coord = dir.dot(coord_frame.x_axis);
	double y_coord = dir.dot(coord_frame.y_axis);

	COORD indices = COORD(
			(x_coord + ModelParams::IMAGE_HALF_SIZE_METER) / resolution,
			(y_coord + ModelParams::IMAGE_HALF_SIZE_METER) / resolution);
	if (indices.x < 0 || indices.y < 0 || indices.x > (dim - 1)
			|| indices.y > (dim - 1))
		return COORD(-1, -1);
	return indices;
}

COORD PedNeuralSolverPrior::point_to_indices_unbounded(COORD pos,
		const CoordFrame& coord_frame, double resolution) const {
	COORD dir = pos - coord_frame.center;
	double x_coord = dir.dot(coord_frame.x_axis);
	double y_coord = dir.dot(coord_frame.y_axis);

	COORD indices = COORD(
			(x_coord + ModelParams::IMAGE_HALF_SIZE_METER) / resolution,
			(y_coord + ModelParams::IMAGE_HALF_SIZE_METER) / resolution);

	return indices;
}

void PedNeuralSolverPrior::add_in_map(cv::Mat map_image, COORD indices,
		double map_intensity, double map_intensity_scale) {

	if (indices.x == -1 || indices.y == -1)
		return;

//	logd << "[add_in_map] " << endl;

	map_image.at<float>((int) round(indices.y), (int) round(indices.x)) =
			map_intensity * map_intensity_scale;

//	logd << "[add_in_map] fill entry " << round(indices.x) << " " << round(indices.y) << endl;

}

std::vector<COORD> PedNeuralSolverPrior::get_image_space_car_state(
		const CarStruct& car, CoordFrame& coord_frame, double resolution,
		double dim) {
	std::vector<COORD> image_space_line;
	COORD car_dir(car.vel * cos(car.heading_dir),
			car.vel * sin(car.heading_dir));
	COORD start = point_to_indices(car.pos, coord_frame, resolution, dim);
	image_space_line.push_back(start);
	COORD end = point_to_indices(car.pos + car_dir, coord_frame, resolution,
			dim);
	image_space_line.push_back(end);
	return image_space_line;
}

std::vector<COORD> PedNeuralSolverPrior::get_image_space_agent(
		const AgentStruct agent, CoordFrame& coord_frame, double resolution,
		double dim) {
	auto start = Time::now();

	float theta = agent.heading_dir;
	float x = agent.pos.x;
	float y = agent.pos.y;

//    agent.text(logd);
//    logd << "=========== transforming agent " << agent.id << "theta: " << theta << " x: " << x << " y: " << y << endl;

	if (agent.bb_extent_y == 0) {
		agent.Text(cerr);
		ERR("agent.bb_extent_y == 0");
	}

	logv << "raw agent shape: \n";
	std::vector<COORD> agent_shape(
			{ COORD(agent.bb_extent_y, agent.bb_extent_x), COORD(
					-agent.bb_extent_y, agent.bb_extent_x), COORD(
					-agent.bb_extent_y, -agent.bb_extent_x), COORD(
					agent.bb_extent_y, -agent.bb_extent_x) });
	for (int i = 0; i < agent_shape.size(); i++) {
		logv << agent_shape[i].x << " " << agent_shape[i].y << endl;
	}
	// rotate and scale the agent

	bool out_of_map = false;
	std::vector<COORD> image_space_polygon;
	for (auto &coord : agent_shape) {
		std::vector<Point3f> original, rotated;
		original.push_back(Point3f(coord.x, coord.y, 1.0));
		rotated.resize(1);
		// rotate w.r.t its local coordinate system and transform to (x, y)
		cv::transform(original, rotated,
				cv::Matx33f(cos(theta), -sin(theta), x, sin(theta), cos(theta),
						y, 0, 0, 1));

		COORD image_space_indices = point_to_indices(
				COORD(rotated[0].x, rotated[0].y), coord_frame, resolution,
				dim);
		image_space_polygon.push_back(image_space_indices);
		if (image_space_indices.x == -1 or image_space_indices.y == -1)
			out_of_map = true;
	}

	logv << "transformed: \n";
	for (int i = 0; i < image_space_polygon.size(); i++) {
		logv << image_space_polygon[i].x << " " << image_space_polygon[i].y
				<< endl;
	}

	logv << "image coord_frame origin " << coord_frame.center.x << " "
			<< coord_frame.center.y << endl;

	if (out_of_map) // agent out side of map should not be considered
		image_space_polygon.resize(0);

	logd << __FUNCTION__ << " " << Globals::ElapsedTime(start) << " s" << endl;

	return image_space_polygon;
}

CoordFrame PedNeuralSolverPrior::select_coord_frame(const CarStruct& car) {
	CoordFrame frame;
	COORD shift(ModelParams::IMAGE_HALF_SIZE_METER,
			ModelParams::IMAGE_HALF_SIZE_METER);
	frame.origin = car.pos - shift;
	frame.center = car.pos;
	frame.x_axis = COORD(cos(car.heading_dir), sin(car.heading_dir));
	frame.y_axis = COORD(-sin(car.heading_dir), cos(car.heading_dir));
	return frame;
}

std::vector<COORD> PedNeuralSolverPrior::get_image_space_car(
		const CarStruct& car, CoordFrame& coord_frame, double resolution) {
	auto start = Time::now();

	logd << "car bb: \n";
	for (int i = 0; i < car_shape.size(); i++) {
		logd << car_shape[i].x << " " << car_shape[i].y << endl;
	}
	// rotate and scale the car
	std::vector<COORD> car_polygon;
	for (auto& point : car_shape) {
		COORD indices = point_to_indices_unbounded(COORD(point.x, point.y),
				coord_frame, resolution);
		car_polygon.push_back(indices);
	}

	logd << "image space bb: \n";
	for (int i = 0; i < car_polygon.size(); i++) {
		logd << car_polygon[i].x << " " << car_polygon[i].y << endl;
	}

	logd << __FUNCTION__ << " " << Globals::ElapsedTime(start) << " s" << endl;

	return car_polygon;
}

void PedNeuralSolverPrior::Process_image_to_tensor(cv::Mat& src_image,
		at::Tensor& des_tensor, string flag) {
	auto start = Time::now();

	logd << "original image size: " << src_image.size[0] << " " << src_image.size[1] << endl;
	logd << "flag " << flag << endl;

	auto rescaled_image = rescale_image(src_image);

	double image_min, image_max;

	normalize(rescaled_image);

//	if(flag.find("map") != std::string::npos){
//
//		merge_images(rescaled_map_, rescaled_image);
//
//		logd << __FUNCTION__<<" merge " << Globals::ElapsedTime(start) << " s" << endl;
//	}
	if (detectNAN(rescaled_image))
		ERR("[Process_image_to_tensor] NAN in rescaled image");

	copy_to_tensor(rescaled_image, des_tensor);

	logd << __FUNCTION__ << " copy " << Globals::ElapsedTime(start) << " s"
			<< endl;
	logd << __FUNCTION__ << " " << Globals::ElapsedTime(start) << " s" << endl;

	if (detectNAN(des_tensor))
		ERR("[Process_image_to_tensor] NAN in converted tensor");

	if (logging::level() >= export_image_level + 1) {
		logd << "[Process_maps] des_tensor address " << &des_tensor << endl;
		export_image(rescaled_image, "Process_" + flag);
		inc_counter();
	}
}

void PedNeuralSolverPrior::Reuse_history(int new_channel, int start_channel,
		int mode) {

	logd << "[Reuse_history] copying data to channel " << new_channel << endl;
	assert(new_channel >= start_channel);
	int old_channel = new_channel - start_channel;

	logd << "[Reuse_history] copying data from " << old_channel
			<< " to new channel " << new_channel << endl;

	if (new_channel != old_channel) {
//		if (mode == FULL){
//
//			map_hist_images_[new_channel] = map_hist_images_[old_channel];
//			car_hist_images_[new_channel] = car_hist_images_[old_channel];
//		}

		map_hist_tensor_[new_channel] = map_hist_tensor_[old_channel];
//		car_hist_tensor_[new_channel] = car_hist_tensor_[old_channel];

		map_hist_links[new_channel] = map_hist_links[old_channel];
		car_hist_links[new_channel] = car_hist_links[old_channel];

		hist_time_stamps[new_channel] = hist_time_stamps[old_channel];
	} else {
		logd << "skipped" << endl;
	}
}

void PedNeuralSolverPrior::Process_ego_car_images(
		const std::vector<PomdpState*>& hist_states,
		const vector<int>& hist_ids) {
	// DONE: Allocate num_history history images, each for a frame of car state
	//		 Refer to python codes: bag_2_hdf5.get_transformed_car, fill_car_edges, fill_image_with_points
	// DONE: get_transformed_car apply the current transformation to the car bounding box
	//	     fill_car_edges fill edges of the car shape with dense points
	//		 fill_image_with_points fills the corresponding entries in the images (with some intensity value)
	for (int i = 0; i < hist_states.size(); i++) {
		int hist_channel = hist_ids[i];
		car_hist_images_[hist_channel].setTo(0.0);

		logd << "[Process_states] reseting image for car hist " << hist_channel
				<< endl;

		if (hist_states[i]) {
			logd << "[Process_states] processing car for hist " << hist_channel
					<< endl;

			CarStruct& car = hist_states[i]->car;
			auto coord_frame = select_coord_frame(car);
			std::vector<COORD> transformed_car = get_image_space_car(car,
					coord_frame, map_prop_.resolution);
			fill_polygon_edges(car_hist_images_[hist_channel], transformed_car);
		}
	}
}

void PedNeuralSolverPrior::Process_exo_agent_images(
		const std::vector<PomdpState*>& hist_states,
		const std::vector<int>& hist_ids) {
	for (int i = 0; i < hist_states.size(); i++) {

		int hist_channel = hist_ids[i];

		logd << "[Process_exo_agent_images] reseting image for map hist "
				<< hist_channel << endl;

		// clear data in the dynamic map
		map_hist_images_[hist_channel].setTo(0.0);

		// get the array of pedestrians (of length ModelParams::N_PED_IN)
		if (hist_states[i]) {
			logd << "[Process_exo_agent_images] start processing peds for "
					<< hist_states[i] << endl;
			//			pomdp_model->PrintState(*hist_states[i]);
			const CarStruct car = hist_states[i]->car;

			CoordFrame coord_frame = select_coord_frame(car);

			auto car_line_coords = get_image_space_car_state(car, coord_frame,
					map_prop_.resolution, map_prop_.dim);
			fill_polygon_edges(map_hist_images_[hist_channel], car_line_coords);

			auto& agent_list = hist_states[i]->agents;

			logd << "[Process_exo_agent_images] iterating peds in agent_list="
					<< &agent_list << endl;

			for (int agent_id = 0; agent_id < ModelParams::N_PED_IN;
					agent_id++) {
				// Process each pedestrian
				AgentStruct agent = agent_list[agent_id];

				if (agent.id != -1) {
					// get position of the ped
					auto image_space_coords = get_image_space_agent(agent,
							coord_frame, map_prop_.resolution, map_prop_.dim);

					// put the point in the dynamic map
					fill_polygon_edges(map_hist_images_[hist_channel],
							image_space_coords);
				}
			}
		}
		if (detectNAN(map_hist_images_[hist_channel]))
			ERR("NAN in map_hist_images_[hist_channel]");
	}
}

void PedNeuralSolverPrior::Process_states(std::vector<despot::VNode*> hist_nodes,
		const std::vector<PomdpState*>& hist_states,
		const std::vector<int> hist_ids) {

	if (hist_states.size() == 0) {
		tout << "[Process_states] skipping empty state list" << endl;
		return;
	}
	auto start_total = Time::now();

	const ContextPomdp* pomdp_model = static_cast<const ContextPomdp*>(model_);
	logd << "Processing states, len=" << hist_states.size() << endl;

	Process_lane_image(hist_states[0]);
	Process_image_to_tensor(lane_image_, lane_tensor, "lanes");
	lane_link = hist_nodes[0];

	Process_exo_agent_images(hist_states, hist_ids);
	for (int i = 0; i < hist_nodes.size(); i++) {
		int hist_channel = hist_ids[i];

		logd << "[Process_states] create new data for channel " << hist_channel
				<< " by node " << hist_nodes[i] << endl;

		Process_image_to_tensor(map_hist_images_[hist_channel], map_hist_tensor_[hist_channel],
				"map_" + std::to_string(hist_channel));
		static_cast<despot::Shared_VNode*>(hist_nodes[i])->map_tensor =
				map_hist_tensor_[hist_channel];

		assert(static_cast<despot::Shared_VNode*>(hist_nodes[i])->map_tensor.defined());
		map_hist_links[hist_channel] = hist_nodes[i];
		hist_time_stamps[hist_channel] = hist_states[i]->time_stamp;
	}

	logd << "[Process_states] done " << endl;
	logd << __FUNCTION__ << " " << Globals::ElapsedTime(start_total) << " s"
			<< endl;
}

at::Tensor PedNeuralSolverPrior::Process_tracked_state_to_car_tensor(
		const State* s) {
	const PomdpState* agent_state = static_cast<const PomdpState*>(s);

	car_hist_image_.setTo(0.0);

	if (agent_state) {
		logd << "[Process_states] processing car for state " << s << endl;

		const CarStruct& car = agent_state->car;
		auto coord_frame = select_coord_frame(car);
		std::vector<COORD> transformed_car = get_image_space_car(car,
				coord_frame, map_prop_.resolution);
		fill_polygon_edges(car_hist_image_, transformed_car);
	}

	at::Tensor car_tensor;
	Process_image_to_tensor(car_hist_image_, car_tensor,
			"car_state_" + std::to_string(long(s)));

	return car_tensor;
}

at::Tensor PedNeuralSolverPrior::Process_track_state_to_map_tensor(
		const State* s) {
	const PomdpState* agent_state = static_cast<const PomdpState*>(s);

	map_hist_image_.setTo(0.0);

	// get the array of pedestrians (of length ModelParams::N_PED_IN)
	if (agent_state) {
		logd << "[Process_states] start processing peds for " << agent_state
				<< endl;
		const auto& agent_list = agent_state->agents;
		int num_valid_ped = 0;
		const auto& car = agent_state->car;
		auto coord_frame = select_coord_frame(car);
		auto car_line_coords = get_image_space_car_state(car, coord_frame,
				map_prop_.resolution, map_prop_.dim);
		fill_polygon_edges(map_hist_image_, car_line_coords);

		logd << "[Process_states] iterating peds in agent_list=" << &agent_list
				<< endl;

		for (int agent_id = 0; agent_id < ModelParams::N_PED_IN; agent_id++) {
			// Process each pedestrian
			AgentStruct agent = agent_list[agent_id];

			if (agent.id != -1) {
				auto image_space_coords = get_image_space_agent(agent,
						coord_frame, map_prop_.resolution, map_prop_.dim);

				// put the point in the dynamic map
				fill_polygon_edges(map_hist_image_, image_space_coords);
			}
		}
	}

	at::Tensor map_tensor;
	Process_image_to_tensor(map_hist_image_, map_tensor,
			"map_state_" + std::to_string(long(s)));

	if (logging::level() >= export_image_level + 1) {
		export_image(map_hist_image_, "tracked_map");
		inc_counter();
	}
	return map_tensor;
}

at::Tensor PedNeuralSolverPrior::Process_tracked_state_to_lane_tensor(
		const State* s) {
	const PomdpState* agent_state = static_cast<const PomdpState*>(s);

	map_hist_image_.setTo(0.0);

	// get the array of pedestrians (of length ModelParams::N_PED_IN)
	if (agent_state != NULL) {
		Process_lane_image(agent_state);
	}

	at::Tensor lane_tensor;
	Process_image_to_tensor(lane_image_, lane_tensor,
			"lanes_state_" + std::to_string(long(s)));

	if (logging::level() >= export_image_level + 1) {
		export_image(lane_image_, "tracked_lanes");
	}

	return lane_tensor;
}

void PedNeuralSolverPrior::Process_lane_image(const PomdpState* agent_state) {
	lane_image_.setTo(0.0);

	auto& car = agent_state->car;
	auto coord_frame = select_coord_frame(car);

	auto& cur_car_pos = car.pos;

	logd << "[Process_lane_image] processing lane list of size "
			<< world_model.local_lane_segments_.size() << endl;

	auto& lanes = world_model.local_lane_segments_;
	if (lanes.size() == 0)
		ERR("No available lane in world_model.local_lane_segments_");

	for (auto& lane_seg : lanes) {
		COORD image_space_start = point_to_indices_unbounded(
				COORD(lane_seg.start.x, lane_seg.start.y), coord_frame,
				map_prop_.resolution);
		COORD image_space_end = point_to_indices_unbounded(
				COORD(lane_seg.end.x, lane_seg.end.y), coord_frame,
				map_prop_.resolution);

		std::vector<COORD> tmp_polygon( { image_space_start, image_space_end });

		//TODO: check whether out-of-bound indices cause unexpected errors!!!!

		fill_polygon_edges(lane_image_, tmp_polygon);
	}

	if (detectNAN(lane_image_))
		ERR("NAN in lane_image_");
	if (cv::sum(lane_image_)[0] < 0.99)
		ERR("No info in lane image");
}

at::Tensor PedNeuralSolverPrior::Process_lane_tensor(const State* s) {
	const PomdpState* agent_state = static_cast<const PomdpState*>(s);

	Process_lane_image(agent_state);

	Process_image_to_tensor(lane_image_, lane_tensor,
			"lanes_state_" + std::to_string(long(s)));

	return lane_tensor;
}

void PedNeuralSolverPrior::Process_path_image(const PomdpState* agent_state) {
	path_image_.setTo(0.0);

	// get distance between cur car pos and car pos at root node
	auto& car = agent_state->car;
	auto coord_frame = select_coord_frame(car);
	auto& cur_car_pos = car.pos;
	float trav_dist_since_root = (cur_car_pos - root_car_pos_).Length();

	logd << "[Process_states] processing path of size "
			<< world_model.path.size() << endl;

	// remove points in path according to the car moving distance
	Path path = world_model.path.CopyWithoutTravelledPoints(
			trav_dist_since_root);

	// use the trimmed path for the goal image
	logd << "[Process_states] after processing: path size " << path.size()
			<< endl;

	for (int i = 0; i < path.size(); i++) {
		COORD point = path[i];
		// process each point
		COORD indices = point_to_indices(point, coord_frame,
				map_prop_.resolution, map_prop_.dim);
		if (indices.x == -1 or indices.y == -1) // path point out of map
			continue;
		// put the point in the goal map
		path_image_.at<float>((int) round(indices.y), (int) round(indices.x)) =
				1.0 * 1.0;
	}
}

at::Tensor PedNeuralSolverPrior::Process_path_tensor(const State* s) {

	const PomdpState* agent_state = static_cast<const PomdpState*>(s);

	Process_path_image(agent_state);

//	at::Tensor path_tensor;
	Process_image_to_tensor(path_image_, path_tensor,
			"path_state_" + std::to_string(long(s)));

	return path_tensor;
}

void PedNeuralSolverPrior::Add_tensor_hist(const State* s) {
	auto state = static_cast<const PomdpState*>(s);
	tracked_map_hist_.push_back(Process_track_state_to_map_tensor(s));
	tracked_semantic_hist_.push_back(state->car.vel);
}

void PedNeuralSolverPrior::Trunc_tensor_hist(int size) {
	tracked_map_hist_.resize(size);
	tracked_semantic_hist_.resize(size);
}

void PedNeuralSolverPrior::Set_tensor_hist(std::vector<torch::Tensor>& from_topic) {
	tracked_map_hist_.resize(0);
	for (auto tensor: from_topic)
		tracked_map_hist_.push_back(tensor.clone());
}

void PedNeuralSolverPrior::Set_semantic_hist(std::vector<float>& from_topic) {
	tracked_semantic_hist_.resize(0);
	for (float vel: from_topic)
		tracked_semantic_hist_.push_back(vel);
}

int PedNeuralSolverPrior::Tensor_hist_size() {
	assert(tracked_semantic_hist_.size() == tracked_map_hist_.size());
	return tracked_map_hist_.size();
}

void PedNeuralSolverPrior::RecordUnlabelledHistImages() {
	unlabelled_hist_images_.resize(0);
	for (torch::Tensor t : map_hist_tensor_)
		unlabelled_hist_images_.push_back(t.clone());
}

std::vector<torch::Tensor> PedNeuralSolverPrior::Process_nodes_input(
		const std::vector<despot::VNode*>& vnodes,
		const std::vector<State*>& vnode_states,
		bool record_unlabelled, int record_child_id) {
	auto start = Time::now();

	logd << "[Process_nodes_input], num=" << vnode_states.size() << endl;

	std::vector<PomdpState*> cur_state;
	std::vector<torch::Tensor> output_images;
	cur_state.resize(1);
	std::vector<int> hist_ids( { 0 }); // use as the last hist step

	get_history_map_tensors(PARTIAL, vnodes[0]->parent()->parent());

	for (int i = 0; i < vnode_states.size(); i++) {

		logd << "[Process_nodes_input] node " << i << endl;
		cur_state[0] = static_cast<PomdpState*>(vnode_states[i]);

		logd << " Using current state of node "
				<< i << " depth " << vnodes[i]->depth() << " as channel 0"
				<< endl;

		Process_states(std::vector<despot::VNode*>( { vnodes[i] }), cur_state,
				hist_ids);

		auto node_nn_input = Combine_images(vnodes[i]);

		if (record_unlabelled && i == record_child_id) {
			logi << "recording search node particles" << endl;
			RecordUnlabelledBelief(vnodes[i]);
			RecordUnlabelledHistImages();
		}

		output_images.push_back(node_nn_input);
	}

	logd << __FUNCTION__ << " " << Globals::ElapsedTime(start) << " s" << endl;

	return output_images;
}

torch::Tensor PedNeuralSolverPrior::Combine_images(despot::VNode* cur_node) {
	auto start = Time::now();

	assert(path_tensor.defined());
	assert(lane_tensor.defined());
	for (at::Tensor& t : map_hist_tensor_) {
		assert(t.defined());
	}

	auto combined = map_hist_tensor_;
	combined.push_back(lane_tensor);
	torch::Tensor result = torch::stack(combined, 0);

	logd << __FUNCTION__ << " " << Globals::ElapsedTime(start) << " s" << endl;

	return result;
}

int BatchSize = 128;

bool PedNeuralSolverPrior::Compute(std::vector<torch::Tensor>& input_batch,
		std::vector<torch::Tensor>& semantic_batch,
		std::vector<despot::VNode*>& vnodes) {
	logd << "[Compute] get " << vnodes.size() << " nodes" << endl;
	auto start = Time::now();

	if (vnodes.size() > BatchSize) {
		logd << "Executing " << (vnodes.size() - 1) / BatchSize << " batches"
				<< endl;
		for (int batch = 0; batch < (vnodes.size() - 1) / BatchSize + 1;
				batch++) {

			int start = batch * BatchSize;
			int end =
					((batch + 1) * BatchSize <= vnodes.size()) ?
							(batch + 1) * BatchSize : vnodes.size();

			logd << "start end: " << start << " " << end << endl;

			std::vector<torch::Tensor> mini_batch(&input_batch[start],
					&input_batch[end]);
			std::vector<torch::Tensor> mini_sem_batch(&semantic_batch[start],
					&semantic_batch[end]);
			std::vector<despot::VNode*> sub_vnodes(&vnodes[batch * BatchSize],
					&vnodes[end]);
			ComputeMiniBatch(mini_batch, mini_sem_batch, sub_vnodes);
		}
	} else {
		if (!ComputeMiniBatch(input_batch, semantic_batch, vnodes))
			return false;
	}

	logd << __FUNCTION__ << " " << vnodes.size() << " nodes "
			<< Globals::ElapsedTime(start) << " s" << endl;
	return true;
}

bool PedNeuralSolverPrior::Compute_val_refracted(torch::Tensor input_tensor,
		torch::Tensor semantic_tensor, const ContextPomdp* ped_model,
		std::vector<despot::VNode*>& vnodes) {

	auto start1 = Time::now();
	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(input_tensor.to(at::kCUDA));
	inputs.push_back(semantic_tensor.to(at::kCUDA));

	auto drive_net_output =
			drive_net_value->forward(inputs).toTuple()->elements();
	logd << "[Compute] Retracting value outputs" << endl;
	// Retracted value output
	auto ncol_value_batch = drive_net_output[0].toTensor().cpu();
	auto col_value_batch = drive_net_output[1].toTensor().cpu();

	logd << "nn query in " << Globals::ElapsedTime(start1) << " s" << endl;
	logd << "[Compute] Retracting value outputs " << endl;

	ncol_value_batch = ncol_value_batch.squeeze(1);
	col_value_batch = col_value_batch.squeeze(1);
	auto ncol_value_double = ncol_value_batch.accessor<float, 1>();
	auto col_value_double = col_value_batch.accessor<float, 1>();

	int node_id = -1;
	for (std::vector<despot::VNode*>::iterator it = vnodes.begin();
			it != vnodes.end(); it++) {
		despot::VNode* vnode = *it;
		node_id++;
		col_value_double[node_id] = max((double)col_value_double[node_id], 0.0);
		ncol_value_double[node_id] = min((double)ncol_value_double[node_id], 0.0);

		double prior_value =
				noncol_value_inv_transform(ncol_value_double[node_id]) +
				col_value_inv_transform(col_value_double[node_id]) +
				ModelParams::GOAL_REWARD;

		if (DESPOT::Print_nodes)
			tout << "vnode " << vnode << " with depth " << vnode->depth() <<
					" learned value " << prior_value  <<
					" " << ncol_value_double[node_id] <<
					" " << col_value_double[node_id] << endl;
		else
			cout << "vnode " << vnode << " with depth " << vnode->depth() <<
					" learned value " << prior_value  <<
					" " << ncol_value_double[node_id] <<
					" " << col_value_double[node_id] << endl;

		if (vnode->depth() == 0) {
			root_value_ = prior_value;
		}

		logd << "getting vnode " << vnode << " value " << prior_value << endl;
		
		if (prior_value > 0.0)
			prior_value = -0.0;

		logd << "assigning vnode " << vnode << " value " << prior_value << endl;
		vnode->factored_prior_value(RWD_TOTAL, prior_value);
		vnode->factored_prior_value(RWD_COL, col_value_double[node_id]);
		vnode->factored_prior_value(RWD_NCOL, ncol_value_double[node_id]);
	}

	return true;
}

bool PedNeuralSolverPrior::Compute_val(torch::Tensor input_tensor,
		torch::Tensor semantic_tensor, const ContextPomdp* ped_model,
		std::vector<despot::VNode*>& vnodes) {

	auto start1 = Time::now();
	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(input_tensor.to(at::kCUDA));
	inputs.push_back(semantic_tensor.to(at::kCUDA));

	auto drive_net_output = drive_net_value->forward(inputs);
	logd << "[Compute] Retracting value outputs " << endl;
	auto value_batch = drive_net_output.toTensor().cpu();
	logd << "Get value output " << value_batch << endl;
	logd << "nn query in " << Globals::ElapsedTime(start1) << " s" << endl;
	logd << "[Compute] Retracting value outputs " << endl;

	value_batch = value_batch.squeeze(1);
	auto value_double = value_batch.accessor<float, 1>();

	int node_id = -1;
	for (std::vector<despot::VNode*>::iterator it = vnodes.begin();
			it != vnodes.end(); it++) {
		despot::VNode* vnode = *it;
		node_id++;

		double prior_value = value_transform_inverse(value_double[node_id])
				+ ModelParams::GOAL_REWARD;

		if (vnode->depth() == 0) {
			root_value_ = value_double[node_id];
		}

		if (prior_value > 0.0)
			prior_value = -0.0;

		logd << "assigning vnode " << vnode << " value " << prior_value << endl;
		vnode->prior_value(prior_value);
	}

	return true;
}

int PedNeuralSolverPrior::ConvertToNNID(int action) {
	// auto context_pomdp = static_cast<const ContextPomdp*>(model_);
	// int laneID = context_pomdp->GetLaneID(action);
	// int accID = context_pomdp->GetAccelerationID(action);

	// int accID_intensor = 0;
	// switch (accID) {
	// case AccCode::MTN:
	// 	accID_intensor = 1;
	// 	break;
	// case AccCode::ACC:
	// 	accID_intensor = 2;
	// 	break;
	// case AccCode::DEC:
	// 	accID_intensor = 0;
	// 	break;
	// }
	// int actionID_intensor = context_pomdp->GetActionID(laneID, accID_intensor);

	// return actionID_intensor;

	return action;
}


void PedNeuralSolverPrior::Compute_pref_libtorch(torch::Tensor input_tensor,
		torch::Tensor semantic_tensor, const ContextPomdp* ped_model,
		std::vector<despot::VNode*>& vnodes) {

	auto start1 = Time::now();
	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(input_tensor.to(at::kCUDA));
	inputs.push_back(semantic_tensor.to(at::kCUDA));

	if (Globals::config.state_source == STATE_FROM_TOPIC) {
		detectNAN(input_tensor);
		detectNAN(semantic_tensor);
	}

	auto action_batch = drive_net->forward(inputs).toTensor().cpu();

	int batchsize = action_batch.size(0);

	logd << "[Compute] Refracting action outputs " << endl;

	logd << "nn query in " << Globals::ElapsedTime(start1) << " s" << endl;

	logd << "Get action probs " << action_batch << endl;

	int node_id = -1;
	for (std::vector<despot::VNode*>::iterator it = vnodes.begin();
			it != vnodes.end(); it++) {
		despot::VNode* vnode = *it;
		node_id++;
		auto action = action_batch[node_id];
		auto action_float = action.accessor<float, 1>();

		if (action.size(0) != ped_model->NumActions())
			ERR("Policy model output size mismatch with NumActions in POMDP.");

		auto action_probs = std::vector<float>(model_->NumActions(), 0.0);
		if (vnode->depth() == 0)
			root_action_probs_.resize(model_->NumActions());
		for (int act = 0; act < model_->NumActions(); act++) {
			int actionID_intensor = ConvertToNNID(act);
			action_probs[act] = action_float[actionID_intensor];

			if (vnode->depth() == 0) {
				root_action_probs_[act] = action_float[act];
			}
		}


		Update_prior_probs(action_probs, vnode);
	}
}

void PedNeuralSolverPrior::Compute_pref(torch::Tensor input_tensor,
		torch::Tensor semantic_tensor, const ContextPomdp* ped_model,
		std::vector<despot::VNode*>& vnodes) {

	auto start1 = Time::now();
	at::Tensor value_batch_dummy, acc_batch, lane_batch;
	bool succeed = false;
	int retry_count = 0;
	while (!succeed) {
		succeed = query_srv(vnodes, input_tensor, semantic_tensor,
				value_batch_dummy, acc_batch, lane_batch);
		retry_count++;
		if (!succeed) {
			logi << "Action model query failed for nodes at level "
					<< vnodes[0]->depth() << "!!!" << endl;
			logi << "retry_count = " << retry_count << ", max_retry="
					<< max_retry_count << endl;
		}
		if (retry_count == max_retry_count)
			break;
	}
	if (!succeed) {
		ERR("ERROR: Policy net query failure !!!!!");
	}

	logd << "Action model query succeeded for nodes at level "
			<< vnodes[0]->depth() << endl;
	logd << "[Compute] Updating prior with nn outputs " << endl;
	int node_id = -1;
	for (std::vector<despot::VNode*>::iterator it = vnodes.begin();
			it != vnodes.end(); it++) {
		despot::VNode* vnode = *it;
		node_id++;

		auto acc = acc_batch[node_id];
		auto lane_probs_Tensor = lane_batch[node_id];
		auto acc_float = acc.accessor<float, 1>();

		if (logging::level() >= logging::DEBUG) {
			cout << "large net raw acc output:" << endl;
			for (int bin_idx = 0; bin_idx < acc.size(0); bin_idx++) {
				if (detectNAN(acc_float[bin_idx])) {
					cout << "input_tensor: \n" << input_tensor << endl;
					ERR("NAN detected in acc_float");
				}
				cout << "acc[" << bin_idx << "]=" << acc_float[bin_idx] << endl;
			}
		}

		int num_accs = ModelParams::NUM_ACC;
		at::Tensor acc_probs_tensor = torch::ones( { num_accs }, at::kFloat);
		for (int acc_id = 0; acc_id < num_accs; acc_id++) {
			float query_acc = ped_model->GetAccelerationNoramlized(acc_id);
			float onehot_resolution = 2.0 / float(num_accs);
			int bin_idx = (int) (std::floor(
					(query_acc + 1.0f) / onehot_resolution));
			bin_idx = min(bin_idx, num_accs - 1);
			bin_idx = max(bin_idx, 0);
			acc_probs_tensor[acc_id] = acc_float[bin_idx];
			logd << "adding query acc: acc_logits_tensor_" << acc_id
					<< "=acc_float_" << bin_idx << "=" << query_acc << endl;
		}

		Update_prior_probs(acc_probs_tensor, lane_probs_Tensor, vnode);
	}
}

void PedNeuralSolverPrior::Compute_pref_hybrid(torch::Tensor input_tensor,
		torch::Tensor semantic_tensor, const ContextPomdp* ped_model,
		std::vector<despot::VNode*>& vnodes) {

	auto start1 = Time::now();
	at::Tensor value_batch_dummy, acc_pi_batch, acc_mu_batch, acc_sigma_batch,
			lane_batch;
	bool succeed = false;
	int retry_count = 0;
	while (!succeed) {
		succeed = query_srv_hybrid(vnodes.size(), input_tensor, semantic_tensor,
				value_batch_dummy, acc_pi_batch, acc_mu_batch, acc_sigma_batch,
				lane_batch);
		retry_count++;
		if (!succeed) {
			logi << "Root node action model query failed !!!" << endl;
			logi << "retry_count = " << retry_count << ", max_retry="
					<< max_retry_count << endl;
		}
		if (retry_count == max_retry_count)
			break;
	}
	if (!succeed) {
		cerr << "ERROR: NN query failure !!!!!" << endl;
		raise (SIGABRT);
	}
	logd << "Root node action model query succeeded" << endl;
	logd << "[Compute] Updating prior with nn outputs " << endl;
	int node_id = -1;
	for (std::vector<despot::VNode*>::iterator it = vnodes.begin();
			it != vnodes.end(); it++) {
		despot::VNode* vnode = *it;
		node_id++;

		auto acc_pi = acc_pi_batch[node_id];
		auto acc_mu = acc_mu_batch[node_id];
		auto acc_sigma = acc_sigma_batch[node_id];
		auto lane_probs_Tensor = lane_batch[node_id];

		if (logging::level() >= logging::INFO) {
			logd << "large net raw acc output:" << endl;
			auto acc_pi_float = acc_pi.accessor<float, 1>();
			auto acc_mu_float = acc_mu.accessor<float, 2>();

			for (int mode = 0; mode < acc_pi.size(0); mode++) {
				logd << "mu[" << mode << "]=" << acc_mu_float[mode][0] << endl;
			}
			for (int mode = 0; mode < acc_pi.size(0); mode++) {
				logd << "pi[" << mode << "]=" << acc_pi_float[mode] << endl;
			}
		}

		int num_accs = ModelParams::NUM_ACC;
		at::Tensor acc_candiates = torch::ones( { num_accs, 1 }, at::kFloat);
		for (int acc_id = 0; acc_id < num_accs; acc_id++) {
			double query_acc = ped_model->GetAccelerationNoramlized(acc_id);
			acc_candiates[acc_id][0] = query_acc;
			logd << "adding query acc: " << acc_id << "=" << query_acc << endl;
		}

		int num_modes = acc_pi.size(0);

		auto acc_pi_actions = acc_pi.unsqueeze(0).expand(
				{ num_accs, num_modes });
		auto acc_mu_actions = acc_mu.unsqueeze(0).expand( { num_accs, num_modes,
				1 });
		auto acc_sigma_actions = acc_sigma.unsqueeze(0).expand( { num_accs,
				num_modes, 1 });

		auto acc_probs_Tensor = gm_pdf(acc_pi_actions, acc_sigma_actions,
				acc_mu_actions, acc_candiates);

		Update_prior_probs(acc_probs_Tensor, lane_probs_Tensor, vnode);
	}
}

void PedNeuralSolverPrior::export_images(std::vector<despot::VNode*>& vnodes) {
	if (logging::level() >= export_image_level + 1) {
		logd << "[Combine_images] vnodes[0]=" << vnodes[0]->depth() << endl;
		string level = std::to_string(vnodes[0]->depth());
		logd << "[Combine_images] exporting images" << endl;

		export_image(path_image_, "level" + level + "path");
		export_image(lane_image_, "level" + level + "lanes");
		for (int i = 0; i < num_hist_channels; i++) {
			int hist_channel = i;
			export_image(map_hist_images_[hist_channel],
					"level" + level + "_map_c" + std::to_string(hist_channel));
			//			export_image(car_hist_images_[hist_channel],
			//					"level" + level + "_car_c" + std::to_string(hist_channel));
		}
		inc_counter();
	}
}

void PedNeuralSolverPrior::SetDefaultPriorPolicy(
		std::vector<despot::VNode*>& vnodes) {
	if (Globals::config.use_prior) {
		for (despot::VNode* vnode : vnodes) {
			if (vnode->depth() == 0) {
				root_action_probs_.resize(model_->NumActions());
				std::fill(root_action_probs_.begin(), root_action_probs_.end(),
						0.0);
			}
			for (auto action : vnode->legal_actions()) {
				vnode->prior_action_probs(action,
						1.0 / vnode->legal_actions().size());
				if (vnode->depth() == 0) {
					root_action_probs_[action] = vnode->prior_action_probs(
							action);
				}
			}
		}
	}
}

void PedNeuralSolverPrior::SetDefaultPriorValue(
		std::vector<despot::VNode*>& vnodes) {
	for (std::vector<despot::VNode*>::iterator it = vnodes.begin();
			it != vnodes.end(); it++) {
		despot::VNode* vnode = *it;
		vnode->prior_value(0.0);
		if (vnode->depth() == 0) {
			root_value_ = 0.0;
		}
	}
}

ACT_TYPE PedNeuralSolverPrior::SamplePriorAction(despot::VNode* vnode) {
	double rand = Random::RANDOM.NextDouble();

	double t = 0;
	for (auto action : vnode->legal_actions()) {
		t += vnode->prior_action_probs(action);
		if (t >= rand)
			return action;
	}
	ERR("Sampling prior action failed");
	return -1;
}

bool PedNeuralSolverPrior::ComputeMiniBatch(
		std::vector<torch::Tensor>& input_batch,
		std::vector<torch::Tensor>& semantic_batch,
		std::vector<despot::VNode*>& vnodes) {

	auto start = Time::now();

	torch::NoGradGuard no_grad;

	const ContextPomdp* ped_model = static_cast<const ContextPomdp*>(model_);
	logd << "[Compute] node depth " << vnodes[0]->depth() << endl;

	export_images(vnodes);

	logd << "[Compute] num_nodes = " << input_batch.size() << endl;

	torch::Tensor input_tensor;
	input_tensor = torch::stack(input_batch, 0);
	input_tensor = input_tensor.contiguous();

	torch::Tensor semantic_tensor;
	semantic_tensor = torch::stack(semantic_batch, 0);
	semantic_tensor = semantic_tensor.contiguous();

	logd << "[Compute] input_tensor dim = \n" << input_tensor.sizes() << endl;

	logd << __FUNCTION__ << " prepare data " << Globals::ElapsedTime(start)
			<< " s" << endl;

	logd << __FUNCTION__ << " query for " << input_tensor.sizes() << " data "
			<< endl;

	// sync_cuda();
	if (SolverPrior::disable_policy_net) {
		SetDefaultPriorPolicy(vnodes);
	} else {
		if (model_file.find("pth") == std::string::npos)
			Compute_pref_libtorch(input_tensor, semantic_tensor, ped_model,
					vnodes);
		else
			Compute_pref(input_tensor, semantic_tensor, ped_model, vnodes);
	}

	if (SolverPrior::disable_value || SolverPrior::prior_min_depth > 0) {
		SetDefaultPriorValue(vnodes);
	} else {
		if (!Compute_val_refracted(input_tensor, semantic_tensor, ped_model, vnodes))
			return false;
	}
	for (std::vector<despot::VNode*>::iterator it = vnodes.begin();
			it != vnodes.end(); it++) {
		despot::VNode* vnode = *it;
		vnode->prior_initialized(true);
	}

	policy_ready_ = true;

	logd << __FUNCTION__ << " " << Globals::ElapsedTime(start) << " s" << endl;

	return true;
}

void PedNeuralSolverPrior::Update_prior_probs(
		std::vector<float>& action_probs_double, despot::VNode* vnode) {
	const ContextPomdp* ped_model = static_cast<const ContextPomdp*>(model_);
	logd << "action_probs = " << action_probs_double << endl;

	logd << "normalizing action_probs" << endl;

	auto action_total_prob = std::accumulate(action_probs_double.begin(),
			action_probs_double.end(), 0.0);

	if (action_total_prob < 0.01) {
		ERR("action_total_prob < 0.01");
	}
	if (detectNAN(action_total_prob)) {
		ERR("NAN in action_total_prob");
	}

	std::transform(action_probs_double.begin(), action_probs_double.end(),
			action_probs_double.begin(),
			std::bind(std::divides<float>(), std::placeholders::_1,
					action_total_prob));

	if (vnode->depth() == 0) {
		logd << "action probs = " << action_probs_double << endl;
	}

	// Update the values in the vnode
	logd << "sharpening action probs" << endl;

	double action_prob_total = 0;
	int sharpen_factor = 1;

	if (sharpen_factor != 1) {
		for (int act_ID = 0; act_ID < ped_model->NumActions(); act_ID++) {
			float act_prob = action_probs_double[act_ID];
			action_prob_total += std::pow(act_prob, sharpen_factor);
		}

		if (action_prob_total < 0.01) {
			cerr << "act_prob_total=" << action_prob_total << endl;
			ERR("act_prob_total unusual");
		}
	}

	logd << "assigning action probs to vnode " << vnode << " at depth "
			<< vnode->depth() << endl;

	double accum_prob = 0;

	for (auto action : vnode->legal_actions()) {
		logd << "legal action " << action << endl;

		float joint_prob = action_probs_double[action];
		if (sharpen_factor != 1)
			joint_prob = pow(joint_prob, sharpen_factor) / action_prob_total; // sharpen and normalize
		logd << "joint prob " << joint_prob << endl;

		vnode->prior_action_probs(action, joint_prob);

		accum_prob += joint_prob;

		logd << "action " << action << " joint_prob = " << joint_prob
				<< " accum_prob = " << accum_prob << endl;
	}

	logd << "normalizing probs" << endl;
	for (auto action : vnode->legal_actions()) {


		if (accum_prob == 0)
			vnode->prior_action_probs(action, 1.0 / vnode->legal_actions().size());
		else
			vnode->prior_action_probs(action, vnode->prior_action_probs(action) / accum_prob);

		logd << action_probs_.size() << endl;

		if (detectNAN(vnode->prior_action_probs(action)))
			ERR("");

		if (vnode->depth() == 0)
			action_probs_[action] = vnode->prior_action_probs(action); // store the root action priors
	}

	logd << "done" << endl;
}

void PedNeuralSolverPrior::Update_prior_probs(at::Tensor& acc_probs_Tensor,
		at::Tensor& lane_probs_Tensor, despot::VNode* vnode) {
	const ContextPomdp* ped_model = static_cast<const ContextPomdp*>(model_);
	logd << "acc probs = " << acc_probs_Tensor << endl;

	logd << "normalizing acc probs" << endl;

	auto acc_sum = acc_probs_Tensor.sum();
	float acc_total_prob = acc_sum.data<float>()[0];

	if (acc_total_prob < 0.01) {
		ERR("acc_total_prob < 0.01");
	}
	if (detectNAN(acc_total_prob)) {
		ERR("NAN in acc_total_prob");
	}
	acc_probs_Tensor = acc_probs_Tensor / acc_total_prob;

	if (ModelParams::NumLaneDecisions != lane_probs_Tensor.size(0))
		ERR("");

	if (vnode->depth() == 0) {
		logd << "acc probs = " << acc_probs_Tensor << endl;
		logd << "lane probs = " << lane_probs_Tensor << endl;
	}

	auto acc_probs_double = acc_probs_Tensor.accessor<float, 1>();

	if (logging::level() >= logging::INFO) {
		logd << "printing acc probs, acc_probs_Tensor dim="
				<< acc_probs_Tensor.sizes() << endl;
		for (int acc_id = 0; acc_id < acc_probs_Tensor.sizes()[0]; acc_id++) {
			double query_acc = ped_model->GetAccelerationNoramlized(acc_id);
			logd << "query acc " << acc_id << "=" << query_acc << ", prob="
					<< acc_probs_double[acc_id] << endl;
		}
	}

	auto lane_probs_double = lane_probs_Tensor.accessor<float, 1>();

	// Update the values in the vnode
	logd << "sharpening acc probs" << endl;

	double act_prob_total = 0;
	int sharpen_factor = 1;

	if (sharpen_factor != 1) {
		for (int acc_ID = 0; acc_ID < ModelParams::NUM_ACC; acc_ID++) {
			float acc_prob = acc_probs_double[acc_ID];
			act_prob_total += std::pow(acc_prob, sharpen_factor);
		}

		if (act_prob_total < 0.01) {
			cerr << "act_prob_total=" << act_prob_total << endl;
			ERR("act_prob_total unusual");
		}
	}

	logd << "assigning action probs to vnode " << vnode << " at depth "
			<< vnode->depth() << endl;

	double accum_prob = 0;

	for (auto action : vnode->legal_actions()) {
		logd << "legal action " << action << endl;

		int acc_ID = ped_model->GetAccelerationID(action);
		int laneID = ped_model->GetLaneID(action);

		float acc_prob = acc_probs_double[acc_ID];
		if (sharpen_factor != 1)
			acc_prob = pow(acc_prob, sharpen_factor) / act_prob_total; // sharpen and normalize
		float lane_prob = lane_probs_double[laneID];

		float joint_prob = acc_prob * lane_prob;

		if (detectNAN(acc_prob)) {
			terr << " acc_prob=" << acc_probs_double[acc_ID] << endl;
			ERR("NAN found in acc_prob");
		}
		if (detectNAN(joint_prob)) {
			ERR("NAN found in joint_prob");
		}

		logd << "joint prob " << joint_prob << endl;

		vnode->prior_action_probs(action, joint_prob);

		accum_prob += joint_prob;

		logd << "action " << acc_ID << " " << laneID << " joint_prob = "
				<< joint_prob << " accum_prob = " << accum_prob << endl;
	}

	logd << "normalizing probs" << endl;

	for (auto action : vnode->legal_actions()) {
		double prob = vnode->prior_action_probs(action);

		prob = prob / accum_prob;

		vnode->prior_action_probs(action, prob);

		logd << action_probs_.size() << endl;

		if (detectNAN(vnode->prior_action_probs(action)))
			ERR("");

		if (vnode->depth() == 0)
			action_probs_[action] = vnode->prior_action_probs(action); // store the root action priors
	}

	logd << "done" << endl;
}

void PedNeuralSolverPrior::ComputePreference(
		std::vector<torch::Tensor>& input_batch,
		std::vector<torch::Tensor>& semantic_batch,
		std::vector<despot::VNode*>& vnodes) {
	logd << "[ComputePreference] get " << vnodes.size() << " nodes" << endl;
	auto start = Time::now();

	if (vnodes.size() > BatchSize) {
		logd << "Executing " << (vnodes.size() - 1) / BatchSize << " batches"
				<< endl;
		for (int batch = 0; batch < (vnodes.size() - 1) / BatchSize + 1;
				batch++) {

			int start = batch * BatchSize;
			int end =
					((batch + 1) * BatchSize <= vnodes.size()) ?
							(batch + 1) * BatchSize : vnodes.size();

			logd << "start end: " << start << " " << end << endl;

			std::vector<torch::Tensor> mini_batch(&input_batch[start],
					&input_batch[end]);
			std::vector<torch::Tensor> mini_sem_batch(&semantic_batch[start],
					&semantic_batch[end]);
			std::vector<despot::VNode*> sub_vnodes(&vnodes[batch * BatchSize],
					&vnodes[end]);
			ComputeMiniBatchPref(mini_batch, mini_sem_batch, sub_vnodes);
		}
	} else
		ComputeMiniBatchPref(input_batch, semantic_batch, vnodes);

	logd << __FUNCTION__ << " " << vnodes.size() << " nodes "
			<< Globals::ElapsedTime(start) << " s" << endl;
}

void PedNeuralSolverPrior::ComputeMiniBatchPref(
		std::vector<torch::Tensor>& input_batch,
		std::vector<torch::Tensor>& semantic_batch,
		std::vector<despot::VNode*>& vnodes) {

	auto start = Time::now();

	torch::NoGradGuard no_grad;

	const ContextPomdp* ped_model = static_cast<const ContextPomdp*>(model_);

	torch::Tensor input_tensor;
	input_tensor = torch::stack(input_batch, 0);
	input_tensor = input_tensor.contiguous();

	torch::Tensor semantic_tensor;
	semantic_tensor = torch::stack(semantic_batch, 0);
	semantic_tensor = semantic_tensor.contiguous();

	auto start1 = Time::now();

	if (model_file.find("pth") == std::string::npos)
		Compute_pref_libtorch(input_tensor, semantic_tensor, ped_model, vnodes);
	else
		Compute_pref(input_tensor, semantic_tensor, ped_model, vnodes);

	export_images(vnodes);

	logd << __FUNCTION__ << " " << Globals::ElapsedTime(start) << " s" << endl;
}

void PedNeuralSolverPrior::ComputeValue(std::vector<torch::Tensor>& input_batch,
		std::vector<torch::Tensor>& semantic_batch,
		std::vector<despot::VNode*>& vnodes) {
	logd << "[ComputeValue] get " << vnodes.size() << " nodes" << endl;
	auto start = Time::now();

	if (vnodes.size() > BatchSize) {
		logd << "Executing " << (vnodes.size() - 1) / BatchSize << " batches"
				<< endl;
		for (int batch = 0; batch < (vnodes.size() - 1) / BatchSize + 1;
				batch++) {

			int start = batch * BatchSize;
			int end =
					((batch + 1) * BatchSize <= vnodes.size()) ?
							(batch + 1) * BatchSize : vnodes.size();

			logd << "start end: " << start << " " << end << endl;

			std::vector<torch::Tensor> mini_batch(&input_batch[start],
					&input_batch[end]);
			std::vector<torch::Tensor> mini_sem_batch(&semantic_batch[start],
					&semantic_batch[end]);
			std::vector<despot::VNode*> sub_vnodes(&vnodes[batch * BatchSize],
					&vnodes[end]);
			ComputeMiniBatchValue(mini_batch, mini_sem_batch, sub_vnodes);
		}
	} else
		ComputeMiniBatchValue(input_batch, semantic_batch, vnodes);

	logd << __FUNCTION__ << " " << vnodes.size() << " nodes "
			<< Globals::ElapsedTime(start) << " s" << endl;
}

void PedNeuralSolverPrior::ComputeMiniBatchValue(
		std::vector<torch::Tensor>& input_batch,
		std::vector<torch::Tensor>& semantic_batch,
		std::vector<despot::VNode*>& vnodes) {

	auto start = Time::now();

	torch::NoGradGuard no_grad;
//	drive_net->eval();

	// DONE: Send nn_input_images_ to drive_net, and get the policy and value output
	const ContextPomdp* ped_model = static_cast<const ContextPomdp*>(model_);

	logd << "[ComputeValue] node depth " << vnodes[0]->depth() << endl;

	export_images(vnodes);

	logd << "[ComputeValue] num_nodes = " << input_batch.size() << endl;

	torch::Tensor input_tensor;
	input_tensor = torch::stack(input_batch, 0);
	input_tensor = input_tensor.contiguous();
	torch::Tensor semantic_tensor;
	semantic_tensor = torch::stack(semantic_batch, 0);
	semantic_tensor = semantic_tensor.contiguous();
	logd << "[ComputeValue] input_tensor dim = \n" << input_tensor.sizes()
			<< endl;
	logd << "[ComputeValue] semantic_tensor dim = \n" << semantic_tensor.sizes()
			<< endl;

	logd << __FUNCTION__ << " prepare data " << Globals::ElapsedTime(start)
			<< " s" << endl;
	logd << __FUNCTION__ << " query for " << input_tensor.sizes() << " data "
			<< endl;

	Compute_val_refracted(input_tensor, semantic_tensor, ped_model, vnodes);
	logd << __FUNCTION__ << " " << Globals::ElapsedTime(start) << " s" << endl;
}

std::vector<ACT_TYPE> PedNeuralSolverPrior::ComputeLegalActions(
		despot::Shared_VNode* vnode, const State* state, const DSPOMDP* model) {
	const PomdpState* pomdp_state = static_cast<const PomdpState*>(state);
	const ContextPomdp* pomdp_model = static_cast<const ContextPomdp*>(model_);

	logd << "================= [ComputeLegalActions] start ================="
			<< endl;

	std::vector<int> legal_lanes;
	if (vnode->depth() == 0)
		legal_lanes = world_model.SetRootPathIDs(vnode);
	else {
		legal_lanes.push_back(LaneCode::KEEP);
		//  world_model.ExtendCurLane(pomdp_state);
		if (world_model.ValidForLaneChange(pomdp_state->car)) {
			if (world_model.ParseLanePath(vnode, pomdp_state, LaneCode::LEFT))
				legal_lanes.push_back(LaneCode::LEFT);
			if (world_model.ParseLanePath(vnode, pomdp_state, LaneCode::RIGHT))
				legal_lanes.push_back(LaneCode::RIGHT);
		}
	}

	std::vector<ACT_TYPE> legal_actions;
	logd << "legal actions for vnode " << vnode << " at level "
			<< vnode->depth() << ":" << endl;
	for (int laneID : legal_lanes) {
		ACT_TYPE act_start = pomdp_model->GetActionID(laneID, 0);
		ACT_TYPE act_end = act_start + ModelParams::NUM_ACC;
		for (ACT_TYPE action = act_start; action < act_end; action++) {
			legal_actions.push_back(action);
			logd << " " << action;
		}
	}
	logd << endl;

	if (vnode->depth() == 0)
		cout << "legal actions at root: " << legal_actions << endl;

	logd << "================= [ComputeLegalActions] end ================="
			<< endl;

	return legal_actions;
}

void PedNeuralSolverPrior::get_history_settings(despot::VNode* cur_node,
		int mode, int &num_history, int &start_channel) {
	if (mode == FULL) { // Full update of the input channels
		num_history = num_hist_channels;
		start_channel = 0;
		logd << "Getting FULL history for node " << cur_node << " at depth "
				<< cur_node->depth() << endl;
	} else if (mode == PARTIAL) { // Partial update of input channels and reuse old channels
		//[1,2,3], id = 0 will be filled in the nodes
		num_history = num_hist_channels - 1;
		start_channel = 1;
		logd << "Getting PARTIAL history for node " << cur_node << " at depth "
				<< cur_node->depth() << endl;
	}
}

void PedNeuralSolverPrior::get_history_map_tensors(int mode, despot::VNode* cur_node) {
	int num_history = 0;
	int start_channel = 0;

	get_history_settings(cur_node, mode, num_history, start_channel);

	despot::VNode* parent = cur_node;
	int t = tracked_map_hist_.size() - 1; // latest pos in tensor history

	for (int i = start_channel; i < start_channel + num_history; i++) {
		if (parent == NULL) {
			try {
				if (!tracked_map_hist_[t].defined()) {
					ERR("tracked_map_hist_[t] not defined");
				}
				map_hist_tensor_[i] = tracked_map_hist_[t];
				t--;
			} catch (...) {
				ERR("Exception when parsing history images");
			}
		} else {
			assert(static_cast<despot::Shared_VNode*>(parent)->map_tensor.defined());
			map_hist_tensor_[i] =
					static_cast<despot::Shared_VNode*>(parent)->map_tensor;
			if (parent->parent()==NULL) // root
				parent = NULL;
			else
				parent = parent->parent()->parent();
		}
	}
}

std::vector<float> PedNeuralSolverPrior::get_semantic_history(
		despot::VNode* cur_node) {
	std::vector<float> semantic;
	despot::VNode* node = cur_node;
	int t = tracked_semantic_hist_.size() - 1; // latest pos in tensor history

	while (semantic.size() < NUM_HIST_CHANNELS) {
		if (node->parent() == NULL) { // parent is root
			semantic.push_back(tracked_semantic_hist_[t]);
			t--;
		} else {
			PomdpState* state = static_cast<PomdpState*>(node->particles()[0]);
			semantic.push_back(state->car.vel);
			node = node->parent()->parent();
		}
	}
	return semantic;
}

void PedNeuralSolverPrior::get_history(int mode, despot::VNode* cur_node,
		std::vector<despot::VNode*>& parents_to_fix_images,
		std::vector<PomdpState*>& hist_states, std::vector<int>& hist_ids) {
	int num_history = 0;
	int start_channel = 0;

	get_history_settings(cur_node, mode, num_history, start_channel);

	despot::VNode* parent = cur_node;

	std::vector<int> reuse_ids;

	for (int i = start_channel; i < start_channel + num_history; i++) {

		if (i > start_channel)
			parent =
					(parent->parent() == NULL) ?
							parent : parent->parent()->parent(); // to handle root node

		if (mode == FULL && cur_node->depth() == 0
				&& !cur_node->prior_initialized()) {
			double cur_ts =
					static_cast<PomdpState*>(cur_node->particles()[0])->time_stamp;
			double hist_ts = cur_ts - i * 1.0 / ModelParams::CONTROL_FREQ;

			logd << "Comparing recorded ts "
					<< hist_time_stamps[i - start_channel]
					<< " with calculated ts " << hist_ts << endl;
			if (abs(hist_time_stamps[i - start_channel] - hist_ts) <= 1e-2) { // can reuse the previous channel
				reuse_ids.push_back(i);
				if (do_print) {
					Globals::lock_process();
					logd << "Thread " << prior_id() << " Reusing channel "
							<< i - start_channel << " to new channel " << i
							<< " for node " << cur_node << endl;
					Globals::unlock_process();
				}
				continue;
			}
		} else {
			if (car_hist_links[i - start_channel] == parent) { // can reuse the previous channel
				reuse_ids.push_back(i);

				if (do_print) {
					Globals::lock_process();
					logd << "Thread " << prior_id() << " Reusing channel "
							<< i - start_channel << " to new channel " << i
							<< " for node " << cur_node << endl;
					Globals::unlock_process();
				}
				continue;
			}
		}

		logd << "add hist id" << i << ", add parent depth = " << parent->depth()
				<< endl;

		hist_ids.push_back(i);
		parents_to_fix_images.push_back(parent);
	}

	for (int i = reuse_ids.size() - 1; i >= 0; i--) {
		Reuse_history(reuse_ids[i], start_channel, mode);
	}

	logd << "hist init len " << init_hist_len << endl;
	logd << "hist_ids len " << hist_ids.size() << endl;

	// DONE: get the 4 latest history states
	int latest = as_history_in_search_.Size() - 1;
	for (int i = 0; i < hist_ids.size(); i++) {
		int t = latest;
		if (mode == FULL)
			t = latest - hist_ids[i];
		else if (mode == PARTIAL)
			t = latest - hist_ids[i] + 1;

		if (t >= 0) {
			PomdpState* car_peds_state =
					static_cast<PomdpState*>(as_history_in_search_.state(t));
			hist_states.push_back(car_peds_state);

			if (do_print) {
				logd << "Thread " << prior_id()
						<< " Using as_history_in_search_ entry " << t << " ts "
						<< car_peds_state->time_stamp << " as new channel "
						<< hist_ids[i] << " node at level " << cur_node->depth()
						<< endl;
			}
		} else {
			hist_states.push_back(NULL);
			logd << " Using NULL state as new channel " << hist_ids[i]
					<< " node at level " << cur_node->depth() << endl;
		}
	}

	logd << "[Get_history] validating history, node " << cur_node << endl;

	parent = cur_node;

	try {
		for (int i = start_channel; i < num_hist_channels; i++) {
			logd << " [Get_history] hist " << i << " should be depth "
					<< parent->depth() << ", get depth "
					<< (car_hist_links[i] ? car_hist_links[i]->depth() : -1)
					<< " node depth " << cur_node->depth() << endl;
			if (car_hist_links[i] != parent)
				logd << "mismatch!!!!!!!!!!!!!!!!!" << endl;
			parent =
					(parent->parent() == NULL) ?
							parent : parent->parent()->parent(); // to handle root node
		}
	} catch (Exception e) {
		logd << " [error] !!!!!!!!!!!!!!!!" << e.what() << endl;
		ERR("");
	}

	logd << "[Get_history] done " << endl;
}

void PedNeuralSolverPrior::Record_hist_len() {
	init_hist_len = as_history_in_search_.Size();
}

void PedNeuralSolverPrior::print_prior_actions(ACT_TYPE action) {
	logd << "action " << action << " (acc/steer) = "
			<< static_cast<const ContextPomdp*>(model_)->GetAcceleration(action)
			<< "/" << static_cast<const ContextPomdp*>(model_)->GetLane(action)
			<< endl;
}

std::vector<torch::Tensor> PedNeuralSolverPrior::Process_semantic_input(
		std::vector<despot::VNode*>& nodes, bool record_unlabelled, int record_child_id) {
	std::vector<torch::Tensor> nn_input;
	int i = 0;
	for (despot::VNode* node : nodes) {
		std::vector<float> semantic = get_semantic_history(node);
		at::Tensor f = torch::tensor(semantic);
		nn_input.push_back(f);
		if (record_unlabelled && i == record_child_id)
			unlabelled_semantic_ = semantic;
		i++;
	}
	return nn_input;
}

void PedNeuralSolverPrior::Process_state(despot::VNode* cur_node) {
	auto cur_state = static_cast<PomdpState*>(cur_node->particles()[0]);
	std::vector<int> hist_ids( { 0 }); // use as the last hist step

	Process_states(std::vector<despot::VNode*>( { cur_node }),
			std::vector<PomdpState*>( { cur_state }), hist_ids);
}

void PedNeuralSolverPrior::RecordUnlabelledBelief(despot::VNode* cur_node) {
	for (State* old_particle : unlabelled_belief_)
		model_->Free(old_particle);
	unlabelled_belief_.resize(0);
	for (State* particle : cur_node->particles())
		unlabelled_belief_.push_back(model_->Copy(particle));
	unlabelled_belief_depth_ = cur_node->depth();
}

void PedNeuralSolverPrior::CleanUnlabelledBelief() {
	for (State* old_particle : unlabelled_belief_)
		model_->Free(old_particle);
	unlabelled_belief_.resize(0);
}

std::vector<torch::Tensor> PedNeuralSolverPrior::Process_node_input(
		despot::VNode* cur_node, bool record_unlabelled) {
	auto start = Time::now();

	logd << "[Process_history_input], len=" << num_hist_channels << endl;

	if (!static_cast<despot::Shared_VNode*>(cur_node)->map_tensor.defined()) {
		Process_state(cur_node);
	} else {
		Process_lane_tensor(cur_node->particles()[0]);
	}

	get_history_map_tensors(FULL, cur_node); // for map and car

	std::vector<torch::Tensor> nn_input;
	nn_input.push_back(Combine_images(cur_node));

	if (record_unlabelled) {
		logd << "recording search node particles" << endl;
		RecordUnlabelledBelief(cur_node);
		RecordUnlabelledHistImages();
	}

	logd << __FUNCTION__ << " " << Globals::ElapsedTime(start) << " s" << endl;
	return nn_input;
}

bool PedNeuralSolverPrior::query_srv(std::vector<despot::VNode*>& vnodes,
		at::Tensor images, at::Tensor semantic, at::Tensor& t_value,
		at::Tensor& t_acc, at::Tensor& t_lane) {
	int num_acc_bins = ModelParams::NUM_ACC;
	int num_lane_bins = ModelParams::NumLaneDecisions;
	int batchsize = vnodes.size();
	int depth = vnodes[0]->depth();

	if (detectNAN(images))
		ERR("NAN in input tensor");

	t_value = torch::zeros( { batchsize });
	t_acc = torch::zeros( { batchsize, num_acc_bins });
	t_lane = torch::zeros( { batchsize, num_lane_bins });

	msg_builder::TensorData message;

	for (despot::VNode* node : vnodes) {
		const PomdpState* pomdp_state =
				static_cast<const PomdpState*>(node->particles()[0]);
		message.request.cur_vel.push_back(pomdp_state->car.vel);
	}

	message.request.tensor = std::vector<SRV_DATA_TYPE>(
			images.data<SRV_DATA_TYPE>(),
			images.data<SRV_DATA_TYPE>() + images.numel());
	message.request.semantic_tensor = std::vector<SRV_DATA_TYPE>(
			semantic.data<SRV_DATA_TYPE>(),
			semantic.data<SRV_DATA_TYPE>() + semantic.numel());
	message.request.batchsize = batchsize;
	message.request.mode = to_string(depth);

	logd << "calling service query" << endl;

	if (nn_client_.call(message)) {
		std::vector<float> value = message.response.value;
		std::vector<float> acc = message.response.acc;
		std::vector<float> lane = message.response.lane;

		logd << "acc" << endl;
		for (int id = 0; id < acc.size(); id++) {
			if (detectNAN(acc[id]))
				ERR("query policy network: NAN in acc");

			int data_id = id / num_acc_bins;
			int mode_id = id % num_acc_bins;
			logd << acc[id] << " ";
			if (mode_id == num_acc_bins - 1) {
				logd << endl;
			}

			t_acc[data_id][mode_id] = acc[id];
		}

		logd << "lane" << lane << endl;
		for (int id = 0; id < lane.size(); id++) {
			if (detectNAN(lane[id]))
				ERR("query policy network: NAN in lane");

			int data_id = id / num_lane_bins;
			int bin_id = id % num_lane_bins;
			logd << lane[id] << " ";
			if (bin_id == num_lane_bins - 1) {
				logd << endl;
			}

			t_lane[data_id][bin_id] = lane[id];
		}

		return true;
	} else {
		ERR("Not able to query the policy network");
		return false;
	}
}

bool PedNeuralSolverPrior::query_srv_hybrid(int batchsize, at::Tensor images,
		at::Tensor semantic, at::Tensor& t_value, at::Tensor& t_acc_pi,
		at::Tensor& t_acc_mu, at::Tensor& t_acc_sigma, at::Tensor& t_lane) {
	int num_guassian_modes = 5;
	int num_lane_bins = ModelParams::NumLaneDecisions;

	t_value = torch::zeros( { batchsize });
	t_acc_pi = torch::zeros( { batchsize, num_guassian_modes });
	t_acc_mu = torch::zeros( { batchsize, num_guassian_modes, 1 });
	t_acc_sigma = torch::zeros( { batchsize, num_guassian_modes, 1 });
	t_lane = torch::zeros( { batchsize, num_lane_bins });

	msg_builder::TensorDataHybrid message;

	message.request.tensor = std::vector<SRV_DATA_TYPE>(
			images.data<SRV_DATA_TYPE>(),
			images.data<SRV_DATA_TYPE>() + images.numel());
	message.request.semantic_tensor = std::vector<SRV_DATA_TYPE>(
			semantic.data<SRV_DATA_TYPE>(),
			semantic.data<SRV_DATA_TYPE>() + semantic.numel());
	message.request.batchsize = batchsize;

	message.request.mode = "-1";

	logd << "calling service query" << endl;

	if (nn_client_.call(message)) {
		std::vector<float> value = message.response.value;
		std::vector<float> acc_pi = message.response.acc_pi;
		std::vector<float> acc_mu = message.response.acc_mu;
		std::vector<float> acc_sigma = message.response.acc_sigma;
		std::vector<float> lane = message.response.lane;

		logd << "acc_pi" << endl;
		for (int id = 0; id < acc_pi.size(); id++) {
			if (detectNAN(acc_pi[id]))
				return false;

			int data_id = id / num_guassian_modes;
			int mode_id = id % num_guassian_modes;
			logd << acc_pi[id] << " ";
			if (mode_id == num_guassian_modes - 1) {
				logd << endl;
			}

			t_acc_pi[data_id][mode_id] = acc_pi[id];
		}

		logd << "acc_mu" << endl;
		for (int id = 0; id < acc_mu.size(); id++) {
			if (detectNAN(acc_mu[id]))
				return false;

			int data_id = id / num_guassian_modes;
			int mode_id = id % num_guassian_modes;
			logd << acc_mu[id] << " ";
			if (mode_id == num_guassian_modes - 1) {
				logd << endl;
			}

			t_acc_mu[data_id][mode_id][0] = acc_mu[id];
		}

		logd << "acc_sigma" << endl;
		for (int id = 0; id < acc_sigma.size(); id++) {
			if (detectNAN(acc_sigma[id]))
				return false;

			int data_id = id / num_guassian_modes;
			int mode_id = id % num_guassian_modes;
			logd << acc_sigma[id] << " ";
			if (mode_id == num_guassian_modes - 1) {
				logd << endl;
			}

			t_acc_sigma[data_id][mode_id][0] = acc_sigma[id];
		}

		logd << "lane" << endl;
		for (int id = 0; id < lane.size(); id++) {
			if (detectNAN(lane[id]))
				return false;

			int data_id = id / num_lane_bins;
			int bin_id = id % num_lane_bins;
			logd << lane[id] << " ";
			if (bin_id == num_lane_bins - 1) {
				logd << endl;
			}

			t_lane[data_id][bin_id] = lane[id];
		}

		return true;
	} else {
		return false;
	}
}

void PedNeuralSolverPrior::Test_all_srv_hybrid(int batchsize,
		int num_guassian_modes, int num_lane_bins) {
	cerr << "Testing all model using ROS service, bs = " << batchsize << "..."
			<< endl;

	ros::NodeHandle n("~");

	nn_client_ = n.serviceClient<msg_builder::TensorDataHybrid>("/query");

	while (true) {
		bool up = nn_client_.waitForExistence(ros::Duration(1));
		if (up) {
			logi << "/query service is ready" << endl;
			break;
		} else
		logi << "waiting for /query service to be ready" << endl;
	}

	for (int i = 0; i < 2; i++) {

		std::vector<torch::jit::IValue> inputs;

		auto images = torch::ones( { batchsize, NUM_CHANNELS, IMSIZE, IMSIZE },
				TORCH_DATA_TYPE);
		auto semantic = torch::ones( { batchsize, NUM_HIST_CHANNELS },
				TORCH_DATA_TYPE);
		auto start1 = Time::now();

		msg_builder::TensorDataHybrid message;

		message.request.tensor = std::vector<SRV_DATA_TYPE>(
				images.data<SRV_DATA_TYPE>(),
				images.data<SRV_DATA_TYPE>() + images.numel());
		message.request.semantic_tensor = std::vector<SRV_DATA_TYPE>(
				semantic.data<SRV_DATA_TYPE>(),
				semantic.data<SRV_DATA_TYPE>() + semantic.numel());
		message.request.batchsize = batchsize;

		message.request.mode = "-1";

		logd << "calling service query" << endl;

		if (nn_client_.call(message)) {
			std::vector<float> value = message.response.value;
			std::vector<float> acc_pi = message.response.acc_pi;
			std::vector<float> acc_mu = message.response.acc_mu;
			std::vector<float> acc_sigma = message.response.acc_sigma;
			std::vector<float> ang = message.response.ang;

			logd << "value" << endl;
			for (int i = 0; i < value.size(); i++) {
				logd << value[i] << " ";
			}
			logd << endl;

			logd << "acc_pi" << endl;
			for (int id = 0; id < acc_pi.size(); id++) {
				int data_id = id / num_guassian_modes;
				int mode_id = id % num_guassian_modes;
				logd << acc_pi[id] << " ";
				if (mode_id == num_guassian_modes - 1) {
					logd << endl;
				}
			}

			logd << "acc_mu" << endl;
			for (int id = 0; id < acc_mu.size(); id++) {
				int data_id = id / num_guassian_modes;
				int mode_id = id % num_guassian_modes;
				logd << acc_mu[id] << " ";
				if (mode_id == num_guassian_modes - 1) {
					logd << endl;
				}
			}

			logd << "ang" << endl;
			for (int id = 0; id < ang.size(); id++) {
				int data_id = id / num_lane_bins;
				int bin_id = id % num_lane_bins;
				logd << ang[id] << " ";
				if (bin_id == num_lane_bins - 1) {
					logd << endl;
				}
			}
			logd << endl;

//			ROS_INFO("value: %f", (long int)message.response.value[0]);
		} else {
			ROS_ERROR("Failed to call service query");
			ERR("");
		}

		logd << "nn query in " << Globals::ElapsedTime(start1) << " s" << endl;
	}
	cerr << "NN ROS servie test done." << endl;
}

void PedNeuralSolverPrior::Test_all_srv(int batchsize, int num_acc_bins,
		int num_lane_bins) {

	if (model_file.find(".pth") == std::string::npos)
		ERR("drive net with non-pth model");

	cerr << "Testing all model using ROS service, bs = " << batchsize << "..."
			<< endl;

	ros::NodeHandle n("~");

	nn_client_ = n.serviceClient<msg_builder::TensorData>("/query");

	while (true) {
		bool up = nn_client_.waitForExistence(ros::Duration(1));
		if (up) {
			logi << "/query service is ready" << endl;
			break;
		} else
		logi << "waiting for /query service to be ready" << endl;
	}

	for (int i = 0; i < 2; i++) {

		std::vector<torch::jit::IValue> inputs;
		auto images = torch::ones( { batchsize, NUM_CHANNELS, IMSIZE, IMSIZE },
				TORCH_DATA_TYPE);
		auto semantic = torch::ones( { batchsize, NUM_HIST_CHANNELS },
				TORCH_DATA_TYPE);
		auto start1 = Time::now();

		msg_builder::TensorData message;
		message.request.cur_vel = std::vector<SRV_DATA_TYPE>(batchsize, 0.0);
		message.request.tensor = std::vector<SRV_DATA_TYPE>(
				images.data<SRV_DATA_TYPE>(),
				images.data<SRV_DATA_TYPE>() + images.numel());
		message.request.semantic_tensor = std::vector<SRV_DATA_TYPE>(
				semantic.data<SRV_DATA_TYPE>(),
				semantic.data<SRV_DATA_TYPE>() + semantic.numel());
		message.request.batchsize = batchsize;
		message.request.mode = "-1";

		logd << "calling service query" << endl;

		if (nn_client_.call(message)) {
			std::vector<float> value = message.response.value;
			std::vector<float> acc = message.response.acc;
			std::vector<float> lane = message.response.lane;

			logd << "value" << endl;
			for (int i = 0; i < value.size(); i++) {
				logd << value[i] << " ";
			}
			logd << endl;

			logd << "acc" << endl;
			for (int id = 0; id < acc.size(); id++) {
				int data_id = id / num_acc_bins;
				int mode_id = id % num_acc_bins;
				logd << acc[id] << " ";
				if (mode_id == num_acc_bins - 1) {
					logd << endl;
				}
			}

			logd << "ang" << endl;
			for (int id = 0; id < lane.size(); id++) {
				int data_id = id / num_lane_bins;
				int bin_id = id % num_lane_bins;
				logd << lane[id] << " ";
				if (bin_id == num_lane_bins - 1) {
					logd << endl;
				}
			}
			logd << endl;
		} else {
			ROS_ERROR("Failed to call service query");
			ERR("");
		}

		logd << "nn query in " << Globals::ElapsedTime(start1) << " s" << endl;
	}
	cerr << "NN ROS servie test done." << endl;
}

void PedNeuralSolverPrior::Test_all_libtorch(int batchsize,
		int num_guassian_modes, int num_lane_bins) {
	cerr << "Testing all model using libtorch, bs = " << batchsize << endl;

//	drive_net->to(at::kCUDA);
//	assert(drive_net);

	Globals::lock_process();

	for (int i = 0; i < 1; i++) {

		std::vector<torch::jit::IValue> inputs;

		auto images = torch::ones( { batchsize, NUM_CHANNELS, IMSIZE, IMSIZE },
				TORCH_DATA_TYPE);
		auto semantic = torch::ones( { batchsize, NUM_HIST_CHANNELS },
				TORCH_DATA_TYPE);

		auto start1 = Time::now();

		inputs.push_back(images.to(at::kCUDA));
		inputs.push_back(semantic.to(at::kCUDA));

		logd << "[Test_model] Query nn for " << inputs.size()
				<< " tensors of dim" << inputs[0].toTensor().sizes() << endl;

		if (true/*Globals::config.close_loop_prior*/) {
			auto action_batch = drive_net->forward(inputs).toTensor().cpu();
			auto act_batch_double = action_batch.accessor<float, 2>();
			logd << "act probs" << endl;
			for (int data_id = 0; data_id < action_batch.size(0); data_id++) {
				for (int bin_id = 0; bin_id < action_batch.size(1); bin_id++) {
					logd << act_batch_double[data_id][bin_id] << " ";
					if (bin_id == model_->NumActions() - 1) {
						logd << endl;
					}
				}
			}
			logd << endl;
		} else {
			auto drive_net_output =
					drive_net->forward(inputs).toTuple()->elements();
			auto vel_batch = drive_net_output[0].toTensor().cpu();
			auto vel_batch_double = vel_batch.accessor<float, 2>();
			auto lane_batch = drive_net_output[1].toTensor().cpu();
			auto lane_batch_double = lane_batch.accessor<float, 2>();

			logd << "lane" << endl;
			for (int data_id = 0; data_id < lane_batch.size(0); data_id++) {
				for (int bin_id = 0; bin_id < lane_batch.size(1); bin_id++) {
					logd << lane_batch_double[data_id][bin_id] << " ";
					if (bin_id == num_lane_bins - 1) {
						logd << endl;
					}
				}
			}
			logd << endl;
		}

		logd << "nn query in " << Globals::ElapsedTime(start1) << " s" << endl;
	}
	Globals::unlock_process();

}

void PedNeuralSolverPrior::Test_val_libtorch(int batchsize,
		int num_guassian_modes, int num_lane_bins) {

	logi << "Testing libtorch value model" << endl;
	Globals::lock_process();
	for (int i = 0; i < 1; i++) {

		std::vector<torch::jit::IValue> inputs;

		auto images = torch::ones( { batchsize, NUM_CHANNELS, IMSIZE, IMSIZE },
				TORCH_DATA_TYPE);
		auto semantic = torch::ones( { batchsize, NUM_HIST_CHANNELS },
				TORCH_DATA_TYPE);

		auto start1 = Time::now();

		inputs.push_back(images.to(at::kCUDA));
		inputs.push_back(semantic.to(at::kCUDA));

		logd << "[Test_model] Query nn for " << inputs.size()
				<< " tensors of dim" << inputs[0].toTensor().sizes() << endl;

		auto drive_net_output = drive_net_value->forward(inputs);

		auto value_batch = drive_net_output.toTensor().cpu();
		auto value_batch_double = value_batch.accessor<float, 2>();

		logd << "value 0: " << value_batch_double[0][0] << endl;
		logd << "nn query in " << Globals::ElapsedTime(start1) << " s" << endl;

		logd << "value" << endl;
		for (int i = 0; i < value_batch.size(0); i++) {
			logd << value_batch_double[i][0] << " ";
		}
		logd << endl;
	}
	Globals::unlock_process();
}

void PedNeuralSolverPrior::Test_val_libtorch_refracted(int batchsize,
		int num_guassian_modes, int num_lane_bins) {

	logi << "Testing libtorch value model" << endl;
	Globals::lock_process();
	for (int i = 0; i < 1; i++) {
		std::vector<torch::jit::IValue> inputs;
		auto images = torch::ones( { batchsize, NUM_CHANNELS, IMSIZE, IMSIZE },
				TORCH_DATA_TYPE);
		auto semantic = torch::ones( { batchsize, NUM_HIST_CHANNELS },
				TORCH_DATA_TYPE);

		auto start1 = Time::now();
		inputs.push_back(images.to(at::kCUDA));
		inputs.push_back(semantic.to(at::kCUDA));

		logd << "[Test_model] Query nn for " << inputs.size()
				<< " tensors of dim" << inputs[0].toTensor().sizes() << endl;

		auto drive_net_output = drive_net_value->forward(inputs).toTuple()->elements();

		auto value_batch = drive_net_output[0].toTensor().cpu();
		auto value_batch_double = value_batch.accessor<float, 2>();
		logd << "value 0: " << value_batch_double[0][0] << endl;
		auto col_prob_batch = drive_net_output[1].toTensor().cpu();
		auto col_prob_batch_double = col_prob_batch.accessor<float, 2>();
		logd << "col_prob 0: " << col_prob_batch_double[0][0] << endl;
		logd << "nn query in " << Globals::ElapsedTime(start1) << " s" << endl;

		logd << "value" << endl;
		for (int i = 0; i < value_batch.size(0); i++) {
			logd << value_batch_double[i][0] << " ";
		}
		logd << endl;
	}
	Globals::unlock_process();
}

void PedNeuralSolverPrior::Test_model(string path) {

	logd << "[Test_model] Query " << endl;

	torch::NoGradGuard no_grad;

	int batchsize = 1;

	int num_guassian_modes = 5;
	int num_acc_bins = ModelParams::NUM_ACC;
	int num_lane_bins = ModelParams::NumLaneDecisions;

//	Test_val_srv_hybrid(batchsize, num_guassian_modes, num_lane_bins);

	if (model_file.find(".pth") == std::string::npos)
		Test_all_libtorch(batchsize, num_guassian_modes, num_lane_bins);
	else
		Test_all_srv(batchsize, num_acc_bins, num_lane_bins);

	if (!SolverPrior::disable_value)
		Test_val_libtorch_refracted(batchsize, num_guassian_modes, num_lane_bins);

	logd << "[Test_model] Done " << endl;

}

State* debug_state = NULL;

void PedNeuralSolverPrior::DebugHistory(string msg) {
	for (int t = 0; t < as_history_in_search_.Size(); t++) {
		auto state = as_history_in_search_.state(t);
		Debug_state(state, msg + "_t_" + std::to_string(t), model_);
	}
}

void Debug_state(State* state, string msg, const DSPOMDP* model) {
	if (state == debug_state) {
		bool mode = DESPOT::Debug_mode;
		DESPOT::Debug_mode = false;

		PomdpState* hist_state = static_cast<PomdpState*>(state);
		static_cast<const ContextPomdp*>(model)->PrintState(*hist_state);

		DESPOT::Debug_mode = mode;

		cerr << "=================== " << msg
				<< " breakpoint ===================" << endl;
		ERR("");
	}
}

void Record_debug_state(State* state) {
	debug_state = state;
}

bool Compare_states(PomdpState* state1, PomdpState* state2) {
	if ((state1->car.pos.x != state2->car.pos.x)
			|| (state1->car.pos.y != state2->car.pos.y)
			|| (state1->car.vel != state2->car.vel)
			|| (state1->car.heading_dir != state2->car.heading_dir)) {
		cerr << "!!!!!!! car diff !!!!!!" << endl;
		return true;
	}

	if (state1->num != state2->num) {
		cerr << "!!!!!!! ped num diff !!!!!!" << endl;
		return true;
	}

	bool diff = false;
	for (int i = 0; i < state1->num; i++) {
		diff = diff || (state1->agents[i].pos.x != state2->agents[i].pos.x);
		diff = diff || (state1->agents[i].pos.y != state2->agents[i].pos.y);
		diff = diff
				|| (state1->agents[i].intention != state2->agents[i].intention);

		if (diff) {
			cerr << "!!!!!!! ped " << i << " diff !!!!!!" << endl;
			return true;
		}
	}

	return false;
}

void PedNeuralSolverPrior::RecordCurHistory() {
	as_history_in_search_recorded.Truncate(0);
	for (int i = 0; i < as_history_in_search_.Size(); i++) {
		as_history_in_search_recorded.Add(as_history_in_search_.Action(i),
				as_history_in_search_.state(i));
	}
}

void PedNeuralSolverPrior::CompareHistoryWithRecorded() {
	if (as_history_in_search_.Size() != as_history_in_search_recorded.Size()) {
		cerr << "ERROR: history length changed after search!!!" << endl;
		cerr << "as_history_in_search_.Size()=" << as_history_in_search_.Size()
				<< ", as_history_in_search_recorded.Size()="
				<< as_history_in_search_recorded.Size() << endl;
		ERR("");
	}
	for (int i = 0; i < as_history_in_search_recorded.Size(); i++) {
		PomdpState* recorded_hist_state =
				static_cast<PomdpState*>(as_history_in_search_recorded.state(i));
		PomdpState* hist_state =
				static_cast<PomdpState*>(as_history_in_search_recorded.state(i));

		bool different = Compare_states(recorded_hist_state, hist_state);

		if (different) {
			cerr << "ERROR: history " << i << " changed after search!!!"
					<< endl;
			static_cast<const ContextPomdp*>(model_)->PrintState(
					*recorded_hist_state, "Recorded hist state");
			static_cast<const ContextPomdp*>(model_)->PrintState(*hist_state,
					"Hist state");

			ERR("");
		}
	}
}

int keep_count = 0;

void PedNeuralSolverPrior::update_ego_car_shape(
		std::vector<geometry_msgs::Point32> points) {
	car_shape.resize(0);
	for (auto &point : points) {
		car_shape.push_back(cv::Point3f(point.x, point.y, 1.0));
	}
}

double noncol_value_transform(double raw_value) {
	return raw_value / 2.0 / ModelParams::REWARD_FACTOR_VEL / 4.0;
}

double col_value_transform(double raw_value) {
	return raw_value / ModelParams::CRASH_PENALTY;
	//	/ std::pow(ModelParams::VEL_MAX, 2);
}

double noncol_value_inv_transform(double value) {
	return value * 2.0 * ModelParams::REWARD_FACTOR_VEL * 4.0;
}

double col_value_inv_transform(double value) {
	return value * ModelParams::CRASH_PENALTY;
	// *	std::pow(ModelParams::VEL_MAX, 2);
}
