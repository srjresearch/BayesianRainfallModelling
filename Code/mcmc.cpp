#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
//~ #include <Eigen/Dense>
#include <vector>
#include <iterator>
#include <string>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_linalg.h>
#include <math.h>
#include <omp.h>
#include <iomanip> 
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>

using namespace Eigen;


//// Global variables ////

//Data setup
int t_final = 72; //number of time points 
int no_col =  72; //number of cols in lattice grid
int no_row = 72; //number of rows in lattice grid
int N = no_col*no_row; //number of (radar) grid squares
int N_g = 15; //number of rain gauges


//MCMC details
int burn_in = 0;
int no_samples = 1;
int thin = 1;
int iters = burn_in + (no_samples*thin); 


//Computatial details
int no_threads = 5; //number of CPUS -- fix = 1 for constant output (with fixed seed)
int fix_seed = 0; //bool: fix seed


//EnKS tuning parameters
int N_ensemble = 100; //number of ensemble members for state sampling. Must ensure N_ensemble/no_threads is an integer.
int inter_time_steps = 7; //number of time steps to insert between observations
int t_final_augment = inter_time_steps*(t_final - 1) + t_final; //length of augmented time-series (\tilde{t})
int window = 3*(inter_time_steps+1); //smoothing window in EnKS -- set to t_final_augment for smoothing all states, l=0,...t.


//Fixed parameters
double phi_gauge = 100.0; //phi_g
double phi_radar = 2.0; //phi_r
double phi_theta = 40.0; //phi_theta
double phi_s = 20.0; //phi_s
double alpha_star = 0.85; //alpha*
double beta_star = 0.15; //beta*
double psi = 1.0; //psi
double alpha_v = 0.95; //alpha_v
double phi_v = 2000; //phi_v
	

//Online forecasts
int generate_forecasts = 1; //bool: set = 1 to generate forecasts n_step_ahead_forecasts for each time point. Otherwise zero
int forecast_thin = 1; //if generate_forecasts = 1, then produce forecasts every forecast_thin iterations (this is applied after the MCMC thin).
int n_step_ahead_forecasts = 6; //numer of (observation) time steps ahead to consider
int n_step_ahead_forecasts_augment = inter_time_steps*(n_step_ahead_forecasts ) + n_step_ahead_forecasts; //length of augmented forecast series


//File paths for inputs
const char* radar_data_file_path = "../Data/Real/radar_data_file.csv";
const char* radar_no_censored_file_path = "../Data/Real/radar_no_censored.csv";
const char* radar_censored_locs_file_path = "../Data/Real/radar_censored_locations.csv";
const char* gauge_data_file_path = "../Data/Real/gauge_data_file.csv";
const char* gauge_locations_file_path = "../Data/Real/gauge_locations.csv";
const char* gauge_no_censored_file_path = "../Data/Real/gauge_no_censored.csv";
const char* gauge_censored_locs_file_path = "../Data/Real/gauge_censored_locations.csv";
const char* neighbour_locs_file_path = "../Data/Real/neighbour_location_data.csv";


//Prior
double mean_mu_radar = 0.0; //prior mean mu_r
double phi_mu_radar = 1.0; //prior precision mu_r

double mean_mu = 0.0; //prior mean mu_r
double phi_mu = 1.0; //prior precision mu_r

double sd_theta_zero = 2.0; //prior SD \theta_0

double mean_alpha = 0.8; //prior mean \beta
double prec_alpha = 250.0; //prior presicion \beta

double mean_beta = 0.1; //prior mean \beta
double prec_beta = 500.0; //prior presicion \beta
																	
double mean_s_zero = 0.0; //prior mean S_0 source sink
double sd_s_zero = 0.5; //prior SD S_0 source sink

double mean_v_x_zero = 0.0; //prior mean of v_x time zero
double mean_v_y_zero = 0.0; //prior mean of v_y time zero
double prec_v_zero = 100.0; //common prior precision



//Functions			
int sample_cum_probs(double U, double* probs, int size); //returns a sample when a cumulative probability array is passed
double sampleNormalTruncatedAbove(gsl_rng* rng, double mu, double sd, double up_lim); //sample from truncated Gaussian in range (-inf, up_lim)
double sampleNormalTruncatedAboveViaLatent(gsl_rng* rng, double current_val, double mu, double sd, double up_lim);
void sampleMultivariateNormal(gsl_rng* rng, int N, Eigen::VectorXd &mean, Eigen::MatrixXd &covar, Eigen::VectorXd &sample);
double LogLikeliMultivariateNormal(Eigen::VectorXd &data_vec, Eigen::VectorXd &mean, Eigen::MatrixXd &prec_mat, int n);
MatrixXd readCSVtoeigenXd(std::string file, int rows, int cols); //read csv file of type double in to eigen matrix
MatrixXi readCSVtoeigenXi(std::string file, int rows, int cols); //read csv file on type int in to eigen matrix 



int main()
{
/////// Read in inputs ///////	
	
	//read in radar related data
	MatrixXd radar_data = readCSVtoeigenXd(radar_data_file_path,N,t_final);
	
	//read in information about radar censoring
	std::ifstream radar_no_cens_file; 
	radar_no_cens_file.open(radar_no_censored_file_path);
	int radar_no_censored;
	radar_no_cens_file >> radar_no_censored; 
	radar_no_cens_file.close();
	MatrixXi radar_censored_locs = readCSVtoeigenXi(radar_censored_locs_file_path, radar_no_censored,1);
	
	//read in gauge related data
	MatrixXd gauge_data = readCSVtoeigenXd(gauge_data_file_path,N_g,t_final);
	
	//read in gauge locations
	MatrixXi gauge_locs = readCSVtoeigenXi(gauge_locations_file_path, N_g,1);
	
	//read in information about gauge censoring
	std::ifstream gauge_no_cens_file; 
	gauge_no_cens_file.open(gauge_no_censored_file_path);
	int gauge_no_censored;
	gauge_no_cens_file >> gauge_no_censored; 
	gauge_no_cens_file.close();
	MatrixXi gauge_censored_locs = readCSVtoeigenXi(gauge_censored_locs_file_path, gauge_no_censored,1);
	
	//read in neighbour locations (in state vector) for each state
	MatrixXi neighbour_locs = readCSVtoeigenXi(neighbour_locs_file_path, N, 5);
		
/////// End of read in inputs ///////


/////// Transform data ///////	

	radar_data.array() += 1.0;
	radar_data.array() = radar_data.array().log();
	gauge_data.array() += 1.0;
	gauge_data.array() = gauge_data.array().log();

/////// End transform data ///////	



/////// Set up/manipulate output streams ///////

	struct stat st = {0};
	if(stat("outputs", &st) == -1) //if output directory does not exists then create it
	{
		mkdir("outputs", 0777);
		printf("Created new output directory as /outputs did not exist\n");
	}
	if(stat("outputs/gauge_states", &st) == -1) //if output directory does not exists then create it
	{
		mkdir("outputs/gauge_states", 0777);
	}
	if(stat("outputs/state_draws", &st) == -1) //if output directory does not exists then create it
	{
		mkdir("outputs/state_draws", 0777);
	}
	if(generate_forecasts && stat("outputs/state_forecasts", &st) == -1) //if forecast output directory does not exists then create it
	{
		mkdir("outputs/state_forecasts", 0777);
	}

	int my_prec = 15; //number of decimal placed to print
	std::ofstream mu_radar_file, mu_file,alpha_file,alpha_latent_file,beta_file, v_x_file,v_y_file,log_observed_file,log_complete_file;//output streams
	mu_radar_file.open("outputs/mu_radar_output.csv"); //open files
	mu_file.open("outputs/mu_output.csv");
	alpha_file.open("outputs/alpha_output.csv");
	alpha_latent_file.open("outputs/alpha_latent_output.csv");
	beta_file.open("outputs/beta_output.csv");
	v_x_file.open("outputs/v_x_output.csv");
	v_y_file.open("outputs/v_y_output.csv");
	log_observed_file.open("outputs/log_observed_output.csv");
	log_complete_file.open("outputs/log_complete_output.csv");
	mu_radar_file << std::fixed << std::setprecision(my_prec); //set presicion
	mu_file << std::fixed << std::setprecision(my_prec);
	alpha_file << std::fixed << std::setprecision(my_prec);
	alpha_latent_file << std::fixed << std::setprecision(my_prec);
	beta_file << std::fixed << std::setprecision(my_prec);
	v_x_file << std::fixed << std::setprecision(my_prec);
	v_y_file << std::fixed << std::setprecision(my_prec);
	log_observed_file<< std::fixed << std::setprecision(my_prec);
	log_complete_file<< std::fixed << std::setprecision(my_prec);
	const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ",", "\n"); //set CSVFormat
	std::cout << std::fixed << std::setprecision(10); //set presicion for std::cout -- used in debugging
	
	//open binary files for saving draws of the (theta) states at the gauge locations
	std::string gauge_state_filenames_bin[N_g];
	std::ofstream gauge_state_outputFiles_bin[N_g];
	for(int i = 0; i < N_g; i++)
	{
		std::ostringstream os;
		os << "outputs/gauge_states/gauge_location_" << i+1 << ".csv";
		gauge_state_filenames_bin[i] =  os.str();
		gauge_state_outputFiles_bin[i].open(gauge_state_filenames_bin[i].c_str());
		gauge_state_outputFiles_bin[i] << std::fixed << std::setprecision(15);
	}
	
	
	std::string state_forecast_filenames_bin[n_step_ahead_forecasts];
	std::ofstream state_forecast_outputFiles_bin[n_step_ahead_forecasts];
	for(int i = 0; i < n_step_ahead_forecasts; i++)
	{
		std::ostringstream os;
		os << "outputs/state_forecasts/state_forecasts_time_" << i+1 << ".bin";
		state_forecast_filenames_bin[i] =  os.str();
		state_forecast_outputFiles_bin[i].open(state_forecast_filenames_bin[i].c_str(), std::ios::out | std::ios::binary);
	}
	
	std::string state_draws_filenames_bin[t_final];
	std::ofstream state_draws_outputFiles_bin[t_final];
	for(int i = 0; i < t_final; i++)
	{
		std::ostringstream os;
		os << "outputs/state_draws/state_draws_time_" << i+1 << ".bin";
		state_draws_filenames_bin[i] =  os.str();
		state_draws_outputFiles_bin[i].open(state_draws_filenames_bin[i].c_str(), std::ios::out | std::ios::binary);
	}
				
	
/////// End set up/manipulate output streams ///////
	
	
	
/////// Manipulate priors into useful form ///////
	
	VectorXd mean_v_zero(2); //load prior for v into (column) vector and matrix
	MatrixXd prec_v_zero_mat(2,2); //identity matrix -- dimension 2
	prec_v_zero_mat.setIdentity();	
	mean_v_zero(0) = mean_v_x_zero; //prior mean of v_x time zero
	mean_v_zero(1) = mean_v_y_zero; //prior mean of v_y time zero
	prec_v_zero_mat *= prec_v_zero;
	
/////// End of manipulate priors into useful form ///////
	
	
/////// Change innovation vairance depending on the number of imputed timesteps ///////

	phi_theta *= (inter_time_steps + 1);
	phi_s *= (inter_time_steps + 1);

/////// End change innovation vairance depending on the number of imputed timesteps ///////


/////// Initialisation ///////

	gsl_rng *rng[no_threads]; //initalise RNG
	int seed = time(NULL);
	for(int i = 0; i < no_threads; i++)
	{
		rng[i]=gsl_rng_alloc(gsl_rng_mt19937);
		if(fix_seed) //fixed initalisation
		{
			gsl_rng_set(rng[i],900000*i+10); 
		}else //random initalisation
		{
			gsl_rng_set(rng[i],900000*i + seed); 
		}
	}
	std::ofstream seed_file; //output seed
	seed_file.open("outputs/seed.txt");
	seed_file << std::fixed << std::setprecision(15);
	seed_file << "seed:" << seed << "\n";
	seed_file.close();
	
	
	double mu_radar = mean_mu_radar + gsl_ran_gaussian(rng[0], sqrt(1.0/phi_mu_radar)); //initalise \mu_r
	
	double mu = mean_mu + gsl_ran_gaussian(rng[0], sqrt(1.0/phi_mu)); //initalise mu (latent mean)
	
	double alpha = 0.9; //alpha, ar param (decay)
	double beta = mean_beta + gsl_ran_gaussian(rng[0], sqrt(1.0/prec_beta)); //beta, spatial (neighbour) weight
	double norm_alpha = (alpha - mean_alpha)/sqrt(1.0/prec_alpha); //normalised alpha - used in FCD sampling
	double log_latent_tn_alpha; //latent variable used in sampling truncated Gaussian (log scale)
				
	
	MatrixXd velocity_mat(2,t_final_augment);
	velocity_mat(0,0) = mean_v_x_zero + gsl_ran_gaussian(rng[0], sqrt(1.0/prec_v_zero)); //initalise \v_x_0
	velocity_mat(1,0) = mean_v_y_zero + gsl_ran_gaussian(rng[0], sqrt(1.0/prec_v_zero)); //initalise \v_x_0
	for(int t = 1; t < (t_final_augment); t++) //evolve
	{
		velocity_mat(0,t) = alpha_v*velocity_mat(0,t-1) + gsl_ran_gaussian(rng[0], sqrt(1.0/phi_v));
		velocity_mat(1,t) = alpha_v*velocity_mat(1,t-1) + gsl_ran_gaussian(rng[0], sqrt(1.0/phi_v));
	}


	//// Augment radar data ////
	for(int l = 0; l < radar_no_censored; l++)
	{
		radar_data(radar_censored_locs(l,0)) = sampleNormalTruncatedAbove(rng[0], mu + mu_radar , 2.0*sqrt(1.0/phi_radar), 0.0);
	}
	//~ //// End augment radar data ////
	
	
	//// Augment gauge data ////
	for(int l = 0; l < gauge_no_censored; l++)
	{
		gauge_data(gauge_censored_locs(l,0)) = sampleNormalTruncatedAbove(rng[0], mu, 2.0*sqrt(1.0/phi_gauge), 0.0);
		
	}
	//// End augment gauge data ////

	
/////// End initialisation ///////
	
	
	
	
	
/////// Declare objects used in computation ///////

	MatrixXd *state_array = new MatrixXd[t_final_augment+1]; //hold latent states in array of eigen matricies, unique eigen matrix for each time point (not ensemble member) as its faster
	for(int j = 0; j < (t_final_augment + 1); j++) //first N entries (in each col) hold \theta, final N entries hold S (source sink)
	{
		state_array[j] = MatrixXd(2*N, N_ensemble); //row is state, col is ensemble member
	}

	MatrixXd state_sample(N, t_final_augment+1); //holds sample of states x = (\theta,S)
	MatrixXd gauge_state_sample(N_g, t_final+1); //holds sample of (theta) states for gauge locations
	MatrixXd radar_state_sample(N,t_final+1); //holds sample of (theta) states for radar locations
	MatrixXd source_sink_sample(N,t_final_augment+1); //holds sample of (S) source-sink states for all locations and (augmented) times
	
	MatrixXd post_mean_state = MatrixXd::Zero(N, t_final+1); //array to hold posterior mean of latent states -- only calulcated if indicator (above) = 1
	MatrixXd post_mean_source_sink = MatrixXd::Zero(N,t_final_augment+1); //array to hold posterior mean of latent states -- only calulcated if indicator (above) = 1
	MatrixXd post_mean_radar_obs = MatrixXd::Zero(N, t_final); //array to hold posterior mean of latent states -- only calulcated if indicator (above) = 1
	MatrixXd post_sd_radar_obs = MatrixXd::Zero(N, t_final); //array to hold posterior mean of latent states -- only calulcated if indicator (above) = 1
	MatrixXd post_mean_gauge_obs = MatrixXd::Zero(N_g, t_final); //array to hold posterior mean of latent states -- only calulcated if indicator (above) = 1
	MatrixXd post_sd_gauge_obs = MatrixXd::Zero(N_g, t_final); //array to hold posterior mean of latent states -- only calulcated if indicator (above) = 1
	MatrixXd tmp_radar_obs = MatrixXd::Zero(N, t_final); //array to hold posterior mean of latent states -- only calulcated if indicator (above) = 1
	MatrixXd tmp_radar_obs_2 = MatrixXd::Zero(N, t_final); //array to hold posterior mean of latent states -- only calulcated if indicator (above) = 1
	MatrixXd tmp_gauge_obs = MatrixXd::Zero(N_g, t_final); //array to hold posterior mean of latent states -- only calulcated if indicator (above) = 1
	MatrixXd tmp_gauge_obs_2 = MatrixXd::Zero(N_g, t_final); //array to hold posterior mean of latent states -- only calulcated if indicator (above) = 1
	
	//matricies used in EnKS
	MatrixXd cent_t(2*N,N_ensemble); //houses centered states at time t
	MatrixXd cent_t_gauge(N_g,N_ensemble); //houses centered states that correspond to gauge locations at time t
	MatrixXd cent_l(2*N,N_ensemble); //houses centered states at time l (theta - \bar{theta})
	MatrixXd Pseudo_obs(N+N_g, N_ensemble); //houses pseudo observations
	MatrixXd A(N+N_g,N_ensemble); //A = F * cent_t
	MatrixXd Binv_P(N+N_g,N_ensemble); //B^{-1}P, where B = (W^{-1} + t(F)V^{-1}F)^{-1}) but is never constructed
	MatrixXd Binv_A(N+N_g,N_ensemble); //B^{-1}A, where B = (W^{-1} + t(F)V^{-1}F)^{-1}) but is never constructed
	MatrixXd AT_Binv_A(N_ensemble,N_ensemble); //t(A)B^{-1}A
	MatrixXd identity_Ne(N_ensemble,N_ensemble); //identity matrix -- dimension N^e
	identity_Ne.setIdentity(); 
	MatrixXd inverse_NeNe(N_ensemble,N_ensemble); //matrix to hold inverse-- dimension N^e
	MatrixXd AT_Binv_P(N_ensemble,N_ensemble); //t(A)B^{-1}P
	MatrixXd tmp_NENE(N_ensemble,N_ensemble);
	MatrixXd tmp_mat_NNg_Ne(N+N_g,N_ensemble);
	MatrixXd R_NENE(N_ensemble,N_ensemble); //temporary matrix
	MatrixXd tmp_mat_2NNE(2*N,N_ensemble); //temporary matrix
	
	
	//matricies using in parameter inference
	MatrixXd tmp_mat_NT(N,t_final);
	MatrixXd tmp_mat_NgT(N_g,t_final);
	MatrixXd g_tilde_theta(N,1); //holds tilde{g}*theta
	VectorXd g_star_tilde_s(N); //holds tilde{g}*theta
	VectorXd tmp_vec_N_1(N);
	VectorXd tmp_vec_N_2(N);
	MatrixXd theta_minus_g_theta(N,t_final_augment);
	MatrixXd identity_2(2,2); //identity matrix -- dimension 2
	identity_2.setIdentity(); 
	
	//beta vec - function of alpha,beta,v_x,v_y, controls theta evolution
	VectorXd beta_vec(5); 
	beta_vec(0) = beta + velocity_mat(0,0); //beta + v_x time zero
	beta_vec(1) = beta - velocity_mat(1,0); //beta - v_y time zero
	beta_vec(2) = beta + velocity_mat(1,0); //beta + v_y time zero
	beta_vec(3) = beta - velocity_mat(0,0); //beta - v_x time zero
	beta_vec(4) = 1.0 - 4.0*beta;
	beta_vec.array() *= alpha;
	
	//beta* vec - function of alpha*,beta*, controls source sink evolution
	VectorXd beta_star_vec(5); 
	beta_star_vec(0) = beta_star;
	beta_star_vec(1) = beta_star;
	beta_star_vec(2) = beta_star;
	beta_star_vec(3) = beta_star;
	beta_star_vec(4) = 1.0 - 4.0*beta_star;
	beta_star_vec.array() *= alpha_star;
	
	//matricies using in generating forecasts
	MatrixXd theta_forecasts(N,n_step_ahead_forecasts_augment+1); //
	MatrixXd source_sink_forecasts(N,n_step_ahead_forecasts_augment+1); //
	MatrixXd observation_forecasts(N+N_g,1); //
	double current_v_x;
	double current_v_y;
	VectorXd beta_vec_forecast(5);
	
	double* int_cum_probs_Ne = (double*)malloc(N_ensemble * sizeof(double)); //used to sample from discrete uniform dist
    int_cum_probs_Ne[0] = 1.0/N_ensemble;
    for(int i = 1; i < N_ensemble; i++)
    {
		int_cum_probs_Ne[i] = int_cum_probs_Ne[i-1] + 1.0/N_ensemble;
	}
	
	RowVectorXi ng_vec = RowVectorXi::Zero(N); //holds number of gauges at each location
	for(int g = 0; g < N_g; g++)
	{
		ng_vec(gauge_locs(g,0)) ++;
	}
	
	
/////// End declare objects used in computation ///////
	
	

	
/////// Manipulate data ///////

	radar_data.array() -= mu_radar; //radar data - mu_radar

/////// End manipulate data ///////	
	
	
/////// Parallel computing ///////
	
	int chunk_size = N_ensemble/no_threads; //set "chunk_size" i.e. how many jobs to place on each thread. Must be integer.
	omp_set_num_threads(no_threads); //set number of threads to use
	
/////// Parallel computing ///////	
	
	
	
	
	
/////// START MCMC ///////

for(int it = 0; it < iters; it++) 
{
	

	//// Draw states via EnKS////
		double state_precision; //possibly inflated covariance \theta evolution
		double source_sink_precision; //possibly inflated covariance S evolution
		state_precision = phi_theta; //no inflation
		source_sink_precision = phi_s; //no inflation
		
		
	
		#pragma omp parallel for schedule(static,chunk_size) default(shared) 
		for(int j = 0; j < N_ensemble; j++) //draw samples for x_0 = (\theta_0,S_0)
		{
			for(int i = 0; i < N; i++)
			{
				state_array[0](i,j) = mu + gsl_ran_gaussian(rng[omp_get_thread_num()], sd_theta_zero); //initalise theta
				state_array[0](N+i,j) = mean_s_zero + gsl_ran_gaussian(rng[omp_get_thread_num()], sd_s_zero); //initalise S source sink
			}
		}		

		int tmp_count = 0;
		double ran_U = gsl_rng_uniform(rng[0]);
		int ensemble_member = sample_cum_probs(ran_U, int_cum_probs_Ne, N_ensemble); //sample ensemble member - need to do this first to generate in-sample predictions
		beta_vec(4) = alpha*(1.0-4*beta); //static
		for(int t = 1; t < (t_final_augment + 1); t++) //loop over (aufmented) time
		{
			
			int obs_time = (((t - 1) % (inter_time_steps + 1)) == 0);
			if(obs_time) tmp_count ++;
			
			
			beta_vec(0) = alpha*(beta + velocity_mat(0,t-1)); //beta + v_x time zero
			beta_vec(1) = alpha*(beta - velocity_mat(1,t-1)); //beta - v_y time zero
			beta_vec(2) = alpha*(beta + velocity_mat(1,t-1)); //beta + v_y time zero
			beta_vec(3) = alpha*(beta - velocity_mat(0,t-1)); //beta - v_x time zero
			
			//compute deterministic forecast ensemble, that is \tilde{G} * theta_{t-1} without matrix multiplication for each ensemble member
			#pragma omp parallel for schedule(static,chunk_size) default(shared)  
			for(int j = 0; j < N_ensemble; j++) //loop over ensembles
			{
				for(int i = 0; i < N; i++) //evolve theta
				{	
					double tmp = 0.0;
					double tmp_s = 0.0;
					for(int k = 0; k < 5; k++)
					{
						if(neighbour_locs(i,k)>-1)
						{
							tmp += beta_vec(k)*(state_array[t-1](neighbour_locs(i,k),j) - mu); //theta evolution
							tmp_s += beta_star_vec(k)*state_array[t-1](N+neighbour_locs(i,k),j); //source sink evolution
						}
					}
					state_array[t](i,j) = tmp + mu + state_array[t-1](N+i,j); //G * \theta + S
					state_array[t](N+i,j) = tmp_s; // G* * S
				}
			}
			
			if(obs_time) //Compute the matricies containing the centered (deterministic) states
			{
				cent_t.noalias() = state_array[t].colwise() - state_array[t].rowwise().mean(); //matrix containing centered states (x) at time t (dimension 2*N by N_ensemble) //Memory being allocated here
				cent_t /= sqrt(N_ensemble - 1.0); //now cent_t * t(cent_t) + W gives sample covar of states
				for(int g = 0; g < N_g; g++) //pull out centred states that correspond to gauge locations
				{
					cent_t_gauge.row(g) = cent_t.row(gauge_locs(g,0));
				}
			}
			
			//compute the noisy ensemble - add noise to the states
			#pragma omp parallel for schedule(static,chunk_size) default(shared)  
			for(int j = 0; j < N_ensemble; j++) //loop over ensembles
			{
				for(int i = 0; i < N; i++)
				{	
					state_array[t](i,j) += gsl_ran_gaussian(rng[omp_get_thread_num()], sqrt(1.0/state_precision)); //add state error for theta
					state_array[t](N+i,j) += gsl_ran_gaussian(rng[omp_get_thread_num()], sqrt(1.0/source_sink_precision)); //add state error for S
				}
			}
			
			if(obs_time)
				{
				//generate pseudo observations, that is, psuedo radar and gauge observations and place in a matrix (P)
				Pseudo_obs.block(0,0,N,N_ensemble).colwise() = radar_data.col(tmp_count-1); //load in radar data
				Pseudo_obs.block(N,0,N_g,N_ensemble).colwise() = gauge_data.col(tmp_count-1); //load in gauge data
				Pseudo_obs.block(0,0,N,N_ensemble) -= psi*state_array[t].block(0,0,N,N_ensemble); //actual radar observations - deterministic pesudo radar obs
				for(int g = 0; g < N_g; g++)
				{
					Pseudo_obs.row(N+g) -= state_array[t].row(gauge_locs(g,0)); //actual gauge observations - deterministic pesudo radar obs
				}
				
				#pragma omp parallel for schedule(static,chunk_size) default(shared) //add noise
				for(int j = 0; j < N_ensemble; j++) //loop over ensemble
				{
					for(int i = 0; i < N; i++) //loop over radar states
					{
						Pseudo_obs(i,j) -= gsl_ran_gaussian(rng[omp_get_thread_num()], sqrt(1.0/phi_radar));
					}
					for(int g = 0; g < N_g; g++) //loop over gauge states
					{
						Pseudo_obs(N+g,j) -= gsl_ran_gaussian(rng[omp_get_thread_num()], sqrt(1.0/phi_gauge));
					}
				}
				
				//compute ingreedients needed for efficient Kalman gain estimation
				A.block(0,0,N,N_ensemble).noalias() = psi*cent_t.block(0,0,N,N_ensemble); //compute A = F cent_t (deterministic states)
				A.block(N,0,N_g,N_ensemble).noalias() = cent_t_gauge;
				
				
				//compute B^{-1}P and B^{-1}A
				//deal with radar only locations first - these are straighforward
				double radar_only_scaling = (1.0 - (psi*psi*phi_radar/(state_precision + psi*psi*phi_radar)));
				#pragma omp parallel for schedule(static,chunk_size) default(shared) 
				for(int i = 0; i < N; i++)
				{
					if(ng_vec(i) == 0) //radar only location
					{
						Binv_P.row(i) = radar_only_scaling*Pseudo_obs.row(i);
						Binv_A.row(i) = radar_only_scaling*A.row(i);
					}
				}
				
				//now deal with the more tricky gauge locations
				RowVectorXd tmp_vec(N_ensemble);
				RowVectorXd tmp_vec_1(N_ensemble);
				for(int g = 0; g < N_g; g++)
				{
					tmp_vec = psi*phi_radar*Pseudo_obs.row(gauge_locs(g,0));
					tmp_vec_1 = psi*phi_radar*A.row(gauge_locs(g,0));
					for(int ell = 0; ell < ng_vec(gauge_locs(g,0)); ell++)
					{
						tmp_vec += phi_gauge*Pseudo_obs.row(N+g+ell);
						tmp_vec_1 += phi_gauge*A.row(N+g+ell);	
					}
					double radar_gauge_scaling = state_precision + psi*psi*phi_radar + ng_vec(gauge_locs(g,0))*phi_gauge;
					tmp_vec /= radar_gauge_scaling;
					tmp_vec_1 /= radar_gauge_scaling;
					Binv_P.row(gauge_locs(g,0)) =  Pseudo_obs.row(gauge_locs(g,0)) - psi*(tmp_vec);
					Binv_A.row(gauge_locs(g,0)) =  A.row(gauge_locs(g,0)) - psi*(tmp_vec_1);
					for(int ell = 0; ell < ng_vec(gauge_locs(g,0)); ell++)
					{
						Binv_P.row(N+g+ell) = Pseudo_obs.row(N+g+ell) - (tmp_vec);
						Binv_A.row(N+g+ell) = A.row(N+g+ell) - (tmp_vec_1);
					}
					g += ng_vec(gauge_locs(g,0)) - 1; //correct the count - need to bump g on if we've processed numerous gauges
					
				}
				
				Binv_P.block(0,0,N,N_ensemble).array() *= phi_radar;
				Binv_P.block(N,0,N_g,N_ensemble).array() *= phi_gauge;

				Binv_A.block(0,0,N,N_ensemble).array() *= phi_radar;
				Binv_A.block(N,0,N_g,N_ensemble).array() *= phi_gauge;
											
				//get final ingreedients through straightforward matrix multplication
				
				AT_Binv_A.noalias() = A.transpose() * Binv_A;
				AT_Binv_A.diagonal().array() += 1.0; //need to invert this
				
				AT_Binv_P.noalias() = A.transpose() * Binv_P;
				
				tmp_NENE.noalias() = AT_Binv_A.llt().solve(AT_Binv_P);
				
				tmp_mat_NNg_Ne.noalias() = Binv_A * tmp_NENE;
				
				Binv_P -= tmp_mat_NNg_Ne; //this is now Q^{-1}P !!!
				
				R_NENE.noalias() = A.transpose() * Binv_P;
				
				//smooth states at time t
				tmp_mat_2NNE.noalias() = cent_t * R_NENE; //smooth states at time t, cent t is centered determinsitc states
				state_array[t] += tmp_mat_2NNE;
				
				//correction term at time t
				int count = 0;
				for(int i = 0; i < N; i++)
				{
					if(ng_vec(i) == 0) //radar only location
					{
						state_array[t].row(i) += (psi/state_precision)*Binv_P.row(i);
					}else{
						tmp_vec = psi*Binv_P.row(i);
						for(int ell = 0; ell < ng_vec(i); ell++)
						{
							tmp_vec += Binv_P.row(N+count);
							if(count != (N_g-1)){
							count ++;
						}
						}
						state_array[t].row(i) += tmp_vec/state_precision;
					}
				}
				
				//smooth hisotic states
				R_NENE.array() /= sqrt(N_ensemble-1); //saves dividing in l loop
				int smooth_limit = 0;
				if(t - window > 0)
				{
					smooth_limit = t - window;
				}
				for(int l = smooth_limit; l < (t); l++) //``smooth'' the states
				{
					cent_l.noalias() = state_array[l].colwise() - state_array[l].rowwise().mean(); //centred states at time l
					tmp_mat_2NNE.noalias() = cent_l * R_NENE;
					state_array[l] += tmp_mat_2NNE;
				}
			}
		}
		tmp_count = 0;
		for(int t = 0; t < (t_final_augment + 1); t++) //loop over time to pull out all states for this ensemble member
		{
			state_sample.col(t) = state_array[t].col(ensemble_member).head(N);
			source_sink_sample.col(t) = state_array[t].col(ensemble_member).tail(N);
			
			if((((t - 1) % (inter_time_steps + 1)) == 0) || t == 0)
			{
				radar_state_sample.col(tmp_count) = state_array[t].col(ensemble_member).head(N);
				tmp_count ++;
			}
		}
		for(int g = 0; g < N_g; g++) //pull out (theta) states that correspond to gauge locations
		{
			gauge_state_sample.row(g) = radar_state_sample.row(gauge_locs(g,0));
		}

	//// End draw states via EnKS////	
	
	
	
	
	//// Augment radar data ////
	#pragma omp parallel for schedule(static,1) default(shared)
	for(int l = 0; l < radar_no_censored; l++)
	{
		radar_data(radar_censored_locs(l,0)) = sampleNormalTruncatedAboveViaLatent(rng[omp_get_thread_num()],radar_data(radar_censored_locs(l,0)) + mu_radar, mu_radar + (psi*radar_state_sample(radar_censored_locs(l,0)+N)), sqrt(1.0/phi_radar), 0.0) - mu_radar;
	}
	//// End augment radar data ////
			
	//// Augment gauge data ////
	#pragma omp parallel for schedule(static,1) default(shared)
	for(int l = 0; l < gauge_no_censored; l++)
	{
		gauge_data(gauge_censored_locs(l,0)) = sampleNormalTruncatedAboveViaLatent(rng[omp_get_thread_num()], gauge_data(gauge_censored_locs(l,0)), gauge_state_sample(gauge_censored_locs(l,0)+N_g), sqrt(1.0/phi_gauge), 0.0);
	}
	//// End augment gauge data ////
		
	
	//// Sample mean mu ////
	
	double mu_fcd_prec = 0; //initalise so we can build the sum
	double mu_fcd_mean = 0;
	mu_fcd_prec = (1.0 - alpha)*(1.0-alpha)*N*t_final_augment*phi_theta + N*sqrt(1.0/sd_theta_zero) + phi_mu;
	beta_vec(4) = alpha*(1.0 - 4.0*beta); //static
	for(int t = 1; t < (t_final_augment+1); t++) //loop over T 
	{
		tmp_vec_N_1.transpose() = state_sample.col(t) - source_sink_sample.col(t-1);
		beta_vec(0) = alpha*(beta + velocity_mat(0,t-1)); //beta + v_x time t-1
		beta_vec(1) = alpha*(beta - velocity_mat(1,t-1)); //beta - v_y time t-1
		beta_vec(2) = alpha*(beta + velocity_mat(1,t-1)); //beta + v_y time t-1
		beta_vec(3) = alpha*(beta - velocity_mat(0,t-1)); //beta - v_x time t-1
		
		for(int i = 0; i < N; i++) //compute \tilde{G} * theta_{t-1}  ( \tilde{G} is just G without being multiplied by alpha)
		{	
			for(int k = 0; k < 5; k++)
			{
				if(neighbour_locs(i,k)>-1)
				{
					tmp_vec_N_1(i) -=  beta_vec(k)*state_sample(neighbour_locs(i,k),t-1);
				}
			}
		}
		mu_fcd_mean += tmp_vec_N_1.sum();
	}
	
	mu_fcd_mean *= (1.0-alpha)*phi_theta;
	mu_fcd_mean += sqrt(1.0/sd_theta_zero)*state_sample.col(0).sum();
	mu_fcd_mean += mean_mu*phi_mu;
	mu_fcd_mean /= mu_fcd_prec;
	mu = mu_fcd_mean + gsl_ran_gaussian(rng[0], sqrt(1.0/mu_fcd_prec)); //sample alpha from Gaussian	
	
	//// End sample mean mu ////
	
		
	state_sample.array() -= mu;
		
		//// Draw alpha  ////
		
		double alpha_fcd_covar = 0; //initalise so we can build the sum
		double alpha_fcd_mean = 0;	
		beta_vec(4) = 1.0 - 4.0*beta; //static
		for(int t = 1; t < (t_final_augment+1); t++) //loop over T 
		{
			tmp_vec_N_1.transpose() = state_sample.col(t) - source_sink_sample.col(t-1);
			beta_vec(0) = (beta + velocity_mat(0,t-1)); //beta + v_x time t-1
			beta_vec(1) = (beta - velocity_mat(1,t-1)); //beta - v_y time t-1
			beta_vec(2) = (beta + velocity_mat(1,t-1)); //beta + v_y time t-1
			beta_vec(3) = (beta - velocity_mat(0,t-1)); //beta - v_x time t-1
			
			for(int i = 0; i < N; i++) //compute \tilde{G} * theta_{t-1}  ( \tilde{G} is just G without being multiplied by alpha)
			{	
				double tmp_sum = 0;
				for(int k = 0; k < 5; k++)
				{
					if(neighbour_locs(i,k)>-1)
					{
						tmp_sum +=  beta_vec(k)*state_sample(neighbour_locs(i,k),t-1);
					}
				}
				tmp_vec_N_2(i) = tmp_sum;
			}
			
			alpha_fcd_covar += tmp_vec_N_2.transpose()*tmp_vec_N_2;
			alpha_fcd_mean +=  tmp_vec_N_1.transpose()*tmp_vec_N_2;
		}

		alpha_fcd_mean *= phi_theta;
		alpha_fcd_covar*= phi_theta;
		alpha_fcd_mean += mean_alpha*prec_alpha;
		alpha_fcd_covar += prec_alpha;
		alpha_fcd_mean /= alpha_fcd_covar;
		
		//sample log latent parameter via invese CDF
		double alpha_sig = sqrt(1.0/alpha_fcd_covar);
		norm_alpha  = (alpha - alpha_fcd_mean)/alpha_sig;
		
		ran_U = gsl_rng_uniform(rng[0]); //uniform rv
		log_latent_tn_alpha = (-0.5*norm_alpha*norm_alpha) + log(ran_U); //Sample log latent variable
		
		//sample alpha from FCD (uniform)
		double srt2ly = sqrt(-2.0*log_latent_tn_alpha);
		double upper_lim = alpha_fcd_mean + srt2ly*alpha_sig; //upper limit
		double lower_lim = alpha_fcd_mean - srt2ly*alpha_sig; //lower limit				
		if(upper_lim > 1.0) upper_lim = 1.0; //truncate
		if(lower_lim < 0.0) lower_lim = 0.0; //trunate
		if(lower_lim > upper_lim) lower_lim = upper_lim - 0.02; //nugget to move lower limit away from upper limit if nessesary
		alpha = gsl_ran_flat(rng[0], lower_lim, upper_lim); //Sample normalised alpha value
		
		//// End of draw alpha////

		
		
		//// Draw beta ////
		
		double beta_fcd_covar = 0.0;
		double beta_fcd_mean = 0.0;
		//#pragma omp parallel for schedule(static,1) default(shared) 
		for(int t = 1; t < (t_final_augment+1); t++) //loop over T 
		{
			tmp_vec_N_1.transpose() = state_sample.col(t) - alpha*state_sample.col(t-1) - source_sink_sample.col(t-1);
			//form theta* -- could we avoid doing this?
			for(int i = 0; i < N; i++) 
			{
				tmp_vec_N_1(i) -= alpha*velocity_mat(0,t-1) * (state_sample(neighbour_locs(i,0),t-1) - state_sample(neighbour_locs(i,3),t-1)); //left - right neighbour
				tmp_vec_N_1(i) -= alpha*velocity_mat(1,t-1) * (state_sample(neighbour_locs(i,2),t-1) - state_sample(neighbour_locs(i,1),t-1)); //down - up neighbour
				
				double tmp = 0;
				for(int k = 0; k < 4; k++) //sum 4 neighbours
				{
					tmp += alpha*state_sample(neighbour_locs(i,k),t-1);
					
				}
				tmp_vec_N_2(i) = tmp - 4.0*alpha*state_sample(i,t-1);
				
			}
			beta_fcd_covar += tmp_vec_N_2.transpose()*tmp_vec_N_2;
			beta_fcd_mean += tmp_vec_N_1.transpose()*tmp_vec_N_2;
			
		}
		beta_fcd_covar *= phi_theta;
		beta_fcd_mean *= phi_theta;
		beta_fcd_covar += prec_beta;
		beta_fcd_mean += prec_beta * mean_beta;			
		beta_fcd_mean /= beta_fcd_covar;
		beta = beta_fcd_mean + gsl_ran_gaussian(rng[0], sqrt(1.0/beta_fcd_covar));
		
		//// End of draw beta////
		
	
	
		//// Start of sampling velocity vectors (v_0,...,v_T) ////	
		
		VectorXd mean_fcd_v(2); //
		VectorXd sample(2); //
		MatrixXd prec_fcd_v(2,2); //identity matrix -- dimension 2
		
		MatrixXd R_mat(N,2); //
		VectorXd A_vec(N); //
	
		//////V_0
		A_vec = state_sample.col(1);
		A_vec -= source_sink_sample.col(0);
		A_vec -= alpha*(1.0-4.0*beta)*state_sample.col(0);
		for(int i = 0; i < N; i++) 
		{
			R_mat(i,0) = alpha * (state_sample(neighbour_locs(i,0),0) - state_sample(neighbour_locs(i,3),0)); //left - right neighbour
			R_mat(i,1) = alpha * (state_sample(neighbour_locs(i,2),0) - state_sample(neighbour_locs(i,1),0)); //down - up neighbour
			
			for(int k = 0; k < 4; k++)
			{
				if(neighbour_locs(i,k)>-1)
				{
					A_vec(i) -= alpha*beta*state_sample(neighbour_locs(i,k),0);
				}
			}
			
		}
		prec_fcd_v = R_mat.transpose()*R_mat;
		prec_fcd_v *= phi_theta;
		prec_fcd_v.diagonal().array() += phi_v*alpha_v*alpha_v;
		prec_fcd_v += prec_v_zero_mat;
		prec_fcd_v = prec_fcd_v.llt().solve(identity_2); //invert
		
		
		mean_fcd_v.transpose() = A_vec.transpose()*R_mat;
		mean_fcd_v *= phi_theta;
		mean_fcd_v += mean_v_zero*prec_v_zero;
		mean_fcd_v += phi_v*alpha_v*velocity_mat.col(1);
		mean_fcd_v = prec_fcd_v*mean_fcd_v; //can not use *= due to right multiplication
		sampleMultivariateNormal(rng[0], 2, mean_fcd_v, prec_fcd_v, sample); //sample 
		velocity_mat.col(0) = sample; //load sample into velocity vector
	
		
		//v_1,...v_{T-2}
		for(int t = 1; t < (t_final_augment-1); t++)
		{
			A_vec = state_sample.col(t+1);
			A_vec -= source_sink_sample.col(t);
			A_vec -= alpha*(1.0-4.0*beta)*state_sample.col(t);
			for(int i = 0; i < N; i++) 
			{
				R_mat(i,0) = alpha * (state_sample(neighbour_locs(i,0),t) - state_sample(neighbour_locs(i,3),t)); //left - right neighbour
				R_mat(i,1) = alpha * (state_sample(neighbour_locs(i,2),t) - state_sample(neighbour_locs(i,1),t)); //down - up neighbour
				
				for(int k = 0; k < 4; k++)
				{
					if(neighbour_locs(i,k)>-1)
					{
						A_vec(i) -= alpha*beta*state_sample(neighbour_locs(i,k),t);
					}
				}
			}
			prec_fcd_v = R_mat.transpose()*R_mat;
			prec_fcd_v *= phi_theta;
			prec_fcd_v.diagonal().array() += phi_v*(1.0 + alpha_v*alpha_v);
			prec_fcd_v = prec_fcd_v.llt().solve(identity_2); //invert
			
			
			mean_fcd_v.transpose() = A_vec.transpose()*R_mat;
			mean_fcd_v *= phi_theta;
			
			mean_fcd_v += phi_v*alpha_v*(velocity_mat.col(t-1) + velocity_mat.col(t+1));
			mean_fcd_v = prec_fcd_v*mean_fcd_v; //can not use *= due to right multiplication
			sampleMultivariateNormal(rng[0], 2, mean_fcd_v, prec_fcd_v, sample); //sample 
			velocity_mat.col(t) = sample; //load sample into velocity vector
		}
		
		
		//v_T-1
		A_vec = state_sample.col(t_final_augment);
		A_vec -= source_sink_sample.col(t_final_augment -1);
		A_vec -= alpha*(1.0-4.0*beta)*state_sample.col(t_final_augment-1);
		for(int i = 0; i < N; i++) 
		{
			R_mat(i,0) = alpha * (state_sample(neighbour_locs(i,0),t_final_augment-1) - state_sample(neighbour_locs(i,3),t_final_augment-1)); //left - right neighbour
			R_mat(i,1) = alpha * (state_sample(neighbour_locs(i,2),t_final_augment-1) - state_sample(neighbour_locs(i,1),t_final_augment-1)); //down - up neighbour
			
			for(int k = 0; k < 4; k++)
			{
				if(neighbour_locs(i,k)>-1)
				{
					A_vec(i) -= alpha*beta*state_sample(neighbour_locs(i,k),t_final_augment-1);
				}
			}
			
		}
		prec_fcd_v = R_mat.transpose()*R_mat;
		prec_fcd_v *= phi_theta;
		prec_fcd_v.diagonal().array() += phi_v;
		prec_fcd_v = prec_fcd_v.llt().solve(identity_2); //invert
		mean_fcd_v.transpose() = A_vec.transpose()*R_mat;
		mean_fcd_v *= phi_theta;
		mean_fcd_v += phi_v*alpha_v*velocity_mat.col(t_final_augment-2);
		mean_fcd_v = prec_fcd_v*mean_fcd_v; //can not use *= due to right multiplication
		sampleMultivariateNormal(rng[0], 2, mean_fcd_v, prec_fcd_v, sample); //sample 
		velocity_mat.col(t_final_augment-1) = sample; //load sample into velocity vector
		
		//// End of sampling velocity vectors (v_0,...,v_T) ////	
	
	
	state_sample.array() += mu;
	
	
	//// Draw mu_radar ////
	
	radar_data.array() += mu_radar; //normal Y now
	tmp_mat_NT.noalias() = radar_data - (psi * radar_state_sample.block(0,1,N,t_final));
	double mu_radar_fcd_prec = phi_mu_radar + N*t_final*phi_radar;
	double mu_radar_fcd_mean = phi_mu_radar*mean_mu_radar + phi_radar*(tmp_mat_NT.sum());
	mu_radar_fcd_mean /= mu_radar_fcd_prec;
	mu_radar = mu_radar_fcd_mean + gsl_ran_gaussian(rng[0], sqrt(1.0/mu_radar_fcd_prec));
	radar_data.array() -= mu_radar; //scale radar data so that we are working with radar_data - mu_radar 
	
	//// End of draw mu_radar ////	
	
	

	
	//// Generate out of sample forecast /////
	
	if(generate_forecasts &&  it >= burn_in && it % thin == 0 && it % forecast_thin == 0)
	{
		//initalise
		theta_forecasts.col(0) = state_sample.col(t_final_augment);
		source_sink_forecasts.col(0) = source_sink_sample.col(t_final_augment);
		beta_vec_forecast(4) = alpha*(1.0 - 4.0*beta);
		current_v_x = velocity_mat(0,t_final_augment-1);
		current_v_y = velocity_mat(1,t_final_augment-1);
		
		
		for(int q = 1; q < (n_step_ahead_forecasts_augment+1); q++) //propogate
		{
			current_v_x *= alpha_v;
			current_v_x += gsl_ran_gaussian(rng[0], sqrt(1.0/phi_v));
			current_v_y *= alpha_v;
			current_v_y += gsl_ran_gaussian(rng[0], sqrt(1.0/phi_v));
			beta_vec_forecast(0) = alpha*(beta + current_v_x); //beta + v_x time t
			beta_vec_forecast(1) = alpha*(beta - current_v_y); //beta - v_y time t
			beta_vec_forecast(2) = alpha*(beta + current_v_y); //beta + v_y time t
			beta_vec_forecast(3) = alpha*(beta - current_v_x); //beta - v_x time t
		
			#pragma omp parallel for schedule(static,1) default(shared)
			for(int i = 0; i < N; i++) //evolve system
			{	
				double tmp = 0.0;
				double tmp_s = 0.0;
				for(int k = 0; k < 5; k++)
				{
					if(neighbour_locs(i,k)>-1)
					{
						tmp += beta_vec_forecast(k)*(theta_forecasts(neighbour_locs(i,k),q-1) - mu); //theta evolution
						tmp_s += beta_star_vec(k)*source_sink_forecasts(neighbour_locs(i,k),q-1); //source sink evolution
					}
				}
				theta_forecasts(i,q) = tmp + mu + source_sink_forecasts(i,q-1) + gsl_ran_gaussian(rng[omp_get_thread_num()], sqrt(1.0/phi_theta));  //G * \theta + S + error
				source_sink_forecasts(i,q) = tmp_s + gsl_ran_gaussian(rng[omp_get_thread_num()], sqrt(1.0/phi_s)); // G* * S + error
			}
		}
		
		int tmp_count = 0;
		for(int q = 0; q < (n_step_ahead_forecasts); q++) //propogate
		{
			tmp_count += inter_time_steps + 1;
			state_forecast_outputFiles_bin[q].write ((char*)theta_forecasts.col(tmp_count).data(), (N)*sizeof (double));
		}
		
		
	}
	
	
	
	///////// Compute log observed and complete data likelihood //////////////
	
	double log_observed = 0.0;
	double log_complete = 0.0;
	double SSE = 0.0;
	if(it >= burn_in && it % thin == 0) 
	{
		double tmp_radar_sd = sqrt(1.0/phi_radar);
		double tmp_gauge_sd = sqrt(1.0/phi_gauge);
		double data_minus_mean, mean;
		double ppd = 0;
		for(int t = 0; t < t_final; t++)
		{
			for(int i = 0; i < N; i++)
			{	
				data_minus_mean  = radar_data(i,t) - (psi*radar_state_sample(i,t+1));
				SSE -= 0.5*data_minus_mean*data_minus_mean/(tmp_radar_sd*tmp_radar_sd);
				if(radar_data(i,t) < -mu_radar)
				{    
					mean = (psi*radar_state_sample(i,t+1) + mu_radar); 
					if(mean < 0) ppd = 1.0 - gsl_cdf_ugaussian_P(mean/tmp_radar_sd);
					if(mean > 0) ppd =  gsl_cdf_ugaussian_P(-(mean/tmp_radar_sd));
				}else{
					ppd = gsl_ran_gaussian_pdf(data_minus_mean, tmp_radar_sd);		
				}
				log_observed +=  log(ppd);
				log_complete +=  log(gsl_ran_gaussian_pdf(data_minus_mean, tmp_radar_sd));
				
			}
			for(int i = 0; i < N_g; i++)
			{
				data_minus_mean =  (gauge_data(i,t) - gauge_state_sample(i,t+1));
				mean = gauge_state_sample(i,t+1);
				SSE -= 0.5*data_minus_mean*data_minus_mean/(tmp_gauge_sd*tmp_gauge_sd);
				if(gauge_data(i,t) <= 0.0)
				{
					if(mean < 0) ppd = 1.0 - gsl_cdf_ugaussian_P((mean/tmp_gauge_sd));
					if(mean > 0) ppd = gsl_cdf_ugaussian_P(-(mean/tmp_gauge_sd));
				}else{
					ppd = gsl_ran_gaussian_pdf(data_minus_mean, tmp_gauge_sd);
				}
				log_observed += log(ppd);
				log_complete +=  log(gsl_ran_gaussian_pdf(data_minus_mean, tmp_gauge_sd));

			}
		}
	}
	
	///////// End compute log observed and complete data likelihood //////////////
	
	
	
	///////// Print statements //////////////
	
	if(it >= burn_in && it % thin == 0) 
	{
		mu_radar_file << mu_radar  << '\n';
		mu_file << mu  << '\n';
		alpha_file << alpha << "\n";
		alpha_latent_file << log_latent_tn_alpha << "\n";
		beta_file << beta << '\n';
		v_x_file << velocity_mat.row(0).format(CSVFormat) << '\n';
		v_y_file << velocity_mat.row(1).format(CSVFormat) << '\n';
		log_observed_file << log_observed << "\n";
		log_complete_file << log_complete << "\n";
		

		for(int i = 0; i < N_g; i++)
		{
			gauge_state_outputFiles_bin[i] << state_sample.row(gauge_locs(i,0)).format(CSVFormat) << "\n";
		}
		
		for(int t = 0; t < t_final; t++) //loop over radar states
		{
			state_draws_outputFiles_bin[t].write ((char*)radar_state_sample.col(t+1).data(), N*sizeof (double));	
		}
		
	}
	
	///////// End print statements //////////////
		
				
		
		
}

/////// END MCMC ///////
	
	
	
	
/////// Close output streams ///////

	mu_radar_file.close();
	mu_file.close();
	alpha_file.close();
	alpha_latent_file.close();
	beta_file.close();
	v_x_file.close();
	v_y_file.close();
	log_observed_file.close();
	log_complete_file.close();
	for(int i = 0; i < n_step_ahead_forecasts; i++)
	{
		state_forecast_outputFiles_bin[i].close();
	}
	for(int i = 0; i < N_g; i++)
	{
		gauge_state_outputFiles_bin[i].close();
	}

/////// End close output streams ///////

  
  return(0);
  
}


double LogLikeliMultivariateNormal(Eigen::VectorXd &data_vec, Eigen::VectorXd &mean, Eigen::MatrixXd &prec_mat, int n)
{
	double log_likeli;
	double tmp;
	
	tmp = prec_mat.determinant();
	tmp *= pow(6.28318530718,n);
	log_likeli = -log(sqrt(tmp));
	
	log_likeli -= 0.5* (data_vec-mean).transpose() * prec_mat * (data_vec-mean);
	
	return(log_likeli);
}


double sampleNormalTruncatedAbove(gsl_rng* rng, double mu, double sd, double up_lim)
{
	double rand_u = gsl_rng_uniform(rng);
	double tmp = gsl_cdf_ugaussian_P((up_lim-mu)/sd) * rand_u;
	tmp = gsl_cdf_ugaussian_Pinv(tmp)*sd + mu;
	
	return(tmp);
}


double sampleNormalTruncatedAboveViaLatent(gsl_rng* rng, double current_val, double mu, double sd, double up_lim)
{
	double tmp = (current_val - mu)/sd;
	double log_latent = -0.5*tmp*tmp + log(gsl_rng_uniform(rng));
	double low = mu - sqrt(-2.0*log_latent)*sd;
	double up = mu + sqrt(-2.0*log_latent)*sd;
	if(up > up_lim) up = up_lim; //truncate
	if(low > up) low = up - 0.02; //nugget to move lower limit away from upper limit if nessesary
	double sample = gsl_ran_flat(rng, low, up); 
	
	return(sample);
}




void sampleMultivariateNormal(gsl_rng* rng, int n, Eigen::VectorXd &mean, Eigen::MatrixXd &covar, Eigen::VectorXd &sample)
{
	
	Eigen::MatrixXd L(covar.llt().matrixL()); //chol decomp
	Eigen::VectorXd norm_samples(n);
	for(int z = 0; z < n; z++)
	{
		norm_samples(z) = gsl_ran_ugaussian(rng);
	}
	sample = mean + (L * norm_samples);
	
}



MatrixXd readCSVtoeigenXd(std::string file, int rows, int cols) {

  std::ifstream in(file.c_str());
  
  std::string line;

  int row = 0;
  int col = 0;

  MatrixXd res = MatrixXd(rows, cols);

  if (in.is_open()) {

    while (std::getline(in, line)) {

      char *ptr = (char *) line.c_str();
      int len = line.length();

      col = 0;

      char *start = ptr;
      for (int i = 0; i < len; i++) {

        if (ptr[i] == ',') {
          res(row, col++) = atof(start);
          start = ptr + i + 1;
        }
      }
      res(row, col) = atof(start);

      row++;
    }

    in.close();
  }
  return res;
}


MatrixXi readCSVtoeigenXi(std::string file, int rows, int cols) {

  std::ifstream in(file.c_str());
  
  std::string line;

  int row = 0;
  int col = 0;

  MatrixXi res = MatrixXi(rows, cols);

  if (in.is_open()) {

    while (std::getline(in, line)) {

      char *ptr = (char *) line.c_str();
      int len = line.length();

      col = 0;

      char *start = ptr;
      for (int i = 0; i < len; i++) {

        if (ptr[i] == ',') {
          res(row, col++) = atof(start);
          start = ptr + i + 1;
        }
      }
      res(row, col) = atof(start);

      row++;
    }

    in.close();
  }
  return res;
}



int sample_cum_probs(double U, double* probs, int size) //returns a sample when a cumulative probability array is passed
{
	int i;

	for(i = 0; i < size; i++)
	{
		if(U <= probs[i])
		{	
			break;
		}
	}
	return(i);
	
}


