# Code

This directory contains C++ code that can be used to obtain posterior samples via the MCMC algorithm outlined in the paper. The code is currently configured to analyse the real data.

--- 

## Compilation

The code can be compiled within a terminal environment using the command

`g++ mcmc.cpp -lgsl -lgslcblas -lm -fopenmp -mfma -O3 -o mcmc_exe`

and executed using the command

`./epl_mcmc_exe`

**WARNING:** Executing the code above will create the directory /outputs in the current working directory if it does not already exist. Alternatively if the directory exists then any previous output files will be overwritten.

This algorithm exploits parallel computation using OpenMP and also utilises the FMA instruction set which may not be available on all machines. If compilation errors relating to "Illegal instruction sets" occur then compiling without the -mfma flag should solve this problem; note that the total execution time of the algorithm will be substantially longer in this case.

### Outputs

As mentioned above all outputs will be written to /outputs in the current working directory. All samples are those obtained after the burn-in period. A summary of the files is given below.

* alpha_output.csv - samples of the auto-regressive parameter alpha

* alpha_latent_output.csv - samples of the latent variable that facilitates a uniform FCD for alpha

* beta_output.csv - samples of the "neighbour weight" parameter beta

* mu_output.csv - samples of the mean parameter mu

* mu_radar_output.csv - samples of the radar bias parameter mu_r

* v_x_output.csv - samples of the velocity parameter v_x for all (augmented) time-steps. Row i column t gives sample i of velocity parameter v_x(t).

* v_y_output.csv - samples of the velocity parameter v_y for all (augmented) time-steps. Row i column t gives sample i of velocity parameter v_y(t).

* log_complete_output.csv - value of the log complete data likelihood evaluated at each iteration

* log_observed_output.csv - value of the log observed data likelihood evaluated at each iteration

* gauge_states directory - samples of the latent process theta for all (augmented) time-steps at each gauge location (written to separate files for each gauge).

* state_draws directory - samples of the latent process theta at all observation times (written to separate files for each time point).

* state_forecasts directory - samples of the latent process theta at all forecast observation times (written to separate files for each time point).

