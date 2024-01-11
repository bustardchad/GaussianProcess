from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import json


class GaussianProcess:
    """Class with Gaussian process functionality
    
    Attributes: None
    """

    @classmethod
    def log_results(cls, results, log):
        """Updates input dictionary (log) to include the results of a Python run

        Args:
            results: A dictionary of the results.
            log: dictionary to update
        """
        if ('df_name' in results.keys()) and ('kernel_before' in results.keys()):
            run_name = results['df_name']+results['kernel_before']
        else:
            raise Exception("df_name and/or kernel are not present, but both are needed to create name for the model run")
        log[run_name] = results

    
    @classmethod
    def write_log(cls, log, log_dir = "../logs/"):
        """Writes log file to JSON

        Args:
            log: dictionary of results over possibly multiple model runs
            log_dir: directory to log results in, filename will be trial_results.json
        """
        os.makedirs(log_dir,exist_ok=True)
        
        with open(log_dir+"trial_results.json", "w") as file:
            json.dump(log,file,skipkeys=True)

    

     
    @classmethod
    def run_GPR(cls, df, measurement, kernel=None, resolution=0.2, log={}):
        """Runs 2D Gaussian process regression
        
        Args:
            df: dataframe to work with
            measurement: name of measurement to run GPR on
            kernel: kernel to use for regression, e.g. Matern, RBF, etc.
            resolution: float that will set the number of points in each direction, 
                i.e. numpoints = int(100.0/resolution)
            log: dictionary to log results

        Returns:
            X0, X1: x and y arrays for the interpolation grid
            Z: Interpolated concentration (2D array of shape (X0, X1) 
            MSE: mean squared error of final Gaussian regression model
            results: dictionary with results
        
        """
        X = df.copy()
        X = X[['x','y']].to_numpy()

        kernel_str = str(kernel)

        numpoints = int(100.0/resolution)

        # Grid we will lay down for interpolation
        x1 = np.linspace(X[:,0].min(), X[:,0].max(), numpoints) 
        x2 = np.linspace(X[:,1].min(), X[:,1].max(), numpoints) 
        x = (np.array([x1, x2])).T

        y = np.array(df[measurement])

        if len(X) != len(y): 
            raise Exception("lengths of training X and y data don't match")


        # Set up a regression model with a given kernel, and allow (default) 15 restarts
        # for the optimizer to find a good solution. 
        #
        # normalize_y = True subtracts the mean and normalizes the variance of the target values, 
        # usually helping achieve better convergence. This normalization is reversed before the predictions
        # are reported
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, normalize_y = True)

        gp.fit(X, y)

        x1x2 = np.array(list(product(x1, x2)))

        # Use gp model for prediction, returning a mesh y (which is the MEAN
        # over the posterior distribution) and a 
        # STANDARD DEVIATION MSE, which crucially quantifies the uncertainty
        y_pred, MSE = gp.predict(x1x2, return_std=True)

        X0, X1 = x1x2[:,0].reshape(numpoints,numpoints), x1x2[:,1].reshape(numpoints,numpoints)
        Z = np.reshape(y_pred,(numpoints,numpoints))
        MSE = np.reshape(MSE,(numpoints,numpoints))


        # log results in a dictionary for later use
        results = {'df_name' : str(df.name), 
                    'measurement' : measurement, 
                    'kernel_before' : kernel_str, 
                    'kernel_after' : str(gp.kernel_),
                    'loglikelihood' : gp.log_marginal_likelihood(gp.kernel_.theta),
                    'mesh_values': Z.tolist(), 
                    'mean_MSE' : np.mean(MSE),
                    'MSE': MSE.tolist()}

        cls.log_results(results, log)

        return X0, X1, Z, MSE, results

    @classmethod
    def plot_mesh(cls, df, measurement, X0, X1, Z):
        """Plot the resulting mesh with original data overplotted as well
        
        Args: 
            df: dataframe with Co or Cu point cloud data
            X0, X1: X and Y values for the interpolation grid
            Z: 2D array of Cu or Co mesh values (same size as the grid)

        Returns:
            fig: Figure showing original measurements as red circles, with 
                size corresponding to concentration, overplotted on interpolated mesh
                of concentration


        """

        # for creating markers with different sizes
        markers = 300.0*np.array(df[measurement])/np.max(np.array(df[measurement]))

        # Create a figure showing measurement prediction results
        # Note this is agnostic to whether input data has been put into log space or not
        fig,axs = plt.subplots(nrows=1,ncols=1,figsize=(8,6))

        sc0 = axs.pcolormesh(X0, X1, Z)
        axs.scatter(np.array(df['x']),np.array(df['y']), s = markers, facecolors='none', edgecolors='r')
        cb = plt.colorbar(sc0)
        
        # Little hack, for now, to figure out whether data is log-scaled or not
        # Would have to do something better before putting this code to use on different data
        if (df.name == f'log10_{measurement}'):
            plt_title = f"log10({measurement}) in {df['units'][0]}"
        else: 
            plt_title = f"{measurement} in {df['units'][0]}"
        axs.set_title(plt_title,fontsize=20)
        plt.xlabel("x",fontsize=16)
        plt.ylabel("y",fontsize=16)

        cb.set_label(plt_title)
        plt.tight_layout()

        return fig

    @classmethod
    def run_trials(cls, df, measurement, log = {}, trial_output_dir="../logs/trial_plots"):
        """ Run a few trials with a grid of input length scales and nu values for a Matern kernel
            Then update a dictionary (log) of results

        Args:
            df: dataframe
            log: dictionary to update with GP regression results
            
        """

        # We'll plot meshes as we go using the plot_mesh function
        # and store them in trial_output_dir

        os.makedirs(trial_output_dir, exist_ok=True)

        # Make separate folders to store the mean predictions and the standard deviations
        mean_dir = trial_output_dir+"/mean/"
        stddev_dir = trial_output_dir+"/standard_deviation/"
        os.makedirs(mean_dir, exist_ok=True)
        os.makedirs(stddev_dir, exist_ok=True)

        # Loop over different values for length_scale and nu
        # Parameters are hard-coded in for now, assuming a Matern kernel
        for length_scale in tqdm([1.0, 10.0]): 
            for nu in tqdm([0.5, 1.0, 1.5, 2.5]): 
                kernel = 1.0 * Matern(length_scale, length_scale_bounds=(1.e-5, 1.e5), nu=nu) 

                X0, X1, Z, MSE, results_dict = cls.run_GPR(df, measurement, kernel, 0.2, log)

                title = results_dict['df_name'] + results_dict['kernel_before'] + '.png'
                fig = cls.plot_mesh(df, measurement, X0, X1, Z)
                fig.savefig(mean_dir + title)
                plt.close()

                # Plot the standard deviation too to get a sense of where results
                # are most uncertain
                fig = cls.plot_mesh(df, measurement, X0, X1, MSE)
                fig.savefig(stddev_dir + title)
                plt.close()

        # # Dump logged results to json file -- commented out because file is large
        # cls.write_log(log, log_dir = "../logs/")

    @classmethod
    def print_results(cls, log):
        """Print results to screen
       
        Specifically, print the before and after kernel, the log-likelihood, and the mean standard deviation
        of the interpolated concentration

        Our goal is to find a model that maximizes the log-likelihood, so we want models with high log-likelihoods

        Mean standard deviation is a measure of the uncertainty for the regression model,
        which can inform us which model we have the most faith in. Models with low MSE are less uncertain.

        Args:
            log: dictionary where results are logged

        
        """

        num_runs = len(log.keys())
        print("Number of total runs: " + str(num_runs))

        for key in log.keys():

            print(
                log[key]['df_name'] + ': ' + log[key]['kernel_before'] + " \n" 
                "-------------------------------------------------- \n"
                "Optimized kernel: " + log[key]['kernel_after'] + " \n"
                "Log-likelihood: " + str(log[key]['loglikelihood']) + " \n"
                "Mean standard deviation (uncertainty): " + str(log[key]['mean_MSE']) + " \n \n "
                )
            


     




