from data_ingestion import Preprocessing
from explore_data import Plotting
from gaussian_process import GaussianProcess
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C


if __name__ == "__main__":

    # What measurement to interpolate
    name = "noise"

    # Resolution of interpolating grid
    resln = 0.001

    # Initial length scale for kernel
    init_ls = 0.01

    log_values = False

    ##### import and transform the dataset #####
    csv_file = "../data/transformed_max_noise_Jan11_2024.csv"

    processing = Preprocessing(csv_file)

    print("Loading and processing data from " + str(csv_file))
    
    # Validate the file path
    processing.valid_file_path(csv_file)

    # Load the data
    full_df = processing.load_data()

    # Split the data into separate Co and Cu files and clean up each one
    clean_df = processing.split_df(full_df, name)

    unit = clean_df['units'][0]


    if log_values==True:
        # If the data is seemingly exponential distributed (more values around zero),
        # it's best to work in log space before we run a Gaussian process regression model
        df_for_gp = processing.create_log_df(clean_df, name)
    else:
        df_for_gp = clean_df
        


    #### Gaussian process regressor -- using final params ####

    GP = GaussianProcess()

    final_results = {}

    print("Fitting data")
    kernel = 1.0 * Matern(length_scale=init_ls, length_scale_bounds=(1.e-5, 1.e5), nu=1.0) 
    X0, X1, Z, MSE, results_dict = GP.run_GPR(df_for_gp, name, kernel, resln, log=final_results)

    title = 'final_mesh_plot.png'
    fig = GP.plot_mesh(df_for_gp, name, X0, X1, Z)
    fig.savefig('../'+title)
    plt.close()

    print("Final figure saved to " + '../'+title)
    
    # Print results to screen
    print("Final results: \n")
    GP.print_results(final_results)


    # Uncertainty quantification
    #---------------------------------------------------------------------------
    # Note we have to be careful when converting from log10 space back
    # to regular space, e.g. a 95% credible interval of mean +- 1.96 sigma
    # in log space will not be symmetric in regular space. 
    # So I'll quantify uncertainty in regular space as the width of the 95%
    # credible interval. It's especially useful to quantify this uncertainty as a
    # percentage of the mean concentration, since the uncertainty is generally
    # larger/lower for larger/lower concentrations, and the average uncertainty 
    # weighted by the mean concentrations

    if log_values == True:
        cred_interval = 10**(Z + 1.96*MSE) - 10**(Z - 1.96*MSE)
        transformed = 10.0**Z
        cb_label = f"Standard Deviation of log10({name}) Prediction in {unit}"
        plt_title = f"Standard Deviation of log10({name} Prediction"
    else:
        cred_interval = (Z + 1.96*MSE) - (Z - 1.96*MSE)
        transformed = Z
        cb_label = f"Standard Deviation of {name} Prediction in {unit}"
        plt_title = f"Standard Deviation of {name} Prediction"
    
    weighted_cred_interval = np.sum(np.multiply(cred_interval,transformed))/np.sum(transformed)
    print("Uncertainty statistics for predicted map:")
    print(f"Average width of 95 percentile credible interval: {(np.mean(cred_interval)):.2f}")
    print(f"Average width of 95 percentile credible interval as percentage of measurement: {(np.mean(np.divide(cred_interval,transformed))*100.0):.2f}%")
    print(f"Average width of 95 percentile credible interval weighted by measurement: {weighted_cred_interval:.2f}  \n \n")


    # plot uncertainties as histograms
    plt.hist(np.divide(cred_interval.flatten(),transformed.flatten()),density=True)
    plt.xlabel(f"Width of 95 percent credible interval / {name}")
    plt.ylabel("Probability Distribution")
    plt.savefig("../CredibleInterval95_hist.png")
    plt.close()


    fig, axs = plt.subplots(nrows=1,ncols=1,figsize=(8,6))
    sc0 = axs.pcolormesh(X0, X1, MSE)
    cb = plt.colorbar(sc0)
    cb.set_label(cb_label)
    plt.xlabel("x",fontsize=16)
    plt.ylabel("y",fontsize=16)
    plt.title(plt_title,fontsize=20)
    plt.tight_layout()
    plt.savefig(f"../{name}_StandardDeviation.png")
    plt.close()




