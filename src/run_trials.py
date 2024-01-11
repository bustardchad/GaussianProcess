from data_ingestion import Preprocessing
from explore_data import Plotting
from gaussian_process import GaussianProcess

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C


if __name__ == "__main__":

    # What measurement to interpolate
    name = "noise"

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


    # Because the data is seemingly exponential distributed (more values around zero),
    # it's best to work in log space before we run a Gaussian process regression model
    clean_df_log = processing.create_log_df(clean_df, name)



    ##### explore the dataset #####
    figure_dir = '../figures/'
    plots = Plotting(figure_dir)

    print("Exploring the dataset and putting figures in " + str(figure_dir))
    
    Plotting.create_figure_dir(figure_dir)

    # Make some histograms to look at the data
    plots.make_hist(clean_df,name, log10=False)

    # Make histograms of the log data
    plots.make_hist(clean_df_log,name, log10=True)

    # Make scatter plots
    plots.make_scatter(clean_df, name, log10=False)

    # Make scatter plots of log10 values
    plots.make_scatter(clean_df_log, name, log10=True)



    ##### Gaussian process regressor -- loop over params ####
    GP = GaussianProcess()

    trials_log = {}
    trial_output_dir = '../logs/trial_plots/'

    print("Running Gaussian process regression on a grid of parameters")
    print("Logging concentration maps in " +str(trial_output_dir))

    GP.run_trials(clean_df, name, trials_log, trial_output_dir)
    GP.run_trials(clean_df_log, name, trials_log, trial_output_dir)

    GP.print_results(trials_log)

