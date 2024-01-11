import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Plotting:

    def __init__(self, figure_dir):
        self.figure_dir = figure_dir

    @staticmethod
    def create_figure_dir(figure_dir):
        """Check that directory exists to put figures in.
        If not, create directory
        
        """
        if os.path.exists(figure_dir):
            pass
        else:
            os.mkdir(figure_dir)

    
    def make_hist(self, df, name, log10):
        """Make histogram of measurements from dataframe df
        
        Args:
            df: dataframe
            log10: bool for whether measurement has been put 
                    into log space (True) or not (False)
        """
        if name in df.columns:
            x = name
            unit = df['units'][0]
            if log10 == True:
                title = f"log10({name}) ({unit})"
            else:
                title = f"{name} ({unit})"
        else:
            raise Exception(f"{name} not in dataframe columns")

        plt_title = "hist_" + str(df.name) + ".png"
        
        sns.histplot(x=x,data=df,kde=True)
        plt.title(title)
        plt.savefig(self.figure_dir + plt_title)
        plt.close()

    def make_scatter(self, df, name, log10):
        """Make scatter plot showing measurements
            from dataframe on (x,y) grid

        Args:
            df: dataframe
            log10: bool for whether measurement has been put 
                    into log space (True) or not (False)

        """
        if name in df.columns:
            pass
        else:
            raise Exception(f"{name} not a column in dataframe")

        unit = df['units'][0]

        plot = sns.scatterplot(x="x", y="y", hue=name, size=name, data=df)
        fig = plot.get_figure()
        plt_filename = 'scatter_' + str(df.name) + '.png'
        if (log10==True):
            plt_title = f"log10({name}) in {unit}"
        else: 
            plt_title = f"{name} in {unit}"

        plot.set(title=plt_title)
        fig.savefig(self.figure_dir + plt_filename) 
        plt.close()

