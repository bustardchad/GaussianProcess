import pandas as pd
import numpy as np
import os

class Preprocessing:
    """
    Class for loading, cleaning, and splitting the data

    Attributes:
        file: string pointing to the data file, i.e. "/path/filename"
    """

    def __init__(self,file):
        self.file = file

    @staticmethod
    def valid_file_path(file):
        if os.path.exists(file): # make sure file path exists
            pass
        else:
            raise FileNotFoundError("File path doesn't exist")

    def load_data(self):
        """Load the data
        
        Args: 
            file: path to file, assumed to be of type CSV
                    e.g. '../files/file.csv'
        """
        full_df = pd.read_csv(self.file)
        
        if len(full_df.index) == 0: # make sure file isn't empty
            raise Exception("File is empty")

        # Check format -- needs to be columns of (value, x, y, units)
        if (('x' in full_df.columns) and ('y' in full_df.columns) and ('units' in full_df.columns)):
            pass
        else:
            raise Exception("Missing either x, y, or units column. Make sure to check column names")


        return full_df

   
    def split_df(self,df,name):
        """Split dataframe containing possibly multiple measurements, and 
        access just the one with measurement = 'name', 
        drop NaN values, and add dataframe name for later use
        
        Args:
          df: full dataframe
          name: name of column to use 

        Returns:
          df_single_measurement: cleaned and split dataframe
        """

        if (name in df.columns):
            pass
        else:
            raise Exception(f"Measurement {name} not in full dataframe")
        
        df_single_measurement = df[[name,'x','y','units']]

        # drop NaN values from this SEPARATE dataframe
        df_single_measurement = df_single_measurement.dropna()

        # ensure all units are the same, otherwise throw an error
        if ((df['units'] == df['units'][0]).all()):
            pass
        else:
            raise Exception("Not all units are the same, please change")

        # Add name to the dataframe for later use with plotfile names, etc.
        df_single_measurement.name = name

        return df_single_measurement

    
    
    def create_log_df(self, df, name):
        """Log10 the concentration
            
        Args: 
            df: dataframe
            name: name of column to take measurements from

        Returns: 
            log10df: A copy of the dataframe with measurements in log10 space

        """

        # Check that no zero or negative values exist
        if (df.iloc[:, 0].le(0).any()):
            raise Exception("Some values <= 0, cannot take log")
        
        log10df = df.copy()

        if name in df.columns:
            log10df[name] = np.log10(df[name])
            log10df.name = f"log10_{name}"
        else: 
            raise Exception(f"Column {name} doesn't exist")

        return log10df
    
    
