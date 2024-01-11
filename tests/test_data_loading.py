import pandas as pd
import numpy as np
import pytest

from src.data_ingestion import Preprocessing as P


class TestLoad:
    def test_file_exists(self):
        file = "not_a_file.csv"
        with pytest.raises(FileNotFoundError) as e_info:
            pclass = P(file)
            df = pclass.load_data()

    def test_non_empty(self):
        file = "non_empty.csv"
        mydf = pd.DataFrame({"measurement" : [0.1,1,2,6]})
        mydf.to_csv(file)

        df = P(file).load_data()
        assert len(df.index) != 0 

    def test_empty(self):
        file = "empty.csv"
        mydf = pd.DataFrame({})
        mydf.to_csv(file)
        with pytest.raises(Exception) as e_info:
            df = P(file).load_soil_data()


class TestSplitClean:
    
    def test_split_df(self):
        df = pd.DataFrame({'Ar': [1.0,1.0,1.0,1.0],
                           'Be': [1.0,1.0,np.NaN,1.0],
                           'x' : [0.1,0.1,0.1,0.1],
                           'y' : [0.2,0.5,0.6,10.0],
                           'units' : ['ppm','ppm','ppm','ppm']})
        
        df_Ar = P("blank.csv").split_df(df, 'Ar')

        assert df_Ar.name == 'Ar'

        df_Be = P("blank.csv").split_df(df, 'Be')

        assert df_Be.name == 'Be'

        good_Be = pd.DataFrame({'Be': [1.0,1.0,1.0],
                           'x' : [0.1,0.1,0.1],
                           'y' : [0.2,0.5,10.0],
                           'units' : ['ppm','ppm','ppm']})
        
        assert df_Be.equals(good_Be)

    def test_bad_units(self):
        df_bad_units = pd.DataFrame({'Ar': [1.0,1.0,1.0,1.0],
                    'Be': [1.0,1.0,np.NaN,1.0],
                    'x' : [0.1,0.1,0.1,0.1],
                    'y' : [0.2,0.5,0.6,10.0],
                    'units' : ['ppm','ppm','pct','pct']})

        with pytest.raises(Exception) as e_info:
            df_Be = P("blank.csv").split_df(df_bad_units,'Be')
       
        
    def test_bad_column_names(self):
        df = pd.DataFrame({'Ar': [1,1,1,1],
                    'Co': [1,1,np.NaN,1],
                    'x' : [0.1,0.1,0.1,0.1],
                    'y' : [0.2,0.5,0.6,10.0],
                    'units' : ['ppm','pct','pct','ppm']})
        
        with pytest.raises(Exception) as e_info:
            df_Be = P("blank.csv").split_df(df, 'Be')
