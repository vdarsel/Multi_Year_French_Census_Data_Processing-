import numpy as np
import pandas as pd

from split_population import generation_split
from data_treatment import process_data_datapaper_global

np.random.seed(5022026)

def data_import_global(target_file, year):
    
    target_file = target_file.replace("XXXX",str(year))
    if(year<2016):
        target_file = target_file.replace(".csv",".txt")
    if (year<2012):
        decimal = ','
    else:
        decimal = '.'
    print(target_file)
    print("Loading data...")
    df = pd.read_csv("%s" % (target_file), sep=";",low_memory=False,decimal=decimal)
    print("Data Loaded")
    return df

split_sizes = [0.03*0.01,1*0.01]

geo_attributes = ["Department", "County", "City"]



dir_data_import = "Data/Raw_census_data_XXXX"
dir_data_save = "Generated_Data/Census_data_XXXX"
filename = "FD_INDCVIZA_XXXX.csv"

for year in range(2021,2006,-1):
    target_file_year = f"{dir_data_import}/{filename}".replace("XXXX",str(year))
    dir_save = dir_data_save.replace("XXXX",str(year))
    data_year = data_import_global(target_file_year, year)
    process_data_datapaper_global(data_year, year, 2021, dir_save)
    generation_split(dir_save)