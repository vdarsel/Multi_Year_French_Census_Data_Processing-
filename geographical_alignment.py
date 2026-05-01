
import pandas as pd
import numpy as np

change_name = {
    "Le Chesnay": "Le Chesnay-Rocquencourt",
    "Rocquencourt": "Le Chesnay-Rocquencourt",
    "Courcouronnes": "Évry-Courcouronnes",
    "Évry": "Évry-Courcouronnes",
    "Saint-Ouen-sur-Seine": "Saint-Ouen",
    "Meulan-en-Yvelines": "Meulan",
    "Arnouville-lès-Gonesse": "Arnouville",
    "Beautheil": "Beautheil-Saints",
    "Saints": "Beautheil-Saints",
    "Chenoise": "Chenoise-Cucharmoy",
    "Cucharmoy": "Chenoise-Cucharmoy",
    "Villiers-Saint-Fréderic": "Villiers-Saint-Frédéric",
    "Herblay": "Herblay-sur-Seine",
    "Hérouville-en-Vexin": "Hérouville",
    "Estouches": "Le Mérévillois",
    "Méréville": "Le Mérévillois",
    "Fourqueux": "Saint-Germain-en-Laye", # Note: Souvent fusionnée avec St-Germain, mais ici on garde la logique de fusion
    "Gadancourt": "Avernes", # Fusion historique commune
    "Jeufosse": "Notre-Dame-de-la-Mer",
    "Port-Villez": "Notre-Dame-de-la-Mer",
    "Moret-sur-Loing": "Moret-Loing-et-Orvanne",
    "Veneux-les-Sablons": "Moret-Loing-et-Orvanne",
    "Moret Loing et Orvanne": "Moret-Loing-et-Orvanne",
    "Orvanne": "Moret-Loing-et-Orvanne",
    "Écuelles": "Moret-Loing-et-Orvanne",
    "Épisy": "Moret-Loing-et-Orvanne",
    "Montarlot": "Moret-Loing-et-Orvanne",
    "Saint-Ange-le-Viel": "Villemaréchal", # Fusion classique
    "Saint-Rémy-la-Vanne": "Saint-Rémy-de-la-Vanne",
    'Crèvecœur-en-Brie':'Crèvecoeur-en-Brie', 
    'Vincy-Manœuvre':'Vincy-Manoeuvre',

    "Gouzangrez": "Commeny",
}

change_col={
    "Department commune":"Commune code",
    "Department - commune":"Commune code",
    "Code géographique":"Commune code",
    "Libellé de commune":"Commune",
    "Libellé de la commune":"Commune",
    "Libellé géographique":"Commune",
    "Canton-ou-ville":"Canton",
    "Canton ville":"Canton",
    }

folder_geo = "Data/Geo"


def get_folder_year(year):
    return f"Rescencement_{year}"

def get_filename_geo_info_year(year):
    return f"table-appartenance-geo-communes-{str(year)[-2:]}.xls{"x" if year>2019 else ""}"

def get_data_file(year, filename):
    folder = get_folder_year(year)
    data = pd.read_csv(f"{folder}/{filename}", sep=";", low_memory=False)
    return data

def get_reference_file_year(year):
    folder = get_folder_year(year)
    data = pd.read_csv(f"{folder}/full_dataset_Individual.csv", sep=";", low_memory=False, usecols=["City","County"])
    return data

def get_geo_file_year(year):
    year_geo = year + 2
    filename_geo = get_filename_geo_info_year(year_geo)
    data = pd.read_excel(f"{folder_geo}/{filename_geo}",header=4,skiprows=[5])
    data.rename(columns=change_col, inplace=True)
    data = data[["Commune","Commune code","Canton","Department"]]
    data["Commune"] = data["Commune"].replace(change_name)
    return data

def get_dictionnary_for_commune_translation(geo_ref, geo_data):
    dataframe_geo_ref_year = pd.merge(geo_ref, geo_data, on=["Commune","Department"], suffixes=[" ref"," year"])
    translation_commune_code = {}
    for com_ref, com_data in zip(dataframe_geo_ref_year["Commune code ref"],dataframe_geo_ref_year["Commune code year"]):
        translation_commune_code[com_data] = com_ref
    return translation_commune_code

def get_dictionnary_for_commune_to_canton(geo_ref):
    translation = {}
    for com_ref, canton_ref in zip(geo_ref["Commune code"],geo_ref["Canton"]):
        translation[com_ref] = canton_ref
    return translation

def translation_for_Z_commune(geo_ref, geo_data, census_data, census_data_ref):
    dict_prop = {}
    dict_values = {}
    dataframe_geo_ref_year = pd.merge(geo_ref, geo_data, on=["Commune","Department"], suffixes=[" ref"," year"])
    county_to_explore_translation = census_data[census_data["City"].apply(lambda x: x[-1])=="Z"]["County"].unique()
    for county in county_to_explore_translation:
        unique_origin_counties = dataframe_geo_ref_year[dataframe_geo_ref_year["Canton year"]==str(county)]["Canton ref"].unique()
        assert(len(unique_origin_counties)>0)
        if len(unique_origin_counties)==1:
            new_county = unique_origin_counties[0]
            dict_prop[county] = [1]
            dict_values[county] = [new_county]
        else:
            filter_census_target = census_data_ref[census_data_ref["County"].isin(unique_origin_counties.astype(int))]
            n_undefined_cities = filter_census_target[filter_census_target["City"].apply(lambda x: x[-1])=="Z"]["County"].value_counts()
            if(np.sum(n_undefined_cities)==0):
                prop = np.ones(len(unique_origin_counties))/len(unique_origin_counties)
                values = unique_origin_counties
            else:
                prop = (n_undefined_cities/n_undefined_cities.sum()).values
                values = (n_undefined_cities).index.to_numpy()
            dict_prop[county] = prop
            dict_values[county] = values
    return dict_prop, dict_values




def geographical_alignment(data, year, year_ref):
    data_geo_ref = get_geo_file_year(year_ref)
    city_data_ref = get_reference_file_year(year_ref)
    data_geo_year = get_geo_file_year(year)
    
    translation_commune = get_dictionnary_for_commune_translation(data_geo_ref, data_geo_year)
    commune_to_canton = get_dictionnary_for_commune_to_canton(data_geo_ref)


    dict_prop_Z_com, dict_translation_Z_com = translation_for_Z_commune(data_geo_ref, data_geo_year, data, city_data_ref)
    
    data["City_new"] = data["City"].map(translation_commune)
    data["County_new"] = data["City_new"].map(commune_to_canton)

    data.loc[data["Department"].astype(str)=="75","County_new"] = data.loc[data["Department"].astype(str)=="75","City"]
    data.loc[data["Department"].astype(str)=="75","City_new"] = data.loc[data["Department"].astype(str)=="75","City"]
    
    assert(np.sum((data["City_new"].isna())&(~(data["City"].apply(lambda x: x[-1])=="Z")))==0)
        
    # Treatment of unknown cities
    
    for initial_county in dict_prop_Z_com.keys():
        prop = dict_prop_Z_com[initial_county]
        values = dict_translation_Z_com[initial_county]
        
        idx = (data["City"]==f"{initial_county}Z")
    
        counties = np.random.choice(values, np.sum(idx), p = prop)
        
        data.loc[idx, "County_new"] = counties
        data.loc[idx, "City_new"] = counties.astype(str)+"Z"
        
    assert(np.sum(data["County_new"].isna())==0)
    assert(np.sum(data["City_new"].isna())==0)
    
    data = data.drop(columns = ["City","County"])
    data = data.rename(columns={"County_new":"County","City_new":"City"})
    
    return data
        
            
            
            