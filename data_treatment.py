import pandas as pd
import numpy as np
from tqdm import tqdm
from geographical_alignment import geographical_alignment


def process_data_datapaper_global(data, year, year_max, dir_save):
    '''year_max: maximum year using this processing, year used to set the geographical reference'''
    data_before_reweighting = pd.DataFrame()

    data["NUMMI_Z_unique"] = data["NUMMI"].copy().astype(str)
    data["CANTVILLE"] = data["CANTVILLE"].astype(str)
    idx_need_change = (data["NUMMI_Z_unique"]=="Z")|(data["NUMMI_Z_unique"].isna())|(data["NUMMI_Z_unique"]=="nan")
    data.loc[idx_need_change,"NUMMI_Z_unique"] = "Z"+np.arange(idx_need_change.sum()).astype(str) 

    # Attributes at the individual level
                    # Attributes used from the original dataset:
                    # - AGEREV
                    # - SEXE
                    # - DIPL
                    # - STAT_CONJ
                    # - COUPLE
                    # - EMPL 
                    # - TACT
                    # - CS1
                    # - NA17
                    # - TP
                    # - TRANS
                    # - LPRM
                    # - LIENF
    ## Age
    data_before_reweighting["Age"] = data["AGEREV"]
    data_before_reweighting.loc[data_before_reweighting["Age"]>99,"Age"] = 99
    ## Sex
    data_before_reweighting["Sex"] = "F"
    data_before_reweighting.loc[data["SEXE"]==1, "Sex"] = "M"
    ## Diploma
    match year:
        case _ if (year<2017)&(year>2012):
            traduction = {"A":2,"B":3,"C":4,"D":5, "Z":1}
            data_before_reweighting["Diploma"] = data["DIPL_15"]
        case _ if (year>2016):
            traduction = {"01":2,"02":2,"03":2,"11":2,"12":2,"13":3,"14":4,"15":4,"16":5,"17":5,"18":5,"19":5, "ZZ":1}
            data_before_reweighting["Diploma"] = data["DIPL"]
        case _ if (year<2013):
            traduction = {"01":2,"02":2,"03":2,"11":2,"12":2,"13":3,"14":3,"15":4,"16":4,"17":5,"18":5, "ZZ":1}
            data_before_reweighting["Diploma"] = data["DIPL"]

    for k,v in traduction.items():
        data_before_reweighting.loc[data_before_reweighting["Diploma"]==k,"Diploma"]=v
    data_before_reweighting["Diploma"] = (data_before_reweighting["Diploma"]).astype(str)
    ## Marital status
    match year:
        case _ if year>2016:
            data_before_reweighting["isMarried"] = (data["STAT_CONJ"]==1)
        case _ if (year<2017)&(year>2012):
            data_before_reweighting["isMarried"] = (data["STAT_CONJ"]=="A")
        case _ if (year<2013):
            data_before_reweighting["isMarried"] = (data["MATR"]==1)
    ## Live as a coupleastype
    data_before_reweighting["Cohabitation"] = (2-data["COUPLE"]).astype(str)
    ## Employment
    traduction = {"ZZ":10,"21":17,"22":18,"23":19}

    data_before_reweighting["Employment"] = data["EMPL"]
    for k,v in traduction.items():
        data_before_reweighting.loc[data["EMPL"]==k,"Employment"]=v

    data_before_reweighting["Employment"] = (data_before_reweighting["Employment"].astype(int)-9).astype(str)

    data_before_reweighting.loc[data_before_reweighting["Employment"]=="1", "Employment"] = data.loc[data_before_reweighting["Employment"]=="1", "TACT"].astype(int)

    traduction = {22:"1",23:"1", 12:"11", 21:"12", 24:"13",25:"14"}
    for k,v in traduction.items():
        data_before_reweighting.loc[data_before_reweighting["Employment"]==k,"Employment"]=v

    ## Socioprofessional category
    match year:
        case _ if year<2022:
            traduction = {}
            data_before_reweighting["Socioprofessional"] = data["CS1"].astype(str)
        case _ if year>2021:
            traduction = {"Z":8}
            data_before_reweighting["Socioprofessional"] = data["GS"].astype(str)
            data_before_reweighting.loc[data["STAT_GSEC"]=="32","Socioprofessional"] = "7"
            
    for k,v in traduction.items():
        data_before_reweighting.loc[data_before_reweighting["Socioprofessional"]==k,"Socioprofessional"]=v


    ## Activity type
    if year>2006:
        traduction = {"AZ":1, "C1": 2, "C2":3, "C3":4, "C4":5, "C5":6, "DE":7, "FZ":8, "GZ":9, "HZ":10, "IZ":11, "JZ":12, "KZ":13, "LZ":14, "MN": 15, "OQ":16, "RU":17, "ZZ":18}

        data_before_reweighting["Activity"] = data["NA17"]
        for k,v in traduction.items():
            data_before_reweighting.loc[data["NA17"]==k,"Activity"]=v

        data_before_reweighting["Activity"] = (data_before_reweighting["Activity"]).astype(str)
    ## Working hours
    traduction = {"Z":"3"}

    data_before_reweighting["Hours"] = data["TP"]
    for k,v in traduction.items():
        data_before_reweighting.loc[data["TP"]==k,"Hours"]=v
        

    ## Tranport means to comute
    match year:
        case _ if year<2017:
            traduction = {"5":"6","6":"7","Z":"7"}
        case _ if year>2016:
            traduction = {"Z":"6"}
            
    data_before_reweighting["Transport"] = data["TRANS"]
    for k,v in traduction.items():
        data_before_reweighting.loc[data["TRANS"]==k,"Transport"]=v
        
    ## Link to reference person in the household
    traduction = {"Z":"10"}

    data_before_reweighting["ReferenceLink"] = data["LPRM"]
    for k,v in traduction.items():
        data_before_reweighting.loc[data["LPRM"]==k,"ReferenceLink"]=v
    ## Family Link
    traduction = {"Z":"1"}

    data_before_reweighting["FamilyLink"] = data["LIENF"]
    for k,v in traduction.items():
        data_before_reweighting.loc[data["LIENF"]==k,"FamilyLink"]=v
        
    data_before_reweighting["FamilyLink"] = (data_before_reweighting["FamilyLink"].astype(int)+1).astype(str)
    # Attributes at the household level
                    # Variables used in the original dataset:
                    # - INPER
                    # - NBPI
                    # - SURF
                    # - GARL
                    # - VOIT
                    # - TYPL
                    # - STOCD
                    
    ## Household size (only for experiments at individual level)
    traduction = {"Z":1}
    if (year<2008):
        nIndiv = data.groupby(["NUMMI_Z_unique","CANTVILLE"]).count()["SEXE"]
        
        # Create a MultiIndex Series for mapping
        nIndiv_index = nIndiv.index.map(lambda x: f"{x[0]}!{x[1]}")
        
        # Map the values directly to the DataFrame
        data_before_reweighting["HouseholdSize"] = (data["NUMMI_Z_unique"] + "!" + data["CANTVILLE"]).map(dict(zip(nIndiv_index, nIndiv)))
    else:
        data_before_reweighting["HouseholdSize"] = data["INPER"]

    for k,v in traduction.items():
        data_before_reweighting.loc[data_before_reweighting["HouseholdSize"]==k,"HouseholdSize"]=v
    household_size_max = 15
    data_before_reweighting["HouseholdSize"] = data_before_reweighting["HouseholdSize"].astype(int)
    data_before_reweighting.loc[data_before_reweighting["HouseholdSize"]>household_size_max,"HouseholdSize"]=household_size_max

    ## Number of children in household (only for experiments at individual level)
    # Create a boolean mask for age < 18
    age_mask = data["AGEREV"] < 18

    # Group by ["NUMMI_Z_unique", "CANTVILLE"] and sum the boolean mask
    nChildren = age_mask.groupby([data["NUMMI_Z_unique"], data["CANTVILLE"]]).sum()

    # Create a MultiIndex Series for mapping
    nChildren_index = nChildren.index.map(lambda x: f"{x[0]}!{x[1]}")

    # Map the values directly to the DataFrame
    data_before_reweighting["nChildren"] = (data["NUMMI_Z_unique"] + "!" + data["CANTVILLE"]).map(dict(zip(nChildren_index, nChildren)))

    ## Number of rooms
    name_var_original="NBPI"
    name_var_final="nRooms"

    traduction = {"ZZ":1}

    data_before_reweighting[name_var_final] = data[name_var_original]

    for k,v in traduction.items():
        data_before_reweighting.loc[data[name_var_original]==k,name_var_final]=v


    data_before_reweighting[name_var_final] = data_before_reweighting[name_var_final].astype(int)
    ## Superficy
    name_var_original="SURF"
    name_var_final="Surface"

    match year:
        case _ if (year>2008)&(year<2013) :
            traduction = {"Z":1}
        case _ if (year>2012):
            traduction = {"1":1,"2":1,"3":2,"4":2,"5":2,"6":3,"7":3,"Z":1,"Y":1}
        case _ if (year<2009):
            traduction = {"1":1,"2":1,"3":2,"4":2,"5":3,"6":3,"Z":1}

    data_before_reweighting[name_var_final] = data[name_var_original]

    for k,v in traduction.items():
        data_before_reweighting.loc[data[name_var_original]==k,name_var_final]=v


    data_before_reweighting[name_var_final] = data_before_reweighting[name_var_final].astype(int)
    ## Parking availibility
    name_var_original="GARL"
    name_var_final="Parking"

    traduction = {"Z":2}

    data_before_reweighting[name_var_final] = data[name_var_original]

    for k,v in traduction.items():
        data_before_reweighting.loc[data[name_var_original]==k,name_var_final]=v


    data_before_reweighting[name_var_final] = (2-data_before_reweighting[name_var_final].astype(int)).astype(str)
    ## Number of cars
    name_var_original="VOIT"
    name_var_final="nCars"

    traduction = {"Z":0}

    data_before_reweighting[name_var_final] = data[name_var_original]

    for k,v in traduction.items():
        data_before_reweighting.loc[data[name_var_original]==k,name_var_final]=v


    data_before_reweighting[name_var_final] = data_before_reweighting[name_var_final].astype(int)
    ## Accomodation Type
    name_var_original="TYPL"
    name_var_final="Accommodation"

    traduction = {"Z":7}

    data_before_reweighting[name_var_final] = data[name_var_original]

    for k,v in traduction.items():
        data_before_reweighting.loc[data[name_var_original]==k,name_var_final]=v


    data_before_reweighting[name_var_final] = data_before_reweighting[name_var_final].astype(str)

    ## Household Type
    name_var_original="TYPMC"
    name_var_final="Household"

    traduction = {"Z":1}

    data_before_reweighting[name_var_final] = data[name_var_original]

    for k,v in traduction.items():
        data_before_reweighting.loc[data[name_var_original]==k,name_var_final]=v


    data_before_reweighting[name_var_final] = data_before_reweighting[name_var_final].astype(str)

    ## Occupancy
    name_var_original="STOCD"
    name_var_final="Occupancy"

    traduction = {"10":1,"21":2,"22":3, "23":4, "30": 5, "ZZ":6}

    data_before_reweighting[name_var_final] = data[name_var_original]

    for k,v in traduction.items():
        data_before_reweighting.loc[data[name_var_original]==k,name_var_final]=v


    data_before_reweighting[name_var_final] = data_before_reweighting[name_var_final].astype(str)
    # Geographic attributes
    ## Department
    name_var_original="DEPT"
    name_var_final="Department"


    data_before_reweighting[name_var_final] = data[name_var_original]

    data_before_reweighting[name_var_final] = data_before_reweighting[name_var_final].astype(str)
        
    if year>2006:
        ## Pseudo-County
        name_var_original="CANTVILLE"
        name_var_original_2="ARM"
        name_var_final="County"


        data_before_reweighting[name_var_final] = data[name_var_original]

        data_before_reweighting.loc[data_before_reweighting["Department"] == "75", "County"] = data.loc[data_before_reweighting["Department"] == "75", name_var_original_2]

        data_before_reweighting[name_var_final] = data_before_reweighting[name_var_final].astype(str)
        ## City
        IRIS_available = data["IRIS"].apply(lambda x: x[:5]!="ZZZZZ")

        data_before_reweighting["City"] = data["IRIS"].apply(lambda x: x[:5])
        data_before_reweighting.loc[~IRIS_available,'City'] = data_before_reweighting.loc[~IRIS_available,'County']+"Z"
        
    else:
        data_before_reweighting["County"] = "ZZZZ"
        data_before_reweighting["City"] = "ZZZZZ"

    # Uniformasiation of geographical attributes based on most recent data
    
    if(year<year_max):
        data_before_reweighting = geographical_alignment(data_before_reweighting, year, year_max)
    
    # Postprocessing at the household level

    data_before_reweighting["NUMMI_Z_unique"] = data["NUMMI_Z_unique"]
    data_before_reweighting["IPONDI"] = data["IPONDI"]

    data_grouped = data_before_reweighting.groupby(["County", "NUMMI_Z_unique"])

    weights = data_grouped["IPONDI"].max().astype(float)

    HH_weights = (np.floor(weights)+np.random.binomial(1,weights-np.floor(weights))).astype(int)

    pbar = tqdm(data_grouped, total=len(data_grouped))
    pbar.set_description(f"Household Sampling")

    household_chunks = []
    household_id = 0

    for key, df_sub in pbar:
        k = HH_weights.loc[key]

        if k == 0:
            continue

        # Duplicate the whole group k times (vectorized)
        df_rep = pd.DataFrame(
            np.tile(df_sub.values, (k, 1)),
            columns=df_sub.columns
        )

        # Assign household IDs efficiently
        df_rep["HouseholdID"] = np.repeat(
            np.arange(household_id, household_id + k), len(df_sub)
        )

        household_id += k

        household_chunks.append(df_rep)
        
    print("Concatenating Household")

    data_final_Household = pd.concat(household_chunks, ignore_index=True)

    data_final_Household = data_final_Household.drop(columns=['HouseholdSize', 'NUMMI_Z_unique', 'IPONDI', 'nChildren'])
    

    data_before_reweighting = data_before_reweighting.drop(columns=['NUMMI_Z_unique','IPONDI'])

    print("Saving Household")

    data_final_Household.to_csv(f"{dir_save}/full_dataset_Household.csv", sep=";", index=False)


    del data_final_Household
    del household_chunks

    # Postprocessing at the individual level


    cols = data_before_reweighting.columns
    individual_chunks = []

    pbar = tqdm(data_before_reweighting.index, total=len(data_before_reweighting.index))
    pbar.set_description(f"Individual Sampling")

    weights = (np.floor(data["IPONDI"])+np.random.binomial(1,data["IPONDI"]-np.floor(data["IPONDI"]))).astype(int)

    for i in pbar:
        
        indiv = data_before_reweighting.loc[i]

        k = weights[i]

        if k == 0:
            continue

        # Duplicate the whole group k times (vectorized)
        df_rep = pd.DataFrame(
            np.tile(indiv.values, (k, 1)),
            columns=cols
        )
        
        individual_chunks.append(df_rep)
        
    print("Concatenating Individual")

    data_final_individual = pd.concat(individual_chunks, ignore_index=True)


    print("Saving Individual")

    data_final_individual.to_csv(f"{dir_save}/full_dataset_Individual.csv", sep=";", index=False)
    
    
