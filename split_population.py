import numpy as np
import pandas as pd
from tqdm import tqdm
import os

split_sizes = [0.03*0.01,1*0.01]

geo_attributes = ["Department","County", "City"]


def generate_balance_dataset_according_to_specific_categories(dataframe: pd.DataFrame,
                                                              criteria: list[pd.Series],
                                                              split_size: float)-> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    dataframe: data to split
    criteria: series that are expected to be balanced in the resulting sets
    split_size: proportion in the first resulting set (the second is composed of the other samples) 
    '''
    if not 0 < split_size < 1:
        raise ValueError("split_size must be between 0 and 1")
    
    codes = ["!".join(str(c) for c in row) for row in zip(*criteria)]
    codes = pd.Series(codes, index=dataframe.index)  
    idx = dataframe.index.to_numpy()
 
    unique_values, inverse, res_counts = np.unique(codes, return_counts=True, return_inverse=True)

    idx_per_value = [[] for _ in unique_values]

    for i,a in enumerate(inverse):
        idx_per_value[a].append(i)
    
    idx_split = []
    
    for j in tqdm(range(len(unique_values))):
        available_idx_rest,res_count = idx_per_value[j],res_counts[j]
        res_count = res_count*split_size
        res_count_int = np.floor(res_count)
        pick_count = int(np.random.binomial(1,res_count-res_count_int)+res_count_int)
        available_idx = idx[available_idx_rest]
        np.random.shuffle(available_idx)
        idx_split.append(available_idx[:pick_count])

    idx_split = np.concatenate(idx_split)
        
    take_split = pd.Series(False,codes.index)
    
    take_split.loc[idx_split] = True
    
    training_set = dataframe.loc[idx_split].copy()
    
    # The testing set includes the samples used for W-DCR evaluation
    testing_set = dataframe.drop(index=idx_split)
        

    return training_set, testing_set



def process_unseen_values_training_individual(df_train, df_test):
    print("Delete samples from testing dataset with unseen categorical modalities in testing dataset")
    df_test = process_unseen_non_geographical_values(df_train, df_test)
    print("Geographical attributes processing")
    df_train, df_test = process_unseen_geographical_values_training(df_train, df_test)
    print("Processing over")
    return df_train, df_test

def process_unseen_values_training_household(df_train, df_test):
    assert len(geo_attributes)>0
    
    print("Delete samples from testing dataset with unseen categorical modalities in testing dataset")
    df_test_2 = process_unseen_non_geographical_values(df_train, df_test)
    print("Geographical attributes processing")
    df_test_by_household = df_test_2.groupby("HouseholdID")[["Department","County", "City"]].first()
    df_train_by_household = df_train.groupby("HouseholdID")[["Department","County", "City"]].first()
    
    df_train_by_household_2, df_test_by_household_2 = process_unseen_geographical_values_training(df_train_by_household, df_test_by_household.copy())
            
    for attribute in geo_attributes:
        df_test_2.loc[:,attribute] = df_test_2["HouseholdID"].map(dict(zip(df_test_by_household_2.index, df_test_by_household_2[attribute])))
        df_train.loc[:,attribute] = df_train["HouseholdID"].map(dict(zip(df_train_by_household_2.index, df_train_by_household_2[attribute])))
    return df_train, df_test_2

def process_unseen_non_geographical_values(df_train, df_test):
    """
    Remove test-set rows containing unseen non-geographical categorical values.

    This function scans all non-geographical columns (assumed to be all columns except
    the last five, which represent hierarchical geographic data) and identifies any
    categories that appear in the test dataset but do not appear in the training dataset.
    Rows in the test set containing these unseen categories are filtered out entirely.

    Steps performed:
    1. For each non-geographical column:
        - Compute normalized value counts for training and test datasets.
        - Identify categories that appear in the test set but are missing from the
          training set.
    2. For each column, flag test rows containing such unseen categories.
    3. Combine all flags to remove any test row that contains at least one unseen value
       across any non-geographical column.

    Parameters
    ----------
    df_train : pandas.DataFrame
        Training dataset containing reference categorical values.
    df_test : pandas.DataFrame
        Test dataset potentially containing unseen non-geographical categories.

    Returns
    -------
    pandas.DataFrame
        Filtered version of `df_test`, where rows containing unseen non-geographical
        values have been removed.

    Notes
    -----
    - Only non-geographical columns (all except the last 5) are processed.
    - The function performs *row-level filtering* rather than imputation.
    - If a row contains even one unseen value in any non-geographical column, it is
      removed from the returned DataFrame.
    - The original `df_train` is not modified.
    """
    keep_index = pd.Series(1, index=df_test.index)
    for col in df_train.columns:
        if (col not in geo_attributes)&(col!="HouseholdID"):
            df = pd.DataFrame([df_train[col].value_counts(normalize=True),df_test[col].value_counts(normalize=True)]).transpose()
            df.columns=["Training","Testing"]
            if (np.sum(df["Training"].isna())>0):
                keep_index = keep_index*(1-df_test[col].isin(df.index[df["Training"].isna()])).astype(int)
    return df_test[keep_index.astype(bool)]

def process_unseen_geographical_values_training(df_train, df_test):
    """
    Aligns unseen geographical values between training and test datasets by propagating parent-level values
    to finer-grained geographical attributes. This function ensures that both datasets share the same
    geographical hierarchy by replacing unseen or mismatched values in finer levels (e.g., City)
    with their corresponding parent-level values (e.g., County, City).

    This is particularly useful when the test dataset contains geographical values not present in the training set,
    or when there are discrepancies in the hierarchical structure of geographical attributes.

    Parameters
    ----------
    df_train : pandas.DataFrame
        Training dataset containing hierarchical geographical attributes.
        Expected to include columns: ["Department", "County", "City"].
    df_test : pandas.DataFrame
        Test dataset where unseen or mismatched geographical values may appear.
        Expected to include columns: ["Department", "County", "City"].

    Returns
    -------
    tuple : (pandas.DataFrame, pandas.DataFrame)
        A tuple containing the modified training and test datasets.
        Both datasets will have aligned geographical values for finer-grained attributes,
        where unseen or mismatched values are replaced by their parent-level values.

    Notes
    -----
    - The function iterates over each geographical level (e.g., County, City).
    - For each level, it identifies values present in one dataset but not the other.
    - For each identified discrepancy, it retrieves the parent geographical level (e.g., County for City).
    - This ensures that the hierarchical structure is preserved and discrepancies are resolved.
    """
    assert len(geo_attributes)>0
    geo_nexts_list = [geo_attributes[i:] for i in range(len(geo_attributes))]

    # Iterate over each pair of previous, current, and next geographical levels
    for geo_previous, geo_current, geo_nexts in zip(geo_attributes[:-1], geo_attributes[1:], geo_nexts_list[1:]):
        # Identify values in the test set that are not present in the training set
        values_in_only_one_set = set(df_test[geo_current].unique()).symmetric_difference(
            set(df_train[geo_current].unique())
        )        
        # If there are unseen values in the test set
        if len(values_in_only_one_set)>0:
            # Get the unique parent geographical values for the unseen values in both test and train sets
            previous_modalities_for_changes = np.unique(
                np.concatenate(
                    [df_test[geo_previous][df_test[geo_current].isin(list(values_in_only_one_set))].unique(), 
                     df_train[geo_previous][df_train[geo_current].isin(list(values_in_only_one_set))].unique()])
                )
            if(geo_current=="County"):
                print(previous_modalities_for_changes)
            # Create a boolean mask for rows in the test set with the identified parent values
            idx_test_update = df_test[geo_previous].isin(previous_modalities_for_changes)
            
            # Create a boolean mask for rows in the training set with the identified parent values
            idx_train_update = df_train[geo_previous].isin(previous_modalities_for_changes)
            
            # Perform updates on all elements for the geographical attribute and thiner geographical attributes that require the same resolution
            for geo_next in geo_nexts:
                df_test.loc[idx_test_update,geo_next] = df_test.loc[idx_test_update,geo_previous]
                df_train.loc[idx_train_update,geo_next] = df_train.loc[idx_train_update,geo_previous]
    return df_train, df_test


def extraction_testing_set_equal_size_training(testing_set_idx, testing_set_equal_size_training_idx, size_training_set):
    testing_set_equal_size_training_idx = np.array(list(set(testing_set_equal_size_training_idx).intersection(set(testing_set_idx))))
    if len(testing_set_equal_size_training_idx)<(size_training_set):
        idx_to_add = np.random.choice(testing_set_idx,size_training_set-len(testing_set_equal_size_training_idx), replace=False)
        testing_set_equal_size_training_idx = np.concat([testing_set_equal_size_training_idx, idx_to_add])
    elif (len(testing_set_equal_size_training_idx)>(size_training_set)):
        testing_set_equal_size_training_idx = np.random.choice(testing_set_equal_size_training_idx,size_training_set, replace=False)
    
    return testing_set_equal_size_training_idx



def generation_split(dir):
    df_individual = pd.read_csv(f"{dir}/full_dataset_Individual.csv", sep=";", low_memory=False)

    for geo_attribute in geo_attributes:
        df_individual[geo_attribute] = df_individual[geo_attribute] .astype(str)

    geo_attribute_stratification = "County"
    
    criteria_individual = [df_individual["Sex"],df_individual["Age"]//5,df_individual[geo_attribute_stratification]]

    print("__________________________________________________________________")
    print("______________________Individual level____________________________")
    print("__________________________________________________________________")

    for size in split_sizes:
        print("***********************************************")
        print(f"***********   Split: {size*100}%     *****************")
        print("***********************************************")
        name_size = str(100*size).replace(".","_") if (100*size)!=np.round(10**size) else str(int(100*size))
        if not os.path.isdir(dir):
            os.makedirs(dir)
        training_set, testing_set = generate_balance_dataset_according_to_specific_categories(df_individual, criteria_individual, size)
        training_set, testing_set = process_unseen_values_training_individual(training_set, testing_set)

        # For W-DCR computation, we construct a set with same size and propreties, see for details: 
        # Darsel, V., Come, E., \& Oukhellou, L. (2025) Robust and Reproducible Evaluation Framework for Population Synthesis Models—Application to Probabilistic and Deep Generative Models. (https://dx.doi.org/10.2139/ssrn.5295092), _Pre-Print available at SSRN 5295092_.
        criteria_individual_equal_size_training = [testing_set["Sex"],testing_set["Age"]//5,testing_set[geo_attribute_stratification]]
        testing_equal_size_training, _ = generate_balance_dataset_according_to_specific_categories(testing_set.copy(), criteria_individual_equal_size_training, size)
        testing_equal_size_training_idx = extraction_testing_set_equal_size_training(testing_set.index.to_numpy(), testing_equal_size_training.index.to_numpy(), len(training_set))
        testing_set_equal_size_training = testing_set.loc[testing_equal_size_training_idx]

        print("Save csv files")
        training_set.to_csv(f"{dir}/training_dataset_Individual_{name_size}.csv", sep=";", index=False)
        testing_set.to_csv(f"{dir}/testing_dataset_Individual_{name_size}.csv", sep=";", index=False)
        testing_set_equal_size_training.to_csv(f"{dir}/testing_dataset_Individual_{name_size}_equal_size_training.csv", sep=";", index=False)
    #     make_archive(dir,'zip', "Generated_Data")


    df_household = pd.read_csv(f"{dir}/full_dataset_Household.csv", sep=";", low_memory=False)

    for geo_attribute in geo_attributes:
        df_household[geo_attribute] = df_household[geo_attribute] .astype(str)

    df_household_group_by = df_household.groupby("HouseholdID")
    criteria_household = [df_household_group_by.size(),df_household_group_by[geo_attribute_stratification].first()]

    household_id_serie = pd.Series(index=criteria_household[0].index.to_numpy())

    print("__________________________________________________________________")
    print("_______________________Household level____________________________")
    print("__________________________________________________________________")
    for size in split_sizes:
        print("***********************************************")
        print(f"***********   Split: {size*100}%     *****************")
        print("***********************************************")
        name_size = str(100*size).replace(".","_") if (100*size)!=np.round(10**size) else str(int(100*size))
        # dir = f"Generated_Data/datasets_Household_{name_size}"
        if not os.path.isdir(dir):
            os.makedirs(dir)
        training_set_household_id, testing_set_household_id = generate_balance_dataset_according_to_specific_categories(household_id_serie, criteria_household, size)
        
        mask = (df_household["HouseholdID"].isin(training_set_household_id.index.to_numpy()))
        
        training_set = df_household[mask]
        testing_set = df_household[~mask]
        
        training_set, testing_set = process_unseen_values_training_household(training_set, testing_set)
        
        df_household_group_by_equal_size_training = testing_set.groupby("HouseholdID")
        criteria_household_equal_size_training = [df_household_group_by_equal_size_training.size(),df_household_group_by_equal_size_training[geo_attribute_stratification].first()]
        household_id_serie_equal_size_training = pd.Series(index=criteria_household_equal_size_training[0].index.to_numpy())
        
        testing_set_household_id_equal_size_training, _ = generate_balance_dataset_according_to_specific_categories(household_id_serie_equal_size_training, criteria_household_equal_size_training, size)
        
        testing_set_household_id_equal_size_training = extraction_testing_set_equal_size_training(testing_set["HouseholdID"].unique(), testing_set_household_id_equal_size_training, training_set["HouseholdID"].nunique())
        
        mask_equal_size_training = (testing_set["HouseholdID"].isin(testing_set_household_id_equal_size_training))
        testing_set_equal_size_training = testing_set[mask_equal_size_training] # same processing as testing, so as training 
        
        training_set.to_csv(f"{dir}/training_dataset_Household_{name_size}.csv", sep=";", index=False)
        testing_set_equal_size_training.to_csv(f"{dir}/testing_dataset_Household_{name_size}_equal_size_training.csv", sep=";", index=False)
        testing_set.to_csv(f"{dir}/testing_dataset_Household_{name_size}.csv", sep=";", index=False)
