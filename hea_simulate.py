#/usr/bin/env python

from hea_compute_props import get_weighted_atomic_radius
from pandas.api.types import is_numeric_dtype
from itertools import combinations, repeat
from datetime import datetime
import pandas as pd
import hea_analysis
import swifter
import logging
import pprint

PREDICTOR_VARS = + ['Alloy'] + hea_analysis.PREDICTOR_VARS 

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def simulate_alloy(metal_combinations):

    alloys_compsn_list = []
    alloys_names_list =  []
    for i,metals  in enumerate(metal_combinations):
        composition = alloy_composition(metals)
        alloys_names_list.append(''.join(metals))
        alloys_compsn_list.append(composition)
        
    #We create the simulate HEA database
    alloy_compsn_df = pd.DataFrame(alloys_compsn_list)
    alloy_compsn_df['Alloy'] = alloys_names_list
    logging.info(f"Added Alloy properties: { ', '.join(sorted(alloy_compsn_df.columns))}")
    
    return alloy_compsn_df



def add_properties(alloy_compsn_df):

   #computer properties using function imported from hea_compute_props.py  
   props_df = alloy_compsn_df.swifter.apply(get_weighted_atomic_radius, axis=1)
   logging.info(f"Added alloy properties: { ', '.join(sorted(props_df.columns))}")
   
   return pd.concat([alloy_compsn_df, props_df], axis=1)


def fixed_props(alloy_props, fixed_fname):

   fixed_props = pd.read_excel(fixed_fname, header=0, index_col=0)
   
   for col in fixed_props:
       alloy_props[col] = fixed_props[col].iloc[0]
   
   return alloy_props
   


def alloy_composition(metals, recipe = "equal"):

    #Can change and define rules for alloy composition here
    if recipe == "equal":
        comp = 100/len(metals)
        #return dictionary of metals with composition
        return {metal: comp for metal in metals}
        

def finalise_data(ref_dataset, HEA_data, impute_missing = True):
    
    ref_cols = set(hea_analysis.PREDICTOR_VARS)
    other_cols = set(HEA_data.columns)
    missing_cols = ref_cols - other_cols
    
    if impute_missing:
       
       for col in missing_cols:
           if col in ref_metals:
              HEA_data.loc[HEA_data.index,col] = 0
           elif is_numeric_dtype(ref_dataset[col]):
              HEA_data.loc[HEA_data.index,col] = ref_dataset[col].mean() 
              logging.info("Imputed mean value for missing column: %s",col)
           else:
              value = ref_dataset[col].sample(1, random_state=42).iloc[0]
              HEA_data[col] = value
              logging.info("Imputed random value for missing column: %s",col)

    #order the columns as they appear in the reference          
    HEA_data = HEA_data[PREDICTOR_VARS]
    HEA_data = HEA_data.fillna(0)
    
    return HEA_data
           
   
   
def progress(other_df):

    
    """
    Compares columns between two DataFrames and logs the percentage overlap,
    using the first DataFrame as the reference.
    
    Parameters:
        ref_df (pd.DataFrame): Reference DataFrame.
        other_df (pd.DataFrame): DataFrame to compare against.
    """
    
    ref_cols = set(PREDICTOR_VARS)
    other_cols = set(other_df.columns)
    common_cols = ref_cols & other_cols
    
    total = len(ref_cols)
    overlap_pct = (len(common_cols) / total * 100) if total > 0 else 0
   
    logging.info("Diff (Columns not matching reference) : %s", ', '.join(sorted(other_cols - ref_cols)))
    logging.info("Diff (Columns missing) : %s", ', '.join(sorted(ref_cols - other_cols)))
    logging.info("Progress (%d added): %.2f%%",len(common_cols), overlap_pct)

    
 
if __name__ == "__main__":
   
     choose_n = 5
     #metals = [ 'Mg', 'Al', 'Si', 'Ca', 'Sc', 'Ti', 'V']
     ref_metals = ['Ag', 'Al', 'Ca', 'Cd', 'Ce', 'Co', 'Cr', 'Cu', 'Fe', 'Ga', 'Gd', 'In', 'Ir', 'La', 'Mg', 'Mn', 'Mo',
                   'Nb', 'Ni', 'Pd', 'Pr', 'Pt', 'Rh', 'Ru', 'Sc', 'Si', 'Sn', 'Tc', 'Ti', 'V', 'Y', 'Zn', 'Zr']

     metals = ref_metals
     args = hea_analysis.main()
     ref_dataset = hea_analysis.get_db(args.filename)
     print(ref_dataset.columns)
     ref_dataset = ref_dataset[PREDICTOR_VARS] 
     ref_n = len(ref_dataset.columns)
     logging.info("Reference columns : %d", ref_n)

     metal_combinations = combinations(metals, choose_n)
     alloy_compsn_df = simulate_alloy(metal_combinations)
     progress(alloy_compsn_df)
     alloy_props_df = add_properties(alloy_compsn_df)
     progress(alloy_props_df)
     alloy_propf_df = fixed_props(alloy_props_df, "Fixed properties.xlsx")
     progress(alloy_propf_df)
     HEA_data_df = finalise_data(ref_dataset, alloy_propf_df)
     progress(HEA_data_df)
     timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
     filename = f'HEA_simulations__{timestamp}.csv'
     HEA_data_df.info()
     HEA_data_df.to_csv(filename, index=False)

