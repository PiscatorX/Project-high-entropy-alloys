#!/usr/bin/env python

import pandas as pd
import numpy as np

elemental_property = pd.read_excel(r"oliynyk_elemental_property_list_HEA ALLOYS.xlsx", header=0, index_col=0)

# Function to compute weighted atomic radius and entropy
def get_weighted_atomic_radius(row):
    #print(f"Processing Alloy: {row['Alloy']}")  # Debugging: Print alloy name

    clean_row = row.dropna()   # Remove NaN values
    
    # Get only elements in the reference dataset
    valid_indexes = elemental_property.index.intersection(clean_row.index)

    # Create a temporary DataFrame for calculations
    df = pd.DataFrame()
    df['atomic_fraction'] = row.loc[valid_indexes]  # Atomic fractions

    # Fix: Use correct indexing to extract Series
    df['atomic_radius'] = elemental_property.loc[valid_indexes, 'Atomic\nradius calculated']
    df['valence_electrons'] = elemental_property.loc[valid_indexes, 'no of valence electrons']
    df['pauling_EN'] = elemental_property.loc[valid_indexes, 'Pauling EN']
    df['Mulliken_EN'] = elemental_property.loc[valid_indexes, 'Mulliken EN']

    # Compute weighted properties
    df['weighted_atomic_radius'] = df['atomic_fraction'] / 100 * df['atomic_radius']
    df['weighted_valence_electrons'] = df['atomic_fraction'] / 100 * df['valence_electrons']
    df['weighted_pauling_EN'] = df['atomic_fraction'] / 100 * df['pauling_EN']
    df['weighted_Mulliken_EN'] = df['atomic_fraction'] / 100 * df['Mulliken_EN']

    # Compute weighted averages
    weighted_avg_radius = df['weighted_atomic_radius'].sum()
    weighted_avg_pauling_EN = df['weighted_pauling_EN'].sum()
    weighted_avg_Mulliken_EN = df['weighted_Mulliken_EN'].sum()
    
    # Compute (Ri - R̄)^2
    df['radius_diff_squared'] = (df['atomic_radius'] - weighted_avg_radius) ** 2  
    df['pauling_EN_squared'] = (df['pauling_EN'] - weighted_avg_pauling_EN) ** 2  
    df['Mulliken_EN_squared'] = (df['Mulliken_EN'] - weighted_avg_Mulliken_EN) ** 2

    # Compute weighted sum of squared differences
    weighted_sum_squares = (df['atomic_fraction'] * df['radius_diff_squared']).sum()  
    weighted_sum_squares_pauling_EN = (df['atomic_fraction'] / 100 * df['pauling_EN_squared']).sum()  
    weighted_sum_squares_Mulliken_EN = (df['atomic_fraction'] / 100 * df['Mulliken_EN_squared']).sum()  

    # Compute final values
    ΔR = np.sqrt(weighted_sum_squares)
    Δ_pauling_EN = np.sqrt(weighted_sum_squares_pauling_EN)
    Δ_Mulliken_EN = np.sqrt(weighted_sum_squares_Mulliken_EN)
    VEC = df['weighted_valence_electrons'].sum()

    # Compute Mixing Entropy (ΔSmix)
    R = 8.314  # J/(mol·K) - Universal gas constant
    df['atomic_fraction'] = df['atomic_fraction'].astype(float).replace(0, 1e-10)  # Prevent log(0)
    df['mixing_entropy'] = -R / 100 * (df['atomic_fraction'] * np.log(df['atomic_fraction']))

    # Fix: Extract entropy properly as a single value
    mixing_entropy = df['mixing_entropy'].sum()

    # (weighted_avg_radius, ΔR, VEC, weighted_avg_pauling_EN, Δ_pauling_EN, weighted_avg_Mulliken_EN, Δ_Mulliken_EN, mixing_entropy) = (0, 0, 0, 0, 0, 0, 0, 0)
    
    # Fix: Correct return structure
    return pd.Series(
        [weighted_avg_radius, ΔR, VEC, weighted_avg_pauling_EN, Δ_pauling_EN, weighted_avg_Mulliken_EN, Δ_Mulliken_EN, mixing_entropy],
  index=['Weighted atomic radius', '$\Delta$R$\mathregular{_{}}$ $\mathregular{^{}}$' , 'VEC', 'Weighted Pauling EN', '$\Delta$Pauling EN$\mathregular{_{}}$ ', 'Weighted Mulliken EN', '$\Delta$Mulliken EN$\mathregular{_{}}$ ', 'Mixing entropy'])
 
