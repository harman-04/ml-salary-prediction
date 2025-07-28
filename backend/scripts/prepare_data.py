import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict # For more robust multi-select parsing
import warnings

# Suppress all warnings for cleaner output during execution
warnings.filterwarnings('ignore')

def preprocess_and_eda(data_path=None):
    """
    Performs advanced preprocessing and generates EDA plots for the raw developer survey data.
    This includes:
    1. Loading data and initial filtering.
    2. Extensive cleaning and mapping of categorical features.
    3. Advanced handling of multi-select features.
    4. Engineering new numerical and interaction features.
    5. Generating diagnostic plots (histograms, box plots, correlation heatmap)
       to understand data distributions, relationships, and outliers.

    Args:
        data_path (str, optional): Path to the raw CSV data.
                                   Defaults to '../data/survey_results_public.csv'.

    Returns:
        pd.DataFrame: Processed DataFrame ready for ColumnTransformer.
        dict: A dictionary containing lists of column names for different transformation types.
        dict: A dictionary containing mappings and other parameters used in preprocessing.
    """
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'survey_results_public.csv')

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print("Dataset loaded successfully!")

    # Define the directory for saving EDA plots
    EDA_PLOTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'model_artifacts', 'eda_plots')
    os.makedirs(EDA_PLOTS_DIR, exist_ok=True)
    print(f"EDA plots will be saved to: {EDA_PLOTS_DIR}")

    # --- Initial Cleaning and Target Variable Handling ---
    initial_rows = df.shape[0]

    # Convert ConvertedCompYearly to numeric, coerce errors to NaN
    df['ConvertedCompYearly'] = pd.to_numeric(df['ConvertedCompYearly'], errors='coerce')

    # Drop rows where 'ConvertedCompYearly' is NaN, as this is our target
    # This is crucial for training, we cannot predict what we don't know
    df_cleaned = df.dropna(subset=['ConvertedCompYearly']).copy()
    print(f"Original data shape: {df.shape}")
    print(f"Data shape after dropping NaN in 'ConvertedCompYearly': {df_cleaned.shape}")

    if df_cleaned.empty:
        raise ValueError("No data remaining after dropping NaN values in 'ConvertedCompYearly'. Cannot proceed.")

    # Remove extreme outliers in ConvertedCompYearly (e.g., top and bottom 0.5%)
    # This prevents extreme values from skewing log transformation and model training
    q_low = df_cleaned['ConvertedCompYearly'].quantile(0.005)
    q_high = df_cleaned['ConvertedCompYearly'].quantile(0.995)
    df_cleaned = df_cleaned[(df_cleaned['ConvertedCompYearly'] >= q_low) & (df_cleaned['ConvertedCompYearly'] <= q_high)]
    print(f"Data shape after removing bottom 0.5% and top 0.5% outliers in 'ConvertedCompYearly': {df_cleaned.shape}")
    print(f"Removed {initial_rows - df_cleaned.shape[0]} rows due to target variable issues.")


    # --- Target Variable Transformation ---
    # Apply log transformation to the target variable, adding 1 to handle potential zeros
    df_cleaned['ConvertedCompYearly_log'] = np.log1p(df_cleaned['ConvertedCompYearly'])
    print("Target variable 'ConvertedCompYearly' log-transformed.")

    # --- EDA: Target Distribution ---
    plt.figure(figsize=(12, 6))
    sns.histplot(df_cleaned['ConvertedCompYearly'], bins=50, kde=True)
    plt.title('Distribution of ConvertedCompYearly (Original Scale)')
    plt.xlabel('Converted Compensation Yearly')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_PLOTS_DIR, 'converted_comp_yearly_distribution_original.png'))
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.histplot(df_cleaned['ConvertedCompYearly_log'], bins=50, kde=True)
    plt.title('Distribution of ConvertedCompYearly (Log-Transformed)')
    plt.xlabel('Log-Transformed Converted Compensation Yearly')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_PLOTS_DIR, 'converted_comp_yearly_distribution_log.png'))
    plt.close()

    preprocessing_params = {} # To store mappings and other info

    # --- Feature Engineering and Cleaning ---

    # 1. Age (Categorical to Numerical)
    age_mapping = {
        'Under 18 years old': 17,
        '18-24 years old': 21,
        '25-34 years old': 29,
        '35-44 years old': 39,
        '45-54 years old': 49,
        '55-64 years old': 59,
        '65 years or older': 65
    }
    df_cleaned['Age_numerical'] = df_cleaned['Age'].map(age_mapping)
    # Impute missing Age with median after mapping
    df_cleaned['Age_numerical'].fillna(df_cleaned['Age_numerical'].median(), inplace=True)
    preprocessing_params['age_mapping'] = age_mapping
    print("Cleaned 'Age' to 'Age_numerical'.")

    # 2. YearsCode and YearsCodePro (Convert to numerical, handle specific strings)
    def clean_years_code_col(df_col):
        df_col = df_col.replace('Less than 1 year', '0') # Treat as 0 for min experience
        df_col = df_col.replace('More than 50 years', '50')
        df_col = pd.to_numeric(df_col, errors='coerce')
        # Fill NaNs with median specific to each column
        return df_col.fillna(df_col.median())
    
    df_cleaned['YearsCode'] = clean_years_code_col(df_cleaned['YearsCode'])
    df_cleaned['YearsCodePro'] = clean_years_code_col(df_cleaned['YearsCodePro'])
    print("Cleaned 'YearsCode' and 'YearsCodePro'.")

    # 3. WorkExp (Convert to numerical, handle NaNs)
    df_cleaned['WorkExp'] = pd.to_numeric(df_cleaned['WorkExp'], errors='coerce')
    df_cleaned['WorkExp'].fillna(df_cleaned['WorkExp'].median(), inplace=True)
    print("Cleaned 'WorkExp'.")

    # 4. OrgSize (Categorical to Numerical)
    org_size_mapping = {
        'Just me - I am a freelancer, sole proprietor, etc.': 1,
        '2 to 9 employees': 5,
        '10 to 19 employees': 15,
        '20 to 99 employees': 60,
        '100 to 499 employees': 300,
        '500 to 999 employees': 750,
        '1,000 to 4,999 employees': 3000,
        '5,000 to 9,999 employees': 7500,
        '10,000 or more employees': 10000,
        'I don’t know': np.nan
    }
    df_cleaned['OrgSize_numerical'] = df_cleaned['OrgSize'].map(org_size_mapping)
    df_cleaned['OrgSize_numerical'].fillna(df_cleaned['OrgSize_numerical'].median(), inplace=True)
    preprocessing_params['org_size_mapping'] = org_size_mapping
    print("Cleaned 'OrgSize' to 'OrgSize_numerical'.")

    # 5. Employment (Extract binary flags)
    df_cleaned['Employed_FullTime'] = df_cleaned['Employment'].apply(lambda x: 1 if 'Employed, full-time' in str(x) else 0)
    df_cleaned['Independent_Contractor'] = df_cleaned['Employment'].apply(lambda x: 1 if 'Independent contractor, freelancer, or self-employed' in str(x) else 0)
    df_cleaned['Student'] = df_cleaned['Employment'].apply(lambda x: 1 if 'Student' in str(x) else 0)
    df_cleaned['Not_Employed'] = df_cleaned['Employment'].apply(lambda x: 1 if 'Not employed' in str(x) else 0) # Capture explicitly not employed
    print("Engineered 'Employment' binary flags.")

    # 6. RemoteWork (Handle 'NA' and create binary flags)
    df_cleaned['RemoteWork'].fillna('Unknown', inplace=True)
    df_cleaned['RemoteWork_is_Remote'] = df_cleaned['RemoteWork'].apply(lambda x: 1 if x == 'Remote' else 0)
    df_cleaned['RemoteWork_is_Hybrid'] = df_cleaned['RemoteWork'].apply(lambda x: 1 if x == 'Hybrid (some remote, some in-person)' else 0)
    df_cleaned['RemoteWork_is_InPerson'] = df_cleaned['RemoteWork'].apply(lambda x: 1 if x == 'In-person' else 0)
    print("Engineered 'RemoteWork' binary flags.")

    # 7. Country (Top N + Other)
    top_countries_threshold = 0.01 # Countries making up at least 1% of the data
    country_counts = df_cleaned['Country'].value_counts(normalize=True)
    top_countries = country_counts[country_counts >= top_countries_threshold].index.tolist()
    df_cleaned['Country_grouped'] = df_cleaned['Country'].apply(lambda x: x if x in top_countries else 'Other_Country')
    preprocessing_params['top_countries'] = top_countries
    print(f"Grouped 'Country' into top {len(top_countries)} and 'Other_Country'.")

    # 8. DevType (Top N + Other)
    top_dev_types_threshold = 0.01 # DevTypes making up at least 1% of the data
    dev_type_counts = df_cleaned['DevType'].value_counts(normalize=True)
    top_dev_types = dev_type_counts[dev_type_counts >= top_dev_types_threshold].index.tolist()
    df_cleaned['DevType_grouped'] = df_cleaned['DevType'].apply(lambda x: x if x in top_dev_types else 'Other_DevType')
    preprocessing_params['top_dev_types'] = top_dev_types
    print(f"Grouped 'DevType' into top {len(top_dev_types)} and 'Other_DevType'.")

    # 9. EdLevel (Map to ordinal numerical values and keep as categorical for OHE)
    ed_level_order = [
        'Primary/elementary school',
        'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)',
        'Some college/university study without earning a degree',
        'Associate degree (A.A., A.S., etc.)',
        'Bachelor’s degree (B.A., B.S., B.Eng., etc.)',
        'Master’s degree (M.A., M.S., M.Eng., MBA, etc.)',
        'Professional degree (JD, MD, Ph.D, Ed.D, etc.)',
        'Something else'
    ]
    df_cleaned['EdLevel'] = pd.Categorical(df_cleaned['EdLevel'], categories=ed_level_order, ordered=True)
    df_cleaned['EdLevel_numerical'] = df_cleaned['EdLevel'].cat.codes
    df_cleaned['EdLevel_numerical'].replace(-1, np.nan, inplace=True) # Replace any not found (shouldn't happen with .cat.codes)
    df_cleaned['EdLevel_numerical'].fillna(df_cleaned['EdLevel_numerical'].median(), inplace=True)
    preprocessing_params['ed_level_order'] = ed_level_order
    print("Cleaned 'EdLevel' and created 'EdLevel_numerical'.")

    # 10. Multi-select Features (e.g., Language, Database, Platform, Webframe, etc.)
    # We will create binary flags for the top N technologies within each category.
    # We'll also create a 'count' feature for each category.

    # Identify potential multi-select columns by looking for 'HaveWorkedWith', 'WantToWorkWith', 'Admired' suffixes
    # and also specific QID columns like Knowledge_, Frequency_, JobSatPoints_
    multi_select_prefixes = [
        'Language', 'Database', 'Platform', 'Webframe', 'Embedded',
        'MiscTech', 'ToolsTech', 'NEWCollabTools', 'OfficeStackAsync',
        'OfficeStackSync', 'AISearchDev', 'AITool'
    ]
    
    # Also include the 'Knowledge', 'Frequency', 'JobSatPoints' columns (these are already split in data)
    # We'll treat these as individual numerical features for now if they are point-based or single selection.
    # If they are multi-select, we'll process them below.

    all_multi_select_base_cols = []
    for prefix in multi_select_prefixes:
        # Check if actual columns exist in the DataFrame for these prefixes
        if any(col.startswith(prefix) and ('HaveWorkedWith' in col or 'WantToWorkWith' in col or 'Admired' in col) for col in df_cleaned.columns):
            all_multi_select_base_cols.append(prefix)

    # Add specific QID multi-select columns
    qid_multi_select_cols = [
        'Knowledge_1', 'Knowledge_2', 'Knowledge_3', 'Knowledge_4', 'Knowledge_5', 
        'Knowledge_6', 'Knowledge_7', 'Knowledge_8', 'Knowledge_9',
        'Frequency_1', 'Frequency_2', 'Frequency_3',
        'JobSatPoints_1', 'JobSatPoints_4', 'JobSatPoints_5', 'JobSatPoints_6',
        'JobSatPoints_7', 'JobSatPoints_8', 'JobSatPoints_9', 'JobSatPoints_10', 'JobSatPoints_11',
        'AIBen', 'AIEthics', 'AIChallenges', 'BuyNewTool', 'TechEndorse', 'Frustration', 'ProfessionalTech', 'CodingActivities'
    ]
    # Filter to only include columns that actually exist in the DataFrame
    qid_multi_select_cols = [col for col in qid_multi_select_cols if col in df_cleaned.columns]


    multi_select_binary_features = []
    engineered_numerical_features = []

    # Process complex multi-selects (e.g., LanguageHaveWorkedWith)
    for prefix in all_multi_select_base_cols:
        for suffix in ['HaveWorkedWith', 'WantToWorkWith', 'Admired']:
            col_name = f"{prefix}{suffix}"
            if col_name in df_cleaned.columns:
                df_cleaned[col_name] = df_cleaned[col_name].fillna('') # Fill NaN with empty string
                
                # Create a count feature
                df_cleaned[f'Num_{col_name}'] = df_cleaned[col_name].apply(
                    lambda x: len([item for item in x.split(';') if item.strip()])
                )
                engineered_numerical_features.append(f'Num_{col_name}')

                # One-hot encode top N most frequent items
                all_items = ';'.join(df_cleaned[col_name].astype(str).dropna()).split(';')
                all_items = [item.strip() for item in all_items if item.strip() != '']
                
                if all_items:
                    item_counts = pd.Series(all_items).value_counts()
                    # Select top items by a frequency threshold to avoid too many features
                    top_n_items_threshold = 0.01 
                    top_n_items = item_counts[item_counts / len(all_items) >= top_n_items_threshold].index.tolist()
                    
                    # Ensure we have at least 5 top items, up to a reasonable max
                    if len(top_n_items) < 5 and len(item_counts) > 0:
                        top_n_items = item_counts.nlargest(min(10, len(item_counts))).index.tolist()
                    elif not top_n_items and len(item_counts) > 0: # If threshold too high, just take top 5
                         top_n_items = item_counts.nlargest(min(5, len(item_counts))).index.tolist()

                    preprocessing_params[f'top_items_{col_name}'] = top_n_items

                    for item in top_n_items:
                        # Sanitize feature name
                        feature_name = f'{col_name}_{item.replace(" ", "_").replace("/", "_").replace("-", "_").replace(".", "").replace("(", "").replace(")", "")}_Flag'
                        df_cleaned[feature_name] = df_cleaned[col_name].apply(lambda x: 1 if item in x else 0)
                        multi_select_binary_features.append(feature_name)
    print("Engineered multi-select binary flags and counts for technologies.")

    # Process QID multi-select columns
    for col in qid_multi_select_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].fillna('') # Fill NaN with empty string
            
            # Create a count feature for each QID multi-select column
            df_cleaned[f'Num_{col}_Selected'] = df_cleaned[col].apply(lambda x: len([item for item in x.split(';') if item.strip()]) if isinstance(x, str) else 0)
            engineered_numerical_features.append(f'Num_{col}_Selected')

            # For QID columns, we often treat responses as binary flags directly for each option
            # This requires parsing the semicolons.
            unique_options = sorted(list(set(item.strip() for sublist in df_cleaned[col].dropna().apply(lambda x: str(x).split(';')) for item in sublist if item.strip())))
            
            if unique_options:
                # Filter out empty strings if any
                unique_options = [opt for opt in unique_options if opt != '']
                for option in unique_options:
                    feature_name = f'{col}_{option.replace(" ", "_").replace("/", "_").replace("-", "_").replace(".", "").replace("(", "").replace(")", "")}_Flag'
                    df_cleaned[feature_name] = df_cleaned[col].apply(lambda x: 1 if option in str(x) else 0)
                    multi_select_binary_features.append(feature_name)
                preprocessing_params[f'qid_options_{col}'] = unique_options
    print("Engineered QID multi-select binary flags and counts.")

    # 11. CompTotal (Numerical, will be handled by preprocessor for scaling, but might need cleaning here)
    # CompTotal had extreme outliers in EDA, ensure it's numeric and impute
    df_cleaned['CompTotal'] = pd.to_numeric(df_cleaned['CompTotal'], errors='coerce')
    df_cleaned['CompTotal'].fillna(df_cleaned['CompTotal'].median(), inplace=True)
    # Cap extreme values for CompTotal before passing to preprocessor to avoid issues with interaction features
    comp_total_q_low = df_cleaned['CompTotal'].quantile(0.005)
    comp_total_q_high = df_cleaned['CompTotal'].quantile(0.995)
    df_cleaned['CompTotal'] = np.clip(df_cleaned['CompTotal'], comp_total_q_low, comp_total_q_high)
    print("Cleaned and capped 'CompTotal'.")


    # 12. Interaction Features (subset of important numericals)
    # Using 'YearsCodePro', 'WorkExp', 'Age_numerical', 'OrgSize_numerical', 'CompTotal'
    # Adding more sophisticated interactions
    numerical_for_poly = [
        'YearsCodePro', 'WorkExp', 'Age_numerical', 'OrgSize_numerical', 'CompTotal'
    ]
    
    # Create pairwise polynomial features (degree 2) for these core numericals
    # Note: PolynomialFeatures in ColumnTransformer will handle this, we just need to list the input columns.
    # We will let the ColumnTransformer handle the PolynomialFeatures.
    
    # Custom interaction: Ratio of professional experience to total coding experience
    df_cleaned['YearsCodePro_Ratio'] = df_cleaned.apply(
        lambda row: row['YearsCodePro'] / row['YearsCode'] if row['YearsCode'] > 0 else 0, axis=1
    )
    engineered_numerical_features.append('YearsCodePro_Ratio')
    print("Engineered 'YearsCodePro_Ratio'.")

    # Add other simple numerical features directly to the list
    engineered_numerical_features.append('Age_numerical')
    engineered_numerical_features.append('OrgSize_numerical')
    engineered_numerical_features.append('Employed_FullTime')
    engineered_numerical_features.append('Independent_Contractor')
    engineered_numerical_features.append('Student')
    engineered_numerical_features.append('Not_Employed')
    engineered_numerical_features.append('RemoteWork_is_Remote')
    engineered_numerical_features.append('RemoteWork_is_Hybrid')
    engineered_numerical_features.append('RemoteWork_is_InPerson')
    engineered_numerical_features.append('EdLevel_numerical')

    # --- START OF JOB_SAT FIX ---
    # This is the new section to map categorical JobSat strings to numerical values
    # These numerical values MUST match what your trained model expects.
    # If your raw survey_results_public.csv 'JobSat' column is already numeric,
    # you can remove this mapping, but ensure the numbers are what your model expects.
    job_sat_mapping_for_backend = {
        "Very dissatisfied": 0,    # <<< VERIFY THIS NUMBER against your existing model's training data
        "Slightly dissatisfied": 4, # <<< VERIFY THIS NUMBER
        "Neither satisfied nor dissatisfied": 5, # <<< VERIFY THIS NUMBER
        "Slightly satisfied": 6,  # <<< VERIFY THIS NUMBER
        "Very satisfied": 10,     # <<< VERIFY THIS NUMBER
    }
    # Apply the mapping
    df_cleaned['JobSat'] = df_cleaned['JobSat'].map(job_sat_mapping_for_backend)
    
    # Impute missing JobSat values (including those that failed to map) with the median
    df_cleaned['JobSat'].fillna(df_cleaned['JobSat'].median(), inplace=True)
    engineered_numerical_features.append('JobSat') # Ensure JobSat is in the list of numerical features

    print("Cleaned 'JobSat' by mapping to numerical values and imputing NaNs.")
    # --- END OF JOB_SAT FIX ---

    # Ensure no duplicates in the engineered_numerical_features list
    engineered_numerical_features = list(set(engineered_numerical_features))


    # --- Define columns for ColumnTransformer ---
    features_info_for_ct = {
        'numerical_for_polynomial_features': numerical_for_poly, # These will get PolyFeatures + Scaling
        'other_numerical_features': [col for col in engineered_numerical_features if col in df_cleaned.columns and col not in numerical_for_poly], # These will get Scaling only
        'single_select_categorical_cols': ['Country_grouped', 'DevType_grouped'], # These will get OHE
        'multi_select_binary_features': [col for col in multi_select_binary_features if col in df_cleaned.columns] # These are already binary, just pass through
    }
    
    # Check if all columns actually exist in the dataframe after preprocessing
    for key, cols in features_info_for_ct.items():
        features_info_for_ct[key] = [col for col in cols if col in df_cleaned.columns]
        if len(cols) != len(features_info_for_ct[key]):
            print(f"Warning: Some columns in '{key}' were not found in df_cleaned and were removed.")

    # --- EDA: Numerical Feature Distributions and Correlations ---
    # Select important numerical features for plotting
    numerical_cols_for_eda = list(set(numerical_for_poly + features_info_for_ct['other_numerical_features']))
    numerical_cols_for_eda = [col for col in numerical_cols_for_eda if col in df_cleaned.columns] # Ensure existence

    print("\nGenerating numerical feature distributions...")
    for col in numerical_cols_for_eda:
        if df_cleaned[col].dtype in ['int64', 'float64'] and not df_cleaned[col].isnull().all():
            plt.figure(figsize=(10, 5))
            sns.histplot(df_cleaned[col], bins=30, kde=True)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(EDA_PLOTS_DIR, f'hist_{col}.png'))
            plt.close()

            plt.figure(figsize=(10, 2))
            sns.boxplot(x=df_cleaned[col])
            plt.title(f'Box Plot of {col}')
            plt.xlabel(col)
            plt.tight_layout()
            plt.savefig(os.path.join(EDA_PLOTS_DIR, f'boxplot_{col}.png'))
            plt.close()
    
    # Correlation Heatmap for important numerical features and target
    print("Generating correlation heatmap...")
    corr_cols = numerical_cols_for_eda + ['ConvertedCompYearly_log']
    corr_df = df_cleaned[corr_cols].corr()

    plt.figure(figsize=(16, 14))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Key Numerical Features and Log-Transformed Target')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_PLOTS_DIR, 'correlation_heatmap.png'))
    plt.close()

    print("\nEDA plots generated and saved to 'model_artifacts/eda_plots/'. Please review them.")
    print("\nPreprocessing complete. Ready for ColumnTransformer.")

    return df_cleaned, features_info_for_ct, preprocessing_params

if __name__ == '__main__':
    # When run as a script, perform the preprocessing and save info
    df_processed, ct_features, pp_params = preprocess_and_eda()
    
    # Save the features_info_for_ct and preprocessing_params for train_model.py
    # This prevents recalculation and ensures consistency.
    MODEL_ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'model_artifacts')
    os.makedirs(MODEL_ARTIFACTS_DIR, exist_ok=True)
    
    joblib.dump(ct_features, os.path.join(MODEL_ARTIFACTS_DIR, 'features_info_for_ct.joblib'))
    joblib.dump(pp_params, os.path.join(MODEL_ARTIFACTS_DIR, 'preprocessing_params_for_data.joblib'))
    
    print(f"\nFinal processed DataFrame shape: {df_processed.shape}")
    print(f"Columns for ColumnTransformer saved to: {os.path.join(MODEL_ARTIFACTS_DIR, 'features_info_for_ct.joblib')}")
    print(f"Preprocessing parameters (mappings, top_techs etc.) saved to: {os.path.join(MODEL_ARTIFACTS_DIR, 'preprocessing_params_for_data.joblib')}")

    print("\nData preparation script finished. Now run train_model.py (if retraining is desired, otherwise the generated artifacts can be used for prediction).")