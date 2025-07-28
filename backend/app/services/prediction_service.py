# app/services/prediction_service.py
import os
import joblib
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline # Import Pipeline to check type
from sklearn.compose import ColumnTransformer # Import ColumnTransformer

# Path to where the trained model artifacts are stored
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'model_artifacts')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model_lightgbm.joblib')
PREPROCESSING_PARAMS_PATH = os.path.join(MODEL_DIR, 'preprocessing_params_for_data.joblib')
FEATURES_INFO_FOR_CT_PATH = os.path.join(MODEL_DIR, 'features_info_for_ct.joblib')
INPUT_FEATURES_TO_CT_PATH = os.path.join(MODEL_DIR, 'input_features_to_ct.joblib')

# Define the path to your raw training data CSV
# IMPORTANT: Adjust this path if your survey_results_public.csv is located elsewhere
TRAINING_DATA_CSV_PATH = os.path.join(MODEL_DIR, '..', 'data', 'survey_results_public.csv')


# Custom Standard Scaler class - MUST be included for joblib.load to work
class CustomStandardScaler(StandardScaler):
    def transform(self, X):
        if isinstance(X, pd.Series):
            return super().transform(X.values.reshape(-1, 1)).flatten()
        elif isinstance(X, pd.DataFrame) and X.shape[1] == 1:
            return super().transform(X.values).flatten()
        elif isinstance(X, np.ndarray) and X.ndim == 1:
            return super().transform(X.reshape(-1, 1)).flatten()
        elif isinstance(X, (np.integer, np.floating)):
            return super().transform(np.array([[X]])).flatten()
        return super().transform(X)

    def fit_transform(self, X, y=None):
        if isinstance(X, pd.Series):
            return super().fit_transform(X.values.reshape(-1, 1)).flatten()
        elif isinstance(X, pd.DataFrame) and X.shape[1] == 1:
            return super().fit_transform(X.values).flatten()
        elif isinstance(X, np.ndarray) and X.ndim == 1:
            return super().fit_transform(X.reshape(-1, 1)).flatten()
        return super().fit_transform(X, y)


class SalaryPredictionService:
    _instance = None # Singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SalaryPredictionService, cls).__new__(cls)
            cls._instance._load_model_and_artifacts()
        return cls._instance

    def _load_model_and_artifacts(self):
        """
        Loads the pre-trained machine learning model pipeline and all necessary
        preprocessing artifacts (mappings, top items lists, feature info).
        Also loads a sample of original training data for salary distribution.
        """
        artifact_paths = {
            'model': MODEL_PATH,
            'preprocessing_params': PREPROCESSING_PARAMS_PATH,
            'features_info_for_ct': FEATURES_INFO_FOR_CT_PATH,
            'input_features_to_ct': INPUT_FEATURES_TO_CT_PATH
        }

        self.model = None # This will hold the full pipeline
        self.lgbm_model_estimator = None # This will hold the extracted LGBM model
        self.column_transformer_preprocessor = None # This will hold the ColumnTransformer
        self.preprocessing_params = None
        self.features_info_for_ct = None
        self.input_features_to_ct = None
        self.training_salary_data = [] # Initialize for salary distribution chart

        for name, path in artifact_paths.items():
            if not os.path.exists(path):
                print(f"Error: {name.replace('_', ' ').capitalize()} file not found at {path}. "
                      f"Please ensure 'train_model.py' and 'prepare_data.py' have been run successfully.")
                continue 

        try:
            self.model = joblib.load(MODEL_PATH)
            self.preprocessing_params = joblib.load(PREPROCESSING_PARAMS_PATH)
            self.features_info_for_ct = joblib.load(FEATURES_INFO_FOR_CT_PATH)
            self.input_features_to_ct = joblib.load(INPUT_FEATURES_TO_CT_PATH)

            print(f"Successfully loaded model from {MODEL_PATH}")
            print(f"Successfully loaded preprocessing parameters from {PREPROCESSING_PARAMS_PATH}")
            print(f"Successfully loaded features info for ColumnTransformer from {FEATURES_INFO_FOR_CT_PATH}")
            print(f"Successfully loaded input features list for ColumnTransformer from {INPUT_FEATURES_TO_CT_PATH}")

            # --- IMPORTANT NEW LOGIC: Extract the actual LightGBM model and ColumnTransformer from the pipeline ---
            if isinstance(self.model, Pipeline):
                # Assuming the LightGBM model is the last step in the pipeline
                self.lgbm_model_estimator = self.model.steps[-1][1]
                print(f"Successfully extracted LightGBM model from pipeline: {self.lgbm_model_estimator.__class__.__name__}")

                # Find the ColumnTransformer in the pipeline
                for name, step in self.model.steps:
                    if isinstance(step, ColumnTransformer):
                        self.column_transformer_preprocessor = step
                        print(f"Successfully extracted ColumnTransformer from pipeline: {name}")
                        break
                if not self.column_transformer_preprocessor:
                    print("Warning: ColumnTransformer not found in the loaded pipeline.")

                if not hasattr(self.lgbm_model_estimator, 'feature_importances_'):
                    print("Warning: Extracted LightGBM estimator does not have 'feature_importances_'.")
            elif hasattr(self.model, 'feature_importances_'):
                # If the loaded model IS directly the LightGBM model (not wrapped in a pipeline)
                self.lgbm_model_estimator = self.model
                print("Loaded model is directly a LightGBM model.")
                print("Warning: No ColumnTransformer was explicitly loaded or found as part of a pipeline. Feature names might be an issue.")
            else:
                self.lgbm_model_estimator = None # Ensure it's explicitly None if not found
                print("Warning: Loaded model is neither a scikit-learn pipeline with a LightGBM estimator nor a direct LightGBM model. Feature importances may not be available.")


            # --- Load data for Salary Distribution Chart ---
            if os.path.exists(TRAINING_DATA_CSV_PATH):
                try:
                    full_df = pd.read_csv(TRAINING_DATA_CSV_PATH, usecols=['ConvertedCompYearly'])
                    self.training_salary_data = full_df['ConvertedCompYearly'].dropna()
                    
                    # Optional: Filter out extreme outliers for better visualization
                    lower_bound = self.training_salary_data.quantile(0.01) # 1st percentile
                    upper_bound = self.training_salary_data.quantile(0.99) # 99th percentile
                    self.training_salary_data = self.training_salary_data[
                        (self.training_salary_data >= lower_bound) & 
                        (self.training_salary_data <= upper_bound)
                    ]

                    # IMPORTANT: Sample the data if it's very large to avoid huge JSON payloads
                    if len(self.training_salary_data) > 10000: # Example: limit to 10,000 samples
                        self.training_salary_data = self.training_salary_data.sample(10000, random_state=42) # Add random_state for reproducibility
                    
                    self.training_salary_data = self.training_salary_data.tolist()

                    print(f"Successfully loaded {len(self.training_salary_data)} training salary data points for distribution from {TRAINING_DATA_CSV_PATH}")
                except Exception as e:
                    print(f"Warning: Could not load or process training data for salary distribution from {TRAINING_DATA_CSV_PATH}: {e}")
                    self.training_salary_data = [] # Fallback to empty list
            else:
                print(f"Warning: Training data CSV not found at {TRAINING_DATA_CSV_PATH}. Salary distribution chart will not be available.")


        except Exception as e:
            print(f"Critical Error loading model or preprocessing artifacts: {e}")
            self.model = None
            self.lgbm_model_estimator = None # Also ensure this is reset on error
            self.column_transformer_preprocessor = None
            self.preprocessing_params = None
            self.features_info_for_ct = None
            self.input_features_to_ct = None

    def _preprocess_input(self, input_data: dict) -> pd.DataFrame:
        """
        Preprocesses the raw input data from the API request to match the format
        expected by the trained ML pipeline's ColumnTransformer.
        This function MUST EXACTLY mirror the logic in prepare_data.py's
        feature engineering steps, creating the same columns with the same values.
        """
        if self.model is None:
            raise RuntimeError("Prediction service not fully initialized. Model not loaded.")
        if self.preprocessing_params is None or self.features_info_for_ct is None or self.input_features_to_ct is None:
            raise RuntimeError("Preprocessing parameters, features info, or input features list not loaded.")

        # Initialize a dictionary to hold all raw input features, with NaN for missing ones
        initial_raw_data = defaultdict(lambda: np.nan) 
        # Using a comprehensive list of columns that might be in the raw survey results
        # This list should ideally come from the prepare_data.py or a config,
        # but for direct porting, let's keep it explicit.
        initial_raw_columns_from_survey = [
            'ResponseId', 'QID', 'MainBranch', 'Employment', 'RemoteWork', 'CodingActivities',
            'EdLevel', 'LearnCode', 'LearnCodeOnline', 'LearnCodeCourses',
            'YearsCode', 'YearsCodePro', 'DevType', 'OrgSize', 'Country', 'Currency',
            'CompTotal', 'CompFreq', 'ConvertedCompYearly', 'WorkExp', 'Knowledge_1',
            'Knowledge_2', 'Knowledge_3', 'Knowledge_4', 'Knowledge_5', 'Knowledge_6',
            'Knowledge_7', 'Knowledge_8', 'Knowledge_9', 'Frequency_1', 'Frequency_2',
            'Frequency_3', 'LanguageHaveWorkedWith', 'LanguageWantToWorkWith',
            'DatabaseHaveWorkedWith', 'DatabaseWantToWorkWith', 'PlatformHaveWorkedWith',
            'PlatformWantToWorkWith', 'WebframeHaveWorkedWith', 'WebframeWantToWorkWith',
            'MiscTechHaveWorkedWith', 'MiscTechWantToWorkWith', 'ToolsTechHaveWorkedWith',
            'ToolsTechWantToWorkWith', 'NEWCollabToolsHaveWorkedWith', 'NEWCollabToolsWantToWorkWith',
            'OfficeStackAsyncHaveWorkedWith', 'OfficeStackAsyncWantToWorkWith',
            'OfficeStackSyncHaveWorkedWith', 'OfficeStackSyncWantToWorkWith',
            'AISearchDevHaveWorkedWith', 'AISearchDevWantToWorkWith', 'AIToolHaveWorkedWith',
            'AIToolWantToWorkWith', 'OpSysPersonal use', 'OpSysProfessional use',
            'ITperson', 'ProfessionalTech', 'IDE', 'DeveloperContainer',
            'DeveloperMachine', 'Blockchain', 'BlockchainIs', 'SOVisitFreq', 'SOTimeSaved',
            'SOHowMuchTime', 'SOAccount', 'SOPartFreq', 'TSFeatures', 'RRoche', 'Age',
            'Gender', 'Trans', 'Sexuality', 'Ethnicity', 'Accessibility', 'MentalHealth',
            'TiredCode', 'SuggestDangerous', 'Metrics', 'Partners', 'SurveyLength',
            'SurveyEase', 'DependVsIndep', 'JobSat', 'JobSatPoints_1', 'JobSatPoints_2',
            'JobSatPoints_3', 'JobSatPoints_4', 'JobSatPoints_5', 'JobSatPoints_6',
            'JobSatPoints_7', 'JobSatPoints_8', 'JobSatPoints_9', 'JobSatPoints_10',
            'JobSatPoints_11', 'CodedProjects',
            'AIBen', 'AIEthics', 'AIChallenges', 'BuyNewTool', 'TechEndorse', 'Frustration' # Ensure these are in the raw columns
        ]
        # Populate initial_raw_data from input_data, ensuring keys are present
        for col in initial_raw_columns_from_survey:
            initial_raw_data[col] = input_data.get(col, np.nan)

        # Create a working DataFrame from the initially raw input data
        df_working = pd.DataFrame([initial_raw_data])

        # Helper for imputation
        def get_imputation_value(param_key, fallback_series):
            # Try to get the pre-calculated imputation value from preprocessing_params
            precalculated_value = self.preprocessing_params.get(param_key)
            
            if precalculated_value is not None:
                return precalculated_value
            
            # If pre-calculated value is None (i.e., not found in params), use fallback_series
            if fallback_series.empty or fallback_series.isnull().all():
                # If fallback_series is also empty/all NaN, return a sensible default
                if pd.api.types.is_numeric_dtype(fallback_series):
                    return 0.0 # Default for numerical if all else fails
                else:
                    return '' # Default for categorical if all else fails
                
            if pd.api.types.is_numeric_dtype(fallback_series):
                return fallback_series.median()
            else:
                # For categorical, mode()[0] can fail if series is all NaN or empty
                if not fallback_series.empty:
                    return fallback_series.mode()[0]
                else:
                    return '' # Fallback for categorical if mode() is not applicable

        # --- Replicate Preprocessing from prepare_data.py ---

        # 1. Age (Categorical to Numerical)
        age_mapping = self.preprocessing_params['age_mapping']
        df_working['Age_numerical'] = df_working['Age'].map(age_mapping)
        df_working['Age_numerical'].fillna(get_imputation_value('median_age_numerical', df_working['Age_numerical']), inplace=True)

        # 2. YearsCode and YearsCodePro (Convert to numerical, handle specific strings)
        def clean_years_code_col_for_prediction(value_series, median_key):
            value_series = value_series.astype(str).replace('Less than 1 year', '0').replace('More than 50 years', '50')
            numeric_series = pd.to_numeric(value_series, errors='coerce')
            return numeric_series.fillna(get_imputation_value(median_key, numeric_series))

        df_working['YearsCode'] = clean_years_code_col_for_prediction(df_working['YearsCode'], 'median_yearscode')
        df_working['YearsCodePro'] = clean_years_code_col_for_prediction(df_working['YearsCodePro'], 'median_yearscodepro')

        # 3. WorkExp (Convert to numerical, handle NaNs)
        df_working['WorkExp'] = pd.to_numeric(df_working['WorkExp'], errors='coerce')
        df_working['WorkExp'].fillna(get_imputation_value('median_workexp', df_working['WorkExp']), inplace=True)

        # 4. OrgSize (Categorical to Numerical)
        org_size_mapping = self.preprocessing_params['org_size_mapping']
        df_working['OrgSize_numerical'] = df_working['OrgSize'].map(org_size_mapping)
        df_working['OrgSize_numerical'].fillna(get_imputation_value('median_orgsize_numerical', df_working['OrgSize_numerical']), inplace=True)

        # 5. Employment (Extract binary flags)
        df_working['Employment'] = df_working['Employment'].fillna('')
        df_working['Employed_FullTime'] = df_working['Employment'].apply(lambda x: 1 if 'Employed, full-time' in str(x) else 0)
        df_working['Independent_Contractor'] = df_working['Employment'].apply(lambda x: 1 if 'Independent contractor, freelancer, or self-employed' in str(x) else 0)
        df_working['Student'] = df_working['Employment'].apply(lambda x: 1 if 'Student' in str(x) else 0)
        df_working['Not_Employed'] = df_working['Employment'].apply(lambda x: 1 if 'Not employed' in str(x) else 0)

        # 6. RemoteWork (Handle 'NA' and create binary flags)
        df_working['RemoteWork'] = df_working['RemoteWork'].fillna('Unknown')
        df_working['RemoteWork_is_Remote'] = df_working['RemoteWork'].apply(lambda x: 1 if x == 'Remote' else 0)
        df_working['RemoteWork_is_Hybrid'] = df_working['RemoteWork'].apply(lambda x: 1 if x == 'Hybrid (some remote, some in-person)' else 0)
        df_working['RemoteWork_is_InPerson'] = df_working['RemoteWork'].apply(lambda x: 1 if x == 'In-person' else 0)

        # 7. Country (Top N + Other)
        top_countries = self.preprocessing_params['top_countries']
        df_working['Country_grouped'] = df_working['Country'].apply(lambda x: x if pd.notna(x) and x in top_countries else 'Other_Country')

        # 8. DevType (Top N + Other)
        top_dev_types = self.preprocessing_params['top_dev_types']
        df_working['DevType_grouped'] = df_working['DevType'].apply(lambda x: x if pd.notna(x) and x in top_dev_types else 'Other_DevType')

        # 9. EdLevel (Map to ordinal numerical values and keep as categorical for OHE)
        ed_level_order = self.preprocessing_params['ed_level_order']
        df_working['EdLevel'] = pd.Categorical(df_working['EdLevel'], categories=ed_level_order, ordered=True)
        df_working['EdLevel_numerical'] = df_working['EdLevel'].cat.codes
        df_working['EdLevel_numerical'].replace(-1, np.nan, inplace=True) # Replace -1 (for missing categories) with NaN
        df_working['EdLevel_numerical'].fillna(get_imputation_value('median_edlevel_numerical', df_working['EdLevel_numerical']), inplace=True)

        # 10. Multi-select Features (e.g., Language, Database, Platform, Webframe, etc.)
        multi_select_prefixes = [
            'Language', 'Database', 'Platform', 'Webframe', 'MiscTech', 'ToolsTech',
            'NEWCollabTools', 'OfficeStackAsync', 'OfficeStackSync',
            'AISearchDev', 'AITool'
        ]
        
        multi_select_suffixes = ['HaveWorkedWith', 'WantToWorkWith'] 


        qid_multi_select_cols = [
            'Knowledge_1', 'Knowledge_2', 'Knowledge_3', 'Knowledge_4', 'Knowledge_5',
            'Knowledge_6', 'Knowledge_7', 'Knowledge_8', 'Knowledge_9',
            'Frequency_1', 'Frequency_2', 'Frequency_3',
            'JobSatPoints_1', 'JobSatPoints_4', 'JobSatPoints_5', 'JobSatPoints_6',
            'JobSatPoints_7', 'JobSatPoints_8', 'JobSatPoints_9', 'JobSatPoints_10', 'JobSatPoints_11',
            'ProfessionalTech', 'CodingActivities',
            'AIBen', 'AIEthics', 'AIChallenges', 'BuyNewTool', 'TechEndorse', 'Frustration' # ADDED THESE
        ]
        
        new_features_data = {}

        # Process multi-select columns
        for prefix in multi_select_prefixes:
            for suffix in multi_select_suffixes:
                col_name = f"{prefix}{suffix}"
                # Use .get with Series(['']) for default to handle potentially missing input columns gracefully
                current_col_data = df_working.get(col_name, pd.Series([''])).iloc[0]
                current_col_data_str = str(current_col_data)

                new_features_data[f'Num_{col_name}'] = len([item for item in current_col_data_str.split(';') if item.strip()])

                top_n_items = self.preprocessing_params.get(f'top_items_{col_name}', [])
                for item in top_n_items:
                    # Clean the item name for valid feature names
                    feature_name = f'{col_name}_{item.replace(" ", "_").replace("/", "_").replace("-", "_").replace(".", "").replace("(", "").replace(")", "")}_Flag'
                    new_features_data[feature_name] = 1 if item in current_col_data_str else 0

        # Process QID multi-select columns
        for col in qid_multi_select_cols:
            current_col_data = df_working.get(col, pd.Series([''])).iloc[0]
            current_col_data_str = str(current_col_data)

            new_features_data[f'Num_{col}_Selected'] = len([item for item in current_col_data_str.split(';') if item.strip()])

            unique_options = self.preprocessing_params.get(f'qid_options_{col}', [])
            for option in unique_options:
                feature_name = f'{col}_{option.replace(" ", "_").replace("/", "_").replace("-", "_").replace(".", "").replace("(", "").replace(")", "")}_Flag'
                new_features_data[feature_name] = 1 if option in current_col_data_str else 0
        
        # Convert the new_features_data dictionary to a DataFrame for concatenation
        df_new_features = pd.DataFrame([new_features_data])

        # Concatenate the original df_working with the new features DataFrame efficiently
        df_working = pd.concat([df_working, df_new_features], axis=1)


        # 11. CompTotal (Numerical, will be handled by preprocessor for scaling)
        df_working['CompTotal'] = pd.to_numeric(df_working['CompTotal'], errors='coerce')
        df_working['CompTotal'].fillna(get_imputation_value('median_comptotal', df_working['CompTotal']), inplace=True)


        # 12. Interaction Features (e.g., YearsCodePro_Ratio)
        df_working['YearsCodePro_Ratio'] = df_working.apply(
            lambda row: row['YearsCodePro'] / row['YearsCode'] if row['YearsCode'] > 0 else 0, axis=1
        )


        # 13. JobSat (Numerical, convert to numeric and fill NaNs)
        df_working['JobSat'] = pd.to_numeric(df_working['JobSat'], errors='coerce')
        df_working['JobSat'].fillna(get_imputation_value('median_jobsat', df_working['JobSat']), inplace=True)


        # --- Final Data Alignment for ColumnTransformer ---
        # Create a DataFrame with all columns expected by the ColumnTransformer, initialized to 0.0.
        # This is crucial to ensure all one-hot encoded features are present, even if their
        # corresponding category is not in the current input.
        final_processed_df = pd.DataFrame(0.0, index=[0], columns=self.input_features_to_ct)

        # Populate final_processed_df with values from df_working.
        # This will overwrite the 0.0 defaults for any columns that were calculated/present.
        for col in self.input_features_to_ct: # Iterate through the exact list of expected columns
            if col in df_working.columns:
                # Assign the scalar value from df_working.
                # Convert boolean columns to integer (0 or 1) for consistency with numerical features.
                if df_working[col].dtype == 'bool':
                    final_processed_df[col] = df_working[col].astype(int).iloc[0]
                else:
                    final_processed_df[col] = df_working[col].iloc[0]
            # If 'col' is not in df_working, it means it's an OHE column not present in the input
            # for this prediction, so it correctly remains 0.0 from initialization.

        # The final_processed_df is now guaranteed to have the correct columns in the correct order.
        return final_processed_df

    def predict(self, input_data: dict):
        """
        Makes a salary prediction and returns additional insights like
        feature importances and salary distribution data.
        """
        if self.model is None: # self.model is the pipeline
            raise RuntimeError("Prediction service not fully initialized. Model not loaded.")
        if self.lgbm_model_estimator is None: # Ensure the actual lgbm model was extracted
            print("Warning: LightGBM model not properly extracted. Feature importances will not be available.")
        if self.column_transformer_preprocessor is None:
            print("Warning: ColumnTransformer not properly extracted. Feature names from CT might be missing for importances.")


        try:
            processed_input_df = self._preprocess_input(input_data)
            
            # Prediction uses the full pipeline (self.model)
            predicted_log_salary = self.model.predict(processed_input_df)[0]
            predicted_salary = np.exp(predicted_log_salary) # exp(x)

            # --- Generate Feature Importances using self.lgbm_model_estimator ---
            feature_importances = []
            if self.lgbm_model_estimator and hasattr(self.lgbm_model_estimator, 'feature_importances_'):
                # Get feature names from the ColumnTransformer if it exists
                if self.column_transformer_preprocessor:
                    # For scikit-learn versions >= 0.23, use get_feature_names_out()
                    # For older versions, you might need custom logic or pre-saved names.
                    try:
                        # Assuming the ColumnTransformer has been fitted during model training
                        # and can provide output feature names.
                        feature_names = self.column_transformer_preprocessor.get_feature_names_out(processed_input_df.columns)
                    except AttributeError:
                        # Fallback for older scikit-learn versions or if get_feature_names_out fails
                        print("Warning: ColumnTransformer.get_feature_names_out() not available or failed. Falling back to input features.")
                        feature_names = self.input_features_to_ct
                    except Exception as e:
                        print(f"Error getting feature names from ColumnTransformer: {e}. Falling back to input features.")
                        feature_names = self.input_features_to_ct
                else:
                    # If ColumnTransformer not found, fall back to input_features_to_ct
                    feature_names = self.input_features_to_ct

                importances = self.lgbm_model_estimator.feature_importances_
                
                if len(feature_names) == len(importances):
                    all_importances = []
                    for i, imp in enumerate(importances):
                        if imp > 0: # Only include features with non-zero importance
                            all_importances.append({"feature": feature_names[i], "importance": float(imp)}) # Ensure importance is float
                    
                    feature_importances = sorted(all_importances, key=lambda x: x['importance'], reverse=True)[:20]
                    print(f"Generated {len(feature_importances)} feature importances.")
                else:
                    print(f"Warning: Feature names ({len(feature_names)}) and importances ({len(importances)}) mismatch. Cannot generate feature importance chart.")
            else:
                print("Warning: LightGBM model not found or does not have 'feature_importances_' attribute for chart.")

            # --- Prepare Salary Distribution Data ---
            salary_distribution = self.training_salary_data 

            return max(0.0, float(predicted_salary)), feature_importances, salary_distribution

        except Exception as e:
            print(f"Error during prediction or insight generation: {e}")
            raise ValueError(f"Failed to make prediction or generate insights: {e}")

# Instantiate the service when the module is loaded
salary_prediction_service = SalaryPredictionService()