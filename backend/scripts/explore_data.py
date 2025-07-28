import pandas as pd
import os

    # Define the path to your dataset
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'survey_results_public.csv')
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'survey_results_schema.csv')

def load_and_explore_data(data_path, schema_path=None):
        """
        Loads the dataset and performs initial exploratory data analysis.
        """
        print(f"Loading data from: {data_path}")
        if not os.path.exists(data_path):
            print(f"Error: Dataset not found at {data_path}. Please check the path and ensure the file is downloaded.")
            return None, None

        try:
            df = pd.read_csv(data_path)
            print("Dataset loaded successfully!")
            print(f"Number of rows: {df.shape[0]}")
            print(f"Number of columns: {df.shape[1]}")
            print("\n--- First 5 rows of the dataset ---")
            print(df.head())
            print("\n--- Dataset Info ---")
            df.info()
            print("\n--- Missing Values (Top 20 Columns) ---")
            print(df.isnull().sum().sort_values(ascending=False).head(20))
            print("\n--- Basic Statistics for Numerical Columns ---")
            print(df.describe())

            # Load schema for better understanding of columns
            if schema_path and os.path.exists(schema_path):
                schema_df = pd.read_csv(schema_path)
                print("\n--- Schema Info (Column Descriptions) ---")
                # Display description for a few key columns if possible
                key_columns = ['ConvertedCompYearly', 'YearsCodePro', 'EdLevel', 'Country', 'DevType', 'OrgSize']
                for col in key_columns:
                    desc = schema_df[schema_df['qname'] == col]['question'].values
                    if len(desc) > 0:
                        print(f"- {col}: {desc[0]}")
            else:
                print(f"Warning: Schema file not found at {schema_path}. Proceeding without detailed schema.")
                schema_df = None


            # Identify and print unique values for some important categorical columns
            print("\n--- Unique values for key categorical columns (first 10) ---")
            categorical_cols_to_check = ['EdLevel', 'Country', 'DevType', 'OrgSize', 'Employment', 'RemoteWork']
            for col in categorical_cols_to_check:
                if col in df.columns:
                    unique_vals = df[col].dropna().unique()
                    print(f"- {col}: {unique_vals[:10]}{'...' if len(unique_vals) > 10 else ''}")
                else:
                    print(f"- {col}: Not found in dataset.")


            return df, schema_df

        except Exception as e:
            print(f"An error occurred during data loading or initial exploration: {e}")
            return None, None

if __name__ == "__main__":
        # Ensure your virtual environment is activated before running this script
        # python scripts/explore_data.py
        data_df, schema_df = load_and_explore_data(DATA_PATH, SCHEMA_PATH)

        if data_df is not None:
            print("\nInitial data exploration complete. Review the output to understand the dataset.")
            print("\nNext steps will involve cleaning and preparing this data for model training.")
    