import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress all warnings for cleaner output during execution
warnings.filterwarnings('ignore')

# Define paths
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'survey_results_public.csv')
MODEL_ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'model_artifacts')
PLOTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'plots') # New directory for plots

# Ensure directories exist
os.makedirs(MODEL_ARTIFACTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# 1. Load data from the preprocessed CSV created by prepare_data.py
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from prepare_data import preprocess_and_eda
sys.path.pop(sys.path.index(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

print("Loading preprocessed data and feature information...")
df_processed, features_info_for_ct, preprocessing_params = preprocess_and_eda(data_path=DATA_PATH)

# Extract column lists for ColumnTransformer from the loaded dictionary
numerical_for_polynomial_features = features_info_for_ct['numerical_for_polynomial_features']
other_numerical_features = features_info_for_ct['other_numerical_features']
single_select_categorical_cols = features_info_for_ct['single_select_categorical_cols']
multi_select_binary_features = features_info_for_ct['multi_select_binary_features']

# Define target variable
TARGET_COL = 'ConvertedCompYearly_log'

# Define features (all columns except the original target and log-transformed target)
features = [col for col in df_processed.columns if col != 'ConvertedCompYearly' and col != TARGET_COL]

# Save the exact list of columns that will be fed into the ColumnTransformer
joblib.dump(features, os.path.join(MODEL_ARTIFACTS_DIR, 'input_features_to_ct.joblib'))
print(f"Input features list for ColumnTransformer saved to: {os.path.join(MODEL_ARTIFACTS_DIR, 'input_features_to_ct.joblib')}")

# 2. Split data into training and testing sets
X = df_processed[features]
y = df_processed[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split into training ({X_train.shape[0]} samples) and testing ({X_test.shape[0]} samples).")

# 3. Custom Standard Scaler to handle potential NumPy scalar issues
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

# 4. Create the ColumnTransformer
numerical_poly_pipeline = Pipeline(steps=[
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', CustomStandardScaler())
])

numerical_scaling_pipeline = Pipeline(steps=[
    ('scaler', CustomStandardScaler())
])

categorical_ohe_pipeline = Pipeline(steps=[
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

binary_pass_through_pipeline = 'passthrough'

preprocessor = ColumnTransformer(
    transformers=[
        ('num_poly', numerical_poly_pipeline, numerical_for_polynomial_features),
        ('num_scale', numerical_scaling_pipeline, other_numerical_features),
        ('cat_ohe', categorical_ohe_pipeline, single_select_categorical_cols),
        ('binary_passthrough', binary_pass_through_pipeline, multi_select_binary_features)
    ],
    remainder='drop'
)

print("ColumnTransformer configured.")

# 5. Define models and their hyperparameter distributions for RandomizedSearchCV
models_config = {
    'RandomForest': {
        'estimator': RandomForestRegressor(random_state=42, n_jobs=-1),
        'param_distributions': {
            'n_estimators': randint(100, 300),
            'max_features': uniform(0.3, 0.4),
            'max_depth': randint(10, 25),
            'min_samples_split': randint(5, 15),
            'min_samples_leaf': randint(3, 8)
        },
        'n_iter': 15
    },
    'XGBoost': {
        'estimator': xgb.XGBRegressor(random_state=42, n_jobs=-1, tree_method='hist', enable_categorical=False),
        'param_distributions': {
            'n_estimators': randint(200, 500),
            'learning_rate': uniform(0.01, 0.08),
            'max_depth': randint(3, 8),
            'subsample': uniform(0.6, 0.35),
            'colsample_bytree': uniform(0.6, 0.35),
            'gamma': uniform(0, 0.2),
            'reg_lambda': uniform(0.5, 1.5),
            'reg_alpha': uniform(0, 0.2)
        },
        'n_iter': 25
    },
    'LightGBM': {
        'estimator': lgb.LGBMRegressor(random_state=42, n_jobs=-1),
        'param_distributions': {
            'n_estimators': randint(200, 500),
            'learning_rate': uniform(0.01, 0.06),
            'num_leaves': randint(20, 60),
            'max_depth': randint(5, 15),
            'subsample': uniform(0.6, 0.35),
            'colsample_bytree': uniform(0.6, 0.35),
            'reg_alpha': uniform(0, 0.1),
            'reg_lambda': uniform(0, 0.1)
        },
        'n_iter': 25
    },
    'CatBoost': {
        'estimator': CatBoostRegressor(random_state=42, verbose=0, thread_count=-1),
        'param_distributions': {
            'iterations': randint(200, 500),
            'learning_rate': uniform(0.01, 0.08),
            'depth': randint(3, 8),
            'l2_leaf_reg': uniform(1, 4),
            'border_count': randint(32, 128),
            'subsample': uniform(0.6, 0.35)
        },
        'n_iter': 25
    },
    'HistGradientBoosting': {
        'estimator': HistGradientBoostingRegressor(random_state=42),
        'param_distributions': {
            'max_iter': randint(200, 500),
            'learning_rate': uniform(0.01, 0.08),
            'max_leaf_nodes': randint(15, 60),
            'max_depth': randint(4, 15),
            'min_samples_leaf': randint(10, 40),
            'l2_regularization': uniform(0, 0.4)
        },
        'n_iter': 25
    }
}

# 6. Train and evaluate models
best_model = None
best_r2_score = -np.inf
best_model_name = ""

results = {}

for name, config in models_config.items():
    print(f"\n--- Training {name} ---")
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', config['estimator'])
    ])

    search_params = {f'regressor__{k}': v for k, v in config['param_distributions'].items()}

    random_search = RandomizedSearchCV(
        estimator=model_pipeline,
        param_distributions=search_params,
        n_iter=config['n_iter'],
        cv=5,
        scoring='r2',
        random_state=42,
        # IMPORTANT CHANGE: Limiting n_jobs to reduce system resource usage
        # You can adjust this number (e.g., 2 or 1) if you still face resource issues.
        n_jobs=4,
        verbose=1
    )

    random_search.fit(X_train, y_train)

    y_pred = random_search.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    results[name] = {
        'best_params': random_search.best_params_,
        'r2_score': r2,
        'mse': mse,
        'rmse': rmse,
        'best_estimator': random_search.best_estimator_
    }

    print(f"Model: {name}")
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Test R2 Score: {r2:.4f}")
    print(f"Test RMSE: {rmse:.4f}")

    if r2 > best_r2_score:
        best_r2_score = r2
        best_model = random_search.best_estimator_
        best_model_name = name

print("\n--- Training Summary ---")
for name, res in results.items():
    print(f"Model: {name}, R2 Score: {res['r2_score']:.4f}, RMSE: {res['rmse']:.4f}")

print(f"\nBest performing model: {best_model_name} with R2 Score: {best_r2_score:.4f}")

# 7. Save the best model
if best_model:
    model_save_path = os.path.join(MODEL_ARTIFACTS_DIR, f'best_model_{best_model_name.lower()}.joblib')
    joblib.dump(best_model, model_save_path)
    print(f"Best model saved to {model_save_path}")

print("\nModel training script finished.")
print("You can now use the saved model for predictions.")

# --- 8. Generate Plots for Documentation ---
print("\n--- Generating Plots ---")

# Set a consistent style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6) # Default figure size

# 8.1. Data Distributions (using df_processed before splitting)
print("Generating data distribution plots...")

# Example numerical features to plot (choose a representative subset)
numerical_cols_to_plot = [
    'Age_numerical', 'YearsCode', 'YearsCodePro', 'WorkExp',
    'OrgSize_numerical', 'CompTotal', 'YearsCodePro_Ratio'
]

for col in numerical_cols_to_plot:
    if col in df_processed.columns:
        plt.figure()
        sns.histplot(df_processed[col], kde=True, bins=30)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'dist_{col}.png'))
        plt.close()
        print(f"Saved: dist_{col}.png")

# Example categorical features to plot (choose a representative subset)
categorical_cols_to_plot = [
    'EdLevel', 'Country_grouped', 'DevType_grouped', 'RemoteWork',
    'Employed_FullTime', 'Independent_Contractor', 'Student', 'Not_Employed'
]

for col in categorical_cols_to_plot:
    if col in df_processed.columns:
        plt.figure()
        # For binary flags or few categories, use countplot
        if df_processed[col].nunique() < 10 and df_processed[col].dtype != 'object': # For binary flags etc.
            sns.countplot(x=df_processed[col])
        else: # For other categorical columns, use value counts and then plot
            top_n = df_processed[col].value_counts().nlargest(10)
            sns.barplot(x=top_n.index, y=top_n.values)
            plt.xticks(rotation=45, ha='right')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'dist_{col}.png'))
        plt.close()
        print(f"Saved: dist_{col}.png")

# 8.2. Model Performance Plots (using the best model)
if best_model:
    print("Generating model performance plots...")
    
    # Get predictions from the best model on the test set
    y_pred_best = best_model.predict(X_test)

    # Actual vs. Predicted Plot
    plt.figure()
    plt.scatter(y_test, y_pred_best, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Log Transformed Salary')
    plt.ylabel('Predicted Log Transformed Salary')
    plt.title(f'Actual vs. Predicted Salaries ({best_model_name})')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'actual_vs_predicted.png'))
    plt.close()
    print("Saved: actual_vs_predicted.png")

    # Residuals Plot
    residuals = y_test - y_pred_best
    plt.figure()
    sns.histplot(residuals, kde=True, bins=50)
    plt.title(f'Residuals Distribution ({best_model_name})')
    plt.xlabel('Residuals (Actual - Predicted)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'residuals_distribution.png'))
    plt.close()
    print("Saved: residuals_distribution.png")

    plt.figure()
    plt.scatter(y_pred_best, residuals, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Log Transformed Salary')
    plt.ylabel('Residuals')
    plt.title(f'Residuals Plot ({best_model_name})')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'residuals_plot.png'))
    plt.close()
    print("Saved: residuals_plot.png")


    # R2 Score Comparison Across Models
    r2_scores = {name: res['r2_score'] for name, res in results.items()}
    model_names = list(r2_scores.keys())
    scores = list(r2_scores.values())

    plt.figure()
    sns.barplot(x=model_names, y=scores, palette='viridis')
    plt.xlabel('Model')
    plt.ylabel('R2 Score (Test Set)')
    plt.title('R2 Score Comparison Across Models')
    plt.ylim(min(scores) * 0.9, max(scores) * 1.1)
    for index, value in enumerate(scores):
        plt.text(index, value, f'{value:.3f}', color='black', ha="center", va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'r2_score_comparison.png'))
    plt.close()
    print("Saved: r2_score_comparison.png")

    # 8.3. Feature Importance (for tree-based models)
    if hasattr(best_model.named_steps['regressor'], 'feature_importances_'):
        print(f"Generating feature importance plot for {best_model_name}...")
        
        transformed_feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
        importances = best_model.named_steps['regressor'].feature_importances_
        
        feature_importance_df = pd.DataFrame({
            'Feature': transformed_feature_names,
            'Importance': importances
        })
        
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        
        top_n_features = 20
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(top_n_features), palette='magma')
        plt.title(f'Top {top_n_features} Feature Importances for {best_model_name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'feature_importance_{best_model_name.lower()}.png'))
        plt.close()
        print(f"Saved: feature_importance_{best_model_name.lower()}.png")
    else:
        print(f"Skipping feature importance plot: {best_model_name} is not a tree-based model or does not expose feature_importances_.")

print("\nAll plots generated and saved to the 'plots' directory.")