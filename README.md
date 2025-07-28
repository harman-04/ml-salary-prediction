# 📊 AI-Powered Developer Salary Predictor

This project implements a full-stack AI application designed to predict developer salaries based on various factors, leveraging the **Stack Overflow Developer Survey 2024 dataset**. The application features a robust Flask backend for machine learning predictions and a modern React frontend for user interaction and visualization.

## ✨ Features

- **Salary Prediction:** Predicts annual developer salaries (`ConvertedCompYearly`) based on user-provided inputs.
- **Intuitive User Interface:** A multi-step form built with React, `shadcn/ui`, and Tailwind CSS for a seamless experience.
- **Data Visualization:** Displays prediction results alongside insightful charts:
  - **Feature Importance Chart:** Visualizes the most influential factors in salary prediction.
  - **Salary Distribution Chart:** Shows how the predicted salary fits within the overall distribution.
- **Robust Backend:** A Flask API serving machine learning models, preprocessors, and data.
- **Pre-trained Models:** Utilizes an ensemble of pre-trained machine learning models for high-accuracy predictions.

## 🚀 Tech Stack

### Backend

- **Framework:** Flask
- **Language:** Python 3.13.4
- **Machine Learning:**
  - `scikit-learn` (for preprocessing and base models)
  - `LightGBM` (Best performing model)
  - `XGBoost`
  - `CatBoost`
  - `mlxtend` (for ensemble/meta-regressor)
  - `joblib` (for model persistence)
- **Data Handling:** `pandas`, `numpy`
- **API Server:** Gunicorn
- **Data Visualization (Backend-generated plots):** `matplotlib`, `seaborn`, `plotly`, `graphviz`

### Frontend

- **Framework:** React (with Vite)
- **UI Components:** `shadcn/ui` (built on Radix UI)
- **Styling:** Tailwind CSS
- **Form Management:** React Hook Form (with Zod for validation)
- **Animation:** Framer Motion
- **API Communication:** Axios
- **Charting:** Recharts

## 🚀 Technologies (with version)

### Backend (Python 3.13.4)

| Component        | Technologies/Packages              |
| ---------------- | ---------------------------------- |
| Framework        | Flask 2.3.3                        |
| Machine Learning | LightGBM 4.6.0, XGBoost 3.0.2      |
|                  | CatBoost 1.2.8, scikit-learn 1.7.1 |
| Data Processing  | pandas 2.3.0, numpy 2.3.1          |
| Visualization    | matplotlib 3.10.3, plotly 6.2.0    |
|                  | seaborn 0.13.2, graphviz 0.21      |
| Server           | Gunicorn 21.2.0                    |

### Frontend (Node.js v22.17.1)

| Component       | Technologies/Packages               |
| --------------- | ----------------------------------- |
| Framework       | React 19 + Vite 7.0.4               |
| UI Components   | shadcn/ui                           |
| Styling         | Tailwind CSS 4.1.11                 |
| Form Management | React Hook Form 7.61.1 + Zod 4.0.10 |
| Charts          | Recharts 3.1.0                      |
| Animation       | Framer Motion 12.23.9               |
| Icons           | Lucide React 0.526.0                |

## 📁 Project Structure

```text
.
├── backend/                         # Backend server and ML components
│   ├── app/                         # Flask application core
│   │   ├── routes/                  # API endpoint definitions
│   │   │   ├── __init__.py
│   │   │   └── prediction.py        # Salary prediction API
│   │   ├── services/                # Business logic
│   │   │   ├── __init__.py
│   │   │   └── prediction_service.py
│   │   ├── __init__.py
│   │   └── config.py                # Configuration settings
│   ├── data/                        # Survey datasets
│   │   ├── 2024 Developer Survey.pdf
│   │   ├── survey_results_processed.csv
│   │   ├── survey_results_public.csv
│   │   └── survey_results_schema.csv
│   ├── model_artifacts/             # Trained models and pipelines
│   │   ├── eda_plots/               # Exploratory analysis (155 files)
│   │   ├── best_model_lightgbm.joblib
│   │   ├── best_model_xgboost.joblib
│   │   ├── best_salary_prediction_pipeline.joblib
│   │   └── ...                      # Other model artifacts
│   ├── plots/                       # Training visualizations (20 files)
│   ├── scripts/                     # Data processing scripts
│   │   ├── explore_data.py
│   │   ├── prepare_data.py
│   │   └── train_model.py
│   ├── .gitignore
│   ├── requirements.txt             # Python dependencies
│   └── run.py                       # Application entry point
├── frontend/                        # React UI components
│   ├── public/                      # Static assets
│   ├── src/
│   │   ├── assets/                  # Default assets
│   │   ├── components/
│   │   │   ├── steps/               # Form wizard components
│   │   │   │   ├── Step1.jsx
│   │   │   │   ├── Step2.jsx
│   │   │   │   └── Step3.jsx
│   │   │   ├── ui/                  # shadcn/ui components
│   │   │   │   ├── accordion.jsx
│   │   │   │   ├── button.jsx
│   │   │   │   └── ...              # Other UI components
│   │   │   ├── ResultDisplay.jsx    # Prediction results view
│   │   │   └── SalaryPredictor.jsx  # Main form component
│   │   ├── lib/                     # Utilities and schemas
│   │   │   ├── data.js              # Form options data
│   │   │   ├── schema.js            # Validation schemas
│   │   │   └── utils.js             # Helper functions
│   │   ├── App.jsx                  # Root component
│   │   ├── App.css
│   │   └── main.jsx                 # Application entry
│   ├── .gitignore
│   ├── package.json                 # Frontend dependencies
│   └── vite.config.js               # Build configuration
├── .gitattributes                   # Git LFS configuration
└── README.md                        # Project documentation
```

````

## ⚙️ Installation & Setup

This project utilizes [Git Large File Storage (Git LFS)](https://git-lfs.com/) for managing large machine learning model and dataset files. Please ensure you have Git LFS installed before cloning the repository.

### Prerequisites

- **Git:** [Download & Install Git](https://git-scm.com/downloads)
- **Git LFS:** [Download & Install Git LFS](https://git-lfs.com/)
  - After installation, run in your terminal:
    ```bash
    git lfs install
    ```
- **Python:** Version 3.13.4 (or compatible 3.x) [Download Python](https://www.python.org/downloads/)
- **Node.js:** Version v22.17.1 (or compatible 20+) [Download Node.js](https://nodejs.org/en/download/) (includes npm)
  - `npm`: Version 10.9.2 (comes with Node.js)

### ⬇️ Cloning the Repository

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/harman-04/ml-salary-prediction.git](https://github.com/harman-04/ml-salary-prediction.git)
    cd ml-salary-prediction
    ```
    Git LFS will automatically download the large files (like `survey_results_public.csv` and `.joblib` models) during the cloning process.

### Backend Setup

1.  **Navigate to the backend directory:**

    ```bash
    cd backend
    ```

2.  **Create and activate a Python virtual environment:**
    It's highly recommended to use a virtual environment to manage dependencies for your Python projects.

            - **Create virtual environment:**
              ```bash
              python -m venv venv
              ```
            - **Activate virtual environment:**
              _ **On Windows (Command Prompt/PowerShell):**
              `bash

        .\venv\Scripts\activate
        `     _ **On macOS/Linux (Bash/Zsh):**

    `bash
    source venv/bin/activate
    ` (You'll see`(venv)` prepended to your terminal prompt, indicating the virtual environment is active.)

3.  **Install Python dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Flask backend application:**
    ```bash
    python run.py
    ```
    The backend server will start and typically run on `http://127.0.0.1:5000`. Keep this terminal window open.

### Frontend Setup

1.  **Open a new terminal window.**

2.  **Navigate to the frontend directory:**
    (Assuming you are in the project root `D:\salary-prediction-ai`)

    ```bash
    cd frontend
    ```

3.  **Install Node.js dependencies:**

    ```bash
    npm install
    ```

4.  **Run the React frontend application:**
    ```bash
    npm run dev
    ```
    The frontend development server will start and typically open in your browser at `http://localhost:5173`.

## 🚀 Usage

Once both the backend and frontend servers are running:

1.  Open your web browser and navigate to `http://localhost:5173`.
2.  Fill in the multi-step form with the required developer details (Age, Country, Org Size, Employment, Education Level, Dev Type, Remote Work, Job Satisfaction, Knowledge Frequency).
3.  Select relevant tech skills across various categories (Language, Database, Platform, Webframe, MiscTech, ToolsTech, NEWCollabTools, OfficeStackAsync, OfficeStackSync, AISearchDev, AITool).
4.  Submit the form to get an instant salary prediction.
5.  View the predicted annual salary along with dynamically generated "Feature Importance Chart" and "Salary Distribution Chart."

## 🧠 Model Details & Performance

The project uses the `stackoverflow-developer-survey-2024` dataset to predict `ConvertedCompYearly`. Various machine learning models were trained and evaluated:

- **Best Model:** LightGBM (R2 Score: 0.734)
- **Other Models Compared:**
  - Random Forest (R2 Score: 0.703)
  - XGBoost (R2 Score: 0.732)
  - CatBoost (R2 Score: 0.727)
  - HistGradientBoosting (R2 Score: 0.725)

The `model_artifacts` folder contains the saved preprocessor and trained models, including the best-performing LightGBM model and the full prediction pipeline. The `plots` folder contains visual comparisons of algorithm results and actual vs. prediction plots from the training phase.

## 🤝 Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.
````
