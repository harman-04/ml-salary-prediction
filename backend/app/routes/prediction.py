# app/routes/prediction.py
from flask import Blueprint, request, jsonify
from app.services.prediction_service import salary_prediction_service # Import the instantiated service

# Create a Blueprint for prediction-related routes
prediction_bp = Blueprint('prediction', __name__)

@prediction_bp.route('/predict', methods=['POST'])
def predict_salary():
    """
    API endpoint to predict employee salary using the trained model,
    and also return feature importances and salary distribution data.
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "No input data provided"}), 400

    # Basic validation for required fields
    required_fields = [
        'YearsCodePro', 'EdLevel', 'Country', 'DevType', 'OrgSize', 'Age',
        'RemoteWork', 'Employment', 'WorkExp', 'JobSat',
        # 'CompTotal' is often an output or not required for input, but include if it's a feature.
    ]
    
    for field in required_fields:
        if field not in data or data[field] is None or (isinstance(data[field], str) and not data[field].strip()):
            return jsonify({"error": f"Missing or empty required field: '{field}'"}), 400

    try:
        # Use the prediction service to get the salary, feature importances, and salary distribution
        predicted_salary, feature_importances_data, salary_distribution_data = salary_prediction_service.predict(data)

        return jsonify({
            "predicted_salary_usd": round(predicted_salary, 2), # Explicitly in USD, rounded for cleaner output
            "message": "Salary prediction successful!",
            "feature_importances": feature_importances_data, # Include feature importances
            "salary_distribution": salary_distribution_data  # Include salary distribution
        })
    except RuntimeError as e:
        # This error typically means the model failed to load or service initialization failed
        return jsonify({"error": f"Service Error: {str(e)}"}), 500
    except ValueError as e:
        # This error is from prediction logic itself (e.g., bad input format causing preprocessing issues)
        return jsonify({"error": f"Input Data Error: {str(e)}"}), 400
    except Exception as e:
        # Catch any other unexpected errors
        return jsonify({"error": f"An unexpected internal error occurred: {str(e)}"}), 500