from flask import Flask

def create_app():
        """
        Factory function to create and configure the Flask application.
        This allows for different configurations (e.g., testing, production).
        """
        app = Flask(__name__)

        # Load configuration from config.py
        app.config.from_object('app.config.Config')

        # Import and register blueprints
        from app.routes import prediction_bp
        app.register_blueprint(prediction_bp, url_prefix='/api') # All prediction routes will start with /api

        # You can add more blueprints, database initialization, etc., here later

        return app

    