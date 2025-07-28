class Config:
        """
        Base configuration for the Flask application.
        """
        SECRET_KEY = 'your_secret_key_here' # IMPORTANT: Change this in production!
        DEBUG = True # Set to False in production
        TESTING = False

        # CORS settings (will be configured more specifically later)
        CORS_HEADERS = 'Content-Type'

class DevelopmentConfig(Config):
        """
        Development specific configurations.
        """
        DEBUG = True

class ProductionConfig(Config):
        """
        Production specific configurations.
        """
        DEBUG = False
        TESTING = False
        # Add production-specific settings like database URLs, logging, etc.

    