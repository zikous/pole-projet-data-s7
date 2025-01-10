from flask import Flask
from routes.main_routes import main_bp
from routes.api_routes import api_bp

# Create a Flask application instance
app = Flask(__name__)

# Register the main blueprint
app.register_blueprint(main_bp)

# Register the API blueprint with a URL prefix of /api
app.register_blueprint(api_bp, url_prefix="/api")

# Run the application in debug mode if this script is executed directly
if __name__ == "__main__":
    app.run(debug=True)
