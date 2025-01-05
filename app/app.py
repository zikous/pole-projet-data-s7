# app.py
from flask import Flask
from routes.main_routes import main_bp
from routes.api_routes import api_bp

app = Flask(__name__)

# Register blueprints
app.register_blueprint(main_bp)
app.register_blueprint(api_bp, url_prefix="/api")

if __name__ == "__main__":
    app.run(debug=True)
