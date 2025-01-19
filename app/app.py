from flask import Flask
from routes.main_routes import main_bp
from routes.api_routes import api_bp

# Crée une instance de l'application Flask
app = Flask(__name__)

# Enregistre le blueprint pour les routes principales
app.register_blueprint(main_bp)

# Enregistre le blueprint pour les routes API avec le préfixe "/api"
app.register_blueprint(api_bp, url_prefix="/api")

# Si le script est exécuté directement, lance l'application en mode debug
if __name__ == "__main__":
    app.run(debug=True)
