"""
Entry point for Azure App Service / Gunicorn
"""

from FlaskWebProject import app

if __name__ == "__main__":
    app.run()
