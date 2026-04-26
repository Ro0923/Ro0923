"""
The flask application package.
"""

import logging
from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_session import Session

app = Flask(__name__)

# Fix HTTPS / proxy headers on Azure App Service
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

app.config.from_object(Config)

# Logging setup
app.logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)

app.logger.addHandler(stream_handler)

# Extensions
Session(app)
db = SQLAlchemy(app)

login = LoginManager(app)
login.login_view = 'login'

import FlaskWebProject.views
