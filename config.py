import os
basedir = os.path.abspath(os.path.dirname(__file__))

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'secret-key'
    BLOB_ACCOUNT = os.environ.get('BLOB_ACCOUNT') or 'iimage11'
    BLOB_STORAGE_KEY = os.environ.get('BLOB_STORAGE_KEY') or '3BNQV4/QczIN/6Si4vm7Uz+4oba0lQgDdYKoSFOvWYrvurVwnQy7u+Nk/bmQ5p/wf1ie6pVFu5nk+AStBi8WrA=='
    BLOB_CONTAINER = os.environ.get('BLOB_CONTAINER') or 'images'
    SQL_SERVER = os.environ.get('SQL_SERVER') or 'cmsserver12.database.windows.net'
    SQL_DATABASE = os.environ.get('SQL_DATABASE') or 'cmsdb'
    SQL_USER_NAME = os.environ.get('SQL_USER_NAME') or 'cmsadmin'
    SQL_PASSWORD = os.environ.get('SQL_PASSWORD') or 'CMS4dmin'
    SQLALCHEMY_DATABASE_URI = 'mssql+pyodbc://' + SQL_USER_NAME + '@' + SQL_SERVER + ':' + SQL_PASSWORD + '@' + SQL_SERVER + ':1433/' + SQL_DATABASE + '?driver=ODBC+Driver+17+for+SQL+Server'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    CLIENT_SECRET = "0MJ8Q~NiucyZwWHTMvXK80W6iZYxP_kdmet3ncU-"
    AUTHORITY = "https://login.microsoftonline.com/f958e84a-92b8-439f-a62d-4f45996b6d07"  # fixed: added missing '7'
    CLIENT_ID = "479be2b2-93ca-4201-bdd1-aff582c2deca"
    REDIRECT_PATH = "/getAToken"
    SCOPE = ["User.Read"]  # fixed: removed broken markdown link
    SESSION_TYPE = "filesystem"
