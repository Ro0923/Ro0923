import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'secret-key'

    # Blob Storage
    BLOB_ACCOUNT = os.environ.get('BLOB_ACCOUNT') or 'image11'
    BLOB_STORAGE_KEY = os.environ.get('BLOB_STORAGE_KEY') or 'qw4epkWNlAcVWj5/WsgYV4WmAGygBSNROZcqsPBEpGmCW6DXEbSZ3BrEmTd3g6wsjxtcPA3C4qrE+AStQrjHSA=='
    BLOB_CONTAINER = os.environ.get('BLOB_CONTAINER') or 'images'
    BLOB_CONNECTION_STRING = os.environ.get('BLOB_CONNECTION_STRING') or 'DefaultEndpointsProtocol=https;AccountName=image11;AccountKey=qw4epkWNlAcVWj5/WsgYV4WmAGygBSNROZcqsPBEpGmCW6DXEbSZ3BrEmTd3g6wsjxtcPA3C4qrE+AStQrjHSA==;EndpointSuffix=core.windows.net'

    # SQL
    SQL_SERVER = os.environ.get('SQL_SERVER') or 'cms12.database.windows.net'
    SQL_DATABASE = os.environ.get('SQL_DATABASE') or 'cms'
    SQL_USER_NAME = os.environ.get('SQL_USER_NAME') or 'cmsadmin'
    SQL_PASSWORD = os.environ.get('SQL_PASSWORD') or 'CMS4dmin'

    SQLALCHEMY_DATABASE_URI = (
        'mssql+pyodbc://'
        + SQL_USER_NAME + '@' + SQL_SERVER + ':'
        + SQL_PASSWORD + '@' + SQL_SERVER
        + ':1433/' + SQL_DATABASE
        + '?driver=ODBC+Driver+17+for+SQL+Server'
    )

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Microsoft Login
    CLIENT_SECRET = os.environ.get('CLIENT_SECRET') or 'Byy8Q~pnkh5gXBozS9Ig5prAvazkPh~8jiFihbAR'
    AUTHORITY = "https://login.microsoftonline.com/common"
    CLIENT_ID = os.environ.get('CLIENT_ID') or 'b23b6190-01fd-4e74-92f6-5f1cb2e37fb4'
    REDIRECT_PATH = "/getAToken"
    SCOPE = ["User.Read"]
    SESSION_TYPE = "filesystem"