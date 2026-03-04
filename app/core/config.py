import os
from dotenv import load_dotenv

load_dotenv()


import os

settings = {
    "APP_NAME": os.environ["APP_NAME"],
    "ENVIRONMENT": os.environ["ENVIRONMENT"],
    "DEBUG": os.environ["ENVIRONMENT"] == "development",

    "SECRET_KEY": os.environ["SECRET_KEY"],
    "ALGORITHM": os.environ["ALGORITHM"],
    "ACCESS_TOKEN_EXPIRE_MINUTES": int(
        os.environ["ACCESS_TOKEN_EXPIRE_MINUTES"]
    ),

    "DB_USER": os.environ["DB_USER"],
    "DB_PASSWORD": os.environ["DB_PASSWORD"],
    "DB_HOST": os.environ["DB_HOST"],
    "DB_PORT": os.environ["DB_PORT"],
    "DB_NAME": os.environ["DB_NAME"],

    "DATABASE_URL": (
        f"mysql+pymysql://{os.environ['DB_USER']}:"
        f"{os.environ['DB_PASSWORD']}@"
        f"{os.environ['DB_HOST']}:"
        f"{os.environ['DB_PORT']}/"
        f"{os.environ['DB_NAME']}"
    ),
    "AI_PROVIDER" : os.environ["AI_PROVIDER"],

    "OLLAMA_BASE_URL": os.environ["OLLAMA_BASE_URL"],
    "DEFAULT_OLLAMA_MODEL": os.environ["OLLAMA_MODEL"],
    "OLLAMA_TIMEOUT": int(os.environ["OLLAMA_TIMEOUT"]),

    "OLLAMA_GENERATE_URL": (
        f"{os.environ['OLLAMA_BASE_URL']}/api/generate"
    ),
    "CLOUD_API_BASE_URL" : os.environ["CLOUD_API_BASE_URL"],
    "CLOUD_API_KEY" : os.environ["CLOUD_API_KEY"],
    "CLOUD_MODEL" : os.environ["CLOUD_MODEL"],
    "CLOUD_TIMEOUT" : os.environ["CLOUD_TIMEOUT"],

    "ALLOWED_ORIGINS": os.environ["ALLOWED_ORIGINS"].split(","),
    "LOG_LEVEL": os.environ["LOG_LEVEL"],
    
}