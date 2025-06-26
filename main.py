# main.py - FastAPI приложение для предсказания риска сердечных приступов
import argparse
import logging

from fastapi import FastAPI, UploadFile
import uvicorn

from model import ModelObject

app = FastAPI()
model = ModelObject()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.get("/health")
def health():
    """Проверка состояния приложения"""
    if model.model is None:
        msg = "Модель не загружена!"
    else:
        msg = "Модель загружена"
    return {"status": "OK", "message": msg}


@app.get("/")
def main_function():
    return {
            "app": "Heart Attack Risk Prediction System (API)",
            "endpoints": {
                "GET /health": "состояние приложения",
                "POST /process": "отправка файла с данными для предсказания"
            }
        }


@app.post("/process")
def process_file(file: UploadFile):
    save_pth = "tmp/" + file.filename
    with open(save_pth, "wb") as fd:
        fd.write(file.file.read())

    return model.prediction(save_pth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8000, type=int, dest="port")
    parser.add_argument("--host", default="0.0.0.0", type=str, dest="host")
    args = vars(parser.parse_args())

    uvicorn.run(app, **args)