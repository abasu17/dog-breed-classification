from fastapi import FastAPI
from pydantic import BaseModel
import json, requests
from src.service import *


class Form(BaseModel):
    image: str

app = FastAPI() 


@app.get("/")
async def index():
    return "Server is running..."


@app.post("/prediction")
async def extract(form: Form):

    req_data = dict(form)
    pred = Prediction()
    pred.save_image(req_data["image"])
    resp = pred.predict_image()
    print(resp)
    return json.dumps(resp)