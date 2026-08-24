import joblib 
import pandas as pd
from fastapi import FastAPI

model = joblib.load("model.pkl")
app = FastAPI()


@app.get("/")
def testapi():
    return {"msg":"your api is rady"}

@app.post("/prediction")
def predict(hours:float):

    new_data = pd.DataFrame({
        "StudyHours":[hours]
    })

    myprediction = model.predict(new_data)

    return {"Prediction":float(myprediction[0])}