from fastapi import FastAPI
import analytics


app = FastAPI()

@app.get("/analytics")
def read_root(company: str, countryCode: str):
    return analytics.findthings(company, countryCode)
