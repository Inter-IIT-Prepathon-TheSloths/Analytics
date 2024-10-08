from fastapi import FastAPI
import uvicorn
import app.analytics as analytics
import time
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

app = FastAPI()

RESP_THRESHOLD = 120  # the amount of seconds the response should take at minimum

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=100)

@app.get("/")
async def health():
    return {"status": "OK"}

@app.get("/analytics")
def read_root(index: int):
    start = time.time()
    res = analytics.get_analytics(index)
    diff = time.time() - start
    if diff < RESP_THRESHOLD:
        time.sleep(RESP_THRESHOLD - diff)
    return res

@app.get("/companies")
def companies():
    return analytics.get_companies()


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, log_level="info")
