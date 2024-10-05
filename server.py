from fastapi import FastAPI
import uvicorn
import analytics
import time
import asyncio

app = FastAPI()

RESP_THRESHOLD = 120  # the amount of seconds the response should take at minimum


@app.get("/analytics")
async def read_root(index: int):
    start = time.time()
    res = analytics.get_analytics(index)
    diff = time.time() - start
    if diff < RESP_THRESHOLD:
        await asyncio.sleep(RESP_THRESHOLD - diff)
    return res


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, log_level="info")
