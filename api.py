from fastapi import FastAPI
from pydantic import BaseModel
from generate import generate_sql

app = FastAPI()

class QueryRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"message": "Welcome to QueryCraft LLM API. Use POST /generate-sql to query."}

@app.post("/generate-sql")
def generate(request: QueryRequest):
    sql = generate_sql(request.prompt)
    return {"sql": sql}
