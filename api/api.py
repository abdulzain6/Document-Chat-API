import threading
import uvicorn
import json
import langchain
import queue
import os

from ai import LocalRetrievalQAWithSQLDatabase
from langchain.cache import InMemoryCache
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from models import ChatRequest, ChatResponse, InjestRequest, InjestResponse



chat = LocalRetrievalQAWithSQLDatabase(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME"),
    table_names=os.getenv("DB_TABLE_NAMES").split(" ")
)



langchain.llm_cache = InMemoryCache()
app = FastAPI()

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_with_bot(chat_request: ChatRequest):
    prompt = chat_request.prompt
    chat_history = [tuple(hist) for hist in chat_request.chat_history[-3:]]
    sql = chat_request.sql
    mmr = chat_request.mmr

    if not prompt:
        return ChatResponse(error="Missing required parameter: prompt")

    resp = debug = error = ""
    try:
        resp = chat.chat(prompt, chat_history=chat_history, sql=sql, mmr=mmr)
    except Exception as e:
        error = str(e)

    return ChatResponse(ai_response=resp, debug_info=json.dumps(debug), error=error)

@app.post("/api/v1/chat_stream")
def chat_with_bot_stream(chat_request: ChatRequest):
    prompt = chat_request.prompt
    chat_history = [tuple(hist) for hist in chat_request.chat_history[-3:]]
    sql = chat_request.sql
    mmr = chat_request.mmr

    if not prompt:
        return ChatResponse(error="Missing required parameter: prompt")      

    data_queue = queue.Queue()
    
    def callback(data):
        data_queue.put(data)
        
    def data_generator():
        yield "[START]"
        while True:
            data = data_queue.get()
            if data is None:
                yield "[END]"
                break
            yield data
            
    def run_chat():
        try:
            chat.chat(prompt, chat_history=chat_history, sql=sql, mmr=mmr, stream=True, callback_func=callback)
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))

    threading.Thread(target=run_chat).start()
            
    return StreamingResponse(data_generator())

@app.post("/api/v1/injest", response_model=InjestResponse)
async def add_data(chat_request: InjestRequest):
    data = chat_request.data
    source = chat_request.source

    if not data:
        return ChatResponse(error="Missing required parameter: data")

    if not source:
        return ChatResponse(error="Missing required parameter: source")

    status = "success"
    try:
        chat.add_docs_to_vectorstore(data, source)
    except Exception as e:
        status = f"Error :{str(e)}"

    return InjestResponse(status=status)



if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
    )