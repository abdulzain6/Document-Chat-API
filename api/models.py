from pydantic import BaseModel
from typing import List, Tuple

class ChatRequest(BaseModel):
    prompt: str
    chat_history: List[Tuple[str, str]]
    sql: bool
    mmr: bool
    
class ChatResponse(BaseModel):
    ai_response: str
    error: str


class InjestRequest(BaseModel):
    data: str
    source: str
    
class InjestResponse(BaseModel):
    status: str