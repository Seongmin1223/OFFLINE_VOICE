from pydantic import BaseModel

class STTResult(BaseModel):

    text: str