from pydantic import BaseModel
from typing import List, Optional


class VerificationRequest(BaseModel):
    word: str
    claim: str
    search_word: Optional[str] = None


class VerificationResponse(BaseModel):
    word: str
    claim: str
    predicted: int
    in_wiki: str
    selected_evidences: Optional[List[dict]] = None


class Example(BaseModel):
    word: str
    definition: str


class Dataset(BaseModel):
    id: int
    name: str
    lang: str
    examples: Optional[List[Example]] = None
