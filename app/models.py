from pydantic import BaseModel
from typing import List

# API models
class Query(BaseModel):
    query: str

class QueryResponse(BaseModel):
    message: str

# Consultant Profile Model
class ConsultantProfile(BaseModel):
    name: str = ''
    title: str = ''
    preamble: str = ''
    article: str = ''
    competence_list: List[str] = []
    cv_list: List[str] = []
    employment_list: List[str] = []
    education_list: List[str] = []
