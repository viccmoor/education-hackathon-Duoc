"""
    Módulo encargado de las interacciónes de API
    EndPoints disponibles:
        /predict:
        /coach:
"""

from typing import Iterable
from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI()
class StudentProfile(BaseModel):
    """Representation of an individual student profile"""

    age: int
    attendance: float
    sex: str
    subject: str
    establishment: str
    grades: Iterable[float]

app.post("/predict")
def predict_result(profile: StudentProfile):
    """
        Returns an estimated risk-score of how close the student is from dropping out
    """
    pass

app.post("/coach")
def determine_coach(profile: StudentProfile):
    """
        Generates a coaching plan for the student
    """
    pass
