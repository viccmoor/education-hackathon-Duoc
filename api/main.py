from typing import Iterable
from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI()
class StudentProfile(BaseModel):
    edad: int
    asistencia: float
    sexo: str
    asignatura: str
    establecimiento: str
    calificaciones: Iterable[float]

app.post("/predict")
def predict_result(profile: StudentProfile):
    pass

app.post("/coach")
def determine_coach(profile: StudentProfile):
    pass