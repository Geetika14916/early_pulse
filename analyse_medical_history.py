from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model="Llama3-8b-8192")
app = FastAPI()


class AnalysisItem(BaseModel):
    LLM_output: str
    language: str
    message: str


class MedicalHistoryRequest(BaseModel):
    email: str
    analysisList: List[AnalysisItem]


def analyze_medical_history(analysis_data):
    system_prompt = (
        "You are a medical expert analyzing a patient's medical history over time. Based on the patient's previous "
        "medical reports, identify trends, improvements, or deteriorations in health conditions. "
        "For each relevant medical marker, please provide the previous and current values in the following format: "
        "Cholesterol:\n\n"
        "Previous Value: x\n"
        "Current Value: y\n"
        "Blood Pressure:\n\n"
        "Previous Value: a\n"
        "Current Value: b\n"
        "Blood Count:\n\n"
        "Previous Value: c\n"
        "Current Value: d\n"
        "Glucose:\n\n"
        "Previous Value: e\n"
        "Current Value: f\n"
        "Other Marker:\n\n"
        "Previous Value: g\n"
        "Current Value: h\n"
        "Ensure that all relevant markers are included and the output is formatted as shown above "
        "If any marker is not available, indicate it as null. "
        "do not bold any word"
        "make sure every thing will be in english, if there is anything which is not understandable feel free to leave that word or sentences"
        "Make sure the report is detailed yet easy to understand. "
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", f"Patient's Medical Reports: {analysis_data}")
        ]
    )

    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    analysis_report = chain.invoke(input={"data": analysis_data})
    analysis_report = analysis_report.strip()
    return analysis_report
