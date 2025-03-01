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

class ResponseItem(BaseModel):
    index: int
    question: str
    response: str

class OnboardingResponses(BaseModel):
    email: str
    responses: List[ResponseItem]

# Function to generate daily routine report
def generate_daily_routine_report(onboarding_data):
    system_prompt = (
        "You are a lifestyle and wellness expert. Based on the user's answers to the onboarding questions, "
        "create a comprehensive daily routine report. Consider the user's responses, habits, lifestyle choices, "
        "and preferences. Offer advice on exercise, diet, sleep, and other aspects of daily life to improve their overall well-being. "
        "Ensure the report is personalized, practical, and easy to follow."
        "do not use any brackets or do not bold any word"
        "use name = Dr. Early Pulse"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", f"User Data: {onboarding_data}")
        ]
    )

    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    routine_report = chain.invoke(input={"data": onboarding_data})
    routine_report = routine_report.strip() 
    
    return routine_report

@app.post("/generate-daily-routine")
async def generate_routine(onboarding_data: OnboardingResponses):
    try:
        responses_str = ", ".join(
            [f"Question: {item.question}, Response: {item.response}" for item in onboarding_data.responses]
        )
        user_data_str = f"Email: {onboarding_data.email}, {responses_str}"
        
        report = generate_daily_routine_report(user_data_str)
        
        return {
            "status": "success",
            "daily_routine_report": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
