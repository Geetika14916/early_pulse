import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="Llama3-8b-8192",
    temperature=0,  
)


def parse_report_2(text_data, language):
    system_prompt_1 = """You are an experienced doctor analyzing medical reports. Format the data into a clear, structured summary with normal ranges and assessments.

Format each finding as:
Parameter: Value - Range (Interpretation if needed)

Example format:
Haemoglobin: 14.2 g/dL - Normal range (13.5-17.5 g/dL for men)
Calcium: 8.9 mg/dL - Normal range (8.5-10.5 mg/dL)
Blood Pressure: 135/90 mmHg - Stage 1 Hypertension

Provide only the formatted data and translated output string without any additional message, LLM_Output, text, headers, or metadata.
At last provide a briefer summary in very easy and short explanation
"""

    prompt1 = ChatPromptTemplate.from_messages([
        ("system", system_prompt_1),
        ("user", "{input}"),
    ])

    prompt2 = ChatPromptTemplate.from_template(
        """Translate the following medical report into {language}. 
        Maintain the exact same format and structure.
        Provide only the translated text without any additional commentary or metadata.
        
        Report to translate:
        {data}"""
    )

    format_chain = (
        prompt1
        | llm
        | StrOutputParser()
    )

    translation_chain = (
        prompt2
        | llm
        | StrOutputParser()
    )

    # Process the text
    try:
        # Get formatted text and clean it
        formatted_text = format_chain.invoke({"input": text_data})
        formatted_text = formatted_text.strip()

        # Get translated text and clean it
        translated_text = translation_chain.invoke({
            "data": formatted_text,
            "language": language
        })
        translated_text = translated_text.strip()

        # Remove any remaining artifacts
        translated_text = translated_text.replace('```', '')
        translated_text = translated_text.replace('Content:', '')
        translated_text = translated_text.replace('Output:', '')

        return translated_text

    except Exception as e:
        return f"Error processing report: {str(e)}"
