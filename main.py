import os
from flask import Flask, request, jsonify
import PyPDF2
from io import BytesIO
from analyse_medical_history import MedicalHistoryRequest, analyze_medical_history
from health_insights import OnboardingResponses, generate_daily_routine_report
import groq

app = Flask(__name__)
groq_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

@app.route("/")
def read_root():
    return jsonify({"Hello": "World"})

def extract_text_from_pdf(uploaded_file):
    pdf_bytes = uploaded_file.read()
    text = ""
    reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text() or ""
    return text

def extract_text_from_pd(uploaded_file):
    pdf_bytes = uploaded_file.read()
    text = ""
    reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text() or ""
    return text

@app.route("/upload_pdf/", methods=["POST"])
def upload_pdf():
    try:
        language = request.form.get("language")
        file_name = request.form.get("file_name")
        file = request.files.get("file")

        if not file or file.content_type != "application/pdf":
            return jsonify({"message": "Only PDF files are allowed."}), 400

        extracted_text = extract_text_from_pdf(file)
        if not extracted_text:
            return jsonify({"message": "Failed to extract text from PDF."}), 400

        formatted_text = parse_and_translate(extracted_text, language)

        return jsonify({
            "message": "File processed successfully.",
            "LLM_output": formatted_text,
            "language": language,
            "file_name": file_name
        }), 200
    except Exception as e:
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500

@app.route("/generate-daily-routine", methods=["POST"])
def generate_routine():
    try:
        onboarding_data = request.get_json()
        responses_str = ", ".join(
            [f"Question: {item['question']}, Response: {item['response']}" for item in onboarding_data["responses"]]
        )
        user_data_str = f"Email: {onboarding_data['email']}, {responses_str}"
        report = generate_daily_routine_report(user_data_str)
        return jsonify({"status": "success", "daily_routine_report": report})
    except Exception as e:
        return jsonify({"message": str(e)}), 500

@app.route("/analyze-medical-history", methods=["POST"])
def generate_medical_history_report():
    try:
        history_data = request.get_json()
        analysis_str = "; ".join(
            [f"Report: {item['LLM_output']}, Language: {item['language']}, Message: {item['message']}" for item in history_data["analysisList"]]
        )
        patient_data_str = f"Email: {history_data['email']}, {analysis_str}"
        report = analyze_medical_history(patient_data_str)
        return jsonify({"status": "success", "medical_history_report": report})
    except Exception as e:
        return jsonify({"message": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    try:
        request_data = request.get_json()
        message_history = format_message_history(request_data["messages"])
        if not message_history or message_history[0]["role"] != "system":
            message_history.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        response = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=message_history,
            temperature=0.7,
            max_tokens=1000,
            top_p=0.95,
            stream=False
        )
        assistant_response = response.choices[0].message.content
        return jsonify({"response": assistant_response})
    except Exception as e:
        return jsonify({"message": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
