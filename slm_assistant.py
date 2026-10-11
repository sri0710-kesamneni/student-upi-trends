
import streamlit as st
import ollama
from google import genai


def generate_financial_advice(
    allowance,
    spending,
    savings,
    financial_score,
    cashback,
    impulse_score,
    provider=None
):
    """
    Supports:
    - Ollama Phi-3 Mini (local)
    - Google Gemini API (cloud)
    - Auto (try Ollama first, then Gemini)
    """

    prompt = f"""
You are an AI financial advisor for college students.
Use simple English and give practical, personalized advice.

Student Details:
- Monthly Allowance: ₹{allowance}
- Monthly UPI Spending: ₹{spending}
- Monthly Savings: ₹{savings}
- Financial Health Score: {financial_score}/100
- Cashback Influence: {cashback}
- Impulse Buying Score: {impulse_score}/10

Provide:
1. Spending Summary
2. Savings Analysis
3. Financial Health Category:
   Excellent, Good, Average, or Poor
4. Five personalized budget recommendations
5. One motivational financial tip

Use the supplied figures. Do not invent facts.
Treat the financial health score as a project-defined indicator,
not a professionally validated financial assessment.
"""

    # Display a provider selector if app.py doesn't pass one.
    if provider is None:
        provider = st.selectbox(
            "Choose AI Provider",
            ["Gemini API", "Ollama Phi-3 Mini", "Auto (Ollama → Gemini)"],
            key="financial_advisor_provider"
        )

    def ask_ollama():
        response = ollama.chat(
            model="phi3",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]

    def ask_gemini():
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing from Streamlit Secrets."
            )

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        if not response.text:
            raise ValueError("Gemini returned an empty response.")

        return response.text

    if provider == "Ollama Phi-3 Mini":
        try:
            return ask_ollama()
        except Exception as e:
            return (
                "Ollama is unavailable. Make sure it is running "
                "and Phi-3 Mini is installed on your computer.\n\n"
                f"Details: {e}"
            )

    if provider == "Gemini API":
        try:
            return ask_gemini()
        except Exception as e:
            return f"Gemini could not generate advice. Details: {e}"

    # Auto mode: try local Ollama, then Gemini.
    try:
        return ask_ollama()
    except Exception:
        try:
            return ask_gemini()
        except Exception as e:
            return (
                "Both AI providers are unavailable. Check that Ollama "
                "is running or configure GEMINI_API_KEY in Streamlit "
                f"Secrets.\n\nDetails: {e}"
            )
