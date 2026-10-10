
import streamlit as st
from google import genai


def generate_financial_advice(
    allowance,
    spending,
    savings,
    financial_score,
    cashback,
    impulse_score
):
    prompt = f"""
You are an AI financial advisor for college students.
Use simple English and give practical budgeting advice.

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
3. Financial Health Category (Excellent, Good, Average, Poor)
4. Five personalized budget recommendations
5. One motivational financial tip.

Use only the supplied figures. Do not invent facts.
Treat the financial health score as a project-defined indicator,
not a validated financial assessment.
"""

    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        if response.text:
            return response.text

        return "The AI advisor returned an empty response. Please try again."

    except Exception as e:
        return (
            "The AI financial advisor is temporarily unavailable. "
            "Please check your Gemini API key and API access. "
            f"Error: {e}"
        )
