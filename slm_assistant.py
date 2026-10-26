import os
from groq import Groq

def generate_financial_advice(
    allowance,
    spending,
    savings,
    financial_score,
    cashback,
    impulse_score
):

from google import genai

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)
    prompt = f"""
You are an AI financial advisor for college students.
Analyze the following student details using simple English.

Monthly Allowance: ₹{allowance}
Monthly UPI Spending: ₹{spending}
Monthly Savings: ₹{savings}
Financial Health Score: {financial_score}/100
Cashback Influence: {cashback}
Impulse Buying Score: {impulse_score}/10

Provide:
1. Spending Summary
2. Savings Analysis
3. Financial Health Category (Excellent, Good, Average, Poor)
4. Five personalized budget recommendations
5. One motivational financial tip.

Base your advice on the supplied values. Do not invent additional facts.
"""

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )

    return response.choices[0].message.content
