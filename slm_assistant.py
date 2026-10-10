# ============================================
# SLM Assistant for Student UPI EDA Project
# Model Used : Phi-3 Mini (Ollama)
# ============================================

import ollama

# Function to generate AI financial advice
def generate_financial_advice(
    allowance,
    spending,
    savings,
    financial_score,
    cashback,
    impulse_score
):

    # Prompt sent to the Small Language Model
    prompt = f"""
You are an AI Financial Advisor for college students.

Analyze the following student's financial details and respond in simple English.

Student Details:
- Monthly Allowance: ₹{allowance}
- Monthly UPI Spending: ₹{spending}
- Monthly Savings: ₹{savings}
- Financial Health Score: {financial_score}/100
- Cashback Influence: {cashback}
- Impulse Buying Score: {impulse_score}/10

Give the output in this format:

1. Spending Summary
2. Savings Analysis
3. Financial Health Category (Excellent, Good, Average, Poor)
4. Five Personalized Budget Recommendations
5. One motivational financial tip.
"""

    # Send prompt to Phi-3 model
    response = ollama.chat(
        model="phi3",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # Return only AI-generated text
    return response["message"]["content"]


# -------------------------------
# Test the assistant
# -------------------------------
if __name__ == "__main__":
    print("Program Started...")

    advice = generate_financial_advice(
        allowance=10000,
        spending=8200,
        savings=900,
        financial_score=38,
        cashback="High",
        impulse_score=8
    )

    print("\n===== AI Financial Report =====\n")
    print(advice)