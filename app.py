import io
import json
import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import ollama
except ImportError:
    ollama = None

try:
    from google import genai
except ImportError:
    genai = None


# ------------------------------------------------------------
# UPI INSIGHT — Student Financial Intelligence
# Streamlit + Pandas + Plotly + Ollama Phi-3 Mini + Gemini API
# ------------------------------------------------------------
st.set_page_config(
    page_title="UPI Insight | Student Financial Intelligence",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_NAME = "phi3"
GEMINI_MODEL = "gemini-flash-latest"
CANDIDATE_FILES = [
    "UPI_Student_Spending_Dataset_1150_01.csv",
    "UPI_Student_Spending_Dataset_Realistic_EDA.csv",
    "Student_UPI_EDA_with_AI.csv",
]

# ---------- UI theme ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root {
      --bg: #07110f; --panel: rgba(16, 34, 30, .78);
      --panel2: rgba(18, 43, 37, .68); --line: rgba(111, 230, 183, .15);
      --emerald: #34d399; --mint: #a7f3d0; --muted: #9ab5ab; --text: #edfdf6;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp {
      background:
        radial-gradient(ellipse at 8% 0%, rgba(16,185,129,.13), transparent 34%),
        radial-gradient(ellipse at 94% 10%, rgba(20,184,166,.10), transparent 28%),
        linear-gradient(135deg, #07110f 0%, #091714 48%, #06100e 100%);
      color: var(--text);
    }
    [data-testid="stHeader"] { background: rgba(7,17,15,.70); }
    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, rgba(10,27,23,.98), rgba(7,17,15,.98));
      border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] * { color: var(--text); }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.025em; }
    p, label, .stMarkdown { color: var(--text); }
    .hero {
      border: 1px solid rgba(110,231,183,.22);
      background: linear-gradient(125deg, rgba(16,54,43,.90), rgba(13,31,28,.70));
      border-radius: 24px; padding: 28px 30px; margin: 4px 0 24px 0;
      box-shadow: 0 18px 60px rgba(0,0,0,.18), inset 0 1px 0 rgba(255,255,255,.04);
    }
    .eyebrow { color: #6ee7b7; text-transform: uppercase; letter-spacing: .16em; font-size: .72rem; font-weight: 700; }
    .hero-title { color: #f0fdf4; font-size: 2.35rem; line-height: 1.12; font-weight: 700; margin: 8px 0; }
    .hero-copy { color: #b7d5c9; font-size: 1rem; max-width: 780px; }
    .pill { display: inline-block; border: 1px solid rgba(52,211,153,.32); background: rgba(16,185,129,.10); color: #a7f3d0; border-radius: 999px; padding: 5px 10px; font-size: .76rem; margin-right: 6px; }
    div[data-testid="stMetric"] {
      background: linear-gradient(145deg, rgba(19,48,40,.84), rgba(11,29,25,.86));
      border: 1px solid rgba(110,231,183,.16); padding: 17px 18px;
      border-radius: 18px; box-shadow: 0 10px 28px rgba(0,0,0,.12);
    }
    div[data-testid="stMetricLabel"] { color: #9ab5ab !important; }
    div[data-testid="stMetricValue"] { color: #eafff5 !important; font-family: 'Space Grotesk', sans-serif; }
    div[data-testid="stMetricDelta"] { color: #6ee7b7 !important; }
    div[data-testid="stPlotlyChart"] {
      background: rgba(13,31,27,.56); border: 1px solid var(--line);
      border-radius: 18px; padding: 8px;
    }
    div[data-testid="stDataFrame"], div[data-testid="stTable"] {
      border: 1px solid var(--line); border-radius: 14px; overflow: hidden;
    }
    .glass-card {
      background: linear-gradient(145deg, rgba(17,43,36,.78), rgba(10,26,23,.78));
      border: 1px solid rgba(110,231,183,.17); border-radius: 18px;
      padding: 20px; min-height: 120px;
    }
    .muted { color: #9ab5ab; }
    .status-good { color: #6ee7b7; font-weight: 600; }
    .status-warn { color: #fbbf24; font-weight: 600; }
    .stButton > button, .stDownloadButton > button {
      border: 1px solid rgba(52,211,153,.35); border-radius: 12px;
      background: linear-gradient(135deg, #10b981, #059669); color: #03140e;
      font-weight: 700; min-height: 42px;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
      border-color: #6ee7b7; background: linear-gradient(135deg, #34d399, #10b981); color: #03140e;
    }
    .stTextInput input, .stTextArea textarea, [data-baseweb="select"] > div {
      background: rgba(7,20,17,.86) !important; color: #eafff5 !important;
      border-color: rgba(110,231,183,.20) !important; border-radius: 10px !important;
    }
    [data-testid="stChatMessage"] { background: rgba(16,42,35,.68); border: 1px solid var(--line); border-radius: 16px; }
    hr { border-color: rgba(110,231,183,.14); }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def money(value):
    if pd.isna(value):
        return "—"
    return "₹" + f"{float(value):,.0f}"


def safe_num(series):
    return pd.to_numeric(series, errors="coerce")


def find_column(columns, candidates):
    lookup = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    # fallback: normalized name matching
    normalized = {
        "".join(ch for ch in str(c).lower() if ch.isalnum()): c
        for c in columns
    }
    for candidate in candidates:
        key = "".join(ch for ch in candidate.lower() if ch.isalnum())
        if key in normalized:
            return normalized[key]
    return None


def load_dataset():
    """Prefer the original/raw CSV. Never treat standardized notebook columns as rupees."""
    uploaded = st.session_state.get("dataset_upload")
    if uploaded is not None:
        try:
            return pd.read_csv(uploaded), uploaded.name, None
        except Exception as exc:
            return None, uploaded.name, f"Could not read uploaded CSV: {exc}"

    for filename in CANDIDATE_FILES:
        if os.path.exists(filename):
            try:
                frame = pd.read_csv(filename)
                # A previously exported app dataset is usable, but raw columns are preferred.
                return frame, filename, None
            except Exception as exc:
                return None, filename, f"Could not read {filename}: {exc}"
    return None, None, "No CSV found in the app folder. Upload your original dataset in the sidebar."


def prepare_data(raw):
    """Map source columns and derive display metrics from the actual loaded data."""
    data = raw.copy()
    cols = data.columns.tolist()

    mapping = {
        "student_id": find_column(cols, ["Student_ID", "Student ID", "student_id"]),
        "allowance": find_column(cols, ["Monthly_Allowance", "Allowance_INR", "Allowance"]),
        "spending": find_column(cols, ["Monthly_UPI_Spending", "Spending_INR", "UPI_Spending"]),
        "savings": find_column(cols, ["Savings", "Savings_INR", "Monthly_Savings"]),
        "app": find_column(cols, ["Preferred_UPI_App", "UPI_App", "Preferred App"]),
        "gender": find_column(cols, ["Gender"]),
        "year": find_column(cols, ["Year_of_Study", "Year of Study"]),
        "department": find_column(cols, ["Department"]),
        "residence": find_column(cols, ["Hosteller_or_DayScholar", "Residence"]),
        "budget": find_column(cols, ["Tracks_Budget", "Tracks Budget"]),
        "transactions": find_column(cols, ["Daily_UPI_Transactions", "Daily Transactions"]),
        "cashback": find_column(cols, ["Cashback_Influence (1-5)", "Cashback_Influence", "Cashback Influence"]),
        "impulse": find_column(cols, ["Impulse_Buying_Score (1-5)", "Impulse_Buying_Score", "Impulse Buying Score"]),
        "age": find_column(cols, ["Age"]),
        "part_time": find_column(cols, ["Part_Time_Job", "Part Time Job"]),
    }

    if mapping["allowance"] is None or mapping["spending"] is None:
        raise ValueError(
            "The CSV must contain allowance and UPI spending columns. "
            "Upload the original CSV with Monthly_Allowance and Monthly_UPI_Spending."
        )

    # Rename selected fields to stable internal names while retaining original fields.
    internal = {
        "allowance": "Monthly_Allowance",
        "spending": "Monthly_UPI_Spending",
        "savings": "Savings",
        "app": "Preferred_UPI_App",
        "gender": "Gender",
        "year": "Year_of_Study",
        "department": "Department",
        "residence": "Hosteller_or_DayScholar",
        "budget": "Tracks_Budget",
        "transactions": "Daily_UPI_Transactions",
        "cashback": "Cashback_Influence",
        "impulse": "Impulse_Buying_Score",
        "age": "Age",
        "part_time": "Part_Time_Job",
    }
    for key, target in internal.items():
        source = mapping.get(key)
        if source is not None and source != target:
            if target not in data.columns:
                data[target] = data[source]

    if mapping["student_id"] is None:
        data["Student_ID"] = [f"Student {i+1:04d}" for i in range(len(data))]
        mapping["student_id"] = "Student_ID"
    elif mapping["student_id"] != "Student_ID" and "Student_ID" not in data.columns:
        data["Student_ID"] = data[mapping["student_id"]]
        mapping["student_id"] = "Student_ID"

    # Convert numerical fields safely. These are expected to be raw monetary values.
    for col in ["Monthly_Allowance", "Monthly_UPI_Spending", "Savings",
                "Daily_UPI_Transactions", "Cashback_Influence",
                "Impulse_Buying_Score", "Age"]:
        if col in data.columns:
            data[col] = safe_num(data[col])

    if "Savings" not in data.columns:
        data["Savings"] = pd.NA

    # If the app is accidentally given standardized columns, stop rather than display z-scores as rupees.
    allowance = data["Monthly_Allowance"].dropna()
    spending = data["Monthly_UPI_Spending"].dropna()
    if (not allowance.empty and allowance.abs().quantile(.95) < 20) or (
        not spending.empty and spending.min() < 0 and spending.abs().quantile(.95) < 20
    ):
        raise ValueError(
            "These values look standardized rather than raw rupee amounts. "
            "Upload the original, unscaled CSV—not the notebook's transformed df export."
        )

    # Filter rows with no essential numeric values; do not silently impute financial data.
    data = data.dropna(subset=["Monthly_Allowance", "Monthly_UPI_Spending"]).copy()
    data = data[data["Monthly_Allowance"] > 0].copy()
    data = data[data["Monthly_UPI_Spending"] >= 0].copy()

    # Derive metrics only from actual source values.
    data["Spending_Percentage"] = (
        data["Monthly_UPI_Spending"] / data["Monthly_Allowance"] * 100
    )
    data["Remaining_Balance"] = data["Monthly_Allowance"] - data["Monthly_UPI_Spending"]
    data["Daily_Average_UPI_Spending"] = data["Monthly_UPI_Spending"] / 30
    data["Savings_Percentage"] = (
        data["Savings"] / data["Monthly_Allowance"] * 100
        if data["Savings"].notna().any() else pd.NA
    )
    data["Financial_Health_Score"] = (
        data["Savings_Percentage"] + (100 - data["Spending_Percentage"])
    ) / 2
    data["Financial_Health_Score"] = pd.to_numeric(
        data["Financial_Health_Score"], errors="coerce"
    ).clip(0, 100)

    def category(score):
        if pd.isna(score):
            return "Insufficient data"
        if score >= 80:
            return "Excellent"
        if score >= 60:
            return "Good"
        if score >= 40:
            return "Average"
        return "Needs improvement"

    data["Financial_Category"] = data["Financial_Health_Score"].apply(category)
    data["Student_ID"] = data["Student_ID"].astype(str)
    return data, mapping


def apply_plot_theme(fig, title=None):
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,22,19,.35)",
        font=dict(family="DM Sans, sans-serif", color="#d8f5e8"),
        margin=dict(l=18, r=18, t=58 if title else 24, b=18),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(148,190,171,.10)", zerolinecolor="rgba(148,190,171,.12)"),
        yaxis=dict(gridcolor="rgba(148,190,171,.10)", zerolinecolor="rgba(148,190,171,.12)"),
    )
    return fig


def get_gemini_api_key():
    """Read the key from Streamlit Secrets or an environment variable."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        return ""


def ask_ai(messages, system_prompt=None, provider="Gemini API"):
    """Generate an answer using Gemini, local Ollama, or automatic selection."""
    prompt_parts = []
    if system_prompt:
        prompt_parts.append("SYSTEM INSTRUCTIONS:\n" + system_prompt)
    for message in messages:
        role = message.get("role", "user").upper()
        prompt_parts.append(f"{role}:\n{message.get('content', '')}")
    full_prompt = "\n\n".join(prompt_parts)

    def use_ollama():
        if ollama is None:
            raise RuntimeError("The 'ollama' package is missing. Add ollama to requirements.txt.")
        response = ollama.chat(
            model=MODEL_NAME,
            messages=([{"role": "system", "content": system_prompt}] if system_prompt else []) + messages,
            options={"temperature": 0.2},
        )
        return response["message"]["content"]

    def use_gemini():
        if genai is None:
            raise RuntimeError("The google-genai package is missing. Add google-genai to requirements.txt.")
        api_key = get_gemini_api_key()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured in Streamlit Cloud → Settings → Secrets.")
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=GEMINI_MODEL, contents=full_prompt)
        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")
        return response.text

    if provider == "Ollama Phi-3 Mini":
        return use_ollama()
    if provider == "Gemini API":
        return use_gemini()

    # Auto: prefer Gemini when a cloud key is configured; otherwise try local Ollama.
    if get_gemini_api_key():
        try:
            return use_gemini()
        except Exception:
            return use_ollama()
    try:
        return use_ollama()
    except Exception:
        return use_gemini()


def make_report(data, source_name):
    total = len(data)
    avg_allowance = data["Monthly_Allowance"].mean()
    avg_spending = data["Monthly_UPI_Spending"].mean()
    avg_savings = data["Savings"].mean() if data["Savings"].notna().any() else None
    corr = data["Monthly_Allowance"].corr(data["Monthly_UPI_Spending"])
    most_used = (
        data["Preferred_UPI_App"].dropna().astype(str).mode().iloc[0]
        if "Preferred_UPI_App" in data and not data["Preferred_UPI_App"].dropna().empty
        else "Unavailable"
    )
    lines = [
        "# UPI Insight — Student Financial Intelligence",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Source file: {source_name}",
        "",
        "## Dataset overview",
        f"- Valid rows analysed: {total}",
        f"- Average monthly allowance: {money(avg_allowance)}",
        f"- Average monthly UPI spending: {money(avg_spending)}",
        f"- Average savings (where provided): {money(avg_savings) if avg_savings is not None else 'Unavailable'}",
        f"- Most common UPI app (where available): {most_used}",
        f"- Allowance/spending correlation: {corr:.3f}" if pd.notna(corr) else "- Allowance/spending correlation: unavailable",
        "",
        "## Finance category counts",
    ]
    counts = data["Financial_Category"].value_counts(dropna=False)
    lines.extend([f"- {key}: {int(value)}" for key, value in counts.items()])
    lines += [
        "",
        "## Interpretation notes",
        "- Metrics are calculated from the loaded dataset; this report does not invent missing values.",
        "- Correlation indicates association, not causation.",
        "- Financial health score is a simple project-specific indicator, not a validated financial assessment.",
        "- Phi-3 Mini outputs are generated text and should be reviewed before relying on them.",
    ]
    return "\n".join(lines)


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown(
        """
        <div style="padding:10px 2px 18px 2px">
          <div style="display:flex;align-items:center;gap:10px">
            <div style="background:linear-gradient(135deg,#34d399,#059669);border-radius:12px;width:42px;height:42px;display:flex;align-items:center;justify-content:center;font-size:22px">↗</div>
            <div>
              <div style="font-family:Space Grotesk;font-size:1.15rem;font-weight:700;color:#eafff5">UPI Insight</div>
              <div style="font-size:.75rem;color:#83a99a">STUDENT FINANCIAL AI</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "EDA Analytics", "Student Profile", "AI Financial Advisor", "Download Reports"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.caption("AI PROVIDERS")
    st.markdown(f"**Local model:** `{MODEL_NAME}` · **Cloud model:** Gemini")
    st.caption("Ollama runs on the computer hosting the app. For the deployed website, choose Gemini and configure GEMINI_API_KEY in Streamlit Secrets.")


# ---------- Load dataset ----------
with st.sidebar:
    st.markdown("### Dataset")
    st.file_uploader(
        "Upload your original student CSV",
        type=["csv"],
        key="dataset_upload",
        help="Prefer the original unscaled CSV. Example: UPI_Student_Spending_Dataset_1150_01.csv",
    )

raw_df, source_name, load_error = load_dataset()

if raw_df is None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">SETUP REQUIRED</div>
          <div class="hero-title">Connect your student dataset</div>
          <div class="hero-copy">Upload your original CSV using the sidebar. Dashboard metrics and charts will be calculated from that file.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning(load_error or "Dataset not available.")
    st.info(
        "Place your original CSV beside app.py, or upload it in the sidebar. "
        "The notebook file alone does not contain a reliable standalone copy of the raw CSV."
    )
    st.stop()

try:
    df, column_mapping = prepare_data(raw_df)
except Exception as exc:
    st.error(f"Dataset validation failed: {exc}")
    st.stop()

if df.empty:
    st.error("No valid rows remain after checking allowance and spending values.")
    st.stop()

# Filters shared across pages
with st.sidebar:
    st.markdown("### Filters")
    filtered = df.copy()
    filter_cols = [
        ("Year_of_Study", "Year of study"),
        ("Department", "Department"),
        ("Gender", "Gender"),
        ("Preferred_UPI_App", "UPI app"),
    ]
    for col, label in filter_cols:
        if col in filtered.columns:
            values = sorted(filtered[col].dropna().astype(str).unique().tolist())
            if values:
                selected = st.multiselect(label, values, default=values, key=f"filter_{col}")
                filtered = filtered[filtered[col].astype(str).isin(selected)]
    if st.button("Clear filters", use_container_width=True):
        for col, _ in filter_cols:
            st.session_state[f"filter_{col}"] = []
        st.rerun()

if filtered.empty:
    st.warning("No rows match the selected filters. Clear or broaden the filters.")
    st.stop()

# App header
st.markdown(
    f"""
    <div class="hero">
      <div class="eyebrow">STUDENT FINANCIAL INTELLIGENCE · EDA + SLM</div>
      <div class="hero-title">See the patterns.<br><span style="color:#6ee7b7">Make smarter money moves.</span></div>
      <div class="hero-copy">Explore real student UPI behaviour, understand spending and savings patterns, and ask Gemini or local Phi-3 Mini for data-grounded financial guidance.</div>
      <div style="margin-top:17px">
        <span class="pill">● Dataset connected</span>
        <span class="pill">Pandas analytics</span>
        <span class="pill">Plotly visuals</span>
        <span class="pill">Gemini API + Ollama</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(f"Source: `{source_name}` · {len(filtered):,} rows in current view · {len(df):,} valid rows loaded")

# ---------- Dashboard ----------
if page == "Dashboard":
    st.subheader("Financial overview")
    avg_allowance = filtered["Monthly_Allowance"].mean()
    avg_spending = filtered["Monthly_UPI_Spending"].mean()
    avg_savings = filtered["Savings"].mean() if filtered["Savings"].notna().any() else None
    avg_score = filtered["Financial_Health_Score"].mean()
    spend_rate = (filtered["Monthly_UPI_Spending"].sum() / filtered["Monthly_Allowance"].sum() * 100) if filtered["Monthly_Allowance"].sum() else float("nan")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students analysed", f"{len(filtered):,}")
    c2.metric("Avg. monthly allowance", money(avg_allowance))
    c3.metric("Avg. monthly UPI spend", money(avg_spending))
    c4.metric("Avg. savings", money(avg_savings) if avg_savings is not None else "Unavailable")
    c1, c2, c3 = st.columns(3)
    c1.metric("Spend / allowance", f"{spend_rate:.1f}%" if pd.notna(spend_rate) else "—")
    c2.metric("Avg. finance indicator", f"{avg_score:.1f}/100" if pd.notna(avg_score) else "Unavailable")
    c3.metric("Allowance vs spend", f"{filtered['Monthly_Allowance'].corr(filtered['Monthly_UPI_Spending']):.2f}" if filtered["Monthly_Allowance"].nunique() > 1 and filtered["Monthly_UPI_Spending"].nunique() > 1 else "Unavailable")

    left, right = st.columns([1.35, 1])
    with left:
        fig = px.histogram(
            filtered, x="Monthly_UPI_Spending", nbins=24,
            labels={"Monthly_UPI_Spending": "Monthly UPI spending (₹)", "count": "Students"},
            color_discrete_sequence=["#34d399"],
        )
        st.plotly_chart(apply_plot_theme(fig, "Monthly spending distribution"), use_container_width=True)
    with right:
        if "Preferred_UPI_App" in filtered and filtered["Preferred_UPI_App"].notna().any():
            app_counts = filtered["Preferred_UPI_App"].astype(str).value_counts().rename_axis("UPI app").reset_index(name="Students")
            fig = px.pie(app_counts, names="UPI app", values="Students", hole=.68, color_discrete_sequence=["#34d399", "#2dd4bf", "#60a5fa", "#a78bfa", "#fbbf24", "#fb7185"])
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(apply_plot_theme(fig, "Preferred UPI apps"), use_container_width=True)
        else:
            st.info("No UPI app column was found in this dataset.")

    left, right = st.columns(2)
    with left:
        fig = px.scatter(
            filtered, x="Monthly_Allowance", y="Monthly_UPI_Spending",
            color="Financial_Category", hover_data=["Student_ID"],
            labels={"Monthly_Allowance": "Monthly allowance (₹)", "Monthly_UPI_Spending": "Monthly UPI spending (₹)"},
            color_discrete_sequence=["#34d399", "#2dd4bf", "#60a5fa", "#fbbf24", "#a78bfa"],
        )
        st.plotly_chart(apply_plot_theme(fig, "Allowance vs UPI spending"), use_container_width=True)
    with right:
        category_counts = filtered["Financial_Category"].value_counts().rename_axis("Category").reset_index(name="Students")
        fig = px.bar(category_counts, x="Category", y="Students", color="Category", color_discrete_sequence=["#34d399", "#2dd4bf", "#60a5fa", "#fbbf24", "#fb7185"])
        st.plotly_chart(apply_plot_theme(fig, "Project financial indicator categories"), use_container_width=True)

    st.subheader("Quick data preview")
    st.dataframe(filtered.head(12), use_container_width=True, hide_index=True)


# ---------- EDA Analytics ----------
elif page == "EDA Analytics":
    st.subheader("Explore the dataset")
    numeric_cols = filtered.select_dtypes(include="number").columns.tolist()
    default_x = "Monthly_Allowance"
    default_y = "Monthly_UPI_Spending"
    left, right = st.columns(2)
    with left:
        x_col = st.selectbox("X-axis", numeric_cols, index=numeric_cols.index(default_x) if default_x in numeric_cols else 0)
    with right:
        y_col = st.selectbox("Y-axis", numeric_cols, index=numeric_cols.index(default_y) if default_y in numeric_cols else min(1, len(numeric_cols)-1))
    chart_type = st.selectbox("Chart type", ["Scatter plot", "Histogram", "Box plot", "Bar chart"])
    if chart_type == "Scatter plot":
        color_options = ["None"] + [c for c in ["Gender", "Department", "Year_of_Study", "Preferred_UPI_App", "Financial_Category"] if c in filtered.columns]
        color_col = st.selectbox("Colour by", color_options)
        fig = px.scatter(filtered, x=x_col, y=y_col, color=None if color_col == "None" else color_col, hover_data=["Student_ID"])
    elif chart_type == "Histogram":
        fig = px.histogram(filtered, x=x_col, nbins=30, color_discrete_sequence=["#34d399"])
    elif chart_type == "Box plot":
        group_options = ["None"] + [c for c in ["Gender", "Department", "Year_of_Study", "Preferred_UPI_App", "Hosteller_or_DayScholar"] if c in filtered.columns]
        group_col = st.selectbox("Group by", group_options)
        fig = px.box(filtered, x=None if group_col == "None" else group_col, y=y_col, color=None if group_col == "None" else group_col)
    else:
        group_options = [c for c in ["Gender", "Department", "Year_of_Study", "Preferred_UPI_App", "Hosteller_or_DayScholar", "Tracks_Budget"] if c in filtered.columns]
        if not group_options:
            st.warning("No categorical columns are available for grouped bar charts.")
            st.stop()
        group_col = st.selectbox("Group by", group_options)
        metric_col = st.selectbox("Metric", numeric_cols, index=numeric_cols.index(y_col) if y_col in numeric_cols else 0)
        summary = filtered.groupby(group_col, observed=True)[metric_col].mean().reset_index()
        fig = px.bar(summary, x=group_col, y=metric_col, color=group_col, color_discrete_sequence=["#34d399", "#2dd4bf", "#60a5fa", "#a78bfa", "#fbbf24"])
    st.plotly_chart(apply_plot_theme(fig, f"{chart_type}"), use_container_width=True)

    st.subheader("Correlation explorer")
    corr_cols = [c for c in numeric_cols if filtered[c].nunique(dropna=True) > 1]
    if len(corr_cols) >= 2:
        corr = filtered[corr_cols].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale=["#0b1f1a", "#0f766e", "#34d399"])
        st.plotly_chart(apply_plot_theme(fig, "Numeric feature correlation matrix"), use_container_width=True)
    else:
        st.info("At least two varying numeric columns are needed for a correlation matrix.")

    st.subheader("Calculated summary")
    summary_df = filtered[numeric_cols].describe().T.reset_index(names="Feature")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ---------- Student Profile ----------
elif page == "Student Profile":
    st.subheader("Student profile")
    ids = filtered["Student_ID"].astype(str).tolist()
    selected_id = st.selectbox("Choose a student", ids)
    student = filtered.loc[filtered["Student_ID"].astype(str) == selected_id].iloc[0]

    st.markdown(
        f"""
        <div class="glass-card">
          <div class="eyebrow">STUDENT SNAPSHOT</div>
          <div style="font-family:Space Grotesk;font-size:1.7rem;font-weight:700;margin:6px 0">{selected_id}</div>
          <div class="muted">Financial category: <span style="color:#6ee7b7;font-weight:700">{student['Financial_Category']}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Allowance", money(student["Monthly_Allowance"]))
    c2.metric("UPI spending", money(student["Monthly_UPI_Spending"]))
    c3.metric("Savings", money(student["Savings"]) if pd.notna(student["Savings"]) else "Unavailable")
    c4.metric("Remaining after UPI spend", money(student["Remaining_Balance"]))

    c1, c2 = st.columns(2)
    with c1:
        spend_pct = max(0, float(student["Spending_Percentage"])) if pd.notna(student["Spending_Percentage"]) else 0
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=spend_pct, number={"suffix": "%"},
            title={"text": "UPI spending / allowance"},
            gauge={"axis": {"range": [0, max(100, min(200, spend_pct * 1.15))]},
                   "bar": {"color": "#34d399"},
                   "bgcolor": "rgba(0,0,0,0)",
                   "bordercolor": "rgba(110,231,183,.2)",
                   "steps": [{"range": [0, 50], "color": "rgba(52,211,153,.10)"},
                             {"range": [50, 80], "color": "rgba(251,191,36,.10)"},
                             {"range": [80, max(100, min(200, spend_pct * 1.15))], "color": "rgba(251,113,133,.10)"}]}
        ))
        st.plotly_chart(apply_plot_theme(fig), use_container_width=True)
    with c2:
        st.markdown("#### Available details")
        profile_cols = ["Age", "Gender", "Year_of_Study", "Department", "Preferred_UPI_App",
                        "Hosteller_or_DayScholar", "Tracks_Budget", "Daily_UPI_Transactions",
                        "Cashback_Influence", "Impulse_Buying_Score", "Part_Time_Job"]
        details = [{"Field": col, "Value": str(student[col]) if col in student.index and pd.notna(student[col]) else "Unavailable"}
                   for col in profile_cols if col in student.index]
        st.dataframe(pd.DataFrame(details), use_container_width=True, hide_index=True)

    st.markdown("#### Student record")
    st.dataframe(student.to_frame(name="Value"), use_container_width=True)


# ---------- AI Financial Advisor ----------
elif page == "AI Financial Advisor":
    st.subheader("AI financial advisor")
    st.write("Ask Gemini or local Phi-3 Mini about a selected student's financial patterns. Responses are educational suggestions, not professional financial advice.")
    ai_provider = st.radio(
        "AI provider",
        ["Gemini API", "Ollama Phi-3 Mini", "Auto (Ollama → Gemini)"],
        horizontal=True,
        index=0,
        key="ai_provider",
        help="Use Gemini for the deployed Streamlit website. Ollama works only where the local Ollama service is running.",
    )
    if ai_provider == "Gemini API" and not get_gemini_api_key():
        st.warning("Gemini API key not found. Add GEMINI_API_KEY under Streamlit Cloud → Settings → Secrets.")
    ids = filtered["Student_ID"].astype(str).tolist()
    selected_id = st.selectbox("Student context", ids, key="ai_student")
    student = filtered.loc[filtered["Student_ID"].astype(str) == selected_id].iloc[0]

    context = {
        "student_id": str(student["Student_ID"]),
        "monthly_allowance_inr": None if pd.isna(student["Monthly_Allowance"]) else round(float(student["Monthly_Allowance"]), 2),
        "monthly_upi_spending_inr": None if pd.isna(student["Monthly_UPI_Spending"]) else round(float(student["Monthly_UPI_Spending"]), 2),
        "savings_inr": None if pd.isna(student["Savings"]) else round(float(student["Savings"]), 2),
        "spending_percentage_of_allowance": None if pd.isna(student["Spending_Percentage"]) else round(float(student["Spending_Percentage"]), 2),
        "remaining_balance_inr": None if pd.isna(student["Remaining_Balance"]) else round(float(student["Remaining_Balance"]), 2),
        "financial_health_indicator_0_to_100": None if pd.isna(student["Financial_Health_Score"]) else round(float(student["Financial_Health_Score"]), 2),
        "cashback_influence_1_to_5": None if "Cashback_Influence" not in student.index or pd.isna(student["Cashback_Influence"]) else float(student["Cashback_Influence"]),
        "impulse_buying_score_1_to_5": None if "Impulse_Buying_Score" not in student.index or pd.isna(student["Impulse_Buying_Score"]) else float(student["Impulse_Buying_Score"]),
        "financial_category": str(student["Financial_Category"]),
    }
    with st.expander("View the exact student data used as AI context"):
        st.json(context)

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if st.button("Clear chat", key="clear_chat"):
        st.session_state.chat_messages = []
        st.rerun()

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt_text = st.chat_input("Ask about spending, savings, or budgeting…")
    if prompt_text:
        st.session_state.chat_messages.append({"role": "user", "content": prompt_text})
        with st.chat_message("user"):
            st.markdown(prompt_text)
        system = (
            "You are UPI Insight, a supportive student financial-literacy assistant. "
            "Use only the supplied student context. Do not invent values. "
            "If a value is null, say it is unavailable. Give practical and non-judgmental "
            "educational suggestions. The financial health indicator is a project-specific "
            "formula, not a validated measure. Do not claim correlation proves causation. "
            "Avoid asking for sensitive bank credentials, UPI PINs, or account numbers."
        )
        user_message = (
            f"Student context (JSON):\n{json.dumps(context, ensure_ascii=False)}\n\n"
            f"Student question: {prompt_text}"
        )
        with st.chat_message("assistant"):
            with st.spinner(f"{ai_provider} is thinking…"):
                try:
                    answer = ask_ai(
                        [{"role": "user", "content": user_message}],
                        system_prompt=system,
                        provider=ai_provider,
                    )
                    st.markdown(answer)
                    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                except Exception as exc:
                    answer = (
                        f"Could not get a response from **{ai_provider}**.\n\n"
                        "For the deployed website, select **Gemini API** and check that `GEMINI_API_KEY` is set in Streamlit Secrets. "
                        "For **Ollama Phi-3 Mini**, the Ollama service and `phi3` model must be running on the same machine as this app.\n\n"
                        f"Technical detail: `{exc}`"
                    )
                    st.error(answer)
                    st.session_state.chat_messages.append({"role": "assistant", "content": answer})


# ---------- Download Reports ----------
elif page == "Download Reports":
    st.subheader("Download reports")
    st.write("Export the currently filtered data and a summary generated from the actual loaded dataset.")
    report_text = make_report(filtered, source_name)
    csv_bytes = filtered.to_csv(index=False).encode("utf-8-sig")
    report_bytes = report_text.encode("utf-8")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="glass-card"><h3>Dataset export</h3><p class="muted">Download the filtered student rows, with calculated metrics.</p></div>', unsafe_allow_html=True)
        st.download_button(
            "⬇ Download filtered CSV",
            data=csv_bytes,
            file_name="upi_insight_student_data.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        st.markdown('<div class="glass-card"><h3>Analysis report</h3><p class="muted">Download actual summary metrics, category counts and interpretation notes.</p></div>', unsafe_allow_html=True)
        st.download_button(
            "⬇ Download report (Markdown)",
            data=report_bytes,
            file_name="upi_insight_analysis_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.markdown("#### Report preview")
    st.markdown(report_text)
    st.markdown("---")
    st.caption("Review the output before submitting. The finance category is a simple project-defined indicator, not a validated financial assessment.")


# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#6f9485;font-size:.78rem;padding:8px 0">UPI INSIGHT · BUILT WITH STREAMLIT, PANDAS, PLOTLY, GEMINI & OLLAMA</div>',
    unsafe_allow_html=True,
)
