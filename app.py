"""
Analytics Assistant - A Streamlit app for natural language data queries.

This app allows users to ask questions about AI product usage data in plain English,
which are then converted to SQL, executed, and visualized.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import hashlib
import io
from typing import Optional

# Configuration
N8N_WEBHOOK_URL = st.secrets.get("N8N_WEBHOOK_URL", "YOUR_WEBHOOK_URL_HERE")
MAX_HISTORY = 10

# Page configuration
st.set_page_config(
    page_title="Analytics Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Example questions for users to try
EXAMPLE_QUESTIONS = [
    "How many CSR AI calls were handled for Pros last week?",
    "What is the percentage of Pros who left feedback on Help AI this month?",
    "Provide a breakdown of AI team usage in 2025. Group by month and exclude CSR AI and Type 1",
    "Which AI Team Member has been used the most this year?",
    "Which AI Team Member gets the most feedback?",
]


def apply_theme():
    """Apply HCP-branded theme styling."""
    st.markdown("""
    <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* Global styles */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* HCP Color Variables */
        :root {
            --hcp-blue: #36A3D9;
            --hcp-blue-dark: #2B8AB8;
            --hcp-blue-light: #E8F4FC;
            --text-dark: #333333;
            --text-light: #666666;
        }

        /* Sidebar branding */
        .sidebar-header {
            background: linear-gradient(135deg, #36A3D9 0%, #2B8AB8 100%);
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            text-align: center;
            box-shadow: 0 4px 15px rgba(54, 163, 217, 0.3);
        }
        .sidebar-header h2 {
            color: white;
            margin: 0;
            font-size: 1.3rem;
            font-weight: 600;
        }
        .sidebar-header .sparkle {
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
            display: block;
        }

        /* Main header */
        .main-header {
            background: linear-gradient(135deg, #36A3D9 0%, #2B8AB8 100%);
            padding: 2rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            box-shadow: 0 8px 30px rgba(54, 163, 217, 0.25);
            animation: fadeInDown 0.5s ease-out;
        }
        .main-header h1 {
            color: white;
            margin: 0;
            font-size: 2.2rem;
            font-weight: 700;
        }
        .main-header p {
            color: rgba(255,255,255,0.95);
            margin: 0.75rem 0 0 0;
            font-size: 1.05rem;
            font-weight: 400;
            line-height: 1.5;
        }

        /* Metric cards */
        .metric-card {
            background: linear-gradient(135deg, #36A3D9 0%, #2B8AB8 100%);
            border-radius: 16px;
            box-shadow: 0 8px 30px rgba(54, 163, 217, 0.25);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(54, 163, 217, 0.35);
        }

        /* Chat bubbles */
        .user-message {
            background: #E8F4FC;
            border-radius: 16px 16px 4px 16px;
            padding: 1.25rem;
            margin: 0.75rem 0;
            border: 1px solid rgba(54, 163, 217, 0.2);
            animation: fadeIn 0.3s ease-out;
        }
        .assistant-message {
            background: #ffffff;
            border-radius: 16px 16px 16px 4px;
            padding: 1.25rem;
            margin: 0.75rem 0;
            border-left: 4px solid #36A3D9;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
            animation: fadeIn 0.3s ease-out;
        }

        /* Error container */
        .error-container {
            background: #FEF2F2;
            border: 1px solid #FECACA;
            border-radius: 12px;
            padding: 1.25rem;
            margin: 1rem 0;
            animation: fadeIn 0.3s ease-out;
        }

        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #36A3D9 0%, #2B8AB8 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.2s ease;
            box-shadow: 0 2px 8px rgba(54, 163, 217, 0.3);
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #2B8AB8 0%, #238BA3 100%);
            box-shadow: 0 4px 15px rgba(54, 163, 217, 0.4);
            transform: translateY(-1px);
        }
        .stButton > button:active {
            transform: translateY(0);
        }

        /* Form submit button */
        .stFormSubmitButton > button {
            background: linear-gradient(135deg, #36A3D9 0%, #2B8AB8 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s ease;
            box-shadow: 0 2px 8px rgba(54, 163, 217, 0.3);
        }
        .stFormSubmitButton > button:hover {
            background: linear-gradient(135deg, #2B8AB8 0%, #238BA3 100%);
            box-shadow: 0 4px 15px rgba(54, 163, 217, 0.4);
        }

        /* Input fields */
        .stTextInput > div > div > input {
            border-radius: 10px;
            border: 2px solid #E5E7EB;
            padding: 0.75rem 1rem;
            transition: all 0.2s ease;
        }
        .stTextInput > div > div > input:focus {
            border-color: #36A3D9;
            box-shadow: 0 0 0 3px rgba(54, 163, 217, 0.15);
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            background: #F8FAFC;
            border-radius: 8px;
            font-weight: 500;
        }

        /* Selectbox styling */
        .stSelectbox > div > div {
            border-radius: 8px;
        }

        /* Download buttons */
        .stDownloadButton > button {
            background: white;
            color: #36A3D9;
            border: 2px solid #36A3D9;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        .stDownloadButton > button:hover {
            background: #E8F4FC;
            border-color: #2B8AB8;
        }

        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Results container animation */
        .results-container {
            animation: fadeIn 0.4s ease-out;
        }

        /* Divider styling */
        hr {
            border: none;
            height: 1px;
            background: linear-gradient(to right, transparent, #E5E7EB, transparent);
            margin: 1.5rem 0;
        }

        /* Card containers */
        .card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
            border: 1px solid #F1F5F9;
            transition: all 0.2s ease;
        }
        .card:hover {
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        }
    </style>
    """, unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_results" not in st.session_state:
        st.session_state.current_results = None
    if "chart_type_override" not in st.session_state:
        st.session_state.chart_type_override = None
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "last_error" not in st.session_state:
        st.session_state.last_error = None
    if "failed_question" not in st.session_state:
        st.session_state.failed_question = None
    # Feedback system state
    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = {}  # query_id -> submission info
    if "feedback_expanded" not in st.session_state:
        st.session_state.feedback_expanded = {}  # query_id -> True/False for form visibility


def query_webhook(question: str) -> Optional[dict]:
    """Send a question to the n8n webhook and return the response."""
    try:
        response = requests.post(
            N8N_WEBHOOK_URL,
            json={"question": question},
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        response.raise_for_status()
        data = response.json()

        # Check for SQL errors in response
        if data.get("error"):
            st.session_state.last_error = data.get("error")
            st.session_state.failed_question = question
            return None

        # Clear any previous errors on success
        st.session_state.last_error = None
        st.session_state.failed_question = None
        return data

    except requests.exceptions.Timeout:
        st.session_state.last_error = "Request timed out. The query might be too complex or the server is busy."
        st.session_state.failed_question = question
        return None
    except requests.exceptions.ConnectionError:
        st.session_state.last_error = "Could not connect to the analytics server. Please check your internet connection."
        st.session_state.failed_question = question
        return None
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            st.session_state.last_error = "Analytics API endpoint not found. Please check the webhook configuration."
        elif e.response.status_code == 500:
            st.session_state.last_error = "Server error occurred. Please try again later."
        else:
            st.session_state.last_error = f"HTTP Error: {e.response.status_code}"
        st.session_state.failed_question = question
        return None
    except requests.exceptions.RequestException as e:
        st.session_state.last_error = f"Request failed: {str(e)}"
        st.session_state.failed_question = question
        return None
    except json.JSONDecodeError:
        st.session_state.last_error = "Received invalid response from the server."
        st.session_state.failed_question = question
        return None


def get_query_identifier(query_data: dict) -> str:
    """Generate a unique identifier for a query based on its content."""
    content = f"{query_data.get('question', '')}{query_data.get('sql', '')}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def send_feedback_to_webhook(payload: dict) -> bool:
    """Send feedback payload to the n8n webhook."""
    try:
        response = requests.post(
            N8N_WEBHOOK_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException:
        return False


def build_feedback_payload(
    sentiment: str,
    selected_options: list,
    free_text: str,
    query_data: dict
) -> dict:
    """Build the feedback payload structure for the webhook."""
    return {
        "type": "user_feedback",
        "timestamp": datetime.now().isoformat(),
        "feedback": {
            "sentiment": sentiment,
            "selected_options": selected_options,
            "free_text": free_text
        },
        "query_context": {
            "question": query_data.get("question", ""),
            "sql": query_data.get("sql", ""),
            "summary": query_data.get("summary", ""),
            "result_count": len(query_data.get("results", [])),
            "query_timestamp": query_data.get("timestamp", "")
        }
    }


def detect_visualization_type(df: pd.DataFrame) -> Optional[str]:
    """
    Auto-detect the appropriate visualization type based on data characteristics.

    Returns: 'metric', 'time_series', 'bar', 'pie', 'histogram', or None
    """
    if df.empty:
        return None

    # Single metric detection: 1 row with numeric value(s)
    if len(df) == 1:
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        if numeric_cols:
            return 'metric'

    if len(df.columns) < 2:
        return None

    # Analyze columns
    date_cols = []
    numeric_cols = []
    categorical_cols = []

    for col in df.columns:
        col_lower = col.lower()

        # Check if column name suggests a date
        if any(hint in col_lower for hint in ['date', 'time', 'day', 'week', 'month', 'year', 'period']):
            date_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        elif df[col].dtype == 'object':
            # Try to detect datetime from values
            try:
                pd.to_datetime(df[col].head())
                date_cols.append(col)
            except (ValueError, TypeError):
                # More lenient categorical detection:
                # If it's a string column with reasonable number of unique values, treat as categorical
                # Allow up to 50 unique values or all unique if less than 20 rows
                unique_count = df[col].nunique()
                if unique_count <= 50 or (len(df) <= 20 and unique_count == len(df)):
                    categorical_cols.append(col)

    # Decision logic - prioritize showing charts
    if date_cols and numeric_cols:
        return 'time_series'
    elif categorical_cols and numeric_cols:
        # Use pie for small datasets, bar for larger
        if len(df) <= 6:
            return 'pie'
        return 'bar'
    elif len(df.columns) == 2 and numeric_cols:
        # If we have 2 columns and one is numeric, assume first is category
        return 'bar'
    elif len(numeric_cols) >= 2:
        return 'bar'
    elif len(numeric_cols) == 1 and len(df) > 10:
        return 'histogram'

    # Last resort: if we have 2+ columns and any numeric, try bar chart
    if len(df.columns) >= 2 and numeric_cols:
        return 'bar'

    return None


def get_available_chart_types(df: pd.DataFrame) -> list:
    """Return list of chart types that make sense for this data."""
    if df.empty or len(df.columns) < 2:
        return ['table']

    types = ['table']
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    if numeric_cols:
        types.extend(['bar', 'pie'])
        if len(df) > 5:
            types.append('line')
        if len(df) > 10:
            types.append('histogram')

    return types


# HCP-themed color palette for charts
HCP_COLORS = [
    '#36A3D9',  # HCP Blue (primary)
    '#2B8AB8',  # HCP Blue Dark
    '#5BB5E0',  # HCP Blue Light
    '#1E7BA5',  # HCP Blue Darker
    '#7CC4E8',  # HCP Blue Lighter
    '#4FADD5',  # HCP Blue Mid
    '#8ED0ED',  # HCP Blue Pale
    '#238BA3',  # HCP Blue Deep
]


def create_visualization(df: pd.DataFrame, viz_type: str) -> Optional[go.Figure]:
    """Create a Plotly visualization based on the specified type with HCP branding."""
    if df.empty:
        return None

    # Identify column types
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    non_numeric_cols = [col for col in df.columns if col not in numeric_cols]

    # Default to first non-numeric as x, first numeric as y
    x_col = non_numeric_cols[0] if non_numeric_cols else df.columns[0]
    y_col = numeric_cols[0] if numeric_cols else df.columns[1] if len(df.columns) > 1 else df.columns[0]

    # Common layout settings for HCP branding
    hcp_layout = dict(
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", color="#333333"),
        title=dict(font=dict(size=18, color="#333333")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#E5E7EB", linecolor="#E5E7EB"),
        yaxis=dict(gridcolor="#E5E7EB", linecolor="#E5E7EB"),
    )

    try:
        if viz_type == 'time_series' or viz_type == 'line':
            fig = px.line(
                df, x=x_col, y=y_col,
                title=f"{y_col} over {x_col}",
                markers=True,
                color_discrete_sequence=HCP_COLORS
            )
            fig.update_layout(**hcp_layout, xaxis_title=x_col, yaxis_title=y_col)
            fig.update_traces(line=dict(width=3), marker=dict(size=8))
            return fig

        elif viz_type == 'bar':
            fig = px.bar(
                df, x=x_col, y=y_col,
                title=f"{y_col} by {x_col}",
                color_discrete_sequence=HCP_COLORS
            )
            fig.update_layout(**hcp_layout, xaxis_title=x_col, yaxis_title=y_col)
            fig.update_traces(marker=dict(cornerradius=6))
            return fig

        elif viz_type == 'pie':
            fig = px.pie(
                df,
                names=x_col,
                values=y_col,
                title=f"Distribution of {y_col} by {x_col}",
                color_discrete_sequence=HCP_COLORS
            )
            fig.update_layout(**hcp_layout)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            return fig

        elif viz_type == 'histogram' and numeric_cols:
            fig = px.histogram(
                df, x=numeric_cols[0],
                title=f"Distribution of {numeric_cols[0]}",
                color_discrete_sequence=HCP_COLORS
            )
            fig.update_layout(**hcp_layout)
            return fig

    except Exception as e:
        st.warning(f"Could not create visualization: {str(e)}")
        return None

    return None


def render_metric_display(df: pd.DataFrame):
    """Render a big number display for single metric results."""
    if df.empty or len(df) != 1:
        return

    row = df.iloc[0]
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    non_numeric_cols = [col for col in df.columns if col not in numeric_cols]

    # Display metrics in columns
    cols = st.columns(len(numeric_cols)) if len(numeric_cols) > 1 else [st.container()]

    for i, col_name in enumerate(numeric_cols):
        value = row[col_name]
        # Format large numbers
        if isinstance(value, (int, float)):
            if abs(value) >= 1_000_000:
                display_value = f"{value/1_000_000:,.1f}M"
            elif abs(value) >= 1_000:
                display_value = f"{value/1_000:,.1f}K"
            elif isinstance(value, float):
                display_value = f"{value:,.2f}"
            else:
                display_value = f"{value:,}"
        else:
            display_value = str(value)

        with cols[i] if len(numeric_cols) > 1 else cols[0]:
            st.markdown(
                f"""
                <div class="metric-card" style="text-align: center; padding: 24px; background: linear-gradient(135deg, #36A3D9 0%, #2B8AB8 100%); border-radius: 16px; margin: 10px 0; box-shadow: 0 8px 30px rgba(54, 163, 217, 0.25);">
                    <p style="color: rgba(255,255,255,0.85); font-size: 13px; margin: 0; text-transform: uppercase; letter-spacing: 1px; font-weight: 500;">{col_name}</p>
                    <p style="color: white; font-size: 52px; font-weight: 700; margin: 12px 0; font-family: 'Inter', sans-serif;">{display_value}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Show any context/labels below
    if non_numeric_cols:
        context_parts = [f"**{col}:** {row[col]}" for col in non_numeric_cols]
        st.markdown(" | ".join(context_parts))


def add_to_history(question: str, response: dict):
    """Add a query and its response to the chat history."""
    st.session_state.chat_history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "summary": response.get("summary", ""),
        "sql": response.get("sql", ""),
        "results": response.get("results", [])
    })
    # Keep only last MAX_HISTORY items
    if len(st.session_state.chat_history) > MAX_HISTORY:
        st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY:]


def render_sidebar():
    """Render the sidebar with examples and history."""
    with st.sidebar:
        # Branded header with sparkle
        st.markdown("""
        <div class="sidebar-header">
            <span class="sparkle">✨</span>
            <h2>Analytics Assistant</h2>
        </div>
        """, unsafe_allow_html=True)

        # Example questions section
        st.markdown("**Try These Examples**")
        for example in EXAMPLE_QUESTIONS:
            if st.button(example, key=f"example_{hash(example)}", use_container_width=True):
                st.session_state.pending_question = example
                st.rerun()

        st.divider()

        # Query history section
        st.markdown("**Recent Queries**")

        if not st.session_state.chat_history:
            st.caption("Your query history will appear here")
        else:
            # Clear history button
            if st.button("Clear History", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.current_results = None
                st.session_state.feedback_submitted = {}
                st.session_state.feedback_expanded = {}
                st.rerun()

            # Display history items (most recent first)
            for i, item in enumerate(reversed(st.session_state.chat_history[-MAX_HISTORY:])):
                idx = len(st.session_state.chat_history) - 1 - i
                truncated = item['question'][:35] + "..." if len(item['question']) > 35 else item['question']
                with st.expander(f"{truncated}", expanded=False):
                    st.caption(f"{item['timestamp']}")
                    if item['summary']:
                        summary_display = item['summary'][:150] + "..." if len(item['summary']) > 150 else item['summary']
                        st.write(summary_display)
                    if st.button("Load Result", key=f"load_{idx}", use_container_width=True):
                        st.session_state.current_results = item
                        st.session_state.chart_type_override = None
                        st.rerun()
                    if st.button("Re-run Query", key=f"rerun_{idx}", use_container_width=True):
                        st.session_state.pending_question = item['question']
                        st.rerun()


def render_feedback_section(query_data: dict):
    """Render the feedback collection UI after results."""
    query_id = get_query_identifier(query_data)

    # Check if feedback already submitted for this query
    if query_id in st.session_state.feedback_submitted:
        st.success("Thank you for your feedback!")
        return

    st.divider()
    st.markdown("**Were you satisfied with this answer?**")

    col1, col2, col3 = st.columns([1, 1, 6])

    with col1:
        if st.button("👍", key=f"thumbs_up_{query_id}", use_container_width=True):
            st.session_state.feedback_expanded[query_id] = "positive"
            st.rerun()

    with col2:
        if st.button("👎", key=f"thumbs_down_{query_id}", use_container_width=True):
            st.session_state.feedback_expanded[query_id] = "negative"
            st.rerun()

    # Show feedback form if thumbs was clicked
    sentiment = st.session_state.feedback_expanded.get(query_id)
    if sentiment:
        render_feedback_form(query_data, query_id, sentiment)


def render_feedback_form(query_data: dict, query_id: str, sentiment: str):
    """Render the detailed feedback form with checkboxes and text."""
    st.markdown("---")

    if sentiment == "positive":
        st.markdown("**What did you like?**")
        options = [
            ("data_accurate", "Data looks accurate"),
            ("summary_helpful", "Summary is helpful"),
            ("visualization_good", "Visualization is convenient"),
            ("other_positive", "Other")
        ]
    else:
        st.markdown("**What could be improved?**")
        options = [
            ("data_incorrect", "Data looks incorrect"),
            ("summary_not_useful", "Summary is not useful"),
            ("visualization_issues", "Visualization has issues"),
            ("other_negative", "Other")
        ]

    # Checkboxes for options
    selected_options = []
    for option_key, option_label in options:
        if st.checkbox(option_label, key=f"fb_{query_id}_{option_key}"):
            selected_options.append(option_key)

    # Free text area
    free_text = st.text_area(
        "Additional comments (optional):",
        key=f"fb_text_{query_id}",
        height=100,
        placeholder="Share any additional thoughts..."
    )

    # Submit button
    if st.button("Submit Feedback", key=f"fb_submit_{query_id}", type="primary"):
        payload = build_feedback_payload(sentiment, selected_options, free_text, query_data)
        success = send_feedback_to_webhook(payload)

        if success:
            st.session_state.feedback_submitted[query_id] = {
                "sentiment": sentiment,
                "timestamp": datetime.now().isoformat()
            }
            # Clear the expanded state
            if query_id in st.session_state.feedback_expanded:
                del st.session_state.feedback_expanded[query_id]
            st.rerun()
        else:
            st.error("Failed to submit feedback. Please try again.")


def render_error_with_retry():
    """Render error message with retry button."""
    if st.session_state.last_error:
        st.markdown(f"""
        <div class="error-container" style="background: #fee2e2; border: 1px solid #ef4444; border-radius: 8px; padding: 1rem; margin: 1rem 0;">
            <strong>Something went wrong</strong><br>
            {st.session_state.last_error}
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Retry", use_container_width=True):
                if st.session_state.failed_question:
                    st.session_state.pending_question = st.session_state.failed_question
                    st.session_state.last_error = None
                    st.rerun()


def render_main_content():
    """Render the main content area."""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>✨ Analytics Assistant</h1>
        <p>Ask questions about your AI product usage in plain English. I'll query the data and visualize the results for you.</p>
    </div>
    """, unsafe_allow_html=True)

    # Check for pending question (from examples or re-run)
    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

        with st.spinner("Analyzing your question..."):
            response = query_webhook(question)

            if response:
                add_to_history(question, response)
                st.session_state.current_results = {
                    "question": question,
                    "summary": response.get("summary", ""),
                    "sql": response.get("sql", ""),
                    "results": response.get("results", []),
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.chart_type_override = None
        st.rerun()

    # Query input
    st.markdown("### Ask a Question")
    with st.form("query_form", clear_on_submit=True):
        question = st.text_input(
            "Your Question",
            placeholder="e.g., How many CSR AI calls were handled for Pros last week?",
            label_visibility="collapsed"
        )
        col1, col2 = st.columns([1, 5])
        with col1:
            submitted = st.form_submit_button("Ask", use_container_width=True)

    # Process query
    if submitted and question:
        with st.spinner("Analyzing your question..."):
            response = query_webhook(question)

            if response:
                add_to_history(question, response)
                st.session_state.current_results = {
                    "question": question,
                    "summary": response.get("summary", ""),
                    "sql": response.get("sql", ""),
                    "results": response.get("results", []),
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.chart_type_override = None
                st.rerun()

    # Show error with retry if present
    render_error_with_retry()

    # Display results
    if st.session_state.current_results:
        render_results(st.session_state.current_results)


def render_results(data: dict):
    """Render the query results in a conversational style."""
    st.divider()

    # User's question (chat bubble style)
    st.markdown(f"""
    <div class="user-message">
        <strong>You asked:</strong> {data['question']}
    </div>
    """, unsafe_allow_html=True)

    # Assistant's response
    if data['summary']:
        st.markdown(f"""
        <div class="assistant-message">
            <strong>Answer:</strong> {data['summary']}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No summary available.")

    # SQL (expandable)
    with st.expander("View Generated SQL", expanded=False):
        if data['sql']:
            st.code(data['sql'], language="sql")
        else:
            st.write("No SQL available.")

    # Results table and visualization
    if data['results']:
        df = pd.DataFrame(data['results'])
        auto_viz_type = detect_visualization_type(df)
        available_types = get_available_chart_types(df)

        # Single metric: show big number prominently, then data below
        if auto_viz_type == 'metric':
            render_metric_display(df)
            st.divider()
            with st.expander("Raw Data", expanded=False):
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            # Chart type selector
            col1, col2 = st.columns([3, 1])
            with col2:
                # Determine current selection
                current_type = st.session_state.chart_type_override or auto_viz_type or 'bar'
                if current_type not in available_types:
                    current_type = available_types[1] if len(available_types) > 1 else available_types[0]

                chart_labels = {
                    'table': 'Table',
                    'bar': 'Bar Chart',
                    'pie': 'Pie Chart',
                    'line': 'Line Chart',
                    'histogram': 'Histogram'
                }

                selected = st.selectbox(
                    "View as:",
                    options=available_types,
                    index=available_types.index(current_type) if current_type in available_types else 0,
                    format_func=lambda x: chart_labels.get(x, x),
                    key="chart_selector"
                )

                if selected != st.session_state.chart_type_override:
                    st.session_state.chart_type_override = selected

            # Use selected or auto-detected type
            viz_type = st.session_state.chart_type_override or auto_viz_type

            # Render based on selection
            fig = None
            if viz_type and viz_type != 'table':
                fig = create_visualization(df, viz_type)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.dataframe(df, use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)

            # Always show data table in expander if chart is shown
            if viz_type and viz_type != 'table':
                with st.expander("View Raw Data", expanded=False):
                    st.dataframe(df, use_container_width=True)

            # Export buttons
            csv = df.to_csv(index=False)

            if fig:
                # Show both CSV and PNG download buttons
                col1, col2, col3 = st.columns([1, 1, 4])
                with col1:
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with col2:
                    try:
                        img_bytes = fig.to_image(format="png", width=1200, height=600, scale=2)
                        st.download_button(
                            label="Download Chart",
                            data=img_bytes,
                            file_name=f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.caption("PNG export unavailable")
            else:
                # Only CSV download when no chart
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

    else:
        st.warning("No data results to display.")

    # Feedback section
    render_feedback_section(data)


def main():
    """Main application entry point."""
    init_session_state()
    apply_theme()
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()
