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

# Custom CSS for better styling
st.markdown("""
<style>
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }

    /* Example question buttons */
    .example-btn {
        background: #f0f2f6;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px 0;
        cursor: pointer;
        transition: all 0.2s;
    }
    .example-btn:hover {
        background: #e0e0e0;
    }

    /* Error container */
    .error-container {
        background: #fee2e2;
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }

    /* Chat message styling */
    .user-message {
        background: #e8f4f8;
        border-radius: 15px 15px 5px 15px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .assistant-message {
        background: #f8f9fa;
        border-radius: 15px 15px 15px 5px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Example questions for users to try
EXAMPLE_QUESTIONS = [
    "How many CSR AI calls were handled for Pros last week?",
    "What is the percentage of Pros who left feedback on Help AI this month?",
    "Provide a breakdown of AI team usage in 2025. Group by month and exclude CSR AI and Type 1",
    "Which AI Team Member has been used the most this year?",
    "Which AI Team Member gets the most feedback?",
]


def init_session_state():
    """Initialize session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_results" not in st.session_state:
        st.session_state.current_results = None
    if "last_error" not in st.session_state:
        st.session_state.last_error = None
    if "failed_question" not in st.session_state:
        st.session_state.failed_question = None


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

    # Check for date/time columns
    date_cols = []
    numeric_cols = []
    categorical_cols = []

    for col in df.columns:
        col_lower = col.lower()
        # Check if column name suggests a date
        if any(hint in col_lower for hint in ['date', 'time', 'day', 'week', 'month', 'year', 'period']):
            date_cols.append(col)
        # Try to detect datetime from values
        elif df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col].head())
                date_cols.append(col)
            except (ValueError, TypeError):
                if df[col].nunique() < len(df) * 0.5:
                    categorical_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        elif df[col].nunique() < len(df) * 0.5:
            categorical_cols.append(col)

    # Decision logic
    if date_cols and numeric_cols:
        return 'time_series'
    elif categorical_cols and numeric_cols:
        if len(df) <= 10:
            return 'pie' if len(categorical_cols) == 1 and len(numeric_cols) == 1 else 'bar'
        return 'bar'
    elif len(numeric_cols) >= 2:
        return 'bar'
    elif len(numeric_cols) == 1 and len(df) > 10:
        return 'histogram'

    return None


def create_visualization(df: pd.DataFrame, viz_type: str) -> Optional[go.Figure]:
    """Create a Plotly visualization based on the detected type."""
    if df.empty:
        return None

    # Identify column types
    date_cols = []
    numeric_cols = []
    categorical_cols = []

    for col in df.columns:
        col_lower = col.lower()
        if any(hint in col_lower for hint in ['date', 'time', 'day', 'week', 'month', 'year', 'period']):
            date_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        elif df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col].head())
                date_cols.append(col)
            except (ValueError, TypeError):
                categorical_cols.append(col)

    try:
        if viz_type == 'time_series' and date_cols and numeric_cols:
            x_col = date_cols[0]
            y_col = numeric_cols[0]
            fig = px.line(
                df, x=x_col, y=y_col,
                title=f"{y_col} over {x_col}",
                markers=True
            )
            fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
            return fig

        elif viz_type == 'bar':
            if categorical_cols and numeric_cols:
                x_col = categorical_cols[0]
                y_col = numeric_cols[0]
            elif len(df.columns) >= 2:
                x_col = df.columns[0]
                y_col = df.columns[1] if pd.api.types.is_numeric_dtype(df[df.columns[1]]) else df.columns[0]
            else:
                return None

            fig = px.bar(
                df, x=x_col, y=y_col,
                title=f"{y_col} by {x_col}"
            )
            fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
            return fig

        elif viz_type == 'pie' and categorical_cols and numeric_cols:
            fig = px.pie(
                df,
                names=categorical_cols[0],
                values=numeric_cols[0],
                title=f"Distribution of {numeric_cols[0]} by {categorical_cols[0]}"
            )
            return fig

        elif viz_type == 'histogram' and numeric_cols:
            fig = px.histogram(
                df, x=numeric_cols[0],
                title=f"Distribution of {numeric_cols[0]}"
            )
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
                <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin: 10px 0;">
                    <p style="color: rgba(255,255,255,0.8); font-size: 14px; margin: 0; text-transform: uppercase;">{col_name}</p>
                    <p style="color: white; font-size: 48px; font-weight: bold; margin: 10px 0;">{display_value}</p>
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


def run_question(question: str):
    """Run a question and update state."""
    st.session_state.pending_question = question


def render_sidebar():
    """Render the chat history sidebar."""
    with st.sidebar:
        # Example questions section
        st.header("💡 Try These Examples")
        for example in EXAMPLE_QUESTIONS:
            if st.button(example, key=f"example_{example}", use_container_width=True):
                st.session_state.pending_question = example
                st.rerun()

        st.divider()

        # Query history section
        st.header("📜 Recent Queries")

        if not st.session_state.chat_history:
            st.caption("Your query history will appear here")
        else:
            # Clear history button
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.current_results = None
                st.rerun()

            # Display history items (most recent first, limit to MAX_HISTORY)
            for i, item in enumerate(reversed(st.session_state.chat_history[-MAX_HISTORY:])):
                idx = len(st.session_state.chat_history) - 1 - i
                truncated = item['question'][:35] + "..." if len(item['question']) > 35 else item['question']
                with st.expander(f"🔍 {truncated}", expanded=False):
                    st.caption(f"🕐 {item['timestamp']}")
                    if item['summary']:
                        summary_display = item['summary'][:150] + "..." if len(item['summary']) > 150 else item['summary']
                        st.write(summary_display)
                    if st.button("Load Result", key=f"load_{idx}", use_container_width=True):
                        st.session_state.current_results = item
                        st.rerun()
                    if st.button("Re-run Query", key=f"rerun_{idx}", use_container_width=True):
                        st.session_state.pending_question = item['question']
                        st.rerun()


def render_error_with_retry():
    """Render error message with retry button."""
    if st.session_state.last_error:
        st.markdown(f"""
        <div class="error-container">
            <strong>⚠️ Something went wrong</strong><br>
            {st.session_state.last_error}
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🔄 Retry", use_container_width=True):
                if st.session_state.failed_question:
                    st.session_state.pending_question = st.session_state.failed_question
                    st.session_state.last_error = None
                    st.rerun()


def render_main_content():
    """Render the main content area."""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Analytics Assistant</h1>
        <p>Ask questions about your AI product usage in plain English. I'll query the data and visualize the results for you.</p>
    </div>
    """, unsafe_allow_html=True)

    # Check for pending question (from examples or re-run)
    if "pending_question" in st.session_state and st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

        with st.spinner("🔍 Analyzing your question..."):
            response = query_webhook(question)

            if response:
                add_to_history(question, response)
                st.session_state.current_results = {
                    "question": question,
                    "summary": response.get("summary", ""),
                    "sql": response.get("sql", ""),
                    "results": response.get("results", [])
                }
        st.rerun()

    # Query input with chat-like styling
    st.markdown("### 💬 Ask a Question")
    with st.form("query_form", clear_on_submit=True):
        question = st.text_input(
            "Your Question",
            placeholder="e.g., How many CSR AI calls were made last week?",
            label_visibility="collapsed"
        )
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            submitted = st.form_submit_button("🔍 Ask", use_container_width=True)

    # Process query
    if submitted and question:
        with st.spinner("🔍 Analyzing your question..."):
            response = query_webhook(question)

            if response:
                add_to_history(question, response)
                st.session_state.current_results = {
                    "question": question,
                    "summary": response.get("summary", ""),
                    "sql": response.get("sql", ""),
                    "results": response.get("results", [])
                }
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
    st.markdown('<div class="assistant-message">', unsafe_allow_html=True)

    # Summary
    if data['summary']:
        st.markdown(f"**📝 Answer:** {data['summary']}")
    else:
        st.markdown("**📝 Answer:** No summary available.")

    st.markdown('</div>', unsafe_allow_html=True)

    # SQL (expandable)
    with st.expander("🔧 View Generated SQL", expanded=False):
        if data['sql']:
            st.code(data['sql'], language="sql")
        else:
            st.write("No SQL available.")

    # Results table and visualization
    if data['results']:
        df = pd.DataFrame(data['results'])
        viz_type = detect_visualization_type(df)

        # Single metric: show big number prominently, then data below
        if viz_type == 'metric':
            render_metric_display(df)
            st.divider()
            with st.expander("📋 Raw Data", expanded=False):
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            # Tabs for data and visualization
            tab1, tab2 = st.tabs(["📈 Visualization", "📋 Data"])

            with tab1:
                if viz_type:
                    fig = create_visualization(df, viz_type)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Could not generate a visualization for this data.")
                else:
                    st.info("📊 This data is best viewed as a table. Check the Data tab.")

            with tab2:
                st.dataframe(df, use_container_width=True)

                # Export button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    else:
        st.warning("No data results to display.")


def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()
