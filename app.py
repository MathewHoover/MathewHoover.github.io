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

# Page configuration
st.set_page_config(
    page_title="Analytics Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """Initialize session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_results" not in st.session_state:
        st.session_state.current_results = None


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
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to the analytics API: {str(e)}")
        return None
    except json.JSONDecodeError:
        st.error("Error parsing response from the analytics API")
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


def render_sidebar():
    """Render the chat history sidebar."""
    with st.sidebar:
        st.header("📜 Query History")

        if not st.session_state.chat_history:
            st.info("No queries yet. Ask a question to get started!")
        else:
            # Clear history button
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.current_results = None
                st.rerun()

            st.divider()

            # Display history items (most recent first)
            for i, item in enumerate(reversed(st.session_state.chat_history)):
                idx = len(st.session_state.chat_history) - 1 - i
                with st.expander(f"🔍 {item['question'][:40]}...", expanded=False):
                    st.caption(f"🕐 {item['timestamp']}")
                    st.write(item['summary'][:200] + "..." if len(item['summary']) > 200 else item['summary'])
                    if st.button("Load", key=f"load_{idx}", use_container_width=True):
                        st.session_state.current_results = item
                        st.rerun()


def render_main_content():
    """Render the main content area."""
    st.title("📊 Analytics Assistant")
    st.markdown("Ask questions about AI product usage data in plain English.")

    # Query input
    with st.form("query_form", clear_on_submit=True):
        question = st.text_input(
            "Your Question",
            placeholder="e.g., How many CSR AI calls were made last week?",
            label_visibility="collapsed"
        )
        col1, col2 = st.columns([1, 5])
        with col1:
            submitted = st.form_submit_button("🔍 Ask", use_container_width=True)

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
                    "results": response.get("results", [])
                }
                st.rerun()

    # Display results
    if st.session_state.current_results:
        render_results(st.session_state.current_results)


def render_results(data: dict):
    """Render the query results."""
    st.divider()

    # Question
    st.subheader(f"❓ {data['question']}")

    # Summary
    st.markdown("### 📝 Summary")
    st.info(data['summary'] if data['summary'] else "No summary available.")

    # SQL (expandable)
    with st.expander("🔧 Generated SQL", expanded=False):
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
            tab1, tab2 = st.tabs(["📋 Data", "📈 Visualization"])

            with tab1:
                st.markdown("### Raw Data")
                st.dataframe(df, use_container_width=True)

                # Export button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            with tab2:
                if viz_type:
                    fig = create_visualization(df, viz_type)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Could not generate a visualization for this data.")
                else:
                    st.info("This data doesn't appear suitable for automatic visualization.")
    else:
        st.warning("No data results to display.")


def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()
