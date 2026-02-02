# Analytics Assistant

A Streamlit web app for querying AI product usage data using natural language.

## Features

- **Natural Language Queries**: Ask questions in plain English (e.g., "How many CSR AI calls last week?")
- **SQL Generation**: Automatically converts questions to SQL via n8n webhook
- **Smart Visualizations**: Auto-detects appropriate chart types (time series, bar, pie, histogram)
- **Query History**: Sidebar tracks all your past queries
- **Data Export**: Download results as CSV

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Webhook URL

Copy the secrets template and add your n8n webhook URL:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml`:

```toml
N8N_WEBHOOK_URL = "https://your-n8n-instance.com/webhook/your-webhook-id"
```

### 3. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## API Contract

The app expects the n8n webhook to:

**Accept (POST):**
```json
{
  "question": "How many CSR AI calls were made last week?"
}
```

**Return:**
```json
{
  "question": "How many CSR AI calls were made last week?",
  "sql": "SELECT date, count FROM usage WHERE ...",
  "summary": "There were 1,234 CSR AI calls last week, up 15% from the previous week.",
  "results": [
    {"date": "2024-01-15", "count": 180},
    {"date": "2024-01-16", "count": 195}
  ]
}
```

## Deployment

### Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Add `N8N_WEBHOOK_URL` in the Secrets section
5. Deploy

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

## Visualization Logic

The app automatically detects visualization types:

| Data Pattern | Chart Type |
|--------------|------------|
| Date + numeric columns | Line chart (time series) |
| Category + numeric (≤10 rows) | Pie chart |
| Category + numeric (>10 rows) | Bar chart |
| Single numeric column | Histogram |
