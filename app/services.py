import os
import re
import json
import pandas as pd
import io
import boto3
import traceback
from dotenv import load_dotenv
from fastapi import HTTPException
from .models import ChatResponse, ChartData, HistoryMessage
from typing import List

# Load environment variables
load_dotenv()

# --- Bedrock Client Configuration ---
try:
    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        region_name=os.getenv("AWS_REGION_NAME", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
        raise ValueError(
            "AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) not found in .env file."
        )
except Exception as e:
    print(f"FATAL ERROR: Failed to initialize Bedrock client: {e}")
    bedrock_client = None

# --- Data Loading and Preparation ---
DATA_FILE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "final_data_for_zoho.csv")
)
DATA_CONTEXT = ""
DF_SALES = None


def load_and_prepare_data():
    """
    Loads the CSV sales data, cleans it, and generates a detailed data context.
    """
    global DF_SALES, DATA_CONTEXT
    print("\n--- Starting Data Loading Sequence ---")
    print(f"Loading data from: {DATA_FILE_PATH}")

    try:
        if not os.path.exists(DATA_FILE_PATH):
            raise FileNotFoundError(f"CSV file not found at: {DATA_FILE_PATH}")
        if not os.access(DATA_FILE_PATH, os.R_OK):
            raise PermissionError(f"No read permission for: {DATA_FILE_PATH}")

        df = pd.read_csv(
            DATA_FILE_PATH, encoding="utf-8", engine="python", on_bad_lines="warn"
        )
        df.columns = df.columns.str.strip()

        required_cols = ["quantity", "retailPrice", "purchaseValue", "year", "category"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: '{col}'")

        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
        df["retailPrice"] = pd.to_numeric(df["retailPrice"], errors="coerce").fillna(0)
        df["purchaseValue"] = pd.to_numeric(
            df["purchaseValue"], errors="coerce"
        ).fillna(0)
        df["year"] = (
            pd.to_numeric(df["year"], errors="coerce")
            .fillna(2025)
            .astype(int)
            .astype(str)
        )

        DF_SALES = df

        buffer = io.StringIO()
        buffer.write("## OFM Retail Data Guide\n\n")
        buffer.write(
            "This guide explains the structure and meaning of the available dataset.\n\n"
        )
        buffer.write("### Column Definitions\n")
        buffer.write(
            "- **articleGroupDescription**: The type or group of the clothing article.\n"
        )
        buffer.write("- **brandDescription**: The name of the brand.\n")
        buffer.write(
            "- **season**: The season the article belongs to ('Summer' or 'Winter').\n"
        )
        buffer.write("- **quantity**: The number of units.\n")
        buffer.write("- **retailPrice**: The revenue generated per unit.\n")
        buffer.write("- **purchaseValue**: The cost to acquire one unit.\n")
        buffer.write(
            "- **Inhouse_Brand**: Indicates if the brand is internal or external.\n"
        )
        buffer.write("- **year**: The year the data pertains to.\n")
        buffer.write("- **category**: Defines what the row represents.\n\n")
        buffer.write("### Category Explanations\n")
        buffer.write(
            "The `category` column is critical. Always filter by a specific category.\n"
        )
        buffer.write("- **'Sales'**: Historical sales for **2024**.\n")
        buffer.write("- **'Forecasted Sales'**: Sales forecast for **2025**.\n")
        buffer.write(
            "- **'Leftover Inventory'**: Forecasted unsold items at the **end of 2025**.\n"
        )
        buffer.write(
            "- **'Lost Sales Opportunity'**: Estimated missed sales in **2024** due to stockouts.\n\n"
        )

        for category in df["category"].unique():
            cat_df = df[df["category"] == category]
            total_qty = int(cat_df["quantity"].sum())
            total_retail = cat_df["retailPrice"].sum()
            buffer.write(
                f"- **{category}**: Total Qty = `{total_qty:,}`, Total Retail Price = `${total_retail:,.2f}`\n"
            )

        article_groups = df["articleGroupDescription"].dropna().unique()
        brand_names = df["brandDescription"].dropna().unique()
        buffer.write("\n#### Unique Values Counts:\n")
        buffer.write(f"- Unique articleGroupDescription: {len(article_groups)}\n")
        buffer.write(f"- Unique brandDescription: {len(brand_names)}\n")

        DATA_CONTEXT = buffer.getvalue()
        print("✅ Data loaded and preparation complete.")

    except Exception as e:
        print(f"\n{'=' * 20} DATA LOADING ERROR {'=' * 20}")
        print(f"Error loading data: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("=" * 67 + "\n")
        DF_SALES, DATA_CONTEXT = None, ""


load_and_prepare_data()


# --- AI Response Parsing ---
def parse_ai_response(bot_response_text: str) -> ChatResponse:
    def clean_json_response(text: str) -> str:
        # Search for content inside <json> tags or markdown code blocks
        match_xml = re.search(r"<json>\s*(\{.*?\})\s*</json>", text, re.DOTALL)
        if match_xml:
            return match_xml.group(1)
        match_md = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match_md:
            return match_md.group(1)
        # As a fallback, try to find any JSON object in the string
        match_json = re.search(r"(\{.*?\})", text, re.DOTALL)
        if match_json:
            return match_json.group(1)
        return text.strip()

    def format_chart_data(chart_object: dict) -> ChartData:
        x_col = chart_object.get("x_axis_column")
        y_cols = chart_object.get("y_axis_columns", [])
        chart_data_list = chart_object.get("data", [])

        if not all([x_col, y_cols, isinstance(chart_data_list, list)]):
            raise ValueError("Invalid chart object structure")

        return ChartData(
            chartData=[
                {"name": row.get(x_col), **{y: row.get(y) for y in y_cols}}
                for row in chart_data_list
            ],
            dataKeys=y_cols,
            chartType=chart_object.get("chart_type", "bar"),
        )

    cleaned_text = clean_json_response(bot_response_text)
    try:
        model_json = json.loads(cleaned_text)
        if (
            "x_axis_column" in model_json
            and "y_axis_columns" in model_json
            and "data" in model_json
        ):
            chart_data = format_chart_data(model_json)
            return ChatResponse(
                sender="bot",
                text=model_json.get(
                    "text_summary", "Here is the visualization you requested:"
                ),
                type="chart",
                **chart_data.model_dump(),
            )
    except (json.JSONDecodeError, TypeError, ValueError):
        # If JSON parsing fails or the structure is wrong, return as plain text
        return ChatResponse(sender="bot", text=bot_response_text, type="text")

    # Fallback for cases where it's JSON but not a chart
    return ChatResponse(sender="bot", text=bot_response_text, type="text")


# --- Main Service Function ---
async def get_ai_response(query: str, history: List[HistoryMessage]) -> ChatResponse:
    if not bedrock_client:
        raise HTTPException(status_code=503, detail="Bedrock client is not available.")
    if DF_SALES is None or DF_SALES.empty:
        raise HTTPException(status_code=503, detail="Sales data is not loaded.")

    csv_data = DF_SALES.to_csv(index=False)

    # --- UPDATED SYSTEM PROMPT ---
    # This is the key change. We now explicitly tell the model how to generate charts.
    system_prompt = f"""
You are 'InsightAI', a data analyst AI for the OFM retail team. Your single and ONLY task is to answer questions based on the data provided below.

**CRITICAL RULES - FOLLOW EXACTLY:**
1.  **SOURCE OF TRUTH:** Answer ONLY using the "OFM Retail Data Guide" and the "Full Dataset (CSV Format)".
2.  **NO HALLUCINATIONS:** If the answer is not in the data, state that clearly. DO NOT invent data.
3.  **NEVER MIX CATEGORIES:** Data has four categories (`Sales`, `Forecasted Sales`, `Leftover Inventory`, `Lost Sales Opportunity`). Never combine them. Infer the category from the user's query.
4.  **RESPONSE FORMAT:**
    - For regular text answers, respond in clear, professional business language.
    - **For visualization requests (chart, graph, plot, etc.), you MUST respond with a JSON object.**

**JSON FORMAT FOR CHARTS:**
When a user asks for a chart, you MUST generate a JSON object with the following structure. Provide a brief text summary in the `text_summary` field. Wrap the entire JSON in `<json>` tags.

```json
<json>
{{
  "chart_type": "bar",
  "x_axis_column": "articleGroupDescription",
  "y_axis_columns": ["quantity", "retailPrice"],
  "data": [
    {{"articleGroupDescription": "T-shirt SS", "quantity": 83425, "retailPrice": 3338414.46}},
    {{"articleGroupDescription": "Polo SS", "quantity": 80812, "retailPrice": 4886982.84}},
    {{"articleGroupDescription": "Jeans", "quantity": 42209, "retailPrice": 4723374.60}}
  ],
  "text_summary": "Here is a chart showing the top 3 forecasted article groups for 2025 by quantity and retail price."
}}
</json>
```

- `chart_type`: Can be 'bar', 'line', 'pie', etc.
- `x_axis_column`: The column name for the X-axis labels.
- `y_axis_columns`: A list of one or more column names for the Y-axis values.
- `data`: A list of dictionaries representing the data points for the chart.
- `text_summary`: A short, user-friendly summary of the chart's findings.

---
**OFM RETAIL DATA GUIDE:**
<data_guide>
{DATA_CONTEXT}
</data_guide>
---
**FULL DATASET (CSV FORMAT):**
<dataset>
{csv_data}
</dataset>
---
"""

    anthropic_messages = []
    for msg in history:
        role = "assistant" if msg.role == "bot" else msg.role
        anthropic_messages.append({"role": role, "content": msg.content})

    anthropic_messages.append({"role": "user", "content": query})

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "system": system_prompt,
        "temperature": 0.0,
        "messages": anthropic_messages,
    }

    try:
        response = bedrock_client.invoke_model(
            body=json.dumps(body),
            modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response.get("body").read())
        bot_response_text = response_body.get("content", [{}])[0].get("text", "")

        return parse_ai_response(bot_response_text)

    except Exception as e:
        print(f"🚨 Error communicating with AI model: {e}")
        traceback.print_exc()
        raise ValueError(f"Error communicating with AI model: {e}")
