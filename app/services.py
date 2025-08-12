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
    Loads the CSV sales data, cleans it, and generates a detailed data context with:
    - Summary stats
    - Unique counts of articles and brands
    - Lists of unique articles and brands for the AI's grounding
    """
    global DF_SALES, DATA_CONTEXT
    print("\n--- Starting Data Loading Sequence ---")
    print(f"Loading data from: {DATA_FILE_PATH}")

    try:
        if not os.path.exists(DATA_FILE_PATH):
            raise FileNotFoundError(f"CSV file not found at: {DATA_FILE_PATH}")
        if not os.access(DATA_FILE_PATH, os.R_OK):
            raise PermissionError(f"No read permission for: {DATA_FILE_PATH}")

        # Load dataframe
        df = pd.read_csv(
            DATA_FILE_PATH, encoding="utf-8", engine="python", on_bad_lines="warn"
        )

        # Clean column headers
        df.columns = df.columns.str.strip()

        # Validate required columns
        required_cols = ["quantity", "retailPrice", "purchaseValue", "year", "category"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: '{col}'")

        # Clean and convert types
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

        # --- Generate the Detailed Data Guide ---
        buffer = io.StringIO()
        buffer.write("## OFM Retail Data Guide\n\n")
        buffer.write(
            "This guide explains the structure and meaning of the available dataset.\n\n"
        )

        # Column Definitions
        buffer.write("### Column Definitions\n")
        buffer.write(
            "- **articleGroupDescription**: The type or group of the clothing article (e.g., 'Jacket', 'Trousers').\n"
        )
        buffer.write("- **brandDescription**: The name of the brand.\n")
        buffer.write(
            "- **season**: The season the article belongs to ('Summer' or 'Winter').\n"
        )
        buffer.write("- **quantity**: The number of units for the article.\n")
        buffer.write(
            "- **retailPrice**: The revenue generated per unit (sale price).\n"
        )
        buffer.write(
            "- **purchaseValue**: The cost to acquire one unit of the article.\n"
        )
        buffer.write(
            "- **Inhouse_Brand**: Indicates if the brand is an internal OFM brand or an external one.\n"
        )
        buffer.write("- **year**: The year the data pertains to.\n")
        buffer.write(
            "- **category**: The most important column for context, defining what the row represents.\n\n"
        )

        # Category Explanations
        buffer.write("### Category Explanations\n")
        buffer.write(
            "The `category` column is critical. You must always filter by a specific category based on the user's query. The categories are:\n"
        )
        buffer.write(
            "- **'Sales'**: Represents historical sales data for the year **2024**.\n"
        )
        buffer.write(
            "- **'Forecasted Sales'**: Represents the sales forecast for the upcoming year **2025**.\n"
        )
        buffer.write(
            "- **'Leftover Inventory'**: Represents the forecasted number of unsold items at the **end of 2025**.\n"
        )
        buffer.write(
            "- **'Lost Sales Opportunity'**: Represents the estimated quantity of sales missed in **2024** due to stockouts.\n\n"
        )

        for category in df["category"].unique():
            cat_df = df[df["category"] == category]
            total_qty = int(cat_df["quantity"].sum())
            total_retail = cat_df["retailPrice"].sum()
            buffer.write(
                f"- **{category}**: Total Quantity = `{total_qty:,}`, Total Retail Price = `${total_retail:,.2f}`\n"
            )

        # Add unique counts and lists for articles and brands:
        article_groups = df["articleGroupDescription"].dropna().unique()
        brand_names = df["brandDescription"].dropna().unique()

        buffer.write("\n#### Unique Values Counts:\n")
        buffer.write(
            f"- Number of unique articleGroupDescription: {len(article_groups)}\n"
        )
        buffer.write(f"- Number of unique brandDescription: {len(brand_names)}\n\n")

        # Optionally include lists, truncated if too long for prompt
        MAX_LIST_ITEMS = 50

        if len(article_groups) > 0:
            articles_list = ", ".join(sorted(article_groups[:MAX_LIST_ITEMS]))
            if len(article_groups) > MAX_LIST_ITEMS:
                articles_list += ", ... (and more)"
            buffer.write(f"- Article types: {articles_list}\n")

        if len(brand_names) > 0:
            brands_list = ", ".join(sorted(brand_names[:MAX_LIST_ITEMS]))
            if len(brand_names) > MAX_LIST_ITEMS:
                brands_list += ", ... (and more)"
            buffer.write(f"- Brands: {brands_list}\n")

        DATA_CONTEXT = buffer.getvalue()

        print("✅ Data loaded and preparation complete.")

    except Exception as e:
        print("\n" + "=" * 20 + " DATA LOADING ERROR " + "=" * 20)
        print(f"Error loading data: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("=" * 67 + "\n")
        DF_SALES, DATA_CONTEXT = None, ""


# Load Data on Module Import
load_and_prepare_data()


# --- AI Response Parsing ---
def parse_ai_response(bot_response_text: str) -> ChatResponse:
    def clean_json_response(text: str) -> str:
        match_xml = re.search(r"<json>\s*(\{.*?\})\s*</json>", text, re.DOTALL)
        if match_xml:
            return match_xml.group(1)
        match_md = re.search(r"``````", text, re.DOTALL)
        if match_md:
            return match_md.group(1)
        return text.strip()

    def format_chart_data(chart_object: dict) -> ChartData:
        x_col, y_cols = (
            chart_object.get("x_axis_column"),
            chart_object.get("y_axis_columns", []),
        )
        chart_data_list = chart_object.get("data", [])
        if not isinstance(chart_data_list, list):
            chart_data_list = []
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
    except (json.JSONDecodeError, TypeError):
        pass
    return ChatResponse(sender="bot", text=bot_response_text, type="text")


# --- Main Service Function ---
async def get_ai_response(query: str, history: List[HistoryMessage]) -> ChatResponse:
    if not bedrock_client:
        raise HTTPException(
            status_code=503,
            detail="Bedrock client is not available. Check server logs.",
        )
    if DF_SALES is None or DF_SALES.empty:
        raise HTTPException(
            status_code=503,
            detail="Sales data is not loaded. Check startup logs for errors.",
        )

    # Updated system prompt tailored for OFM retail team internal users
    system_prompt = f"""
You are 'InsightAI', a data analyst AI for the OFM retail team. Your single and ONLY purpose is to answer questions by analyzing the data provided in the 'OFM Retail Data Guide'. You must adhere to the following rules without exception.

**CRITICAL RULES:**
1.  **STRICT GROUNDING:** Your entire knowledge base is the 'OFM Retail Data Guide' below. You CANNOT use any information not present in this guide. You MUST NOT invent, assume, or hallucinate any data, numbers, or facts.
2.  **DATA-DRIVEN ANSWERS ONLY:** Every part of your answer must be directly supported by the data. If a user's question cannot be answered with the provided data, you MUST respond by saying, "I cannot answer that question as the necessary information is not available in the dataset."
3.  **NEVER MIX CATEGORIES:** The data is split into four distinct categories (`Sales`, `Forecasted Sales`, `Leftover Inventory`, `Lost Sales Opportunity`). You must *never* combine or aggregate data across these categories. Always infer from the user's query which specific category they are interested in. For example, if they ask for "sales," assume they mean 2024 `Sales` unless they specify "forecast."
4.  **PROFESSIONAL TONE:** Your audience is the OFM retail team. Use clear, professional business language.
5.  **NO RAW DATA:** Do not show raw data tables, code, or internal JSON in your final response to the user. Synthesize the findings into natural language.

**OUTPUT FORMATTING:**
-   For text answers, provide a concise, professional summary.
-   For visualization requests, you MUST respond ONLY with a JSON object inside `<json></json>` tags. The JSON must have these keys: `chart_type`, `x_axis_column`, `y_axis_columns`, `data`, `text_summary`.

---
**OFM RETAIL DATA GUIDE (Your ONLY source of truth):**
<data_context>
{DATA_CONTEXT}
</data_context>
---
"""

    anthropic_messages = [
        {"role": "user", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Hello. I am InsightAI, your retail data analyst. How can I assist with our sales and inventory data today?",
                }
            ],
        },
    ]

    # Append conversation history messages properly
    for msg in history:
        role = "assistant" if msg.role == "bot" else "user"
        anthropic_messages.append(
            {"role": role, "content": [{"type": "text", "text": msg.content}]}
        )

    anthropic_messages.append(
        {"role": "user", "content": [{"type": "text", "text": query}]}
    )

    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "temperature": 0.0,
            "top_p": 0.9,
            "messages": anthropic_messages,
        }
    )

    try:
        response = bedrock_client.invoke_model(
            body=body,
            modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response.get("body").read())
        if "content" in response_body and response_body["content"]:
            bot_response_text = response_body["content"][0].get("text")
            return parse_ai_response(bot_response_text)
        else:
            raise ValueError("Unexpected AI model response format.")
    except Exception as e:
        raise ValueError(f"Error communicating with AI model: {e}")
