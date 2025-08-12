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

        # Prepare data context string for grounding the AI
        buffer = io.StringIO()
        buffer.write("### OFM Sales & Inventory Data Profile (Year 2025)\n\n")
        buffer.write(
            f"This dataset contains **{df.shape[0]} rows** and **{df.shape[1]} columns** "
            "detailing sales and inventory data for the fashion company OFM.\n\n"
        )
        buffer.write("#### Key Data Categories:\n")
        buffer.write("The 'category' column indicates rows for:\n")
        buffer.write("- **Forecasted Sales:** Sales predictions for 2025.\n")
        buffer.write("- **Leftover Inventory:** Unsold stock forecasted for 2025.\n")
        buffer.write(
            "- **Lost Sales Opportunity:** Missed sales due to stockouts for the year 2025.\n"
        )
        buffer.write("- **Actual Sales:** Historical sales data for 2024.\n\n")
        buffer.write("#### Core Columns:\n")
        buffer.write(
            "- **articleGroupDescription:** Article or Type of clothing, e.g., Jacket, Trousers.\n"
        )
        buffer.write(
            "- **brandDescription:** Brand name, e.g., The BLUEPRINT Premium.\n"
        )
        buffer.write("- **season:** Summer or Winter.\n")
        buffer.write("- **quantity, retailPrice, purchaseValue, year**\n\n")
        buffer.write("#### Summary Stats by Category:\n")

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
Human: You are 'InsightAI', an expert data analyst AI designed exclusively for the OFM retail team. OFM is a large fashion company operating numerous offline stores and a growing online store. Your entire knowledge base is the detailed sales and inventory data provided below, which encompasses both offline and online sales channels for the years 2024 and 2025.

You are speaking to retail managers, supply chain analysts, and business leaders at OFM who are familiar with retail operations but want clear, concise data-driven insights to make informed decisions.

**CRITICAL GUIDELINES:**
1. Base all answers 100% on the provided **Data Context**.
2. Understand and distinguish between:
   - **Offline store sales**
   - **Online store sales**
   - **Forecasted sales for 2025**
   - **Leftover inventory** from previous years
   - **Lost sales opportunities** due to stockouts.
3. Use terminology consistent with retail analytics, inventory management, and omnichannel sales.
4. Provide actionable insights and recommendations where appropriate.
5. Deliver concise, clear, and professional answers suitable for OFM retail teams.
6. Never reveal raw data, code, or internal JSON; synthesize the results naturally.
7. If the data doesn't support an answer, state that explicitly.

**OUTPUT FORMATTING:**
- Natural language text answers suitable for OFM retail teams.
- For visualization requests, only respond with JSON object inside <json></json> tags with keys: chart_type, x_axis_column, y_axis_columns, data, text_summary.

**Data Context from OFM sales and inventory data:**
<data_context>
{DATA_CONTEXT}
</data_context>
"""

    anthropic_messages = [
        {"role": "user", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Hello OFM team. I am InsightAI, your dedicated retail data analyst. How can I assist you with our sales and inventory insights today?",
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
