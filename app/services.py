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
    Loads the CSV sales data and generates a highly detailed and structured
    Data Guide for the AI to use as its only source of truth.
    """
    global DF_SALES, DATA_CONTEXT
    print("\n--- Starting Data Loading Sequence ---")
    print(f"Loading data from: {DATA_FILE_PATH}")

    try:
        if not os.path.exists(DATA_FILE_PATH):
            raise FileNotFoundError(f"CSV file not found at: {DATA_FILE_PATH}")

        df = pd.read_csv(
            DATA_FILE_PATH, encoding="utf-8", engine="python", on_bad_lines="warn"
        )
        df.columns = df.columns.str.strip()

        # Data cleaning
        for col in ["quantity", "retailPrice", "purchaseValue"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        if "year" in df.columns:
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

        DATA_CONTEXT = buffer.getvalue()
        print("✅ Data loaded and detailed Data Guide prepared.")

    except Exception as e:
        print(f"\n🚨 DATA LOADING ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        DF_SALES, DATA_CONTEXT = None, ""


# Load Data on Module Import
load_and_prepare_data()


# --- AI Response Parsing ---
def parse_ai_response(bot_response_text: str) -> ChatResponse:
    """
    Parses the AI's response, extracting chart JSON if present.
    """
    json_match = re.search(
        r"<json>\s*(\{.*?\})\s*</json>", bot_response_text, re.DOTALL
    )

    if json_match:
        json_str = json_match.group(1)
        try:
            chart_object = json.loads(json_str)
            if "x_axis_column" in chart_object and "y_axis_columns" in chart_object:
                chart_data_list = chart_object.get("data", [])
                formatted_chart = ChartData(
                    chartData=[
                        {
                            "name": row.get(chart_object["x_axis_column"]),
                            **{y: row.get(y) for y in chart_object["y_axis_columns"]},
                        }
                        for row in chart_data_list
                    ],
                    dataKeys=chart_object["y_axis_columns"],
                    chartType=chart_object.get("chart_type", "bar"),
                )
                return ChatResponse(
                    sender="assistant",
                    text=chart_object.get(
                        "text_summary", "Here is the visualization you requested:"
                    ),
                    type="chart",
                    **formatted_chart.model_dump(),
                )
        except (json.JSONDecodeError, TypeError):
            pass

    # If no valid chart JSON is found, return a plain text response.
    return ChatResponse(sender="assistant", text=bot_response_text, type="text")


# --- Main Service Function ---
async def get_ai_response(query: str, history: List[HistoryMessage]) -> ChatResponse:
    if not bedrock_client:
        raise HTTPException(status_code=503, detail="Bedrock client is not available.")
    if DF_SALES is None or DF_SALES.empty:
        raise HTTPException(status_code=503, detail="Sales data is not loaded.")

    # A more forceful and detailed system prompt to strictly ground the model.
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

    for msg in history:
        # The API expects 'assistant' for the AI's role.
        role = "assistant" if msg.role == "assistant" else "user"
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
            "temperature": 0.0,  # Set to 0 for deterministic, fact-based responses
            "top_p": 0.9,
            "messages": anthropic_messages,
        }
    )

    try:
        # This part of the code is not used in the agentic workflow, but kept for reference
        # response = bedrock_client.invoke_model(
        #     body=body,
        #     modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
        #     accept="application/json",
        #     contentType="application/json",
        # )
        # response_body = json.loads(response.get("body").read())
        # if "content" in response_body and response_body["content"]:
        #     bot_response_text = response_body["content"][0].get("text")
        #     return parse_ai_response(bot_response_text)
        # else:
        #     raise ValueError("Unexpected AI model response format.")

        # This is a simplified placeholder for the agentic execution logic
        # In a real scenario, you would have the AI generate pandas code here
        # For now, we simulate a direct response based on the grounded prompt

        # This is a placeholder for a more complex agentic response
        # For now, we will just return a simple text response to demonstrate the prompt is working
        # In a real agentic setup, you would parse the AI's code, execute it, and then get a final summary

        # For demonstration, we'll just send the query to the model with the strong prompt
        # and let it generate a direct answer without the code execution step.
        # This simplifies the flow while still benefiting from the strong grounding.

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
        print(f"🚨 Error communicating with AI model: {e}")
        traceback.print_exc()
        raise ValueError(f"Error communicating with AI model: {e}")
