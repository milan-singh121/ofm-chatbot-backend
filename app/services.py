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


def execute_python_code(code: str) -> dict:
    """
    Safely executes Python code generated by the AI.
    """
    if DF_SALES is None:
        return {"error": "DataFrame not loaded."}

    print(f"🐍 Executing AI-generated code:\n---\n{code}\n---")

    scope = {"df": DF_SALES.copy(), "pd": pd}

    try:
        exec(code, scope)
        if "result" in scope:
            output = scope["result"]
            if isinstance(output, pd.DataFrame):
                return {"data": output.to_dict(orient="records")}
            else:
                if hasattr(output, "item"):
                    return {"data": output.item()}
                return {"data": output}
        else:
            return {"message": "Code executed, but no 'result' variable was found."}

    except Exception as e:
        print(f"💥 Code Execution Error: {e}")
        traceback.print_exc()
        return {"error": f"Error executing code: {type(e).__name__}: {e}"}


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
# --- Main Service Function ---
async def get_ai_response(query: str, history: List[HistoryMessage]) -> ChatResponse:
    if not bedrock_client:
        raise HTTPException(status_code=503, detail="Bedrock client is not available.")
    if DF_SALES is None or DF_SALES.empty:
        raise HTTPException(status_code=503, detail="Sales data is not loaded.")

    # This agentic prompt instructs the AI to generate code first.
    system_prompt = f"""
You are 'InsightAI', a data analyst AI. Your ONLY task is to answer questions by writing Python code to analyze a pandas DataFrame that is ALREADY loaded and available to you under the name `df`.

**CRITICAL RULES - FOLLOW EXACTLY:**
1.  **DO NOT write any code to load data (e.g., `pd.read_csv`). The DataFrame `df` is ALWAYS available to you.**
2.  Analyze the user's query to understand the required analysis. This may involve filtering by multiple columns (e.g., `category` and `season`).
3.  Write Python code using the `df` DataFrame. Your code must be valid and runnable.
4.  Your final line of code MUST assign the result to a variable named `result`.
5.  You MUST respond *only* with a JSON object wrapped in ```json ... ```. The JSON must contain two keys: "thought" (your reasoning) and "code" (your Python code).

**DATA CONTEXT (Schema of the `df` DataFrame):**
{DATA_CONTEXT}
"""

    anthropic_messages = []
    for msg in history:
        anthropic_messages.append({"role": msg.role, "content": msg.content})
    anthropic_messages.append({"role": "user", "content": query})

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "system": system_prompt,
        "temperature": 0.0,
        "messages": anthropic_messages,
    }

    try:
        # Step 1: Get the AI to generate the Python code
        response = bedrock_client.invoke_model(
            body=json.dumps(body),
            modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response.get("body").read())
        ai_response_text = response_body.get("content", [{}])[0].get("text", "")

        json_part_match = re.search(
            r"```json\s*(\{.*?\})\s*```", ai_response_text, re.DOTALL
        )

        if not json_part_match:
            # If the AI gives a direct answer, return it
            return ChatResponse(sender="assistant", text=ai_response_text, type="text")

        json_str = json_part_match.group(1)
        ai_json = json.loads(json_str, strict=False)
        code_to_execute = ai_json.get("code")

        if not code_to_execute:
            raise ValueError("AI response did not contain code to execute.")

        # Step 2: Execute the generated code
        execution_result = execute_python_code(code_to_execute)

        # Step 3: Ask the AI to summarize the result
        anthropic_messages.append({"role": "assistant", "content": ai_response_text})

        final_prompt = f"""
I have executed your code. The result is in the JSON below.
Your task is to now act as a friendly business analyst and provide a final, user-friendly answer to the original user query, based ONLY on this data.

**CRITICAL INSTRUCTIONS FOR THIS FINAL RESPONSE:**
- DO NOT mention the code, the execution, or the JSON structure.
- If the execution result contains an error, apologize and state that a technical error occurred.
- Speak in natural language.
- If the original query asked for a chart or visualization, provide the necessary JSON object for the chart wrapped in `<json></json>` tags in your response.

**Execution Result:**
```json
{json.dumps(execution_result, indent=2)}
```
"""
        anthropic_messages.append({"role": "user", "content": final_prompt})

        body["messages"] = anthropic_messages

        final_response = bedrock_client.invoke_model(
            body=json.dumps(body),
            modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
            accept="application/json",
            contentType="application/json",
        )
        final_response_body = json.loads(final_response.get("body").read())
        final_bot_text = final_response_body.get("content", [{}])[0].get("text", "")

        return parse_ai_response(final_bot_text)

    except Exception as e:
        print(f"🚨 Error communicating with AI model: {e}")
        traceback.print_exc()
        raise ValueError(f"Error communicating with AI model: {e}")
