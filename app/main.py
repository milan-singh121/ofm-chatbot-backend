import json
import warnings
from typing import List

warnings.filterwarnings("ignore")
from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from .services import get_ai_response, load_and_prepare_data
from .models import ChatResponse, HistoryMessage

# --- App Initialization ---
app = FastAPI(
    title="OFM Sales Insights AI",
    description="API for analyzing OFM sales and inventory data with an intelligent agent.",
    version="4.0.0",
)

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://ofm-chatbot.netlify.app",
    ],  # IMPORTANT: For production, restrict this to your frontend's actual URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Event Handlers ---
@app.on_event("startup")
async def startup_event():
    """
    Load data when the application starts.
    """
    print("Server starting up...")
    load_and_prepare_data()


# --- Exception Handlers ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Custom handler for Pydantic validation errors.
    """
    serializable_errors = [
        {"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()
    ]
    print(f"Caught validation error: {serializable_errors}")
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation Error", "errors": serializable_errors},
    )


# --- API Endpoints ---
@app.post("/api/chat", response_model=ChatResponse)
async def chat_handler(
    query: str = Form(...),
    conversation_id: str = Form(...),
    history: str = Form(...),
):
    """
    Main chat endpoint that receives user queries and conversation history.
    """
    try:
        # The history from frontend is a JSON string, so we parse it.
        history_data = json.loads(history)
        # Pydantic models validate the structure of each message in the history.
        history_messages = [HistoryMessage(**message) for message in history_data]

        # Call the core AI service function to get a response.
        response = await get_ai_response(query, history_messages)
        return response
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid history JSON format.")
    except Exception as e:
        # Catch-all for any other unexpected errors in the pipeline.
        print(f"Error in chat_handler: {e}")
        # It's better to return a generic error message to the user.
        raise HTTPException(
            status_code=500, detail="An internal server error occurred."
        )


@app.get("/")
async def read_root():
    """
    Root endpoint for health checks.
    """
    return {"message": "OFM Sales Insights AI Backend is running."}
