from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# Pydantic models define the data structures for API requests and responses.
# They provide automatic validation and documentation, ensuring data consistency.


class HistoryMessage(BaseModel):
    """
    Represents a single message in the conversation history sent from the frontend.
    """

    # The role must be either 'user' or 'bot'.
    role: Literal["user", "bot"] = Field(
        ..., description="The role of the message sender."
    )
    content: str = Field(..., description="The text content of the message.")


class ChartData(BaseModel):
    """
    Defines a standardized structure for chart visualizations.
    This ensures the frontend knows how to render the data.
    """

    # The actual data points for the chart.
    # Example: [{'name': 'Summer', 'Sales': 5000}]
    chartData: List[dict] = Field(
        ...,
        description="A list of data points for the chart.",
    )
    # A list of keys used for the Y-axis values.
    # Example: ['Sales', 'Forecast']
    dataKeys: List[str] = Field(
        ...,
        description="List of keys for the Y-axis values.",
    )
    # The type of chart to render.
    chartType: Optional[str] = Field(
        "bar", description="Type of chart (e.g., 'bar', 'line', 'pie')."
    )


class ChatResponse(BaseModel):
    """
    The main response model sent from the backend to the frontend.
    This structure is flexible to handle plain text, charts, or errors.
    """

    # Indicates the message is always from the 'bot'.
    sender: Literal["bot"] = Field(
        "bot", description="Indicates the message is from the bot."
    )
    # The primary text response from the bot.
    text: str = Field(..., description="The primary text response from the bot.")
    # The type of the response, which tells the frontend how to display it.
    type: str = Field(
        "text", description="The type of the response ('text', 'chart', 'error')."
    )

    # The following fields are optional and are used only for 'chart' type responses.
    # We embed the ChartData fields directly for a flatter, simpler JSON structure.
    chartData: Optional[List[dict]] = None
    dataKeys: Optional[List[str]] = None
    chartType: Optional[str] = None

    class Config:
        # Allows other potential fields to be included without causing validation errors.
        extra = "allow"
