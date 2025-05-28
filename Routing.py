import os
import json 
from google import genai
import enum
from dotenv import load_dotenv
from pydantic import BaseModel
load_dotenv()


client= genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class Category(enum.Enum):
    WEATHER = "weather"
    UNKNOWN = "unknown"
    SIENCE = "sience"


class ResponseSchema(BaseModel):
    category: Category
    reasoning: str
    
user_query = "What is the weather like in New York today?"

prompt_router = f"""
Analyze the user query below and determine its category.
Categories:
- weather: For questions about weather conditions.
- science: For questions about science.
- unknown: If the category is unclear.
 
Query: {user_query}
"""

response_router = client.models.generate_content(
    model = 'gemini-2.0-flash',
    contents = prompt_router,
    config = {
        'response_mime_type': 'application/json',
        'response_schema': ResponseSchema,
        }
)

print(f"Routing Decision: Category={response_router.parsed.category}, Reasoning={response_router.parsed.reasoning}")

if response_router.parsed.category == Category.WEATHER:
    prompt_weather = f"""
    Provide a detailed weather report for New York City today.
    """
    
    response_weather = client.models.generate_content(
        model = 'gemini-2.0-flash',
        contents = prompt_weather,
        config = {
            'response_mime_type': 'application/json',
            }
    )
    
    print(f"Weather Report: {response_weather.text}")
    
elif response_router.parsed.category == Category.SIENCE:
    prompt_science = f"""
    Provide a detailed explanation of a scientific concept related to the query.
    """
    
    response_science = client.models.generate_content(
        model = 'gemini-2.0-flash',
        contents = prompt_science,
        config = {
            'response_mime_type': 'application/json',
            }
    )
    
    print(f"Science Explanation: {response_science.text}")
    
else:
    print("The query category is unknown. No further action taken.")