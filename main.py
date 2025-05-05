import os
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
from face_detection import FaceRecognitionClass
from ocr_service import OCRService
from scene_service import SceneService

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Smart Glasses API")

# Initialize services
face_recognition = FaceRecognitionClass()
ocr_service = OCRService()
scene_service = SceneService()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('models/gemini-1.5-pro-001')

# Pydantic models for request/response
class ImageRequest(BaseModel):
    image_url: str

class FaceResponse(BaseModel):
    matches: List[Dict]

class OCRResponse(BaseModel):
    text_lines: List[str]

class QueryRequest(BaseModel):
    query: str

class SaveFaceRequest(BaseModel):
    image_url: str
    identity: str

class SceneDescriptionRequest(BaseModel):
    image_url: str

class DailyRecapRequest(BaseModel):
    date: Optional[str] = None

# Function definitions for Gemini
functions = [
    {
        "name": "recognize_face",
        "description": "Recognize a face in an image from the stored database. Use this function when someone asks 'who is this?' or 'who is in this image?' or similar questions about identifying a person in an image.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_url": {
                    "type": "string",
                    "description": "URL of the image containing the face to identify"
                }
            },
            "required": ["image_url"]
        }
    },
    {
        "name": "extract_text",
        "description": "Extract text from an image using OCR. Use this function when someone asks to read text from an image or identify text in an image.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_url": {
                    "type": "string",
                    "description": "URL of the image containing text to extract"
                }
            },
            "required": ["image_url"]
        }
    },
    {
        "name": "save_face",
        "description": "Save a new face to the database with a given identity. Use this function when someone wants to add a new person to the face recognition system.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_url": {
                    "type": "string",
                    "description": "URL of the image containing the face to save"
                },
                "identity": {
                    "type": "string",
                    "description": "Name or identifier for the face"
                }
            },
            "required": ["image_url", "identity"]
        }
    },
    {
        "name": "save_screenshot",
        "description": "Save a screenshot/scene from the current view. Use this function when someone wants to capture the current scene or take a screenshot.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_url": {
                    "type": "string",
                    "description": "URL of the image to save as a screenshot"
                }
            },
            "required": ["image_url"]
        }
    },
    {
        "name": "describe_scene",
        "description": "Describe a single scene from an image. Use this function when someone asks 'what's in this image?' or similar questions about describing a specific scene.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_url": {
                    "type": "string",
                    "description": "URL of the image to describe"
                }
            },
            "required": ["image_url"]
        }
    },
    {
        "name": "get_daily_recap",
        "description": "Get a comprehensive description of all scenes from a specific date. Use this function when someone asks 'what happened today?' or 'what happened yesterday?' or similar questions about daily activities.",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Date to get scenes from (optional, format: YYYYMMDD)"
                }
            }
        }
    }
]

# API endpoints
@app.post("/recognize_face", response_model=FaceResponse)
async def recognize_face(request: ImageRequest):
    """Recognize a face in the given image."""
    matches = face_recognition.find_face(request.image_url)
    return FaceResponse(matches=matches)

@app.post("/extract_text", response_model=OCRResponse)
async def extract_text(request: ImageRequest):
    """Extract text from the given image."""
    text_lines = ocr_service.extract_text(request.image_url)
    return OCRResponse(text_lines=text_lines)

@app.post("/save_face")
async def save_face(request: SaveFaceRequest):
    """Save a new face to the database."""
    success = face_recognition.add_face(request.image_url, request.identity)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to save face")
    return {"status": "success", "message": f"Face saved as {request.identity}"}

@app.post("/save_screenshot")
async def save_screenshot(request: ImageRequest):
    """Save a screenshot from the given image."""
    result = scene_service.save_screenshot(request.image_url)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result

@app.post("/describe_scene")
async def describe_scene(request: SceneDescriptionRequest):
    """Describe a single scene from the provided image."""
    result = scene_service.describe_scene(request.image_url)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result

@app.post("/daily_recap")
async def get_daily_recap(request: DailyRecapRequest):
    """Get a comprehensive description of all scenes from a specific date."""
    result = scene_service.get_daily_recap(request.date)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result

@app.post("/query")
async def process_query(request: QueryRequest):
    """Process a natural language query and determine which function to call."""
    try:
        # Generate function calling response from Gemini
        response = model.generate_content(
            request.query,
            generation_config={
                "temperature": 0,
                "top_p": 0.95,
                "top_k": 40
            },
            tools=[{"function_declarations": functions}]
        )
        
        # Extract function call details
        function_call = response.candidates[0].content.parts[0].function_call
        
        if function_call.name == "recognize_face":
            matches = face_recognition.find_face(function_call.args["image_url"])
            if not matches:
                return {
                    "function": "recognize_face",
                    "result": {"status": "not_found", "message": "No matching faces found in the database"}
                }
            return {
                "function": "recognize_face",
                "result": {
                    "status": "success",
                    "matches": matches,
                    "message": f"Found {len(matches)} potential matches"
                }
            }
        elif function_call.name == "extract_text":
            text_lines = ocr_service.extract_text(function_call.args["image_url"])
            return {
                "function": "extract_text",
                "result": {
                    "status": "success",
                    "text": text_lines
                }
            }
        elif function_call.name == "save_face":
            success = face_recognition.add_face(
                function_call.args["image_url"],
                function_call.args["identity"]
            )
            if not success:
                raise HTTPException(status_code=400, detail="Failed to save face")
            return {
                "function": "save_face",
                "result": {
                    "status": "success",
                    "message": f"Face saved as {function_call.args['identity']}"
                }
            }
        elif function_call.name == "save_screenshot":
            result = scene_service.save_screenshot(function_call.args["image_url"])
            if result["status"] == "error":
                raise HTTPException(status_code=400, detail=result["message"])
            return {
                "function": "save_screenshot",
                "result": result
            }
        elif function_call.name == "describe_scene":
            result = scene_service.describe_scene(function_call.args["image_url"])
            if result["status"] == "error":
                raise HTTPException(status_code=400, detail=result["message"])
            return {
                "function": "describe_scene",
                "result": result
            }
        elif function_call.name == "get_daily_recap":
            result = scene_service.get_daily_recap(function_call.args.get("date"))
            if result["status"] == "error":
                raise HTTPException(status_code=400, detail=result["message"])
            return {
                "function": "get_daily_recap",
                "result": result
            }
        else:
            raise HTTPException(status_code=400, detail="Unsupported function")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
