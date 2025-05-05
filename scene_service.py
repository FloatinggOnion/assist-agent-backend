import os
import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SceneService:
    def __init__(self):
        self.scenes_dir = "scenes"
        self.ensure_scenes_directory()
        self.model = genai.GenerativeModel('models/gemini-1.5-pro-001')
        
    def ensure_scenes_directory(self):
        """Ensure the scenes directory exists."""
        if not os.path.exists(self.scenes_dir):
            os.makedirs(self.scenes_dir)
            
    def save_screenshot(self, image_url: str) -> Dict:
        """Save a screenshot from the given image URL."""
        try:
            # Create a unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scene_{timestamp}.jpg"
            filepath = os.path.join(self.scenes_dir, filename)
            
            # Download and save the image
            import requests
            response = requests.get(image_url)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                return {
                    "status": "success",
                    "filepath": filepath,
                    "timestamp": timestamp
                }
            else:
                return {"status": "error", "message": "Failed to download image"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
            
    def get_daily_scenes(self, date: Optional[str] = None) -> List[str]:
        """Get all scenes from a specific date or today if no date provided."""
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
            
        scenes = []
        for filename in os.listdir(self.scenes_dir):
            if filename.startswith(f"scene_{date}"):
                scenes.append(os.path.join(self.scenes_dir, filename))
        return scenes
        
    def describe_scene(self, image_url: str) -> Dict:
        """Describe a single scene from the provided image URL."""
        try:
            response = self.model.generate_content(
                f"Describe this scene in detail: {image_url}",
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 40
                }
            )
            return {
                "status": "success",
                "description": response.text,
                "source": "provided_image"
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
            
    def get_daily_recap(self, date: Optional[str] = None) -> Dict:
        """Get a comprehensive description of all scenes from a specific date."""
        try:
            # Get scenes from the specified date
            scenes = self.get_daily_scenes(date)
            if not scenes:
                return {
                    "status": "error",
                    "message": "No scenes found for the specified date"
                }
            
            # Create a prompt with all scenes
            prompt = "Here are several scenes from the same day. Please provide a comprehensive description of what happened throughout the day based on these scenes:\n"
            for scene in scenes:
                prompt += f"Scene: {scene}\n"
            
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 40
                }
            )
            
            return {
                "status": "success",
                "description": response.text,
                "source": "daily_recap",
                "scenes_used": scenes
            }
        except Exception as e:
            return {"status": "error", "message": str(e)} 