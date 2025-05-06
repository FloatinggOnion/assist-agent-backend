import os
import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
import google.generativeai as genai
from dotenv import load_dotenv
import time

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
            
            # Analyze each scene individually with rate limiting
            scene_descriptions = []
            for scene in scenes:
                try:
                    with open(scene, 'rb') as f:
                        image_data = f.read()
                        response = self.model.generate_content(
                            [
                                "Describe this scene in detail from my own point of view, as they are my experiences. Use pronouns like 'You' and 'Your' instead of 'I' and 'My' or 'The narrator' or 'The author':",
                                {"mime_type": "image/jpeg", "data": image_data}
                            ],
                            generation_config={
                                "temperature": 0.7,
                                "top_p": 0.95,
                                "top_k": 40
                            }
                        )
                        scene_descriptions.append(response.text)
                    # Add a delay between API calls to avoid rate limits
                    time.sleep(5)  # Wait 20 seconds between calls
                except Exception as e:
                    # If we hit a rate limit, wait longer and retry
                    if "429" in str(e):
                        time.sleep(30)  # Wait 30 seconds before retrying
                        try:
                            with open(scene, 'rb') as f:
                                image_data = f.read()
                                response = self.model.generate_content(
                                    [
                                        "Describe this scene in detail:",
                                        {"mime_type": "image/jpeg", "data": image_data}
                                    ],
                                    generation_config={
                                        "temperature": 0.7,
                                        "top_p": 0.95,
                                        "top_k": 40
                                    }
                                )
                                scene_descriptions.append(response.text)
                        except Exception as retry_e:
                            return {"status": "error", "message": f"Failed to analyze scene after retry: {str(retry_e)}"}
                    else:
                        return {"status": "error", "message": f"Failed to analyze scene: {str(e)}"}
            
            if not scene_descriptions:
                return {"status": "error", "message": "Failed to analyze any scenes"}
            
            # Combine all scene descriptions and ask for a comprehensive summary
            combined_prompt = "Here are descriptions of several scenes from the same day. Please provide a comprehensive summary of what happened throughout the day based on these scenes:\n\n"
            for i, desc in enumerate(scene_descriptions, 1):
                combined_prompt += f"Scene {i}:\n{desc}\n\n"
            
            try:
                final_response = self.model.generate_content(
                    combined_prompt,
                    generation_config={
                        "temperature": 0.7,
                        "top_p": 0.95,
                        "top_k": 40
                    }
                )
            except Exception as e:
                if "429" in str(e):
                    time.sleep(30)  # Wait 30 seconds before retrying
                    try:
                        final_response = self.model.generate_content(
                            combined_prompt,
                            generation_config={
                                "temperature": 0.7,
                                "top_p": 0.95,
                                "top_k": 40
                            }
                        )
                    except Exception as retry_e:
                        return {"status": "error", "message": f"Failed to generate summary after retry: {str(retry_e)}"}
                else:
                    return {"status": "error", "message": f"Failed to generate summary: {str(e)}"}
            
            return {
                "status": "success",
                "description": final_response.text,
                "source": "daily_recap",
                "scenes_used": scenes
            }
        except Exception as e:
            return {"status": "error", "message": str(e)} 