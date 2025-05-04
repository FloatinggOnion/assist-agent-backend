import os
from typing import List, Optional
from google.cloud import vision
import requests
from io import BytesIO

class OCRService:
    def __init__(self):
        """
        Initialize the OCR service using Google Cloud Vision API.
        Requires GOOGLE_APPLICATION_CREDENTIALS environment variable to be set.
        """
        self.client = vision.ImageAnnotatorClient()
    
    def _download_image(self, image_url: str) -> Optional[bytes]:
        """
        Download image from URL.
        
        Args:
            image_url (str): URL of the image to download
            
        Returns:
            Optional[bytes]: Image bytes or None if download fails
        """
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            return response.content
        except Exception as e:
            print(f"Error downloading image: {e}")
            return None
    
    def extract_text(self, image_url: str) -> List[str]:
        """
        Extract text from an image using Google Cloud Vision API.
        
        Args:
            image_url (str): URL of the image to analyze
            
        Returns:
            List[str]: List of extracted text blocks
        """
        # Download the image
        image_content = self._download_image(image_url)
        if image_content is None:
            return []
        
        try:
            # Create image object for Google Cloud Vision
            image = vision.Image(content=image_content)
            
            # Perform text detection
            response = self.client.text_detection(image=image)
            texts = response.text_annotations
            
            if not texts:
                return []
            
            # Extract text from annotations
            # First annotation contains the entire text
            full_text = texts[0].description
            
            # Split into lines and clean up
            text_lines = [line.strip() for line in full_text.split('\n') if line.strip()]
            
            return text_lines
            
        except Exception as e:
            print(f"Error in OCR processing: {e}")
            return [] 