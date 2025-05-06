import os
import requests
from typing import List, Optional, Dict
from deepface import DeepFace
from PIL import Image
import numpy as np
from io import BytesIO

class FaceRecognitionClass:
    def __init__(self, faces_dir: str = "faces"):
        """
        Initialize the face recognition system.
        
        Args:
            faces_dir (str): Directory containing reference face images
        """
        self.faces_dir = faces_dir
        if not os.path.exists(faces_dir):
            os.makedirs(faces_dir)
    
    def _download_image(self, image_url: str) -> Optional[np.ndarray]:
        """
        Download image from URL and convert to numpy array.
        
        Args:
            image_url (str): URL of the image to download
            
        Returns:
            Optional[np.ndarray]: Image array or None if download fails
        """
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            return np.array(image)
        except Exception as e:
            print(f"Error downloading image: {e}")
            return None

    def find_face(self, image_url: str) -> List[Dict]:
        """
        Find matching faces from the reference database given an image URL.
        
        Args:
            image_url (str): URL of the image to analyze
            
        Returns:
            List[Dict]: List of matches with confidence scores and identities
        """
        # Download and process the image
        image = self._download_image(image_url)
        if image is None:
            return []
        
        try:
            # Analyze the image using DeepFace
            results = DeepFace.find(
                img_path=image,
                db_path=self.faces_dir,
                enforce_detection=False,
                model_name="VGG-Face"
            )
            
            # Process and format results
            matches = []
            for result in results:
                # Get the best match (first result)
                if len(result) > 0:
                    match = result.iloc[0]  # Get the first row of the DataFrame
                    identity = os.path.basename(match['identity']).split('.')[0]
                    # Convert distance to confidence score (0-100)
                    confidence = float((1 - match['distance']) * 100)
                    matches.append({
                        'identity': identity,
                        'confidence': confidence,
                        'path': match['identity']
                    })
            
            return matches
            
        except Exception as e:
            print(f"Error in face recognition: {e}")
            return []
    
    def add_face(self, image_url: str, identity: str) -> bool:
        """
        Add a new face to the reference database.
        
        Args:
            image_url (str): URL of the image containing the face
            identity (str): Identifier for the face
            
        Returns:
            bool: True if successful, False otherwise
        """
        image = self._download_image(image_url)
        if image is None:
            return False
            
        try:
            # Save the image with the identity as filename
            save_path = os.path.join(self.faces_dir, f"{identity}.jpg")
            Image.fromarray(image).save(save_path)
            return True
        except Exception as e:
            print(f"Error adding face: {e}")
            return False



face_rec = FaceRecognitionClass()
# face_rec.add_face("https://pbs.twimg.com/profile_images/1711104177521266688/vv54lurV_400x400.jpg", "JP")
# face_rec.find_face("./me.jpeg")
face_rec.find_face("https://pbs.twimg.com/profile_images/1711104177521266688/vv54lurV_400x400.jpg")