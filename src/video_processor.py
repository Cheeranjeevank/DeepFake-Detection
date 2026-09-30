import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from src.config import config
from src.utils import logger
from src.inference import ImagePredictor

class VideoPredictor:
    def __init__(self, predictor=None):
        if predictor is None:
            self.predictor = ImagePredictor()
        else:
            self.predictor = predictor
            
        self.frames_to_sample = config["inference"]["video_frames_to_sample"]

    def extract_frames(self, video_path, num_frames=None):
        """
        Extract evenly spaced frames from a video.
        """
        if num_frames is None:
            num_frames = self.frames_to_sample
            
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video {video_path}")
            return []
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames <= 0:
            return []
            
        # Determine which frames to extract
        step = max(1, total_frames // num_frames)
        frame_indices = range(0, total_frames, step)[:num_frames]
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                # Calculate timestamp
                timestamp = idx / fps if fps > 0 else 0
                
                frames.append({
                    "frame_num": idx,
                    "timestamp": timestamp,
                    "image": pil_img
                })
                
        cap.release()
        return frames

    def predict_video(self, video_path):
        """
        Process a video, extract frames, detect faces, predict and aggregate.
        """
        logger.info(f"Processing video: {video_path}")
        frames = self.extract_frames(video_path)
        
        if not frames:
            return {"error": "Failed to extract frames from video."}
            
        frame_results = []
        suspicious_frames_count = 0
        total_fake_prob = 0.0
        analyzed_frames = 0
        
        for frame_data in frames:
            result = self.predictor.predict_image(frame_data["image"])
            
            if "error" in result:
                continue # Skip frames without faces
                
            analyzed_frames += 1
            fake_prob = result["fake_probability"]
            total_fake_prob += fake_prob
            
            if result["prediction"] == "DEEPFAKE":
                suspicious_frames_count += 1
                
            frame_results.append({
                "frame_num": frame_data["frame_num"],
                "timestamp": frame_data["timestamp"],
                "prediction": result["prediction"],
                "fake_probability": fake_prob,
                "real_probability": result["real_probability"],
                "face_crop": result["face_crop"]
            })
            
        if analyzed_frames == 0:
            return {"error": "No faces detected in any of the extracted frames."}
            
        # Aggregate logic: mean fake probability
        mean_fake_prob = total_fake_prob / analyzed_frames
        mean_real_prob = 1.0 - mean_fake_prob
        
        if mean_fake_prob > 0.5:
            final_prediction = "DEEPFAKE"
            confidence = mean_fake_prob
        else:
            final_prediction = "REAL"
            confidence = mean_real_prob
            
        return {
            "final_prediction": final_prediction,
            "confidence": confidence,
            "frames_analyzed": analyzed_frames,
            "suspicious_frames": suspicious_frames_count,
            "mean_fake_probability": mean_fake_prob,
            "mean_real_probability": mean_real_prob,
            "frame_results": frame_results
        }
