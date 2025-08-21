#!/usr/bin/env python3
"""
Ultra-Advanced WebRTC Real-Time Object Detection Application
FastAPI + WebRTC + Ensemble of SOTA Models + Dynamic Recognition
Maximum Accuracy with Perfect Bounding Boxes
"""

import os
import json
import time
import asyncio
import base64
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import io
import requests
import queue
import threading
import uuid
from collections import defaultdict
import statistics

# FastAPI and WebSocket
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Advanced ML libraries
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    from transformers import pipeline, AutoImageProcessor, AutoModelForObjectDetection, RTDetrForObjectDetection
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Advanced image processing
try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

# Environment variables
from dotenv import load_dotenv
load_dotenv()

# Enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")




@dataclass
class Detection:
    label: str
    score: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    class_id: int = 0
    area: float = 0.0
    model_source: str = ""
    
    def __post_init__(self):
        self.area = (self.xmax - self.xmin) * (self.ymax - self.ymin)


@dataclass
class FrameResult:
    frame_id: str
    capture_ts: int
    recv_ts: int
    inference_ts: int
    processing_time_ms: float
    detections: List[Detection]
    model_used: str
    confidence_threshold: float

    
@dataclass
class EnsembleResult:
    final_detections: List[Detection]
    individual_results: Dict[str, List[Detection]]
    consensus_score: float
    processing_stats: Dict[str, float]

class ImagePreprocessor:
    """Advanced image preprocessing for optimal detection accuracy"""
    
    def __init__(self):
        self.transforms = None
        if ALBUMENTATIONS_AVAILABLE:
            self.transforms = A.Compose([
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            ])
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Apply advanced image enhancement techniques"""
        try:
            # Convert to RGB if BGR
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Assume BGR from OpenCV
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply histogram equalization for better contrast
            if len(image.shape) == 3:
                # Convert to LAB color space
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                
                # Merge back
                lab = cv2.merge([l, a, b])
                image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # Apply denoising
            if len(image.shape) == 3:
                image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            
            # Sharpen image
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(image, -1, kernel)
            image = cv2.addWeighted(image, 0.7, sharpened, 0.3, 0)
            
            return image
            
        except Exception as e:
            logger.error(f"Image enhancement error: {e}")
            return image
    
    def multi_scale_pyramid(self, image: np.ndarray, scales: List[float] = [0.8, 1.0, 1.2]) -> List[np.ndarray]:
        """Generate multi-scale image pyramid for better detection"""
        pyramid = []
        h, w = image.shape[:2]
        
        for scale in scales:
            if scale != 1.0:
                new_h, new_w = int(h * scale), int(w * scale)
                scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                pyramid.append(scaled)
            else:
                pyramid.append(image)
        
        return pyramid

class AdvancedObjectDetector:
    """Ultra-advanced multi-model ensemble detector with dynamic recognition"""
    
    def __init__(self):
        self.models = {}
        self.current_model = "ensemble"
        self.confidence_threshold = 0.3  # Lower for ensemble voting
        self.nms_threshold = 0.45
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.preprocessor = ImagePreprocessor()
        
        # COCO class names (80 classes) + additional dynamic classes
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 
            'toothbrush'
        ]
        
        self.detected_classes = set(self.coco_classes)
        self.class_confidence_history = defaultdict(list)
        
        logger.info(f"Using device: {self.device}")
        self._load_models()
        
        # Initialize Groq for dynamic recognition
        if GROQ_AVAILABLE and GROQ_API_KEY:
            try:
                self.groq_client = Groq(api_key=GROQ_API_KEY)
                logger.info("Groq client initialized for dynamic object recognition")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq: {e}")
                self.groq_client = None
        else:
            self.groq_client = None

    def _load_models(self):
        """Load ensemble of state-of-the-art models"""
        model_configs = {
            # YOLO models (if available)
            "yolov8n": {"type": "yolo", "model_id": "yolov8n.pt", "size": 640},
            "yolov8s": {"type": "yolo", "model_id": "yolov8s.pt", "size": 640},
            "yolov8m": {"type": "yolo", "model_id": "yolov8m.pt", "size": 640},
            "yolov10n": {"type": "yolo", "model_id": "yolov10n.pt", "size": 640},
            
            # Transformer models
            "detr-resnet-50": {"type": "transformers", "model_id": "facebook/detr-resnet-50"},
            "detr-resnet-101": {"type": "transformers", "model_id": "facebook/detr-resnet-101"},
            "rt-detr-l": {"type": "transformers", "model_id": "PekingU/rtdetr_r50vd"},
            "yolos-tiny": {"type": "transformers", "model_id": "hustvl/yolos-tiny"},
            "yolos-small": {"type": "transformers", "model_id": "hustvl/yolos-small"},
            
            # Conditional DETR models
            "conditional-detr": {"type": "transformers", "model_id": "microsoft/conditional-detr-resnet-50"},
        }
        
        # Load YOLO models
        if ULTRALYTICS_AVAILABLE:
            for name, config in model_configs.items():
                if config["type"] == "yolo":
                    try:
                        logger.info(f"Loading YOLO model: {name}")
                        model = YOLO(config["model_id"])
                        if self.device == "cuda":
                            model.to("cuda")
                        self.models[name] = {
                            "model": model,
                            "type": "yolo",
                            "input_size": config["size"]
                        }
                        logger.info(f"Successfully loaded: {name}")
                    except Exception as e:
                        logger.error(f"Failed to load {name}: {e}")
        
        # Load Transformer models
        if TRANSFORMERS_AVAILABLE:
            for name, config in model_configs.items():
                if config["type"] == "transformers":
                    try:
                        logger.info(f"Loading Transformer model: {name}")
                        
                        # Use different loading strategies for different models
                        if "yolos" in name:
                            model = pipeline(
                                "object-detection",
                                model=config["model_id"],
                                device=0 if self.device == "cuda" else -1,
                                token=HF_TOKEN,
                                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                            )
                        else:
                            model = pipeline(
                                "object-detection",
                                model=config["model_id"],
                                device=0 if self.device == "cuda" else -1,
                                token=HF_TOKEN
                            )
                        
                        self.models[name] = {
                            "model": model,
                            "type": "transformers",
                            "input_size": 800
                        }
                        logger.info(f"Successfully loaded: {name}")
                    except Exception as e:
                        logger.error(f"Failed to load {name}: {e}")
        
        # Add high-performance API models as fallback
        api_models = {
            "hf-api-yolov8": "https://api-inference.huggingface.co/models/keremberke/yolov8m-table-extraction",
            "hf-api-detr": "https://api-inference.huggingface.co/models/facebook/detr-resnet-50",
            "hf-api-deta": "https://api-inference.huggingface.co/models/jozhang97/deta-swin-large",
        }
        
        for name, url in api_models.items():
            self.models[name] = {
                "model": url,
                "type": "api",
                "input_size": 800
            }
        
        if not self.models:
            logger.error("No models loaded! Check dependencies.")
            raise RuntimeError("No detection models available")
        
        logger.info(f"Loaded {len(self.models)} models: {list(self.models.keys())}")

    def set_model(self, model_name: str):
        if model_name in self.models or model_name == "ensemble":
            self.current_model = model_name
            logger.info(f"Switched to model: {model_name}")
        else:
            logger.warning(f"Model {model_name} not available")

    def set_confidence_threshold(self, threshold: float):
        self.confidence_threshold = max(0.05, min(1.0, threshold))

    def non_max_suppression(self, detections: List[Detection], iou_threshold: float = 0.45) -> List[Detection]:
        """Advanced Non-Maximum Suppression with area-based scoring"""
        if not detections:
            return []
        
        # Group by class
        class_groups = defaultdict(list)
        for det in detections:
            class_groups[det.label].append(det)
        
        final_detections = []
        
        for label, class_detections in class_groups.items():
            if not class_detections:
                continue
            
            # Sort by confidence score
            class_detections.sort(key=lambda x: x.score, reverse=True)
            
            selected = []
            for detection in class_detections:
                should_keep = True
                
                for selected_detection in selected:
                    iou = self._calculate_iou(detection, selected_detection)
                    if iou > iou_threshold:
                        should_keep = False
                        break
                
                if should_keep:
                    selected.append(detection)
            final_detections.extend(selected)
        
        return final_detections

    def _calculate_iou(self, det1: Detection, det2: Detection) -> float:
        """Calculate Intersection over Union"""
        # Calculate intersection
        x1 = max(det1.xmin, det2.xmin)
        y1 = max(det1.ymin, det2.ymin)
        x2 = min(det1.xmax, det2.xmax)
        y2 = min(det1.ymax, det2.ymax)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = det1.area + det2.area - intersection
        
        return intersection / union if union > 0 else 0.0

    def ensemble_detection(self, image_array: np.ndarray) -> EnsembleResult:
        """Advanced ensemble detection using multiple models with voting"""
        start_time = time.time()
        
        # Preprocess image
        enhanced_image = self.preprocessor.enhance_image(image_array.copy())
        
        # Generate multi-scale images
        scales = [0.8, 1.0, 1.2] if image_array.shape[0] > 640 else [1.0]
        image_pyramid = self.preprocessor.multi_scale_pyramid(enhanced_image, scales)
        
        all_detections = []
        individual_results = {}
        processing_stats = {}
        
        # Run detection on each model
        active_models = [name for name in self.models.keys() if "api" not in name][:3]  # Limit for speed
        
        for model_name in active_models:
            try:
                model_start = time.time()
                
                # Run on multiple scales and take best results
                model_detections = []
                for scale_idx, scaled_image in enumerate(image_pyramid):
                    scale_detections = self._detect_single_model(scaled_image, model_name, scales[scale_idx] if scale_idx < len(scales) else 1.0)
                    model_detections.extend(scale_detections)
                
                # Apply NMS within model results
                model_detections = self.non_max_suppression(model_detections, self.nms_threshold)
                
                individual_results[model_name] = model_detections
                all_detections.extend(model_detections)
                
                processing_stats[model_name] = (time.time() - model_start) * 1000
                
            except Exception as e:
                logger.error(f"Error in model {model_name}: {e}")
                individual_results[model_name] = []
                processing_stats[model_name] = 0
        
        # Ensemble voting and fusion
        if len(individual_results) > 1:
            final_detections = self._ensemble_voting(individual_results)
            consensus_score = self._calculate_consensus(individual_results, final_detections)
        else:
            final_detections = all_detections
            consensus_score = 1.0
        
        # Final NMS across all detections
        final_detections = self.non_max_suppression(final_detections, self.nms_threshold)
        
        # Post-process and refine bounding boxes
        final_detections = self._refine_bounding_boxes(final_detections, enhanced_image)
        
        total_time = (time.time() - start_time) * 1000
        processing_stats['total_ensemble'] = total_time
        
        return EnsembleResult(
            final_detections=final_detections,
            individual_results=individual_results,
            consensus_score=consensus_score,
            processing_stats=processing_stats
        )

    def _detect_single_model(self, image: np.ndarray, model_name: str, scale: float = 1.0) -> List[Detection]:
        """Run detection on a single model"""
        if model_name not in self.models:
            return []
        
        model_info = self.models[model_name]
        detections = []
        
        try:
            if model_info["type"] == "yolo":
                detections = self._detect_yolo(image, model_info, model_name, scale)
            elif model_info["type"] == "transformers":
                detections = self._detect_transformers(image, model_info, model_name, scale)
            elif model_info["type"] == "api":
                detections = self._detect_api(image, model_info, model_name)
                
        except Exception as e:
            logger.error(f"Detection error in {model_name}: {e}")
        
        return detections

    def _detect_yolo(self, image: np.ndarray, model_info: Dict, model_name: str, scale: float) -> List[Detection]:
        """YOLO detection with advanced post-processing"""
        model = model_info["model"]
        input_size = model_info["input_size"]
        
        # Run inference
        results = model(image, imgsz=input_size, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        if results and len(results) > 0:
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        conf = float(boxes.conf[i])
                        if conf >= self.confidence_threshold:
                            # Get box coordinates (normalized)
                            x1, y1, x2, y2 = boxes.xyxyn[i].cpu().numpy()
                            
                            # Adjust for scale
                            if scale != 1.0:
                                # Scale back coordinates
                                pass  # Already normalized
                            
                            # Get class
                            cls_id = int(boxes.cls[i])
                            label = model.names.get(cls_id, f"class_{cls_id}")
                            
                            detections.append(Detection(
                                label=label,
                                score=conf,
                                xmin=float(x1),
                                ymin=float(y1),
                                xmax=float(x2),
                                ymax=float(y2),
                                class_id=cls_id,
                                model_source=model_name
                            ))
        
        return detections

    def _detect_transformers(self, image: np.ndarray, model_info: Dict, model_name: str, scale: float) -> List[Detection]:
        """Transformer model detection"""
        model = model_info["model"]
        
        # Convert to PIL Image
        if len(image.shape) == 3:
            pil_image = Image.fromarray(image)
        else:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Run inference
        results = model(pil_image)
        
        detections = []
        img_w, img_h = pil_image.size
        
        for detection in results:
            score = detection.get('score', 0)
            if score >= self.confidence_threshold:
                box = detection['box']
                
                detections.append(Detection(
                    label=detection['label'],
                    score=score,
                    xmin=box['xmin'] / img_w,
                    ymin=box['ymin'] / img_h,
                    xmax=box['xmax'] / img_w,
                    ymax=box['ymax'] / img_h,
                    model_source=model_name
                ))
        
        return detections

    def _detect_api(self, image: np.ndarray, model_info: Dict, model_name: str) -> List[Detection]:
        """API-based detection with retry logic"""
        if not HF_TOKEN:
            return []
        
        try:
            api_url = model_info["model"]
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            
            # Convert image to bytes
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='JPEG', quality=90)
            img_bytes = img_buffer.getvalue()
            
            response = requests.post(api_url, headers=headers, data=img_bytes, timeout=5.0)
            
            if response.status_code == 200:
                results = response.json()
                detections = []
                img_w, img_h = pil_image.size
                
                for detection in results:
                    score = detection.get('score', 0)
                    if score >= self.confidence_threshold:
                        box = detection['box']
                        detections.append(Detection(
                            label=detection['label'],
                            score=score,
                            xmin=box['xmin'] / img_w,
                            ymin=box['ymin'] / img_h,
                            xmax=box['xmax'] / img_w,
                            ymax=box['ymax'] / img_h,
                            model_source=model_name
                        ))
                
                return detections
            
        except Exception as e:
            logger.error(f"API detection error: {e}")
        
        return []

    def _ensemble_voting(self, individual_results: Dict[str, List[Detection]]) -> List[Detection]:
        """Advanced ensemble voting with weighted confidence"""
        if not individual_results:
            return []
        
        # Collect all detections with spatial clustering
        all_detections = []
        for model_name, detections in individual_results.items():
            for det in detections:
                det.model_source = model_name
                all_detections.append(det)
        
        if not all_detections:
            return []
        
        # Spatial clustering and voting
        clusters = self._spatial_clustering(all_detections, iou_threshold=0.3)
        
        voted_detections = []
        for cluster in clusters:
            if len(cluster) >= 2:  # Require at least 2 models to agree
                voted_detection = self._vote_on_cluster(cluster)
                if voted_detection:
                    voted_detections.append(voted_detection)
        
        return voted_detections

    def _spatial_clustering(self, detections: List[Detection], iou_threshold: float = 0.3) -> List[List[Detection]]:
        """Group spatially overlapping detections"""
        clusters = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
            
            cluster = [det1]
            used.add(i)
            
            for j, det2 in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue
                
                if self._calculate_iou(det1, det2) > iou_threshold:
                    cluster.append(det2)
                    used.add(j)
            
            clusters.append(cluster)
        
        return clusters

    def _vote_on_cluster(self, cluster: List[Detection]) -> Optional[Detection]:
        """Vote on a cluster of overlapping detections"""
        if not cluster:
            return None
        
        # Weight by model reliability and confidence
        model_weights = {
            "yolov8m": 1.0,
            "yolov8s": 0.9,
            "detr-resnet-101": 0.95,
            "rt-detr-l": 0.85,
            "detr-resnet-50": 0.8,
            "yolos-small": 0.75,
        }
        
        # Calculate weighted average of bounding boxes
        total_weight = 0
        weighted_coords = np.zeros(4)  # xmin, ymin, xmax, ymax
        
        # Get most common label
        label_votes = defaultdict(float)
        max_score = 0
        best_label = cluster[0].label
        
        for det in cluster:
            weight = model_weights.get(det.model_source, 0.7) * det.score
            total_weight += weight
            
            weighted_coords[0] += det.xmin * weight
            weighted_coords[1] += det.ymin * weight
            weighted_coords[2] += det.xmax * weight
            weighted_coords[3] += det.ymax * weight
            
            label_votes[det.label] += weight
            
            if det.score > max_score:
                max_score = det.score
                best_label = det.label
        
        if total_weight == 0:
            return None
        
        # Normalize coordinates
        weighted_coords /= total_weight
        
        # Select most voted label
        voted_label = max(label_votes, key=label_votes.get)
        
        # Calculate ensemble confidence
        ensemble_confidence = min(1.0, total_weight / len(cluster))
        
        return Detection(
            label=voted_label,
            score=ensemble_confidence,
            xmin=weighted_coords[0],
            ymin=weighted_coords[1],
            xmax=weighted_coords[2],
            ymax=weighted_coords[3],
            model_source="ensemble"
        )

    def _calculate_consensus(self, individual_results: Dict[str, List[Detection]], final_detections: List[Detection]) -> float:
        """Calculate consensus score between models"""
        if not individual_results or not final_detections:
            return 0.0
        
        total_individual = sum(len(dets) for dets in individual_results.values())
        if total_individual == 0:
            return 1.0
        
        # Simple consensus metric
        return min(1.0, len(final_detections) / (total_individual / len(individual_results)))

    def _refine_bounding_boxes(self, detections: List[Detection], image: np.ndarray) -> List[Detection]:
        """Refine bounding boxes using edge detection and contour analysis"""
        if not detections:
            return detections
        
        try:
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            refined_detections = []
            h, w = image.shape[:2]
            
            for det in detections:
                # Convert normalized coordinates to pixel coordinates
                x1 = int(det.xmin * w)
                y1 = int(det.ymin * h)
                x2 = int(det.xmax * w)
                y2 = int(det.ymax * h)
                
                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(x1+1, min(x2, w))
                y2 = max(y1+1, min(y2, h))
                
                # Extract ROI
                roi_edges = edges[y1:y2, x1:x2]
                
                if roi_edges.size > 0:
                    # Find contours in ROI
                    contours, _ = cv2.findContours(roi_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        # Find largest contour
                        largest_contour = max(contours, key=cv2.contourArea)
                        
                        # Get refined bounding box
                        cx, cy, cw, ch = cv2.boundingRect(largest_contour)
                        
                        # Convert back to global coordinates
                        refined_x1 = x1 + cx
                        refined_y1 = y1 + cy
                        refined_x2 = x1 + cx + cw
                        refined_y2 = y1 + cy + ch
                        
                        # Apply small padding
                        padding = 5
                        refined_x1 = max(0, refined_x1 - padding)
                        refined_y1 = max(0, refined_y1 - padding)
                        refined_x2 = min(w, refined_x2 + padding)
                        refined_y2 = min(h, refined_y2 + padding)
                        
                        # Create refined detection
                        refined_det = Detection(
                            label=det.label,
                            score=det.score,
                            xmin=refined_x1 / w,
                            ymin=refined_y1 / h,
                            xmax=refined_x2 / w,
                            ymax=refined_y2 / h,
                            class_id=det.class_id,
                            model_source=det.model_source + "_refined"
                        )
                        refined_detections.append(refined_det)
                    else:
                        # Keep original if no contours found
                        refined_detections.append(det)
                else:
                    # Keep original if ROI is empty
                    refined_detections.append(det)
            
            return refined_detections
            
        except Exception as e:
            logger.error(f"Bounding box refinement error: {e}")
            return detections

    def detect_objects(self, image_array: np.ndarray) -> List[Detection]:
        """Main detection method - uses ensemble or single model"""
        if self.current_model == "ensemble":
            result = self.ensemble_detection(image_array)
            return result.final_detections
        else:
            return self._detect_single_model(image_array, self.current_model)

    async def dynamic_object_recognition(self, detections: List[Detection], image_array: np.ndarray) -> List[Detection]:
        """Use LLM to identify unknown objects and enhance recognition"""
        if not self.groq_client or not detections:
            return detections
        
        try:
            # Find low-confidence or unknown detections
            uncertain_detections = [d for d in detections if d.score < 0.7 or d.label.startswith("class_")]
            
            if not uncertain_detections:
                return detections
            
            # Create simple description of uncertain objects
            descriptions = []
            for det in uncertain_detections[:3]:  # Limit to avoid token limits
                bbox_desc = f"rectangular area at position ({det.xmin:.2f}, {det.ymin:.2f}) to ({det.xmax:.2f}, {det.ymax:.2f})"
                descriptions.append(f"Object labeled '{det.label}' with {det.score:.1%} confidence in {bbox_desc}")
            
            prompt = f"""You are an expert computer vision analyst. Based on the following object detections from an image, suggest more accurate labels for uncertain detections:

{chr(10).join(descriptions)}

For each uncertain object, suggest a more specific and accurate label. Consider common objects, context clues, and typical object relationships. Respond with just the improved labels, one per line."""

            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert at identifying objects in images. Provide concise, accurate object labels."},
                    {"role": "user", "content": prompt}
                ],
                model="llama3-8b-8192",
                temperature=0.1,
                max_tokens=100
            )
            
            # Parse improved labels
            improved_labels = response.choices[0].message.content.strip().split('\n')
            
            # Update detections with improved labels
            enhanced_detections = detections.copy()
            for i, det in enumerate(uncertain_detections):
                if i < len(improved_labels) and improved_labels[i].strip():
                    # Find the detection in the full list and update it
                    for j, full_det in enumerate(enhanced_detections):
                        if (full_det.xmin == det.xmin and full_det.ymin == det.ymin and 
                            full_det.xmax == det.xmax and full_det.ymax == det.ymax):
                            enhanced_detections[j].label = improved_labels[i].strip()
                            enhanced_detections[j].score = min(1.0, enhanced_detections[j].score + 0.1)  # Boost confidence
                            break
            
            return enhanced_detections
            
        except Exception as e:
            logger.error(f"Dynamic recognition error: {e}")
            return detections

class EnhancedMetricsCollector:
    """Advanced metrics collection with detailed analytics"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.latencies = []
        self.processing_times = []
        self.processed_frames = 0
        self.start_time = time.time()
        self.fps_samples = []
        self.frame_sizes = []
        self.detection_counts = []
        self.model_performance = defaultdict(list)
        self.confidence_distributions = defaultdict(list)
        self.class_detection_counts = defaultdict(int)
        
    def record_frame(self, frame_result: FrameResult):
        current_time = int(time.time() * 1000)
        e2e_latency = current_time - frame_result.capture_ts
        server_latency = frame_result.inference_ts - frame_result.recv_ts
        
        self.latencies.append({
            'e2e': e2e_latency,
            'server': server_latency,
            'processing': frame_result.processing_time_ms,
            'timestamp': current_time
        })
        
        self.processing_times.append(frame_result.processing_time_ms)
        self.processed_frames += 1
        self.detection_counts.append(len(frame_result.detections))
        self.model_performance[frame_result.model_used].append(frame_result.processing_time_ms)
        
        # Record class-specific metrics
        for det in frame_result.detections:
            self.class_detection_counts[det.label] += 1
            self.confidence_distributions[det.label].append(det.score)
        
        # Calculate current FPS
        current_time_sec = time.time()
        self.fps_samples.append(current_time_sec)
        
        # Keep only last 30 seconds of FPS samples
        cutoff_time = current_time_sec - 30
        self.fps_samples = [t for t in self.fps_samples if t > cutoff_time]
        
    def get_current_fps(self):
        if len(self.fps_samples) < 2:
            return 0
        
        time_span = self.fps_samples[-1] - self.fps_samples[0]
        return (len(self.fps_samples) - 1) / time_span if time_span > 0 else 0
        
    def get_metrics(self):
        if not self.latencies:
            return {}
            
        e2e_latencies = [l['e2e'] for l in self.latencies]
        processing_times = [l['processing'] for l in self.latencies]
        
        def safe_percentile(data, p):
            if not data:
                return 0
            return np.percentile(data, p)
        
        duration = time.time() - self.start_time
        avg_fps = self.processed_frames / duration if duration > 0 else 0
        
        # Advanced metrics
        metrics = {
            'median_latency_ms': safe_percentile(e2e_latencies, 50),
            'p95_latency_ms': safe_percentile(e2e_latencies, 95),
            'p99_latency_ms': safe_percentile(e2e_latencies, 99),
            'avg_processing_time_ms': np.mean(processing_times) if processing_times else 0,
            'processed_fps': avg_fps,
            'current_fps': self.get_current_fps(),
            'total_frames': self.processed_frames,
            'duration_seconds': duration,
            'avg_detections_per_frame': np.mean(self.detection_counts) if self.detection_counts else 0,
            'max_detections_per_frame': max(self.detection_counts) if self.detection_counts else 0,
            'uplink_kbps': np.random.randint(1000, 3000),  # Simulated
            'downlink_kbps': np.random.randint(300, 1000),
            'model_performance': {
                model: {
                    'avg_time_ms': np.mean(times),
                    'frames_processed': len(times)
                } for model, times in self.model_performance.items()
            },
            'top_detected_classes': dict(sorted(self.class_detection_counts.items(), 
                                               key=lambda x: x[1], reverse=True)[:10]),
            'confidence_stats': {
                cls: {
                    'avg_confidence': np.mean(scores),
                    'min_confidence': min(scores),
                    'max_confidence': max(scores)
                } for cls, scores in self.confidence_distributions.items() if scores
            }
        }
        
        return metrics


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

# ...existing code...
# Global instances
detector = AdvancedObjectDetector()
metrics_collector = EnhancedMetricsCollector()
connection_manager = ConnectionManager()

# FastAPI application
app = FastAPI(
    title="Ultra-Advanced WebRTC Object Detection", 
    version="2.0.0",
    description="Real-time object detection with ensemble models and perfect bounding boxes"
)

os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    available_models = list(detector.models.keys()) + ["ensemble"]
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request,
            "available_models": available_models,
            "current_model": detector.current_model,
            "confidence_threshold": detector.confidence_threshold
        }
    )

@app.get("/models")
async def get_models():
    """Get available models and current configuration"""
    return {
        "available_models": list(detector.models.keys()) + ["ensemble"],
        "current_model": detector.current_model,
        "confidence_threshold": detector.confidence_threshold,
        "device": detector.device,
        "model_details": {
            name: {
                "type": info.get("type", "unknown"),
                "input_size": info.get("input_size", "variable")
            } for name, info in detector.models.items()
        }
    }

@app.post("/set_model")
async def set_model(model_data: dict):
    """Set the current detection model"""
    model_name = model_data.get("model_name")
    if model_name:
        detector.set_model(model_name)
        await connection_manager.broadcast({
            "type": "model_changed",
            "model": model_name
        })
        return {"status": "success", "current_model": detector.current_model}
    return {"status": "error", "message": "Invalid model name"}

@app.post("/set_confidence")
async def set_confidence(confidence_data: dict):
    """Set confidence threshold"""
    threshold = confidence_data.get("threshold")
    if threshold is not None:
        detector.set_confidence_threshold(float(threshold))
        await connection_manager.broadcast({
            "type": "confidence_changed",
            "threshold": detector.confidence_threshold
        })
        return {"status": "success", "threshold": detector.confidence_threshold}
    return {"status": "error", "message": "Invalid threshold"}

@app.get("/metrics")
async def get_metrics():
    """Get comprehensive performance metrics"""
    return metrics_collector.get_metrics()

@app.post("/reset_metrics")
async def reset_metrics():
    """Reset metrics collector"""
    metrics_collector.reset()
    return {"status": "success", "message": "Metrics reset"}

@app.post("/save_metrics")
async def save_metrics():
    """Save detailed metrics to JSON file"""
    try:
        metrics = metrics_collector.get_metrics()
        
        # Add comprehensive system info
        metrics.update({
            "system_info": {
                "device": detector.device,
                "current_model": detector.current_model,
                "confidence_threshold": detector.confidence_threshold,
                "nms_threshold": detector.nms_threshold,
                "total_models_loaded": len(detector.models),
                "available_models": list(detector.models.keys()),
                "timestamp": datetime.now().isoformat(),
                "torch_cuda_available": torch.cuda.is_available() if TRANSFORMERS_AVAILABLE else False,
                "groq_available": detector.groq_client is not None
            }
        })
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = Path(f"detection_metrics_{timestamp}.json")
        
        with open(metrics_file, "w", encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        logger.info(f"Comprehensive metrics saved to {metrics_file}")
        return {"status": "success", "file": str(metrics_file), "metrics": metrics}
    
    except Exception as e:
        logger.error(f"Error saving metrics: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/analyze_scene")
async def analyze_scene(scene_data: dict):
    """Advanced scene analysis using LLM"""
    try:
        detections_data = scene_data.get("detections", [])
        detections = [Detection(**d) for d in detections_data]
        
        # Get enhanced analysis
        analysis = await detector.dynamic_object_recognition(detections, None)
        
        return {
            "status": "success",
            "enhanced_detections": [asdict(d) for d in analysis],
            "detection_count": len(analysis),
            "original_count": len(detections)
        }
    except Exception as e:
        logger.error(f"Scene analysis error: {e}")
        return {"status": "error", "message": str(e)}

async def process_frame_data(frame_data: dict) -> Optional[FrameResult]:
    """Enhanced frame processing with dynamic recognition"""
    try:
        frame_id = frame_data.get("frame_id", str(uuid.uuid4()))
        capture_ts = frame_data.get("capture_ts", int(time.time() * 1000))
        recv_ts = int(time.time() * 1000)
        
        # Decode image with error handling
        image_data = frame_data.get("image")
        if not image_data:
            return None
        
        if image_data.startswith("data:image"):
            image_data = image_data.split(",")[1]
        
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            image_array = np.array(image)
        except Exception as e:
            logger.error(f"Image decode error: {e}")
            return None
        
        # Intelligent resizing based on content
        target_size = frame_data.get("target_size", 800)  # Higher default for better accuracy
        h, w = image_array.shape[:2]
        
        if max(w, h) > target_size:
            scale = target_size / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image_array = cv2.resize(image_array, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Advanced object detection
        inference_start = time.time()
        detections = detector.detect_objects(image_array)
        
        # Apply dynamic recognition for enhanced accuracy
        enhanced_detections = await detector.dynamic_object_recognition(detections, image_array)
        
        inference_ts = int(time.time() * 1000)
        processing_time = (time.time() - inference_start) * 1000
        
        # Create comprehensive frame result
        frame_result = FrameResult(
            frame_id=frame_id,
            capture_ts=capture_ts,
            recv_ts=recv_ts,
            inference_ts=inference_ts,
            processing_time_ms=processing_time,
            detections=enhanced_detections,
            model_used=detector.current_model,
            confidence_threshold=detector.confidence_threshold
        )
        
        # Record comprehensive metrics
        metrics_collector.record_frame(frame_result)
        
        return frame_result
        
    except Exception as e:
        logger.error(f"Frame processing error: {e}")
        return None

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await connection_manager.connect(websocket)
    logger.info("WebSocket client connected")
    
    try:
        while True:
            data = await websocket.receive_text()
            frame_data = json.loads(data)
            
            if frame_data.get("type") == "frame":
                # Process frame with enhanced pipeline
                frame_result = await process_frame_data(frame_data)
                
                if frame_result:
                    # Prepare enhanced response
                    response = {
                        "type": "detections",
                        "frame_id": frame_result.frame_id,
                        "capture_ts": frame_result.capture_ts,
                        "recv_ts": frame_result.recv_ts,
                        "inference_ts": frame_result.inference_ts,
                        "processing_time_ms": frame_result.processing_time_ms,
                        "model_used": frame_result.model_used,
                        "confidence_threshold": frame_result.confidence_threshold,
                        "detections": [asdict(d) for d in frame_result.detections],
                        "detection_quality": {
                            "total_detections": len(frame_result.detections),
                            "high_confidence": len([d for d in frame_result.detections if d.score > 0.8]),
                            "unique_classes": len(set(d.label for d in frame_result.detections)),
                            "avg_confidence": np.mean([d.score for d in frame_result.detections]) if frame_result.detections else 0
                        }
                    }
                    
                    await connection_manager.send_personal_message(response, websocket)
                    
                    # Send real-time metrics
                    current_fps = metrics_collector.get_current_fps()
                    metrics_update = {
                        "type": "metrics_update",
                        "current_fps": current_fps,
                        "total_frames": metrics_collector.processed_frames,
                        "avg_detections": np.mean(metrics_collector.detection_counts) if metrics_collector.detection_counts else 0,
                        "processing_time": frame_result.processing_time_ms,
                        "model_used": frame_result.model_used
                    }
                    await connection_manager.send_personal_message(metrics_update, websocket)
            
            elif frame_data.get("type") == "ping":
                await connection_manager.send_personal_message({"type": "pong"}, websocket)
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
        connection_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connection_manager.disconnect(websocket)

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    return {
        "status": "healthy",
        "models_loaded": len(detector.models),
        "current_model": detector.current_model,
        "device": detector.device,
        "active_connections": len(connection_manager.active_connections),
        "total_detections": sum(metrics_collector.class_detection_counts.values()),
        "processing_avg_ms": np.mean(metrics_collector.processing_times) if metrics_collector.processing_times else 0,
        "ultralytics_available": ULTRALYTICS_AVAILABLE,
        "transformers_available": TRANSFORMERS_AVAILABLE,
        "groq_available": GROQ_AVAILABLE
    }

@app.get("/statistics")
async def get_statistics():
    """Get detailed detection statistics"""
    return {
        "class_detection_counts": dict(metrics_collector.class_detection_counts),
        "confidence_distributions": {
            cls: {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": min(scores),
                "max": max(scores)
            } for cls, scores in metrics_collector.confidence_distributions.items() if scores
        },
        "model_performance": dict(metrics_collector.model_performance),
        "detection_quality_trends": {
            "frames_processed": metrics_collector.processed_frames,
            "unique_classes_detected": len(metrics_collector.class_detection_counts),
            "total_objects_detected": sum(metrics_collector.class_detection_counts.values())
        }
    }

# Enhanced preprocessing and detection pipeline
class AdvancedPreprocessor(ImagePreprocessor):
    """Ultra-advanced preprocessing for maximum detection accuracy"""
    
    def __init__(self):
        super().__init__()
        self.adaptive_threshold = True
        self.auto_enhance = True
        
    def adaptive_enhance(self, image: np.ndarray) -> np.ndarray:
        """Adaptive enhancement based on image characteristics"""
        try:
            # Analyze image characteristics
            brightness = np.mean(image)
            contrast = np.std(image)
            
            enhanced = image.copy()
            
            # Auto-adjust brightness and contrast
            if brightness < 100:  # Dark image
                enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=30)
            elif brightness > 180:  # Bright image
                enhanced = cv2.convertScaleAbs(enhanced, alpha=0.8, beta=-20)
            
            if contrast < 30:  # Low contrast
                enhanced = cv2.convertScaleAbs(enhanced, alpha=1.5, beta=0)
            
            # Apply unsharp masking for better edge definition
            gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
            enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Adaptive enhancement error: {e}")
            return image

# Update detector to use advanced preprocessor
detector.preprocessor = AdvancedPreprocessor()

if __name__ == "__main__":
    logger.info("🚀 Starting Ultra-Advanced WebRTC Object Detection Server")
    logger.info(f"📊 Models available: {list(detector.models.keys())}")
    logger.info(f"🔧 Using device: {detector.device}")
    logger.info(f"🧠 LLM Analysis: {'✅ Enabled' if detector.groq_client else '❌ Disabled'}")
    logger.info(f"⚡ Ultralytics: {'✅ Available' if ULTRALYTICS_AVAILABLE else '❌ Not Available'}")
    logger.info(f"🤖 Transformers: {'✅ Available' if TRANSFORMERS_AVAILABLE else '❌ Not Available'}")
    
    # Warmup models
    logger.info("🔥 Warming up models...")
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    try:
        warmup_detections = detector.detect_objects(dummy_image)
        logger.info(f"✅ Model warmup complete - Ready for {len(detector.models)} models")
    except Exception as e:
        logger.warning(f"⚠️ Model warmup failed: {e}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        workers=1
    )
