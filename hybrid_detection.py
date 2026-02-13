import numpy as np
from typing import Tuple, Optional, List
from PIL import Image
import torch

class HybridDetection:
    """Enhanced Detection class that supports both YOLO and U-Net results"""
    
    def __init__(self, xyxy: Tuple[float, float, float, float], 
                 conf: float, cls_id: int, 
                 yolo_mask: Optional[np.ndarray] = None,
                 unet_mask: Optional[np.ndarray] = None):
        self.xyxy = xyxy
        self.conf = conf
        self.cls_id = cls_id
        
        # Store both types of masks
        self.yolo_mask = yolo_mask  # Original YOLO segmentation (if available)
        self.unet_mask = unet_mask  # U-Net segmentation (more precise)
        
        # Metadata
        self.detection_method = "yolo"
        self.segmentation_method = "unet" if unet_mask is not None else "yolo"
        self.segmentation_quality = self._assess_segmentation_quality()

    @property
    def mask(self) -> Optional[np.ndarray]:
        """Return the best available mask (prefer U-Net over YOLO)"""
        if self.unet_mask is not None:
            return self.unet_mask
        return self.yolo_mask

    def get_tumor_area(self, use_unet: bool = True) -> float:
        """Calculate tumor area using specified segmentation method"""
        if use_unet and self.unet_mask is not None:
            return float((self.unet_mask > 0).sum())
        elif self.yolo_mask is not None:
            return float((self.yolo_mask > 0.5).sum())
        else:
            # Fallback to bounding box area
            x1, y1, x2, y2 = self.xyxy
            return (x2 - x1) * (y2 - y1)

    def get_precise_measurements(self, image_size: Tuple[int, int], 
                               mm_per_pixel: float = 1.0) -> dict:
        """Get precise measurements using U-Net segmentation"""
        measurements = {
            'method': self.segmentation_method,
            'confidence': self.conf,
            'quality_score': self.segmentation_quality
        }
        
        if self.unet_mask is not None:
            # Use U-Net for precise measurements
            tumor_pixels = (self.unet_mask > 0).sum()
            
            # Find bounding box of actual tumor pixels
            tumor_coords = np.where(self.unet_mask > 0)
            if len(tumor_coords[0]) > 0:
                min_y, max_y = tumor_coords[0].min(), tumor_coords[0].max()
                min_x, max_x = tumor_coords[1].min(), tumor_coords[1].max()
                
                width_px = max_x - min_x + 1
                height_px = max_y - min_y + 1
            else:
                width_px = height_px = 0
            
            area_mm2 = tumor_pixels * (mm_per_pixel ** 2)
            measurements.update({
                'area_pixels': float(tumor_pixels),
                'width_pixels': float(width_px),
                'height_pixels': float(height_px),
                'area_mm2': area_mm2,
                'area_cm2': area_mm2 / 100.0,  # Convert mm² to cm²
                'width_mm': width_px * mm_per_pixel,
                'height_mm': height_px * mm_per_pixel,
                'precision': 'pixel-perfect'
            })
        else:
            # Fallback to bounding box measurements
            x1, y1, x2, y2 = self.xyxy
            width_px = x2 - x1
            height_px = y2 - y1
            area_px = width_px * height_px
            
            area_mm2 = area_px * (mm_per_pixel ** 2)
            measurements.update({
                'area_pixels': float(area_px),
                'width_pixels': float(width_px),
                'height_pixels': float(height_px),
                'area_mm2': area_mm2,
                'area_cm2': area_mm2 / 100.0,  # Convert mm² to cm²
                'width_mm': width_px * mm_per_pixel,
                'height_mm': height_px * mm_per_pixel,
                'precision': 'bounding-box-estimate'
            })
        
        return measurements

    def _assess_segmentation_quality(self) -> float:
        """Assess quality of segmentation (0.0 to 1.0)"""
        if self.unet_mask is not None:
            # U-Net quality assessment
            mask = self.unet_mask
            tumor_pixels = (mask > 0).sum()
            
            if tumor_pixels == 0:
                return 0.0
            
            # Calculate compactness (how circular/compact the tumor is)
            try:
                import cv2
                mask_uint8 = (mask > 0).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    perimeter = cv2.arcLength(largest_contour, True)
                    
                    if perimeter > 0:
                        compactness = (4 * np.pi * area) / (perimeter ** 2)
                        return min(1.0, compactness * self.conf)
                
            except ImportError:
                pass
            
            # Simple quality based on confidence and relative size
            bbox_area = (self.xyxy[2] - self.xyxy[0]) * (self.xyxy[3] - self.xyxy[1])
            fill_ratio = tumor_pixels / bbox_area if bbox_area > 0 else 0
            
            return min(1.0, fill_ratio * self.conf)
        
        elif self.yolo_mask is not None:
            # YOLO mask quality (generally lower than U-Net)
            return self.conf * 0.7  # Scale down for YOLO masks
        
        else:
            # No segmentation, only bounding box
            return self.conf * 0.5

    def get_visualization_data(self) -> dict:
        """Get data for visualization purposes"""
        return {
            'bbox': self.xyxy,
            'confidence': self.conf,
            'class_id': self.cls_id,
            'mask': self.mask,
            'detection_method': self.detection_method,
            'segmentation_method': self.segmentation_method,
            'quality_score': self.segmentation_quality,
            'has_unet_segmentation': self.unet_mask is not None,
            'has_yolo_segmentation': self.yolo_mask is not None
        }

    def to_dict(self) -> dict:
        """Convert detection to dictionary for serialization"""
        return {
            'bbox': list(self.xyxy),
            'confidence': float(self.conf),
            'class_id': int(self.cls_id),
            'detection_method': self.detection_method,
            'segmentation_method': self.segmentation_method,
            'quality_score': float(self.segmentation_quality),
            'tumor_area': float(self.get_tumor_area()),
            'has_unet_mask': self.unet_mask is not None,
            'has_yolo_mask': self.yolo_mask is not None
        }

    def __repr__(self) -> str:
        return (f"HybridDetection(bbox={self.xyxy}, conf={self.conf:.3f}, "
                f"cls={self.cls_id}, method={self.segmentation_method}, "
                f"quality={self.segmentation_quality:.3f})")


class HybridInferenceEngine:
    """Engine that combines YOLO detection with U-Net segmentation"""
    
    def __init__(self, yolo_model, unet_model):
        self.yolo_model = yolo_model
        self.unet_model = unet_model
        
    def run_hybrid_inference(self, image_path: str, conf_threshold: float = 0.25) -> List[HybridDetection]:
        """Run YOLO detection followed by U-Net segmentation"""
        
        # Step 1: YOLO Detection
        print("🔍 Running YOLO detection...")
        yolo_results = self.yolo_model.predict(source=image_path, conf=conf_threshold, verbose=False)
        
        # Step 2: Parse YOLO results
        detections = self._parse_yolo_results(yolo_results)
        
        if not detections:
            print("ℹ️ No tumors detected by YOLO")
            return []
        
        # Step 3: Load image for U-Net segmentation
        image = Image.open(image_path).convert('RGB')
        
        # Step 4: Run U-Net segmentation on each detection
        print(f"🎯 Running U-Net segmentation on {len(detections)} detections...")
        
        enhanced_detections = []
        for i, detection in enumerate(detections):
            print(f"  Processing detection {i+1}/{len(detections)}...")
            
            try:
                # Get U-Net segmentation for this region
                unet_mask = self.unet_model.segment_region(image, detection.xyxy)
                
                # Create enhanced detection with U-Net mask
                enhanced_detection = HybridDetection(
                    xyxy=detection.xyxy,
                    conf=detection.conf,
                    cls_id=detection.cls_id,
                    yolo_mask=detection.yolo_mask,
                    unet_mask=unet_mask
                )
                
                enhanced_detections.append(enhanced_detection)
                
            except Exception as e:
                print(f"⚠️ U-Net segmentation failed for detection {i+1}: {e}")
                # Keep original detection without U-Net enhancement
                enhanced_detections.append(detection)
        
        print(f"✓ Completed hybrid inference: {len(enhanced_detections)} enhanced detections")
        return enhanced_detections
    
    def _parse_yolo_results(self, results) -> List[HybridDetection]:
        """Parse YOLO results into HybridDetection objects"""
        detections = []
        
        if not results:
            return detections
            
        result = results[0]
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        
        # Get mask data if available
        mask_data = None
        if masks is not None and getattr(masks, "data", None) is not None:
            mask_data = masks.data.cpu().numpy()
        
        if boxes is not None and getattr(boxes, "xyxy", None) is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else np.ones((xyxy.shape[0],), dtype=float)
            cls = boxes.cls.cpu().numpy().astype(int) if getattr(boxes, "cls", None) is not None else np.zeros((xyxy.shape[0],), dtype=int)
            
            for i in range(xyxy.shape[0]):
                yolo_mask = mask_data[i] if mask_data is not None and i < mask_data.shape[0] else None
                
                detection = HybridDetection(
                    xyxy=tuple(map(float, xyxy[i].tolist())),
                    conf=float(conf[i]),
                    cls_id=int(cls[i]),
                    yolo_mask=yolo_mask,
                    unet_mask=None  # Will be filled by U-Net
                )
                
                detections.append(detection)
        
        return detections