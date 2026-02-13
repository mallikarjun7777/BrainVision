#!/usr/bin/env python3
"""
Demo script for Hybrid Brain Tumor Detection System (YOLO + U-Net)
This script demonstrates the hybrid detection capabilities without GUI
"""

import os
import sys
import time
from pathlib import Path
import argparse
import json
from typing import List, Dict, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our hybrid system components
from unet_model import TumorUNet, create_pretrained_unet
from hybrid_detection import HybridDetection, HybridInferenceEngine

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Ultralytics not installed. Install with: pip install ultralytics")
    YOLO = None

try:
    import torch
except ImportError:
    print("❌ PyTorch not installed. Install with: pip install torch torchvision")
    torch = None


class HybridDemo:
    """Demonstration class for hybrid brain tumor detection"""
    
    def __init__(self, yolo_model_path: str = "yolo11n.pt", 
                 unet_model_path: str = "unet_brain_tumor.pth"):
        self.yolo_model_path = yolo_model_path
        self.unet_model_path = unet_model_path
        
        # Models (will be loaded on demand)
        self.yolo_model = None
        self.unet_model = None
        self.hybrid_engine = None
        
        # Results storage
        self.last_results = []
        
        print("🧠 Hybrid Brain Tumor Detection Demo")
        print("=" * 50)
        
    def initialize_models(self) -> bool:
        """Initialize YOLO and U-Net models"""
        print("🚀 Initializing hybrid detection system...")
        
        # Check dependencies
        if YOLO is None or torch is None:
            print("❌ Missing dependencies. Please install ultralytics and torch.")
            return False
        
        try:
            # Initialize YOLO model
            print(f"📦 Loading YOLO model: {self.yolo_model_path}")
            if not os.path.exists(self.yolo_model_path):
                print(f"⚠️ YOLO model not found at {self.yolo_model_path}")
                print("📥 Downloading default YOLO model...")
            
            self.yolo_model = YOLO(self.yolo_model_path)
            print("✓ YOLO model loaded successfully")
            
            # Initialize U-Net model
            print(f"🎯 Loading U-Net model: {self.unet_model_path}")
            if not os.path.exists(self.unet_model_path):
                print("⚠️ U-Net model not found. Creating demonstration model...")
                create_pretrained_unet(self.unet_model_path)
            
            self.unet_model = TumorUNet(self.unet_model_path)
            print("✓ U-Net model loaded successfully")
            
            # Create hybrid engine
            self.hybrid_engine = HybridInferenceEngine(self.yolo_model, self.unet_model)
            print("✓ Hybrid inference engine initialized")
            
            return True
            
        except Exception as e:
            print(f"❌ Model initialization failed: {e}")
            return False
    
    def analyze_image(self, image_path: str, conf_threshold: float = 0.25, 
                     save_results: bool = True) -> List[HybridDetection]:
        """Analyze a single brain scan image"""
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return []
        
        if self.hybrid_engine is None:
            print("❌ Models not initialized. Call initialize_models() first.")
            return []
        
        print(f"\n🔍 Analyzing: {os.path.basename(image_path)}")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            # Run hybrid inference
            detections = self.hybrid_engine.run_hybrid_inference(
                image_path, conf_threshold
            )
            
            analysis_time = time.time() - start_time
            
            # Print results
            self._print_analysis_results(detections, analysis_time)
            
            # Save results if requested
            if save_results:
                self._save_analysis_results(image_path, detections, analysis_time)
            
            self.last_results = detections
            return detections
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return []
    
    def _print_analysis_results(self, detections: List[HybridDetection], 
                               analysis_time: float) -> None:
        """Print analysis results to console"""
        
        print(f"⏱️ Analysis completed in {analysis_time:.2f} seconds")
        print(f"🎯 Found {len(detections)} tumor(s)")
        
        if not detections:
            print("✅ Normal brain scan - No tumors detected")
            return
        
        print("\n📊 DETECTION RESULTS:")
        print("=" * 60)
        
        for i, detection in enumerate(detections, 1):
            cls_name = self._get_class_name(detection.cls_id)
            
            print(f"\n🔸 TUMOR #{i}: {cls_name}")
            print(f"   Confidence: {detection.conf:.1%}")
            print(f"   Detection Method: YOLO")
            print(f"   Segmentation Method: {detection.segmentation_method.upper()}")
            print(f"   Quality Score: {detection.segmentation_quality:.1%}")
            
            # Get measurements
            measurements = detection.get_precise_measurements((640, 640), mm_per_pixel=0.5)
            print(f"   Size: {measurements['width_mm']:.1f} × {measurements['height_mm']:.1f} mm")
            print(f"   Area: {measurements['area_mm2']:.1f} mm² ({measurements['area_cm2']:.2f} cm²)")
            print(f"   Precision: {measurements['precision']}")
            
            # Bounding box
            x1, y1, x2, y2 = detection.xyxy
            print(f"   Location: ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})")
    
    def _save_analysis_results(self, image_path: str, detections: List[HybridDetection], 
                              analysis_time: float) -> None:
        """Save analysis results to JSON file"""
        
        results = {
            'image_path': image_path,
            'analysis_time': analysis_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_detections': len(detections),
            'detections': []
        }
        
        for detection in detections:
            det_data = detection.to_dict()
            det_data['class_name'] = self._get_class_name(detection.cls_id)
            measurements = detection.get_precise_measurements((640, 640), mm_per_pixel=0.5)
            det_data['measurements'] = measurements
            results['detections'].append(det_data)
        
        # Save to JSON
        output_dir = "demo_results"
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = Path(image_path).stem
        json_path = os.path.join(output_dir, f"{base_name}_hybrid_results.json")
        
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {json_path}")
    
    def create_visualization(self, image_path: str, detections: List[HybridDetection], 
                           save_path: str = None) -> None:
        """Create visualization of hybrid detection results"""
        
        if not detections:
            print("ℹ️ No detections to visualize")
            return
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Brain Scan', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Annotated image
        annotated = self._create_annotated_image(image, detections)
        axes[1].imshow(annotated)
        axes[1].set_title('Hybrid Detection Results (YOLO + U-Net)', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Add detection info as text
        info_text = self._create_info_text(detections)
        fig.text(0.02, 0.02, info_text, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"🖼️ Visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _create_annotated_image(self, image: Image.Image, 
                               detections: List[HybridDetection]) -> Image.Image:
        """Create annotated image with detection results"""
        
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection.xyxy
            
            # Choose color based on segmentation method
            if detection.segmentation_method == 'unet':
                color = (255, 0, 0)  # Red for U-Net
                method_label = "U-Net"
            else:
                color = (255, 165, 0)  # Orange for YOLO
                method_label = "YOLO"
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Create label
            cls_name = self._get_class_name(detection.cls_id)
            label = f"{cls_name} {detection.conf:.2f} ({method_label})"
            
            # Draw label background
            bbox = draw.textbbox((x1, y1-25), label, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, y1-25), label, fill=(255, 255, 255), font=font)
            
            # Draw tumor number
            number_label = f"#{i+1}"
            draw.text((x1+5, y1+5), number_label, fill=color, font=font)
        
        return annotated
    
    def _create_info_text(self, detections: List[HybridDetection]) -> str:
        """Create information text for visualization"""
        
        lines = [f"Hybrid Analysis Results: {len(detections)} tumor(s) detected"]
        lines.append("=" * 50)
        
        for i, detection in enumerate(detections, 1):
            cls_name = self._get_class_name(detection.cls_id)
            measurements = detection.get_precise_measurements((640, 640), mm_per_pixel=0.5)
            
            lines.append(f"#{i} {cls_name}: {detection.conf:.1%} conf, "
                        f"{measurements['area_cm2']:.2f} cm², "
                        f"{detection.segmentation_method} seg")
        
        return "\n".join(lines)
    
    def batch_analyze(self, image_directory: str, pattern: str = "*.jpg", 
                     conf_threshold: float = 0.25) -> Dict[str, Any]:
        """Analyze multiple images in a directory"""
        
        image_dir = Path(image_directory)
        if not image_dir.exists():
            print(f"❌ Directory not found: {image_directory}")
            return {}
        
        # Find images
        image_files = list(image_dir.glob(pattern))
        if not image_files:
            print(f"❌ No images found matching pattern: {pattern}")
            return {}
        
        print(f"\n📁 Batch Analysis: {len(image_files)} images")
        print("=" * 50)
        
        batch_results = {
            'total_images': len(image_files),
            'total_tumors': 0,
            'analysis_time': 0,
            'results': []
        }
        
        start_time = time.time()
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing: {image_path.name}")
            
            detections = self.analyze_image(str(image_path), conf_threshold, save_results=False)
            
            batch_results['total_tumors'] += len(detections)
            batch_results['results'].append({
                'image': image_path.name,
                'num_tumors': len(detections),
                'detections': [d.to_dict() for d in detections]
            })
        
        batch_results['analysis_time'] = time.time() - start_time
        
        # Print summary
        print(f"\n📊 BATCH ANALYSIS SUMMARY")
        print("=" * 40)
        print(f"Images processed: {batch_results['total_images']}")
        print(f"Total tumors found: {batch_results['total_tumors']}")
        print(f"Average per image: {batch_results['total_tumors']/batch_results['total_images']:.1f}")
        print(f"Total time: {batch_results['analysis_time']:.1f} seconds")
        print(f"Average per image: {batch_results['analysis_time']/batch_results['total_images']:.1f} seconds")
        
        # Save batch results
        output_path = "demo_results/batch_analysis_results.json"
        os.makedirs("demo_results", exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(batch_results, f, indent=2)
        print(f"💾 Batch results saved to: {output_path}")
        
        return batch_results
    
    def compare_methods(self, image_path: str) -> None:
        """Compare YOLO-only vs Hybrid (YOLO+U-Net) results"""
        
        if self.yolo_model is None:
            print("❌ Models not initialized")
            return
        
        print(f"\n🔬 Method Comparison: {os.path.basename(image_path)}")
        print("=" * 60)
        
        # YOLO-only analysis
        print("1️⃣ YOLO-Only Analysis:")
        yolo_start = time.time()
        yolo_results = self.yolo_model.predict(image_path, conf=0.25, verbose=False)
        yolo_time = time.time() - yolo_start
        yolo_detections = len(yolo_results[0].boxes) if yolo_results[0].boxes is not None else 0
        print(f"   Time: {yolo_time:.2f}s | Detections: {yolo_detections}")
        
        # Hybrid analysis
        print("2️⃣ Hybrid (YOLO + U-Net) Analysis:")
        hybrid_start = time.time()
        hybrid_detections = self.analyze_image(image_path, save_results=False)
        hybrid_time = time.time() - hybrid_start
        print(f"   Time: {hybrid_time:.2f}s | Enhanced Detections: {len(hybrid_detections)}")
        
        # Comparison
        print("\n📈 COMPARISON RESULTS:")
        print(f"   Speed Difference: {hybrid_time/yolo_time:.1f}x slower (expected due to U-Net)")
        print(f"   Precision Gain: U-Net provides pixel-level segmentation")
        print(f"   Clinical Value: Hybrid approach suitable for treatment planning")
        
        # Create comparison visualization
        if hybrid_detections:
            output_path = f"demo_results/{Path(image_path).stem}_comparison.png"
            self.create_visualization(image_path, hybrid_detections, output_path)
    
    def _get_class_name(self, cls_id: int) -> str:
        """Get class name from ID"""
        class_names = {
            0: 'NO_tumor',
            1: 'glioma', 
            2: 'meningioma',
            3: 'pituitary',
            4: 'space-occupying lesion'
        }
        return class_names.get(cls_id, f'class_{cls_id}')


def main():
    """Main demonstration function"""
    parser = argparse.ArgumentParser(description='Hybrid Brain Tumor Detection Demo')
    parser.add_argument('--image', type=str, help='Single image to analyze')
    parser.add_argument('--batch', type=str, help='Directory for batch analysis')
    parser.add_argument('--pattern', type=str, default='*.jpg', help='File pattern for batch analysis')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--yolo_model', type=str, default='yolo11n.pt', help='YOLO model path')
    parser.add_argument('--unet_model', type=str, default='projectYolov9/unet_brain_tumor.pth', 
                       help='U-Net model path')
    parser.add_argument('--compare', action='store_true', help='Compare YOLO vs Hybrid methods')
    parser.add_argument('--visualize', action='store_true', help='Create visualization')
    
    args = parser.parse_args()
    
    # Create demo instance
    demo = HybridDemo(args.yolo_model, args.unet_model)
    
    # Initialize models
    if not demo.initialize_models():
        print("❌ Failed to initialize models. Exiting.")
        return
    
    # Single image analysis
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ Image not found: {args.image}")
            return
        
        detections = demo.analyze_image(args.image, args.conf)
        
        if args.visualize and detections:
            output_path = f"demo_results/{Path(args.image).stem}_hybrid_result.png"
            demo.create_visualization(args.image, detections, output_path)
        
        if args.compare:
            demo.compare_methods(args.image)
    
    # Batch analysis
    elif args.batch:
        demo.batch_analyze(args.batch, args.pattern, args.conf)
    
    # Demo with sample images
    else:
        print("\n🎯 DEMO MODE: Looking for sample images...")
        
        # Look for sample images in common locations
        sample_locations = [
            "projectYolov9/Tumor-Detection-8/test/images",
            "projectYolov9/Tumor-Detection-8/valid/images", 
            "sample_images",
            "."
        ]
        
        sample_found = False
        for location in sample_locations:
            if os.path.exists(location):
                image_files = list(Path(location).glob("*.jpg"))
                if image_files:
                    print(f"📁 Found sample images in: {location}")
                    sample_image = str(image_files[0])
                    print(f"🖼️ Analyzing sample: {sample_image}")
                    
                    detections = demo.analyze_image(sample_image, args.conf)
                    
                    if detections:
                        output_path = f"demo_results/sample_hybrid_result.png"
                        demo.create_visualization(sample_image, detections, output_path)
                        
                        if args.compare:
                            demo.compare_methods(sample_image)
                    
                    sample_found = True
                    break
        
        if not sample_found:
            print("ℹ️ No sample images found. Usage examples:")
            print("   python demo_hybrid.py --image path/to/brain_scan.jpg")
            print("   python demo_hybrid.py --batch path/to/image_directory")
            print("   python demo_hybrid.py --image scan.jpg --visualize --compare")


if __name__ == "__main__":
    main()