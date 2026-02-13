#!/usr/bin/env python3
"""
Simple test script for the hybrid detection system
"""

import os
from unet_model import TumorUNet, create_pretrained_unet
from hybrid_detection import HybridInferenceEngine

try:
    from ultralytics import YOLO
    print("✓ Ultralytics available")
except ImportError:
    print("❌ Ultralytics not available")
    YOLO = None

def test_hybrid_system():
    """Test the hybrid detection system"""
    print("🧠 Testing Hybrid Brain Tumor Detection System")
    print("=" * 50)
    
    # Check if models exist
    yolo_path = "runs/detect/train/weights/best.pt"  # Use trained model
    unet_path = "unet_brain_tumor.pth"
    
    print(f"📦 YOLO model: {yolo_path} - {'✓' if os.path.exists(yolo_path) else '❌'}")
    print(f"🎯 U-Net model: {unet_path} - {'✓' if os.path.exists(unet_path) else '❌'}")
    
    # Create U-Net if needed
    if not os.path.exists(unet_path):
        print("🔧 Creating U-Net model...")
        create_pretrained_unet(unet_path)
        print("✓ U-Net model created")
    
    # Initialize models
    if YOLO is None:
        print("❌ Cannot test - YOLO not available")
        return
    
    try:
        print("🚀 Initializing models...")
        yolo_model = YOLO(yolo_path)
        unet_model = TumorUNet(unet_path)
        hybrid_engine = HybridInferenceEngine(yolo_model, unet_model)
        print("✓ Models initialized successfully")
        
        # Test with a sample image
        test_images = [
            "Tumor-Detection-8/test/images/glioma_1025_jpg.rf.16f677de0fc84afa6b702f2fb8bdb3c2.jpg",
            "Tumor-Detection-8/test/images/meningioma_1022_jpg.rf.a3ae957a204e1f240de0d48f7c95c0aa.jpg",
            "Tumor-Detection-8/test/images/pituitary_1006_jpg.rf.ea728f46f66b6ef65c3e4f11028bd72b.jpg"
        ]
        
        for test_image in test_images:
            if os.path.exists(test_image):
                print(f"\n🔍 Testing with: {os.path.basename(test_image)}")
                
                detections = hybrid_engine.run_hybrid_inference(test_image, conf_threshold=0.25)
                
                print(f"📊 Results: {len(detections)} detections")
                
                for i, detection in enumerate(detections, 1):
                    measurements = detection.get_precise_measurements((640, 640), mm_per_pixel=0.5)
                    print(f"   #{i}: {detection.conf:.1%} confidence, "
                          f"{detection.segmentation_method} segmentation, "
                          f"{measurements['area_cm2']:.2f} cm²")
                
                break
        else:
            print("❌ No test images found")
        
        print("\n✅ Hybrid system test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hybrid_system()