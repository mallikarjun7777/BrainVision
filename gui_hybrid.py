import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List, Optional, Tuple, Dict
from datetime import datetime
import math
import webbrowser

import numpy as np
from PIL import Image, ImageTk, ImageDraw

# Import our hybrid detection system
from unet_model import TumorUNet, create_pretrained_unet
from hybrid_detection import HybridDetection, HybridInferenceEngine

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as pdf_canvas
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
except Exception:
    A4 = None
    pdf_canvas = None
    mm = 1
    ImageReader = None

try:
    from ultralytics import YOLO
except Exception as exc:
    YOLO = None

# Medical knowledge base (same as before)
TUMOR_KNOWLEDGE_BASE = {
    "meningioma": {
        "name": "Meningioma",
        "description": "Typically benign tumor arising from meninges (brain covering)",
        "daily_tips": [
            "Monitor for changes in headache patterns or vision",
            "Avoid head trauma - wear protective gear during activities",
            "Stay hydrated (8-10 glasses water daily)",
            "Limit alcohol and caffeine intake",
            "Maintain consistent sleep schedule (7-8 hours)",
            "Practice stress-reduction techniques (meditation, yoga)"
        ],
        "warning_signs": [
            "Sudden severe headache (thunderclap headache)",
            "New onset seizures or worsening seizure activity",
            "Vision changes (double vision, blurriness, loss of vision)",
            "Weakness or numbness in limbs",
            "Changes in personality or cognitive function"
        ]
    },
    "glioma": {
        "name": "Glioma",
        "description": "Tumor arising from glial cells; varies from low-grade to high-grade (aggressive)",
        "daily_tips": [
            "Take anti-seizure medication exactly as prescribed",
            "Avoid driving if experiencing seizures",
            "Use pillbox organizer for complex medication schedules",
            "Gentle exercise as tolerated (walking, tai chi)",
            "Cognitive exercises: puzzles, reading, memory games"
        ],
        "warning_signs": [
            "New or worsening seizures",
            "Severe persistent headaches not relieved by medication",
            "Sudden confusion or disorientation",
            "Progressive weakness on one side of body"
        ]
    },
    "pituitary": {
        "name": "Pituitary Adenoma",
        "description": "Usually benign tumor of pituitary gland; may affect hormone levels",
        "daily_tips": [
            "Take hormone replacement medications at same time daily",
            "Monitor for vision changes (peripheral vision loss)",
            "Keep glucose monitoring if diabetic or at risk",
            "Wear medical alert bracelet if on hormone replacement"
        ],
        "warning_signs": [
            "Sudden severe headache with vision loss (pituitary apoplexy - EMERGENCY)",
            "Rapid vision changes or loss",
            "Severe fatigue or weakness (adrenal crisis)",
            "Excessive thirst and urination (diabetes insipidus)"
        ]
    }
}

DEFAULT_TUMOR_INFO = {
    "name": "Brain Lesion",
    "description": "Abnormal growth detected in brain tissue requiring further evaluation",
    "daily_tips": [
        "Monitor for any new or worsening neurological symptoms",
        "Maintain regular medication schedule if prescribed",
        "Avoid activities that could result in head injury"
    ],
    "warning_signs": [
        "Sudden severe headache",
        "New onset seizures",
        "Vision changes or loss",
        "Weakness or numbness in limbs"
    ]
}

class HybridTumorApp(tk.Tk):
    """Enhanced tumor detection app using YOLO + U-Net hybrid approach"""
    
    def __init__(self) -> None:
        super().__init__()
        self.title("🧠 Hybrid Brain Tumor Detection System (YOLO + U-Net)")
        self.geometry("1300x800")
        self.minsize(1200, 700)
        
        # Professional styling
        self.configure(bg='#f0f4f8')
        style = ttk.Style()
        
        
        # Variables
        self.model_path_var = tk.StringVar(value=self._default_model_path())
        self.image_path_var = tk.StringVar()
        self.name_var = tk.StringVar()
        self.age_var = tk.StringVar()
        self.gender_var = tk.StringVar(value="Male")
        self.patient_id_var = tk.StringVar()
        self.conf_var = tk.DoubleVar(value=0.25)
        self.imgsz_var = tk.IntVar(value=640)
        
        # Image and model state
        self.original_image: Optional[Image.Image] = None
        self.annotated_image: Optional[Image.Image] = None
        self.photo_left: Optional[ImageTk.PhotoImage] = None
        self.photo_right: Optional[ImageTk.PhotoImage] = None
        
        # Models
        self.yolo_model: Optional[YOLO] = None
        self.unet_model: Optional[TumorUNet] = None
        self.hybrid_engine: Optional[HybridInferenceEngine] = None
        
        # Results
        self.last_detections: List[HybridDetection] = []
        self.last_seg_overlay: Optional[Image.Image] = None
        self.last_tumor_crops: List[Image.Image] = []
        
        self._build_ui()
        self._initialize_models()

    def _initialize_models(self):
        """Initialize YOLO and U-Net models"""
        print("🚀 Initializing hybrid detection system...")
        
        # Create U-Net model if it doesn't exist
        unet_path = "unet_brain_tumor.pth"
        if not os.path.exists(unet_path):
            print("📦 Creating demonstration U-Net model...")
            create_pretrained_unet(unet_path)
        
        try:
            # Initialize U-Net
            self.unet_model = TumorUNet(unet_path)
            print("✓ U-Net model initialized")
            
            # Update status
            self._update_status("✓ Hybrid system ready (YOLO + U-Net)")
            
        except Exception as e:
            print(f"⚠️ Error initializing models: {e}")
            self._update_status("⚠️ Model initialization failed")

    def _build_ui(self) -> None:
        # Header
        header = tk.Frame(self, bg='#2c5f7c', height=60)
        header.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        header.pack_propagate(False)
        
        title_label = tk.Label(header, 
                              text="🧠 HYBRID BRAIN TUMOR DETECTION (YOLO Detection + U-Net Segmentation)", 
                              bg='#2c5f7c', fg='#ffffff', 
                              font=('Segoe UI', 14, 'bold'))
        title_label.pack(side=tk.LEFT, padx=20, pady=15)
        
        # Status label
        self.status_label = tk.Label(header, text="Initializing...", 
                                   bg='#2c5f7c', fg='#ffff99', 
                                   font=('Segoe UI', 10))
        self.status_label.pack(side=tk.RIGHT, padx=20, pady=15)

        # Main container
        main_container = ttk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Controls area
        controls = ttk.Frame(main_container)
        controls.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        # Patient Information
        patient_frame = ttk.LabelFrame(controls, text="📋 Patient Information", padding=15)
        patient_frame.pack(side=tk.LEFT, padx=(0, 10), pady=5)

        self._add_labeled_entry(patient_frame, "Patient ID", self.patient_id_var, width=15)
        self._add_labeled_entry(patient_frame, "Full Name", self.name_var, width=20)
        self._add_labeled_entry(patient_frame, "Age", self.age_var, width=6)
        
        gender_row = ttk.Frame(patient_frame)
        gender_row.pack(side=tk.TOP, anchor=tk.W, pady=4)
        ttk.Label(gender_row, text="Gender", font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(0, 10))
        gender_combo = ttk.Combobox(gender_row, textvariable=self.gender_var, 
                                   values=["Male", "Female", "Other"], width=12, state="readonly")
        gender_combo.pack(side=tk.LEFT)

        # Model Configuration
        model_frame = ttk.LabelFrame(controls, text="⚙️ Hybrid Model Configuration", padding=15)
        model_frame.pack(side=tk.LEFT, padx=10, pady=5)

        self._add_path_selector(model_frame, "YOLO Model", self.model_path_var, self._choose_model, width=30)
        
        settings_row = ttk.Frame(model_frame)
        settings_row.pack(side=tk.TOP, anchor=tk.W, pady=4)
        ttk.Label(settings_row, text="Confidence", font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(0, 6))
        conf_spin = ttk.Spinbox(settings_row, from_=0.05, to=0.95, increment=0.05, 
                               textvariable=self.conf_var, width=8)
        conf_spin.pack(side=tk.LEFT, padx=(0, 15))
        
        ttk.Label(settings_row, text="Image Size", font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(0, 6))
        imgsz_spin = ttk.Spinbox(settings_row, from_=320, to=1280, increment=64, 
                                textvariable=self.imgsz_var, width=8)
        imgsz_spin.pack(side=tk.LEFT)

        # Analysis Actions
        action_frame = ttk.LabelFrame(controls, text="🔬 Hybrid Analysis", padding=15)
        action_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.BOTH, expand=True)

        self._add_path_selector(action_frame, "Brain Scan", self.image_path_var, 
                               self._choose_image, width=25)

        btn_container = ttk.Frame(action_frame)
        btn_container.pack(side=tk.TOP, anchor=tk.W, pady=(8, 0))

        ttk.Button(btn_container, text="🔄 Load YOLO", command=self._on_load_yolo).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_container, text="🎯 Run Hybrid Analysis", command=self._on_run_hybrid).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_container, text="🔍 Zoom View", command=self._on_zoom).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_container, text="✂️ Extract Tumors", command=self._on_extract).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_container, text="📄 Generate Report", command=self._on_report).pack(side=tk.LEFT, padx=5)

        # Display panels
        display = ttk.Frame(main_container)
        display.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=10)

        left_panel = ttk.LabelFrame(display, text="📷 Original Scan", padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        right_panel = ttk.LabelFrame(display, text="🎯 Hybrid Detection Results (YOLO + U-Net)", padding=10)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.left_image_label = ttk.Label(left_panel, background='#000000')
        self.left_image_label.pack(fill=tk.BOTH, expand=True)
        
        self.right_image_label = ttk.Label(right_panel, background='#000000')
        self.right_image_label.pack(fill=tk.BOTH, expand=True)

        # Results panel
        results_frame = ttk.LabelFrame(main_container, text="📊 Hybrid Analysis Results & Clinical Assessment", padding=15)
        results_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=(10, 0))
        
        text_container = ttk.Frame(results_frame)
        text_container.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(text_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.results_text = tk.Text(text_container, wrap=tk.WORD, 
                                   font=('Segoe UI', 9), 
                                   bg='#f8f9fa', fg='#2c3e50',
                                   relief='flat', borderwidth=2,
                                   yscrollcommand=scrollbar.set,
                                   cursor="arrow")
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.config(command=self.results_text.yview)
        
        self.results_text.configure(state=tk.DISABLED)

    def _update_status(self, message: str):
        """Update status label"""
        self.status_label.config(text=message)
        self.update_idletasks()

    def _add_labeled_entry(self, parent: tk.Widget, label: str, var: tk.StringVar, width: int = 12) -> None:
        row = ttk.Frame(parent)
        row.pack(side=tk.TOP, anchor=tk.W, pady=4)
        ttk.Label(row, text=label, font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(0, 10))
        entry = ttk.Entry(row, textvariable=var, width=width, font=('Segoe UI', 9))
        entry.pack(side=tk.LEFT)

    def _add_path_selector(self, parent: tk.Widget, label: str, var: tk.StringVar, 
                          chooser, width: int = 40) -> None:
        row = ttk.Frame(parent)
        row.pack(side=tk.TOP, anchor=tk.W, pady=4, fill=tk.X)
        ttk.Label(row, text=label, font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(0, 10))
        entry = ttk.Entry(row, textvariable=var, width=width, font=('Segoe UI', 8))
        entry.pack(side=tk.LEFT, padx=(0, 6), fill=tk.X, expand=True)
        ttk.Button(row, text="Browse", command=chooser).pack(side=tk.LEFT)

    def _choose_model(self) -> None:
        path = filedialog.askopenfilename(title="Select YOLO model .pt file", 
                                         filetypes=[("PyTorch Weights", "*.pt"), ("All Files", "*.*")])
        if path:
            self.model_path_var.set(path)

    def _choose_image(self) -> None:
        path = filedialog.askopenfilename(title="Select brain scan image", 
                                         filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff"), 
                                                   ("All Files", "*.*")])
        if path:
            self.image_path_var.set(path)
            self._load_and_show_original()

    def _load_and_show_original(self) -> None:
        img_path = self.image_path_var.get().strip()
        if not img_path or not os.path.isfile(img_path):
            return
        try:
            self.original_image = Image.open(img_path).convert("RGB")
            self._show_image(self.original_image, target="left")
            self._update_status("✓ Image loaded - Ready for hybrid analysis")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load image: {exc}")

    def _on_load_yolo(self) -> None:
        if YOLO is None:
            messagebox.showerror("Ultralytics not found", 
                               "Ultralytics is not installed. Please install 'ultralytics' in your environment.")
            return
        
        model_path = self.model_path_var.get().strip()
        if not model_path or not os.path.isfile(model_path):
            messagebox.showerror("Model not found", "Please select a valid YOLO model .pt file.")
            return
        
        try:
            self._update_status("🔄 Loading YOLO model...")
            self.yolo_model = YOLO(model_path)
            
            # Create hybrid engine
            if self.unet_model is not None:
                self.hybrid_engine = HybridInferenceEngine(self.yolo_model, self.unet_model)
                self._update_status("✓ Hybrid system ready (YOLO + U-Net)")
                messagebox.showinfo("Models Loaded", 
                                  f"✓ YOLO model loaded: {os.path.basename(model_path)}\n"
                                  f"✓ U-Net model ready\n"
                                  f"✓ Hybrid detection system initialized")
            else:
                self._update_status("⚠️ YOLO loaded, U-Net unavailable")
                messagebox.showwarning("Partial Load", "YOLO loaded but U-Net unavailable")
                
        except Exception as exc:
            self._update_status("❌ Model loading failed")
            messagebox.showerror("Error", f"Failed to load YOLO model: {exc}")

    def _on_run_hybrid(self) -> None:
        if self.hybrid_engine is None:
            self._on_load_yolo()
            if self.hybrid_engine is None:
                return
        
        img_path = self.image_path_var.get().strip()
        if not img_path or not os.path.isfile(img_path):
            messagebox.showwarning("Image Required", "Please select a brain scan image to analyze.")
            return
        
        try:
            age_val = int(self.age_var.get()) if self.age_var.get().strip() else None
        except ValueError:
            messagebox.showwarning("Invalid Age", "Please enter a valid integer for age.")
            return

        # Run hybrid analysis in separate thread
        threading.Thread(target=self._run_hybrid_analysis_thread, 
                        args=(img_path, age_val, self.gender_var.get()), daemon=True).start()

    def _run_hybrid_analysis_thread(self, img_path: str, age: Optional[int], gender: str) -> None:
        try:
            self.after(0, lambda: self._update_status("🔍 Running YOLO detection..."))
            
            # Run hybrid inference
            detections = self.hybrid_engine.run_hybrid_inference(img_path, float(self.conf_var.get()))
            
            self.after(0, lambda: self._update_status("🎨 Creating visualizations..."))
            
            # Create visualizations
            annotated = self._create_hybrid_visualization(Image.open(img_path).convert("RGB"), detections)
            seg_overlay, crops = self._create_segmentation_overlay(Image.open(img_path).convert("RGB"), detections)
            
            def update_ui():
                self.last_detections = detections
                self.annotated_image = annotated
                self.last_seg_overlay = seg_overlay
                self.last_tumor_crops = crops
                
                self._show_image(annotated, target="right")
                
                # Generate comprehensive results
                summary = self._format_hybrid_results(detections, annotated.size)
                recommendations = self._generate_clinical_recommendations(age, gender, detections, annotated.size)
                
                full_results = summary + "\n\n" + recommendations
                self._set_results_text(full_results)
                
                self._update_status(f"✓ Analysis complete: {len(detections)} detections with U-Net segmentation")
            
            self.after(0, update_ui)
            
        except Exception as exc:
            self.after(0, lambda: self._update_status("❌ Analysis failed"))
            self.after(0, lambda: self._set_results_text(f"⚠️ Error during hybrid analysis: {exc}"))

    def _create_hybrid_visualization(self, image: Image.Image, detections: List[HybridDetection]) -> Image.Image:
        """Create visualization showing both YOLO detection and U-Net segmentation"""
        annotated = image.copy()
        overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        draw = ImageDraw.Draw(annotated)

        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection.xyxy
            
            # Use U-Net mask if available, otherwise YOLO mask
            if detection.unet_mask is not None:
                # U-Net segmentation (more precise) - use brighter red
                mask = self._resize_mask_to_image(detection.unet_mask, annotated.size)
                ys, xs = np.where(mask > 0)
                for y, x in zip(ys, xs):
                    overlay.putpixel((int(x), int(y)), (255, 50, 50, 140))  # Bright red for U-Net
                    
                segmentation_type = "U-Net"
                box_color = (255, 0, 0)  # Red for U-Net
                
            elif detection.yolo_mask is not None:
                # YOLO segmentation - use orange
                mask = self._resize_mask_to_image(detection.yolo_mask, annotated.size)
                ys, xs = np.where(mask > 0.5)
                for y, x in zip(ys, xs):
                    overlay.putpixel((int(x), int(y)), (255, 165, 0, 110))  # Orange for YOLO
                    
                segmentation_type = "YOLO"
                box_color = (255, 165, 0)  # Orange for YOLO
                
            else:
                # Bounding box only
                overlay_draw.rectangle([x1, y1, x2, y2], fill=(128, 128, 128, 80))
                segmentation_type = "Box"
                box_color = (128, 128, 128)  # Gray for box only
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
            
            # Create label with segmentation method
            cls_name = self._get_class_name(detection.cls_id)
            label = f"{cls_name} {detection.conf:.2f} ({segmentation_type})"
            
            # Draw label background and text
            try:
                bbox = overlay_draw.textbbox((0, 0), label)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            except Exception:
                text_w = len(label) * 8
                text_h = 16
            
            padding = 4
            text_bg = [x1, max(0, y1 - (text_h + 2 * padding)), x1 + text_w + 2 * padding, y1]
            overlay_draw.rectangle(text_bg, fill=(0, 0, 0, 200))
            overlay_draw.text((x1 + padding, y1 - text_h - padding), label, fill=(255, 255, 255, 255))

        # Combine original image with overlay
        annotated = Image.alpha_composite(annotated.convert("RGBA"), overlay)
        return annotated.convert("RGB")

    def _create_segmentation_overlay(self, image: Image.Image, detections: List[HybridDetection]) -> Tuple[Optional[Image.Image], List[Image.Image]]:
        """Create segmentation overlay and tumor crops"""
        if not detections:
            return None, []
        
        w, h = image.size
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ov = ImageDraw.Draw(overlay)
        crops = []

        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.xyxy)
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            
            # Use U-Net mask for overlay if available
            if detection.unet_mask is not None:
                # Resize U-Net mask to crop size, then place in full image
                crop_w, crop_h = x2 - x1, y2 - y1
                if crop_w > 0 and crop_h > 0:
                    mask_resized = Image.fromarray((detection.unet_mask * 255).astype(np.uint8))
                    mask_resized = mask_resized.resize((crop_w, crop_h), Image.NEAREST)
                    mask_array = np.array(mask_resized) / 255.0
                    
                    # Place in overlay
                    for dy in range(crop_h):
                        for dx in range(crop_w):
                            if mask_array[dy, dx] > 0:
                                overlay.putpixel((x1 + dx, y1 + dy), (255, 0, 0, 140))
            else:
                # Fallback to bounding box
                ov.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, 110))
            
            # Create crop
            try:
                crop = image.crop((x1, y1, x2, y2))
                crops.append(crop)
            except Exception:
                pass

        overlay_img = overlay.convert("RGBA") if detections else None
        return overlay_img, crops

    def _resize_mask_to_image(self, mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize mask to match image size"""
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img = mask_img.resize(size, resample=Image.NEAREST)
        return np.array(mask_img) / 255.0

    def _format_hybrid_results(self, detections: List[HybridDetection], image_size: Tuple[int, int]) -> str:
        """Format hybrid detection results"""
        if not detections:
            return "✅ NORMAL BRAIN SCAN - No tumors detected"
        
        lines = [f"🎯 HYBRID DETECTION RESULTS ({len(detections)} tumor(s) found)"]
        lines.append("=" * 60)
        
        for i, detection in enumerate(detections, 1):
            cls_name = self._get_class_name(detection.cls_id)
            measurements = detection.get_precise_measurements(image_size, mm_per_pixel=0.5)
            
            lines.append(f"\n📍 TUMOR #{i}: {cls_name}")
            lines.append(f"   Detection Method: YOLO (confidence: {detection.conf:.1%})")
            lines.append(f"   Segmentation Method: {detection.segmentation_method.upper()}")
            lines.append(f"   Quality Score: {detection.segmentation_quality:.1%}")
            lines.append(f"   Size: {measurements['width_mm']:.1f} × {measurements['height_mm']:.1f} mm")
            lines.append(f"   Area: {measurements['area_mm2']:.1f} mm² ({measurements['area_cm2']:.2f} cm²)")
            lines.append(f"   Precision: {measurements['precision']}")
            
            # Calculate tumor-to-brain ratio
            total_area = image_size[0] * image_size[1] * (0.5 ** 2)  # mm²
            ratio = (measurements['area_mm2'] / total_area) * 100
            stage = self._determine_stage(ratio / 100)
            lines.append(f"   Tumor-to-Brain Ratio: {ratio:.2f}% (Stage {stage[0]} - {stage[1]})")
        
        return "\n".join(lines)

    def _generate_clinical_recommendations(self, age: Optional[int], gender: str, 
                                         detections: List[HybridDetection], 
                                         image_size: Tuple[int, int]) -> str:
        """Generate clinical recommendations based on hybrid analysis"""
        if not detections:
            return ""
        
        # Get primary tumor info
        primary_detection = detections[0]
        cls_name = self._get_class_name(primary_detection.cls_id)
        tumor_info = self._get_tumor_info(cls_name)
        
        patient_name = self.name_var.get().strip() or "Patient"
        age_txt = f", {age} years old" if age is not None else ""
        
        sections = []
        
        # Clinical assessment
        sections.append("🏥 CLINICAL ASSESSMENT")
        sections.append(f"Patient: {patient_name}{age_txt}")
        sections.append(f"Primary Finding: {tumor_info['name']}")
        sections.append(f"Detection Confidence: {primary_detection.conf:.1%}")
        sections.append(f"Segmentation Quality: {primary_detection.segmentation_quality:.1%}")
        sections.append(f"Analysis Method: Hybrid (YOLO + U-Net)")
        sections.append("")
        
        # Tumor description
        sections.append(f"📋 ABOUT {tumor_info['name'].upper()}")
        sections.append(tumor_info['description'])
        sections.append("")
        
        # Recommendations
        sections.append("💊 CLINICAL RECOMMENDATIONS")
        for tip in tumor_info.get('daily_tips', [])[:5]:
            sections.append(f"  • {tip}")
        sections.append("")
        
        # Warning signs
        sections.append("⚠️ WARNING SIGNS - SEEK IMMEDIATE CARE:")
        for sign in tumor_info.get('warning_signs', [])[:5]:
            sections.append(f"  • {sign}")
        sections.append("")
        
        # Technical notes
        sections.append("🔬 TECHNICAL NOTES")
        sections.append("• Detection performed using YOLO v9 object detection")
        sections.append("• Segmentation refined using U-Net neural network")
        sections.append("• Size measurements based on pixel-level segmentation")
        sections.append("• This analysis requires professional medical review")
        
        return "\n".join(sections)

    def _get_class_name(self, cls_id: int) -> str:
        """Get class name from ID"""
        class_names = {0: 'NO_tumor', 1: 'glioma', 2: 'meningioma', 3: 'pituitary', 4: 'space-occupying lesion'}
        return class_names.get(cls_id, f'class_{cls_id}')

    def _get_tumor_info(self, cls_name: str) -> dict:
        """Get tumor information from knowledge base"""
        if not cls_name:
            return DEFAULT_TUMOR_INFO
        
        normalized = cls_name.lower().strip()
        return TUMOR_KNOWLEDGE_BASE.get(normalized, DEFAULT_TUMOR_INFO)

    def _determine_stage(self, area_ratio: float) -> Tuple[str, str]:
        """Determine tumor stage based on size ratio"""
        if area_ratio < 0.05:
            return ("1", "Mild")
        elif area_ratio <= 0.15:
            return ("2", "Moderate")
        else:
            return ("3", "Severe")

    def _set_results_text(self, text: str) -> None:
        """Set results text"""
        self.results_text.configure(state=tk.NORMAL)
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, text)
        self.results_text.configure(state=tk.DISABLED)

    def _show_image(self, img: Image.Image, target: str) -> None:
        """Display image in specified panel"""
        label = self.left_image_label if target == "left" else self.right_image_label
        label.update_idletasks()
        w = max(200, label.winfo_width() or 500)
        h = max(200, label.winfo_height() or 400)
        img_copy = img.copy()
        img_copy.thumbnail((w, h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img_copy)
        label.configure(image=photo)
        if target == "left":
            self.photo_left = photo
        else:
            self.photo_right = photo

    def _on_zoom(self) -> None:
        """Show zoomed view of results"""
        if self.annotated_image is None:
            messagebox.showinfo("Zoom", "Please run hybrid analysis first.")
            return
        
        # Create zoom window with hybrid results
        zoom_window = tk.Toplevel(self)
        zoom_window.title("🔍 Hybrid Detection Results - Zoomed View")
        zoom_window.geometry("1000x800")
        
        # Scale up image
        big_img = self.annotated_image.copy()
        w, h = big_img.size
        big_img = big_img.resize((int(w * 2), int(h * 2)), Image.NEAREST)
        
        # Create scrollable canvas
        container = ttk.Frame(zoom_window)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        canvas = tk.Canvas(container, bg='#000000')
        h_scroll = ttk.Scrollbar(container, orient=tk.HORIZONTAL, command=canvas.xview)
        v_scroll = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        
        canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        photo = ImageTk.PhotoImage(big_img)
        canvas.create_image(0, 0, image=photo, anchor="nw")
        canvas.configure(scrollregion=canvas.bbox("all"))
        
        zoom_window._photo = photo  # Keep reference

    def _on_extract(self) -> None:
        """Show tumor extraction window"""
        if not self.last_tumor_crops:
            messagebox.showinfo("Extract", "Please run hybrid analysis first.")
            return
        
        # Create extraction window
        extract_window = tk.Toplevel(self)
        extract_window.title("✂️ Hybrid Tumor Extraction Results")
        extract_window.geometry("1200x800")
        
        # Show extracted tumors with U-Net precision
        main_frame = ttk.Frame(extract_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Info label
        info_label = ttk.Label(main_frame, 
                              text="🎯 Tumor regions extracted using hybrid YOLO+U-Net analysis",
                              font=('Segoe UI', 12, 'bold'))
        info_label.pack(pady=(0, 10))
        
        # Scrollable frame for crops
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Display crops in grid
        cols = 3
        for i, crop in enumerate(self.last_tumor_crops):
            row = i // cols
            col = i % cols
            
            crop_frame = ttk.LabelFrame(scrollable_frame, text=f"Tumor #{i+1}", padding=10)
            crop_frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            # Resize crop for display
            display_crop = crop.copy()
            display_crop.thumbnail((250, 250), Image.LANCZOS)
            crop_photo = ImageTk.PhotoImage(display_crop)
            
            crop_label = ttk.Label(crop_frame, image=crop_photo)
            crop_label.pack()
            
            # Save button
            def make_save_func(crop_img=crop, idx=i):
                return lambda: self._save_crop(crop_img, idx)
            
            save_btn = ttk.Button(crop_frame, text="💾 Save Tumor", command=make_save_func())
            save_btn.pack(pady=(5, 0))
            
            # Keep photo reference
            crop_label._photo = crop_photo
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _save_crop(self, crop: Image.Image, idx: int) -> None:
        """Save individual tumor crop"""
        try:
            filename = f"hybrid_tumor_{idx+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            path = filedialog.asksaveasfilename(
                defaultextension=".png",
                initialfile=filename,
                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All Files", "*.*")]
            )
            if path:
                crop.save(path)
                messagebox.showinfo("Saved", f"✓ Tumor region saved: {path}")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to save crop: {exc}")

    def _on_report(self) -> None:
        """Generate comprehensive PDF report"""
        if not self.last_detections:
            messagebox.showinfo("Report", "Please run hybrid analysis first.")
            return
        
        if pdf_canvas is None:
            messagebox.showerror("Report", "ReportLab not installed. Install with 'pip install reportlab'.")
            return
        
        patient_name = self.name_var.get().strip() or "Patient"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Hybrid_Brain_Tumor_Report_{patient_name}_{timestamp}.pdf"
        
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            initialfile=filename,
            filetypes=[("PDF", "*.pdf"), ("All Files", "*.*")]
        )
        
        if path:
            try:
                self._generate_hybrid_pdf_report(path)
                messagebox.showinfo("Report Generated", f"✓ Hybrid analysis report saved:\n{path}")
            except Exception as exc:
                messagebox.showerror("Error", f"Failed to generate report: {exc}")

    def _generate_hybrid_pdf_report(self, output_path: str) -> None:
        """Generate comprehensive PDF report with hybrid analysis results"""
        c = pdf_canvas.Canvas(output_path, pagesize=A4)
        page_w, page_h = A4
        margin = 20 * mm
        y = page_h - margin

        # Header
        c.setFillColorRGB(0.1, 0.3, 0.5)
        c.rect(0, page_h - 40*mm, page_w, 40*mm, fill=True, stroke=False)
        
        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(margin, page_h - 20*mm, "HYBRID BRAIN TUMOR ANALYSIS REPORT")
        c.setFont("Helvetica", 11)
        c.drawString(margin, page_h - 30*mm, "YOLO Detection + U-Net Segmentation System")
        
        report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        c.drawString(margin, page_h - 35*mm, f"Generated: {report_date}")
        
        y = page_h - 50*mm

        # Patient info
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y, "PATIENT INFORMATION")
        y -= 8*mm
        
        c.setFont("Helvetica", 11)
        patient_id = self.patient_id_var.get().strip() or "N/A"
        name = self.name_var.get().strip() or "N/A"
        age = self.age_var.get().strip() or "N/A"
        gender = self.gender_var.get().strip() or "N/A"
        
        c.drawString(margin, y, f"Patient ID: {patient_id}")
        y -= 6*mm
        c.drawString(margin, y, f"Name: {name}")
        y -= 6*mm
        c.drawString(margin, y, f"Age: {age}")
        c.drawString(margin + 60*mm, y, f"Gender: {gender}")
        y -= 15*mm

        # Hybrid analysis results
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y, "HYBRID ANALYSIS RESULTS")
        y -= 8*mm
        
        c.setFont("Helvetica", 10)
        if self.last_detections:
            c.drawString(margin, y, f"Detection Method: YOLO v9 + U-Net Segmentation")
            y -= 5*mm
            c.drawString(margin, y, f"Number of tumors detected: {len(self.last_detections)}")
            y -= 5*mm
            
            for i, detection in enumerate(self.last_detections, 1):
                if y < margin + 60*mm:
                    c.showPage()
                    y = page_h - margin
                    c.setFont("Helvetica", 10)
                
                cls_name = self._get_class_name(detection.cls_id)
                measurements = detection.get_precise_measurements(
                    self.annotated_image.size if self.annotated_image else (640, 640)
                )
                
                c.setFont("Helvetica-Bold", 11)
                c.drawString(margin, y, f"Tumor #{i}: {cls_name}")
                y -= 6*mm
                
                c.setFont("Helvetica", 10)
                c.drawString(margin + 5*mm, y, f"• Confidence: {detection.conf:.1%}")
                y -= 4*mm
                c.drawString(margin + 5*mm, y, f"• Segmentation: {detection.segmentation_method.upper()}")
                y -= 4*mm
                c.drawString(margin + 5*mm, y, f"• Quality Score: {detection.segmentation_quality:.1%}")
                y -= 4*mm
                c.drawString(margin + 5*mm, y, f"• Size: {measurements['width_mm']:.1f} × {measurements['height_mm']:.1f} mm")
                y -= 4*mm
                c.drawString(margin + 5*mm, y, f"• Area: {measurements['area_mm2']:.1f} mm² ({measurements['area_cm2']:.2f} cm²)")
                y -= 4*mm
                c.drawString(margin + 5*mm, y, f"• Precision: {measurements['precision']}")
                y -= 8*mm
        else:
            c.drawString(margin, y, "No tumors detected - Normal brain scan")
            y -= 10*mm

        # Add annotated image
        if self.annotated_image and y > margin + 100*mm:
            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin, y, "HYBRID DETECTION VISUALIZATION")
            y -= 8*mm
            
            # Scale image to fit
            img = self.annotated_image.copy()
            max_w = page_w - 2*margin
            max_h = 80*mm
            img_w, img_h = img.size
            scale = min(max_w / img_w, max_h / img_h)
            resized = img.resize((int(img_w * scale), int(img_h * scale)), Image.LANCZOS)
            
            c.drawImage(ImageReader(resized), margin, y - resized.height, 
                       width=resized.width, height=resized.height)
            y -= resized.height + 10*mm

        # Technical notes
        if y < margin + 40*mm:
            c.showPage()
            y = page_h - margin
        
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "TECHNICAL METHODOLOGY")
        y -= 8*mm
        
        c.setFont("Helvetica", 9)
        tech_notes = [
            "• Detection: YOLOv9 neural network for real-time tumor localization",
            "• Segmentation: U-Net architecture for pixel-precise tumor boundaries", 
            "• Hybrid Approach: Combines speed of YOLO with accuracy of U-Net",
            "• Size Measurements: Based on pixel-level segmentation masks",
            "• Quality Assessment: Automated evaluation of segmentation precision"
        ]
        
        for note in tech_notes:
            c.drawString(margin, y, note)
            y -= 5*mm

        # Disclaimer
        y -= 10*mm
        c.setFont("Helvetica-Bold", 10)
        c.drawString(margin, y, "IMPORTANT DISCLAIMER")
        y -= 6*mm
        
        c.setFont("Helvetica-Oblique", 9)
        disclaimer_text = [
            "This report is generated by an AI-assisted hybrid detection system combining YOLO and U-Net.",
            "Results must be reviewed and validated by qualified medical professionals.",
            "This system is intended for research and educational purposes only.",
            "Clinical decisions should not be based solely on this automated analysis."
        ]
        
        for line in disclaimer_text:
            c.drawString(margin, y, line)
            y -= 4*mm

        c.save()

    def _default_model_path(self) -> str:
        """Get default YOLO model path"""
        candidates = [
            "runs/detect/train/weights/best.pt",  # Trained model first
            "projectYolov9/runs/detect/train/weights/best.pt",
            "projectYolov9/yolo11n.pt",
            "yolo11n.pt"
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return ""


def main():
    """Launch the hybrid tumor detection application"""
    print("🚀 Starting Hybrid Brain Tumor Detection System...")
    print("📋 Features:")
    print("   • YOLO v9 for fast tumor detection")
    print("   • U-Net for precise segmentation")
    print("   • Hybrid analysis combining both methods")
    print("   • Professional medical reporting")
    print("   • Clinical recommendations")
    
    app = HybridTumorApp()
    app.mainloop()


if __name__ == "__main__":
    main()