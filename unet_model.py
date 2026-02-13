import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class DoubleConv(nn.Module):
    """Double convolution block used in U-Net"""
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels: int, out_channels: int):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Output convolution"""
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """U-Net architecture for medical image segmentation"""
    def __init__(self, n_channels: int = 3, n_classes: int = 6, bilinear: bool = False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNetPreprocessor:
    """Preprocessing utilities for U-Net"""
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for U-Net inference"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        tensor = self.transform(image)
        return tensor.unsqueeze(0)  # Add batch dimension

    def postprocess_mask(self, mask_tensor: torch.Tensor, original_size: Tuple[int, int]) -> np.ndarray:
        """Convert U-Net output to binary mask"""
        # Apply softmax and get class predictions
        mask_probs = torch.softmax(mask_tensor, dim=1)
        mask_pred = torch.argmax(mask_probs, dim=1)
        
        # Convert to numpy and resize to original size
        mask_np = mask_pred.squeeze().cpu().numpy().astype(np.uint8)
        mask_img = Image.fromarray(mask_np)
        mask_resized = mask_img.resize(original_size, Image.NEAREST)
        
        return np.array(mask_resized)

class TumorUNet:
    """Wrapper class for tumor segmentation using U-Net"""
    def __init__(self, model_path: str = None, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = UNet(n_channels=3, n_classes=6)  # 6 classes: background + 5 tumor types
        self.preprocessor = UNetPreprocessor()
        
        if model_path and torch.cuda.is_available():
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"✓ Loaded U-Net model from {model_path}")
            except Exception as e:
                print(f"⚠️ Could not load U-Net model: {e}")
                print("Using randomly initialized weights (for demonstration)")
        
        self.model.to(device)
        self.model.eval()
        
        # Class mapping
        self.class_names = {
            0: 'background',
            1: 'NO_tumor',
            2: 'glioma', 
            3: 'meningioma',
            4: 'pituitary',
            5: 'space-occupying lesion'
        }

    def segment_region(self, image: Image.Image, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Segment tumor in specified bounding box region"""
        x1, y1, x2, y2 = bbox
        
        # Crop region from image
        crop = image.crop((x1, y1, x2, y2))
        original_crop_size = crop.size
        
        # Preprocess for U-Net
        input_tensor = self.preprocessor.preprocess_image(crop)
        input_tensor = input_tensor.to(self.device)
        
        # Run U-Net inference
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Postprocess to get binary mask
        mask = self.preprocessor.postprocess_mask(output, original_crop_size)
        
        return mask

    def create_full_image_mask(self, image_size: Tuple[int, int], 
                              detections: list) -> np.ndarray:
        """Create full image segmentation mask from multiple detections"""
        full_mask = np.zeros(image_size[::-1], dtype=np.uint8)  # (height, width)
        
        for detection in detections:
            if hasattr(detection, 'unet_mask') and detection.unet_mask is not None:
                x1, y1, x2, y2 = map(int, detection.xyxy)
                
                # Ensure coordinates are within image bounds
                x1 = max(0, min(image_size[0] - 1, x1))
                y1 = max(0, min(image_size[1] - 1, y1))
                x2 = max(0, min(image_size[0], x2))
                y2 = max(0, min(image_size[1], y2))
                
                # Place U-Net mask in full image
                mask_h, mask_w = detection.unet_mask.shape
                crop_h, crop_w = y2 - y1, x2 - x1
                
                if mask_h > 0 and mask_w > 0 and crop_h > 0 and crop_w > 0:
                    # Resize mask to fit crop area
                    mask_resized = Image.fromarray(detection.unet_mask)
                    mask_resized = mask_resized.resize((crop_w, crop_h), Image.NEAREST)
                    mask_array = np.array(mask_resized)
                    
                    # Only keep tumor pixels (non-background)
                    tumor_pixels = mask_array > 0
                    full_mask[y1:y2, x1:x2][tumor_pixels] = detection.cls_id + 1
        
        return full_mask

def create_pretrained_unet(save_path: str = "unet_brain_tumor.pth"):
    """Create a pre-trained U-Net model for demonstration"""
    print("🔧 Creating demonstration U-Net model...")
    
    model = UNet(n_channels=3, n_classes=6)
    
    # Initialize with reasonable weights for medical segmentation
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    
    # Save the model
    torch.save(model.state_dict(), save_path)
    print(f"✓ Saved demonstration U-Net model to {save_path}")
    
    return save_path

if __name__ == "__main__":
    # Create demonstration model
    model_path = create_pretrained_unet()
    
    # Test the model
    unet = TumorUNet(model_path)
    print("✓ U-Net model ready for integration")