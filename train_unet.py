import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
import json

from unet_model import UNet, UNetPreprocessor

class BrainTumorDataset(Dataset):
    """Dataset class for brain tumor segmentation"""
    
    def __init__(self, images_dir: str, labels_dir: str, transform=None, target_size=(256, 256)):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.target_size = target_size
        
        # Get list of image files
        self.image_files = [f for f in os.listdir(images_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Found {len(self.image_files)} images in {images_dir}")
        
        # Class mapping for tumor types
        self.class_mapping = {
            'NO_tumor': 1,
            'glioma': 2, 
            'meningioma': 3,
            'pituitary': 4,
            'space-occupying lesion': 5
        }
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Create synthetic mask based on filename (for demonstration)
        # In real training, you'd load actual segmentation masks
        mask = self._create_synthetic_mask(img_name, image.size)
        
        # Resize image and mask
        image = image.resize(self.target_size, Image.LANCZOS)
        mask = mask.resize(self.target_size, Image.NEAREST)
        
        # Convert to tensors
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        mask = torch.from_numpy(np.array(mask)).long()
        
        return image, mask
    
    def _create_synthetic_mask(self, filename: str, image_size: tuple) -> Image.Image:
        """Create synthetic segmentation mask based on filename"""
        # This is a placeholder - in real training you'd load actual masks
        mask = np.zeros(image_size[::-1], dtype=np.uint8)  # (height, width)
        
        # Determine class from filename
        class_id = 0  # background
        for tumor_type, type_id in self.class_mapping.items():
            if tumor_type.lower() in filename.lower():
                class_id = type_id
                break
        
        if class_id > 0:
            # Create synthetic tumor region (circular)
            h, w = mask.shape
            center_x, center_y = w // 2, h // 2
            radius = min(w, h) // 6
            
            y, x = np.ogrid[:h, :w]
            mask_circle = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            mask[mask_circle] = class_id
        
        return Image.fromarray(mask)

class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        # Apply softmax to predictions
        predictions = torch.softmax(predictions, dim=1)
        
        # One-hot encode targets
        num_classes = predictions.shape[1]
        targets_one_hot = torch.zeros_like(predictions)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Calculate Dice coefficient for each class
        dice_scores = []
        for i in range(num_classes):
            pred_i = predictions[:, i]
            target_i = targets_one_hot[:, i]
            
            intersection = (pred_i * target_i).sum()
            union = pred_i.sum() + target_i.sum()
            
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        # Return 1 - mean dice (loss)
        return 1 - torch.stack(dice_scores).mean()

class UNetTrainer:
    """Trainer class for U-Net model"""
    
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss functions
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_dice = DiceLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_ious = []
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Calculate combined loss
            loss_ce = self.criterion_ce(outputs, masks)
            loss_dice = self.criterion_dice(outputs, masks)
            loss = loss_ce + loss_dice
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_ious = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                loss_ce = self.criterion_ce(outputs, masks)
                loss_dice = self.criterion_dice(outputs, masks)
                loss = loss_ce + loss_dice
                
                total_loss += loss.item()
                
                # Calculate IoU
                predictions = torch.argmax(outputs, dim=1)
                iou = self._calculate_iou(predictions, masks)
                all_ious.append(iou)
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'iou': f'{iou:.4f}'})
        
        avg_loss = total_loss / len(self.val_loader)
        avg_iou = np.mean(all_ious)
        
        return avg_loss, avg_iou
    
    def _calculate_iou(self, predictions, targets):
        """Calculate Intersection over Union"""
        predictions = predictions.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()
        
        # Calculate IoU for non-background classes
        iou_scores = []
        for class_id in range(1, 6):  # Classes 1-5 (excluding background)
            pred_mask = (predictions == class_id)
            target_mask = (targets == class_id)
            
            if target_mask.sum() == 0 and pred_mask.sum() == 0:
                continue  # Skip if no ground truth and no prediction
            
            intersection = (pred_mask & target_mask).sum()
            union = (pred_mask | target_mask).sum()
            
            if union > 0:
                iou_scores.append(intersection / union)
        
        return np.mean(iou_scores) if iou_scores else 0.0
    
    def train(self, num_epochs, save_dir='projectYolov9/unet_checkpoints'):
        """Train the model"""
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        best_iou = 0.0
        
        print(f"Starting U-Net training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_iou = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_ious.append(val_iou)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 
                          os.path.join(save_dir, 'best_unet_model.pth'))
                print("✓ Saved best model (lowest validation loss)")
            
            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(self.model.state_dict(), 
                          os.path.join(save_dir, 'best_iou_unet_model.pth'))
                print("✓ Saved best IoU model")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'val_ious': self.val_ious
                }
                torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Save final model
        torch.save(self.model.state_dict(), 
                  os.path.join(save_dir, 'final_unet_model.pth'))
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_ious': self.val_ious
        }
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f)
        
        print(f"\n✓ Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best validation IoU: {best_iou:.4f}")
        
        return history
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # IoU plot
        ax2.plot(self.val_ious, label='Validation IoU', color='green')
        ax2.set_title('Validation IoU')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('IoU')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Training plots saved to {save_path}")
        
        plt.show()

def create_data_loaders(data_dir, batch_size=8, val_split=0.2):
    """Create training and validation data loaders"""
    
    # For this example, we'll use the existing dataset structure
    train_images_dir = os.path.join(data_dir, 'train', 'images')
    val_images_dir = os.path.join(data_dir, 'valid', 'images')
    
    # Create dummy labels directories (in real scenario, you'd have actual segmentation masks)
    train_labels_dir = os.path.join(data_dir, 'train', 'masks')  # Would contain segmentation masks
    val_labels_dir = os.path.join(data_dir, 'valid', 'masks')
    
    # Create datasets
    train_dataset = BrainTumorDataset(train_images_dir, train_labels_dir)
    val_dataset = BrainTumorDataset(val_images_dir, val_labels_dir)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader

def main():
    parser = argparse.ArgumentParser(description='Train U-Net for brain tumor segmentation')
    parser.add_argument('--data_dir', type=str, default='projectYolov9/Tumor-Detection-8',
                       help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training')
    parser.add_argument('--save_dir', type=str, default='projectYolov9/unet_checkpoints',
                       help='Directory to save model checkpoints')
    
    args = parser.parse_args()
    
    print("🧠 U-Net Brain Tumor Segmentation Training")
    print("=" * 50)
    print(f"Device: {args.device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        print("Please ensure the Tumor-Detection-8 dataset is available")
        return
    
    try:
        # Create data loaders
        print("\n📊 Creating data loaders...")
        train_loader, val_loader = create_data_loaders(args.data_dir, args.batch_size)
        print(f"✓ Training samples: {len(train_loader.dataset)}")
        print(f"✓ Validation samples: {len(val_loader.dataset)}")
        
        # Create model
        print("\n🏗️ Creating U-Net model...")
        model = UNet(n_channels=3, n_classes=6)  # 6 classes: background + 5 tumor types
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        
        # Create trainer
        print(f"\n🚀 Initializing trainer on {args.device}...")
        trainer = UNetTrainer(model, train_loader, val_loader, args.device)
        
        # Train model
        print("\n🎯 Starting training...")
        history = trainer.train(args.epochs, args.save_dir)
        
        # Plot results
        print("\n📈 Generating training plots...")
        plot_path = os.path.join(args.save_dir, 'training_plots.png')
        trainer.plot_training_history(plot_path)
        
        # Copy best model to main directory for use in GUI
        import shutil
        best_model_path = os.path.join(args.save_dir, 'best_unet_model.pth')
        target_path = 'projectYolov9/unet_brain_tumor.pth'
        
        if os.path.exists(best_model_path):
            shutil.copy2(best_model_path, target_path)
            print(f"✓ Copied best model to {target_path}")
        
        print("\n🎉 Training completed successfully!")
        print(f"📁 Models saved in: {args.save_dir}")
        print(f"📊 Training history saved")
        print(f"📈 Training plots saved")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()