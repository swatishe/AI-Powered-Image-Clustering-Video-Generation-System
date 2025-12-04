"""
Feature extraction module using pre-trained deep learning models
"""

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple


class FeatureExtractor:
    """
    Extract feature embeddings from images using pre-trained CNN models
    
    Model Choice Justification:
    - ResNet50: Excellent balance of accuracy and speed, 2048-dim features
    - Proven effective for transfer learning and image similarity tasks
    - Pre-trained on ImageNet (1000 classes, 1.2M images)
    - Residual connections help capture both low and high-level features
    """
    
    def __init__(self, model_name: str = "resnet50"):
        """
        Initialize feature extractor
        
        Args:
            model_name: Model to use ('resnet50', 'vgg16', 'mobilenet')
        """
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Using device: {self.device}")
        
        self.model = self._load_model()
        self.transform = self._get_transforms()
        
    def _load_model(self):
        """Load pre-trained model and remove classification head"""
        print(f" Loading {self.model_name} model...")
        
        if self.model_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            # Remove final classification layer to get 2048-dim embeddings
            model = torch.nn.Sequential(*list(model.children())[:-1])
            
        elif self.model_name == "vgg16":
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            # Use features only (before classifier)
            model = model.features
            
        elif self.model_name == "mobilenet":
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            # Remove classifier
            model = torch.nn.Sequential(*list(model.children())[:-1])
            
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        model = model.to(self.device)
        model.eval()  # Set to evaluation mode
        
        print(f" Model loaded successfully")
        return model
    
    def _get_transforms(self):
        """Get image preprocessing transforms"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_single_image(self, image_path: str) -> np.ndarray:
        """
        Extract features from a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Feature vector as numpy array
        """
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(img_tensor)
            
            # Flatten and convert to numpy
            features = features.squeeze().cpu().numpy()
            if len(features.shape) > 1:
                features = features.flatten()
            
            return features
            
        except Exception as e:
            print(f"  Error processing {image_path}: {e}")
            # Return zero vector on error
            return np.zeros(2048 if self.model_name == "resnet50" else 512)
    
    def extract_features(
        self, 
        image_paths: List[str],
        batch_size: int = 32,
        save_path: str = None,
        paths_file: str = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features from multiple images
        
        Args:
            image_paths: List of image file paths
            batch_size: Number of images to process at once
            save_path: Path to save embeddings (optional)
            paths_file: Path to save image paths list (optional)
            
        Returns:
            Tuple of (embeddings array, list of image paths)
        """
        print(f"\n Extracting features from {len(image_paths)} images...")
        print(f"   Model: {self.model_name}")
        print(f"   Batch size: {batch_size}")
        
        embeddings = []
        valid_paths = []
        
        # Process images in batches for efficiency
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            batch_valid_paths = []
            
            # Load batch
            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = self.transform(img)
                    batch_tensors.append(img_tensor)
                    batch_valid_paths.append(img_path)
                except Exception as e:
                    print(f" Skipping {img_path}: {e}")
            
            if not batch_tensors:
                continue
            
            # Stack tensors and move to device
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Extract features
            with torch.no_grad():
                batch_features = self.model(batch_tensor)
            
            # Convert to numpy
            batch_features = batch_features.squeeze().cpu().numpy()
            if len(batch_features.shape) == 1:
                batch_features = batch_features.reshape(1, -1)
            
            # Flatten if needed
            if len(batch_features.shape) > 2:
                batch_features = batch_features.reshape(batch_features.shape[0], -1)
            
            embeddings.append(batch_features)
            valid_paths.extend(batch_valid_paths)
        
        # Concatenate all embeddings
        embeddings = np.vstack(embeddings)
        
        print(f" Extracted features shape: {embeddings.shape}")
        print(f"   Feature dimension: {embeddings.shape[1]}")
        
        # Save if requested
        if save_path:
            np.save(save_path, embeddings)
            print(f"💾 Saved embeddings to: {save_path}")
        
        if paths_file:
            with open(paths_file, 'w') as f:
                for path in valid_paths:
                    f.write(path + '\n')
            print(f" Saved image paths to: {paths_file}")
        
        return embeddings, valid_paths
    
    @staticmethod
    def load_embeddings(embeddings_path: str, paths_file: str = None) -> Tuple[np.ndarray, List[str]]:
        """
        Load previously saved embeddings
        
        Args:
            embeddings_path: Path to embeddings .npy file
            paths_file: Path to image paths text file (optional)
            
        Returns:
            Tuple of (embeddings array, list of image paths)
        """
        embeddings = np.load(embeddings_path)
        print(f" Loaded embeddings from: {embeddings_path}")
        print(f"   Shape: {embeddings.shape}")
        
        paths = []
        if paths_file:
            with open(paths_file, 'r') as f:
                paths = [line.strip() for line in f]
            print(f" Loaded {len(paths)} image paths")
        
        return embeddings, paths