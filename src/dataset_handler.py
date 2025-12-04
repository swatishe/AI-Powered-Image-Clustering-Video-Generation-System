"""
Dataset handling and analysis module
@author sshende
"""

import os
from pathlib import Path
from typing import List, Dict
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter


class DatasetHandler:
    """Handles image dataset loading, validation, and analysis"""
    
    def __init__(self, image_dir: str, min_images: int = 200):
        """
        Initialize dataset handler
        
        Args:
            image_dir: Directory containing images
            min_images: Minimum required images
        """
        self.image_dir = Path(image_dir)
        self.min_images = min_images
        self.image_paths = []
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        self._load_image_paths()
    
    def _load_image_paths(self):
        """Load all valid image paths from directory"""
        if not self.image_dir.exists():
            print(f"  Directory not found: {self.image_dir}")
            return
        
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if Path(file).suffix.lower() in self.supported_formats:
                    self.image_paths.append(str(Path(root) / file))
        
        self.image_paths.sort()
        print(f" Found {len(self.image_paths)} images in {self.image_dir}")
    
    def validate_dataset(self) -> bool:
        """Check if dataset meets minimum requirements"""
        return len(self.image_paths) >= self.min_images
    
    def analyze_dataset(self) -> Dict:
        """
        Analyze dataset characteristics
        
        Returns:
            Dictionary with dataset statistics
        """
        print("\n Analyzing dataset...")
        
        widths, heights = [], []
        color_modes = []
        file_sizes = []
        
        # Sample images for analysis
        sample_size = min(100, len(self.image_paths))
        sample_paths = np.random.choice(self.image_paths, sample_size, replace=False)
        
        for img_path in sample_paths:
            try:
                with Image.open(img_path) as img:
                    widths.append(img.width)
                    heights.append(img.height)
                    color_modes.append(img.mode)
                    file_sizes.append(os.path.getsize(img_path) / 1024)  # KB
            except Exception as e:
                print(f"  Error loading {img_path}: {e}")
        
        mode_counts = Counter(color_modes)
        
        analysis = {
            'total_images': len(self.image_paths),
            'avg_width': np.mean(widths),
            'avg_height': np.mean(heights),
            'min_width': np.min(widths),
            'max_width': np.max(widths),
            'min_height': np.min(heights),
            'max_height': np.max(heights),
            'avg_file_size_kb': np.mean(file_sizes),
            'color_formats': dict(mode_counts)
        }
        
        print(f"\n Dataset Analysis:")
        print(f"   Total images: {analysis['total_images']}")
        print(f"   Dimensions: {analysis['avg_width']:.0f}x{analysis['avg_height']:.0f} (avg)")
        print(f"   Range: {analysis['min_width']}x{analysis['min_height']} to {analysis['max_width']}x{analysis['max_height']}")
        print(f"   Avg file size: {analysis['avg_file_size_kb']:.1f} KB")
        print(f"   Color formats: {analysis['color_formats']}")
        
        return analysis
    
    def display_samples(self, n_samples: int = 6):
        """Display sample images from dataset"""
        sample_indices = np.random.choice(len(self.image_paths), min(n_samples, len(self.image_paths)), replace=False)
        
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        
        for idx, ax in zip(sample_indices, axes):
            try:
                img = Image.open(self.image_paths[idx])
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f"Image {idx}", fontsize=10)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading\nimage {idx}", 
                       ha='center', va='center')
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('output/sample_images.png', dpi=100, bbox_inches='tight')
        print(f" Sample images saved to output/sample_images.png")
        plt.close()
    
    def save_analysis_report(self, output_path: str):
        """Save detailed analysis report to file"""
        analysis = self.analyze_dataset()
        
        with open(output_path, 'w') as f:
            f.write("DATASET ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Dataset Directory: {self.image_dir}\n")
            f.write(f"Total Images: {analysis['total_images']}\n\n")
            
            f.write("Image Dimensions:\n")
            f.write(f"  Average: {analysis['avg_width']:.0f}x{analysis['avg_height']:.0f}\n")
            f.write(f"  Minimum: {analysis['min_width']}x{analysis['min_height']}\n")
            f.write(f"  Maximum: {analysis['max_width']}x{analysis['max_height']}\n\n")
            
            f.write(f"Average File Size: {analysis['avg_file_size_kb']:.1f} KB\n\n")
            
            f.write("Color Formats:\n")
            for mode, count in analysis['color_formats'].items():
                f.write(f"  {mode}: {count} images\n")
        
        print(f" Analysis report saved to {output_path}")


# Example: Download dataset from Kaggle
def download_kaggle_dataset(dataset_name: str, output_dir: str = "data/images"):
    """
    Example function to download dataset from Kaggle
    
    Args:
        dataset_name: Kaggle dataset identifier (e.g., 'alessiocorrado99/animals10')
        output_dir: Directory to extract images
    
    Requirements:
        - Kaggle API installed: pip install kaggle
        - Kaggle credentials configured (~/.kaggle/kaggle.json)
    """
    import subprocess
    
    print(f" Downloading dataset: {dataset_name}")
    
    try:
        # Download dataset
        subprocess.run(['kaggle', 'datasets', 'download', '-d', dataset_name], check=True)
        
        # Extract
        zip_file = dataset_name.split('/')[-1] + '.zip'
        subprocess.run(['unzip', '-q', zip_file, '-d', output_dir], check=True)
        
        # Cleanup
        os.remove(zip_file)
        
        print(f" Dataset downloaded and extracted to {output_dir}")
        
    except subprocess.CalledProcessError as e:
        print(f" Error downloading dataset: {e}")
        print("Please ensure Kaggle API is configured correctly")
    except FileNotFoundError:
        print(" Kaggle CLI not found. Install with: pip install kaggle")