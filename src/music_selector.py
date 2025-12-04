"""
Algorithmic music selection based on image cluster characteristics
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict


class MusicSelector:
    """
    Algorithmically select background music based on visual characteristics
    
    Selection Algorithm:
    - Analyzes brightness, colorfulness, and contrast of image cluster
    - Maps visual features to emotional/musical categories
    - Selects appropriate music file from library
    """
    
    def __init__(self, music_dir: str = "music"):
        """
        Initialize music selector
        
        Args:
            music_dir: Directory containing music files
        """
        self.music_dir = Path(music_dir)
        self.music_files = self._load_music_files()
        
        print(f" Music Selector initialized")
        print(f"   Music directory: {self.music_dir}")
        print(f"   Available tracks: {len(self.music_files)}")
    
    def _load_music_files(self) -> List[str]:
        """Load available music files"""
        if not self.music_dir.exists():
            print(f"  Music directory not found: {self.music_dir}")
            self.music_dir.mkdir(parents=True, exist_ok=True)
            return []
        
        music_files = []
        for ext in ['.mp3', '.wav', '.m4a', '.ogg']:
            music_files.extend(list(self.music_dir.glob(f'*{ext}')))
        
        return [str(f) for f in music_files]
    
    def analyze_image_cluster(self, image_paths: List[str]) -> Dict:
        """
        Analyze visual characteristics of image cluster
        
        Args:
            image_paths: List of image paths in cluster
            
        Returns:
            Dictionary of visual features
        """
        print(f"\n Analyzing cluster characteristics for music selection...")
        
        brightness_values = []
        colorfulness_values = []
        contrast_values = []
        saturation_values = []
        
        # Sample images for analysis
        sample_size = min(30, len(image_paths))
        sample_paths = np.random.choice(image_paths, sample_size, replace=False)
        
        for img_path in sample_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img)
                
                # Brightness: average pixel intensity (0-255)
                brightness = np.mean(img_array)
                brightness_values.append(brightness)
                
                # Colorfulness: standard deviation across RGB channels
                colorfulness = np.std(img_array)
                colorfulness_values.append(colorfulness)
                
                # Contrast: difference between max and min intensities
                contrast = np.max(img_array) - np.min(img_array)
                contrast_values.append(contrast)
                
                # Saturation: difference from grayscale
                gray = np.mean(img_array, axis=2, keepdims=True)
                saturation = np.mean(np.abs(img_array - gray))
                saturation_values.append(saturation)
                
            except Exception as e:
                pass
        
        features = {
            'brightness': np.mean(brightness_values) if brightness_values else 128,
            'colorfulness': np.mean(colorfulness_values) if colorfulness_values else 50,
            'contrast': np.mean(contrast_values) if contrast_values else 128,
            'saturation': np.mean(saturation_values) if saturation_values else 30
        }
        
        # Normalize features to 0-1 range
        features['brightness_norm'] = features['brightness'] / 255.0
        features['colorfulness_norm'] = min(features['colorfulness'] / 100.0, 1.0)
        features['contrast_norm'] = features['contrast'] / 255.0
        features['saturation_norm'] = min(features['saturation'] / 50.0, 1.0)
        
        print(f"   Brightness: {features['brightness']:.1f}/255 (norm: {features['brightness_norm']:.2f})")
        print(f"   Colorfulness: {features['colorfulness']:.1f} (norm: {features['colorfulness_norm']:.2f})")
        print(f"   Contrast: {features['contrast']:.1f}/255 (norm: {features['contrast_norm']:.2f})")
        print(f"   Saturation: {features['saturation']:.1f} (norm: {features['saturation_norm']:.2f})")
        
        return features
    
    def select_music(self, features: Dict) -> str:
        """
        Select appropriate music based on visual features
        
        Selection Logic:
        - Bright + Colorful → Upbeat/Energetic
        - Dark + Low Saturation → Calm/Ambient
        - High Contrast → Energetic/Dynamic
        - Mid-range values → Balanced/Neutral
        
        Args:
            features: Visual feature dictionary
            
        Returns:
            Path to selected music file
        """
        if not self.music_files:
            print("  No music files available")
            return None
        
        print(f"\n Selecting music based on cluster characteristics...")
        
        brightness = features['brightness_norm']
        colorfulness = features['colorfulness_norm']
        contrast = features['contrast_norm']
        saturation = features['saturation_norm']
        
        # Calculate music mood score
        # High energy: bright, colorful, high contrast
        energy_score = (brightness * 0.3 + colorfulness * 0.3 + contrast * 0.4)
        
        # Calmness: low energy, low saturation
        calm_score = (1 - energy_score) * (1 - saturation)
        
        print(f"   Energy score: {energy_score:.2f}")
        print(f"   Calm score: {calm_score:.2f}")
        
        # Select music based on mood
        selected_music = None
        
        # Try to match music filename to mood
        music_keywords = {
            'upbeat': ['upbeat', 'energetic', 'happy', 'bright', 'positive'],
            'calm': ['calm', 'ambient', 'peaceful', 'slow', 'relaxing'],
            'energetic': ['energetic', 'dynamic', 'intense', 'fast'],
            'neutral': ['neutral', 'balanced', 'medium', 'background']
        }
        
        if energy_score > 0.6:
            # High energy - look for upbeat/energetic music
            selected_music = self._find_music_by_keywords(music_keywords['upbeat'] + music_keywords['energetic'])
            mood = "upbeat/energetic"
        elif calm_score > 0.6:
            # Calm mood - look for calm/ambient music
            selected_music = self._find_music_by_keywords(music_keywords['calm'])
            mood = "calm/ambient"
        else:
            # Neutral - look for balanced music
            selected_music = self._find_music_by_keywords(music_keywords['neutral'])
            mood = "balanced/neutral"
        
        # Fallback to first available music
        if selected_music is None:
            selected_music = self.music_files[0]
            mood = "default"
        
        print(f"   Selected mood: {mood}")
        print(f"   Music file: {Path(selected_music).name}")
        
        return selected_music
    
    def _find_music_by_keywords(self, keywords: List[str]) -> str:
        """
        Find music file matching keywords
        
        Args:
            keywords: List of keywords to search for
            
        Returns:
            Path to matching music file or None
        """
        for keyword in keywords:
            for music_file in self.music_files:
                if keyword.lower() in Path(music_file).stem.lower():
                    return music_file
        return None
    
    def create_default_music_files_guide(self):
        """Print guide for creating/downloading music files"""
        guide = """
        📁 Music Library Setup Guide
        ═══════════════════════════════════════════════════════════════
        
        Place music files in the 'music/' directory with these suggested names:
        
        1. upbeat.mp3 / energetic.mp3
           - For bright, colorful image clusters
           - Examples: pop, electronic, upbeat instrumental
        
        2. calm.mp3 / ambient.mp3
           - For dark or monochrome clusters
           - Examples: ambient, minimal, slow instrumental
        
        3. dynamic.mp3 / intense.mp3
           - For high-contrast clusters
           - Examples: cinematic, dramatic
        
        4. neutral.mp3 / background.mp3
           - For balanced clusters
           - Examples: corporate, background music
        
        Free Music Sources:
        - YouTube Audio Library (royalty-free)
        - Free Music Archive (freemusicarchive.org)
        - Incompetech (incompetech.com)
        - Bensound (bensound.com)
        
        Supported formats: MP3, WAV, M4A, OGG
        """
        
        print(guide)
        
        # Save guide to file
        guide_path = self.music_dir / "MUSIC_GUIDE.txt"
        with open(guide_path, 'w') as f:
            f.write(guide)
        
        print(f" Music guide saved to: {guide_path}")