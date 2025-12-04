"""
AI-Powered Image Clustering & Video Generation System
"""

__version__ = "1.0.0"
__author__ = "AI Video Generator"

from .dataset_handler import DatasetHandler
from .feature_extractor import FeatureExtractor
from .clustering import ImageClusterer
from .video_generator import VideoGenerator
from .music_selector import MusicSelector

__all__ = [
    'DatasetHandler',
    'FeatureExtractor',
    'ImageClusterer',
    'VideoGenerator',
    'MusicSelector'
]