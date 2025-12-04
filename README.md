# AI-Powered Image Clustering & Video Generation System

Complete end-to-end pipeline for clustering images and generating slideshow videos with algorithmic music selection.

## Features

- **Dataset Handling**: Load and analyze 200+ images from local folders or Kaggle
- **Feature Extraction**: ResNet50-based deep learning embeddings
- **Clustering**: K-Means clustering with visualization
- **Video Generation**: Slideshow videos with smooth transitions
- **Smart Music Selection**: Algorithmic music selection based on visual features
- **Complete Pipeline**: One command to run everything

## Project Structure

```
image-clustering-video/
├── data/
│   └── images/              # Place your images here (≥200 images)
├── output/
│   ├── embeddings/          # Saved feature embeddings
│   ├── clusters/            # Cluster assignments and visualizations
│   └── videos/              # Generated videos
├── music/                   # Background music files
├── src/
│   ├── dataset_handler.py   # Dataset loading and analysis
│   ├── feature_extractor.py # Deep learning feature extraction
│   ├── clustering.py        # K-Means clustering
│   ├── video_generator.py   # Video creation with transitions
│   └── music_selector.py    # Algorithmic music selection
├── main.py                  # Main execution pipeline
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Installation

```bash
# Clone or create project directory
mkdir image-clustering-video
cd image-clustering-video

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

**Option A: Use Your Own Images**
```bash
# Place 200+ images in data/images/
cp /path/to/your/images/* data/images/
```

**Option B: Download Sample Dataset from Kaggle**
```bash
# Install Kaggle CLI
pip install kaggle

# Configure Kaggle API (place kaggle.json in ~/.kaggle/)
# Download sample dataset
kaggle datasets download -d anthonytherrien/dog-vs-cat
unzip dog-vs-cat.zip -d data/images/

# Alternative datasets:
# kaggle datasets download -d puneet6060/intel-image-classification
# kaggle datasets download -d prasunroy/natural-images
```

**Option C: Download from OpenImages**
```bash
# Install OpenImages downloader
pip install openimages

# Download sample images
openimages download --download_dir data/images --limit 300
```

### 3. Prepare Music Files

Place 3-5 music files in the `music/` directory:

```bash
mkdir -p music
# Add music files:
# - upbeat.mp3 (for bright, colorful clusters)
# - calm.mp3 (for dark/monochrome clusters)
# - energetic.mp3 (for high-contrast clusters)
```

### 4. Run the Pipeline

```bash
# Execute complete pipeline
python main.py
```

This will:
1. Load and analyze dataset
2. Extract feature embeddings
3. Cluster images
4. Generate cluster visualization
5. Create slideshow video with music
6. Save all outputs and reports

## Output Files

After running, you'll find:

```
output/
├── embeddings/
│   ├── embeddings.npy           # Feature vectors (2048-dim)
│   └── image_paths.txt          # Image file paths
├── clusters/
│   ├── cluster_assignments.csv  # Image → Cluster mapping
│   └── cluster_visualization.png # PCA visualization
├── videos/
│   └── cluster_X_slideshow_with_music.mp4  # Final video
├── dataset_analysis.txt         # Dataset statistics
└── summary_report.txt           # Complete pipeline report
```

## Customization

### Change Number of Clusters

Edit `main.py`:
```python
N_CLUSTERS = 5  # Change this value
```

### Use Different Model

Edit `main.py`:
```python
extractor = FeatureExtractor(model_name="vgg16")  # or "mobilenet"
```

### Adjust Video Settings

Edit `main.py`:
```python
video_gen = VideoGenerator(
    fps=2,                    # Frames per second
    resolution=(1920, 1080),  # Video resolution
    transition_frames=10      # Transition smoothness
)
```

### Use Different Clustering Algorithm

Edit `main.py`:
```python
clusterer = ImageClusterer(n_clusters=5, algorithm="hierarchical")  # or "dbscan"
```

## Advanced Usage

### Load Pre-computed Embeddings

```python
from src.feature_extractor import FeatureExtractor

# Load saved embeddings
embeddings, paths = FeatureExtractor.load_embeddings(
    'output/embeddings/embeddings.npy',
    'output/embeddings/image_paths.txt'
)
```

### Generate Videos for All Clusters

```python
from src.video_generator import VideoGenerator
from src.clustering import ImageClusterer

video_gen = VideoGenerator()

for cluster_id in range(N_CLUSTERS):
    cluster_images = [
        img for img, label in zip(image_paths, labels)
        if label == cluster_id
    ]
    
    video_gen.create_slideshow(
        cluster_images,
        output_name=f"cluster_{cluster_id}.mp4"
    )
```

### Analyze Specific Cluster

```python
from src.clustering import ImageClusterer

# Get images in cluster 2
cluster_2_images = [
    img for img, label in zip(image_paths, labels)
    if label == 2
]

print(f"Cluster 2 has {len(cluster_2_images)} images")
```

## Music Selection Algorithm

The system automatically selects music based on:

- **Brightness**: Average pixel intensity
- **Colorfulness**: Standard deviation of RGB channels
- **Contrast**: Range of pixel values
- **Saturation**: Distance from grayscale

**Mapping:**
- Bright + Colorful → Upbeat/Energetic music
- Dark + Low Saturation → Calm/Ambient music
- High Contrast → Dynamic/Intense music
- Balanced → Neutral/Background music

## Performance

**Typical Processing Times** (on modern CPU):
- 200 images: ~2-3 minutes (feature extraction)
- Clustering: <10 seconds
- Video generation (50 images): ~1-2 minutes

**GPU Acceleration:**
System automatically uses CUDA if available for faster feature extraction.

## Troubleshooting

### "Not enough images" error
- Ensure `data/images/` contains ≥200 images
- Check supported formats: JPG, PNG, BMP, WEBP

### "No music files available" warning
- Add music files to `music/` directory
- Video will be created without audio

### Out of memory error
- Reduce batch size in `feature_extractor.py`
- Process fewer images at once

### Video playback issues
- Ensure you have appropriate codecs installed
- Try VLC media player for best compatibility

## Dependencies

Core libraries:
- PyTorch (deep learning)
- OpenCV (video generation)
- scikit-learn (clustering)
- Pillow (image processing)
- MoviePy (audio integration)

See `requirements.txt` for complete list.

## Model Details

**ResNet50** (chosen model):
- Pre-trained on ImageNet (1.2M images, 1000 classes)
- 2048-dimensional feature embeddings
- Excellent for transfer learning and similarity tasks
- Captures hierarchical visual features

**Why ResNet50?**
- Balance of accuracy and computational efficiency
- Proven effective for image similarity
- Residual connections capture multi-scale features
- Widely used in production systems

## License

This project is provided as-is for educational and research purposes.