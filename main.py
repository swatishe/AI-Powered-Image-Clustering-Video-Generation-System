"""
AI-Powered Image Clustering & Video Generation System
Main execution pipeline
@author sshende
"""

import os
import sys
from pathlib import Path

from src.dataset_handler import DatasetHandler
from src.feature_extractor import FeatureExtractor
from src.clustering import ImageClusterer
from src.video_generator import VideoGenerator
from src.music_selector import MusicSelector


def main():
    """Execute the complete pipeline"""
    
    print("=" * 80)
    print(" AI-Powered Image Clustering & Video Generation System")
    print("=" * 80)
    
    # Configuration
    IMAGE_DIR = "data/images"
    OUTPUT_DIR = "output"
    MUSIC_DIR = "music"
    N_CLUSTERS = 5
    MIN_IMAGES = 200
    
    # Create directories
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    Path(f"{OUTPUT_DIR}/embeddings").mkdir(exist_ok=True)
    Path(f"{OUTPUT_DIR}/clusters").mkdir(exist_ok=True)
    Path(f"{OUTPUT_DIR}/videos").mkdir(exist_ok=True)
    
    # Step 1: Dataset Handling
    print("\n Step 1: Loading and Analyzing Dataset")
    print("-" * 80)
    handler = DatasetHandler(IMAGE_DIR, min_images=MIN_IMAGES)
    
    if not handler.validate_dataset():
        print(f" Error: Need at least {MIN_IMAGES} images in {IMAGE_DIR}")
        print(f" Download a dataset using:")
        print(f" kaggle datasets download -d anthonytherrien/dog-vs-cat")
        print(f" unzip dog-vs-cat.zip -d {IMAGE_DIR}/")
        sys.exit(1)
    
    # Analyze dataset
    analysis = handler.analyze_dataset()
    handler.display_samples(n_samples=6)
    handler.save_analysis_report(f"{OUTPUT_DIR}/dataset_analysis.txt")
    
    # Step 2: Feature Extraction
    print("\n Step 2: Extracting Image Features")
    print("-" * 80)
    extractor = FeatureExtractor(model_name="resnet50")
    
    embeddings_path = f"{OUTPUT_DIR}/embeddings/embeddings.npy"
    image_paths_file = f"{OUTPUT_DIR}/embeddings/image_paths.txt"
    
    embeddings, image_paths = extractor.extract_features(
        handler.image_paths,
        save_path=embeddings_path,
        paths_file=image_paths_file
    )
    
    print(f" Extracted embeddings shape: {embeddings.shape}")
    print(f" Saved to: {embeddings_path}")
    
    # Step 3: Clustering
    print("\n Step 3: Clustering Images")
    print("-" * 80)
    clusterer = ImageClusterer(n_clusters=N_CLUSTERS)
    
    cluster_labels = clusterer.fit_predict(embeddings)
    clusterer.visualize_clusters(
        embeddings, 
        cluster_labels,
        save_path=f"{OUTPUT_DIR}/clusters/cluster_visualization.png"
    )
    
    # Save cluster assignments
    clusterer.save_cluster_assignments(
        image_paths,
        cluster_labels,
        f"{OUTPUT_DIR}/clusters/cluster_assignments.csv"
    )
    
    # Evaluate clusters
    clusterer.evaluate_clusters(embeddings, cluster_labels)
    clusterer.print_cluster_summary(cluster_labels, image_paths)
    
    # Step 4: Video Generation
    print("\n Step 4: Generating Videos")
    print("-" * 80)
    
    video_gen = VideoGenerator(
        output_dir=f"{OUTPUT_DIR}/videos",
        fps=2,
        transition_frames=10
    )
    
    music_selector = MusicSelector(MUSIC_DIR)
    
    # Generate video for largest cluster
    largest_cluster = clusterer.get_largest_cluster(cluster_labels)
    cluster_images = [
        img for img, label in zip(image_paths, cluster_labels)
        if label == largest_cluster
    ]
    
    print(f"\n Largest cluster: {largest_cluster}")
    print(f" Images in cluster: {len(cluster_images)}")
    
    if len(cluster_images) < 3:
        print("  Warning: Too few images in largest cluster, using all images")
        cluster_images = image_paths[:50]  # Use first 50 images
    
    # Sort images by similarity
    cluster_indices = [i for i, label in enumerate(cluster_labels) if label == largest_cluster]
    if len(cluster_indices) >= 3:
        sorted_images = video_gen.sort_by_similarity(
            cluster_images,
            embeddings[cluster_indices]
        )
    else:
        sorted_images = cluster_images
    
    # Analyze cluster for music selection
    cluster_features = music_selector.analyze_image_cluster(sorted_images[:30])
    selected_music = music_selector.select_music(cluster_features)
    
    print(f"\n Selected music: {selected_music}")
    print(f"   Based on: brightness={cluster_features['brightness']:.2f}, "
          f"colorfulness={cluster_features['colorfulness']:.2f}")
    
    # Generate video
    video_path = video_gen.create_slideshow(
        sorted_images,
        music_path=selected_music,
        output_name=f"cluster_{largest_cluster}_slideshow.mp4"
    )
    
    # Step 5: Generate Summary Report
    print("\n Step 5: Generating Summary Report")
    
    with open(f"{OUTPUT_DIR}/summary_report.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("IMAGE CLUSTERING & VIDEO GENERATION - SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Dataset Statistics:\n")
        f.write(f"  - Total images: {analysis['total_images']}\n")
        f.write(f"  - Average dimensions: {analysis['avg_width']:.0f}x{analysis['avg_height']:.0f}\n")
        f.write(f"  - Color formats: {analysis['color_formats']}\n\n")
        
        f.write(f"Feature Extraction:\n")
        f.write(f"  - Model: ResNet50 (pre-trained on ImageNet)\n")
        f.write(f"  - Embedding dimensions: {embeddings.shape[1]}\n\n")
        
        f.write(f"Clustering:\n")
        f.write(f"  - Algorithm: K-Means\n")
        f.write(f"  - Number of clusters: {N_CLUSTERS}\n")
        f.write(f"  - Largest cluster: {largest_cluster} ({len(cluster_images)} images)\n\n")
        
        f.write(f"Video Generation:\n")
        f.write(f"  - Output file: {video_path}\n")
        f.write(f"  - Images used: {len(sorted_images)}\n")
        f.write(f"  - Music selected: {selected_music}\n")
        f.write(f"  - Cluster characteristics: {cluster_features}\n\n")
        
        f.write(f"Output Files:\n")
        f.write(f"  - {embeddings_path}\n")
        f.write(f"  - {OUTPUT_DIR}/clusters/cluster_assignments.csv\n")
        f.write(f"  - {video_path}\n")
        f.write(f"  - {OUTPUT_DIR}/dataset_analysis.txt\n")
    
    print("\n" + "=" * 80)
    print(" PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\n Output files located in: {OUTPUT_DIR}/")
    print(f" Generated video: {video_path}")
    print(f" View cluster assignments: {OUTPUT_DIR}/clusters/cluster_assignments.csv")
    print(f" Full report: {OUTPUT_DIR}/summary_report.txt\n")


if __name__ == "__main__":
    main()