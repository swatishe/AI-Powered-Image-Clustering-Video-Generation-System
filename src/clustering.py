"""
Image clustering module using various algorithms
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import List
from collections import Counter


class ImageClusterer:
    """
    Cluster images based on feature embeddings
    
    Algorithm Choice:
    - K-Means: Fast, scalable, works well with high-dimensional data
    - Produces compact, spherical clusters ideal for image similarity
    - Easy to interpret and visualize
    """
    
    def __init__(self, n_clusters: int = 5, algorithm: str = "kmeans"):
        """
        Initialize clusterer
        
        Args:
            n_clusters: Number of clusters to create
            algorithm: Clustering algorithm ('kmeans', 'dbscan', 'hierarchical')
        """
        self.n_clusters = n_clusters
        self.algorithm = algorithm
        self.model = None
        
        self._init_model()
    
    def _init_model(self):
        """Initialize clustering model"""
        if self.algorithm == "kmeans":
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10
            )
        elif self.algorithm == "dbscan":
            self.model = DBSCAN(
                eps=0.5,
                min_samples=5,
                metric='euclidean'
            )
        elif self.algorithm == "hierarchical":
            self.model = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage='ward'
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        print(f" Initialized {self.algorithm} clusterer")
    
    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit clustering model and predict cluster labels
        
        Args:
            embeddings: Feature embeddings array
            
        Returns:
            Cluster labels for each image
        """
        print(f"\n Clustering {len(embeddings)} images into {self.n_clusters} clusters...")
        
        # Normalize embeddings for better clustering
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        embeddings_normalized = scaler.fit_transform(embeddings)
        
        # Fit and predict
        labels = self.model.fit_predict(embeddings_normalized)
        
        # Count cluster sizes
        cluster_counts = Counter(labels)
        print(f"\n Cluster distribution:")
        for cluster_id, count in sorted(cluster_counts.items()):
            print(f"   Cluster {cluster_id}: {count} images ({count/len(labels)*100:.1f}%)")
        
        return labels
    
    def evaluate_clusters(self, embeddings: np.ndarray, labels: np.ndarray):
        """
        Evaluate clustering quality
        
        Args:
            embeddings: Feature embeddings
            labels: Cluster labels
        """
        print(f"\n Evaluating clustering quality...")
        
        # Filter out noise points (label=-1 in DBSCAN)
        valid_mask = labels >= 0
        if not valid_mask.all():
            embeddings_valid = embeddings[valid_mask]
            labels_valid = labels[valid_mask]
        else:
            embeddings_valid = embeddings
            labels_valid = labels
        
        # Silhouette score (higher is better, range [-1, 1])
        if len(np.unique(labels_valid)) > 1:
            silhouette = silhouette_score(embeddings_valid, labels_valid)
            print(f"   Silhouette Score: {silhouette:.3f}")
            print(f"   (Range: -1 to 1, higher is better)")
            
            # Davies-Bouldin score (lower is better)
            db_score = davies_bouldin_score(embeddings_valid, labels_valid)
            print(f"   Davies-Bouldin Score: {db_score:.3f}")
            print(f"   (Lower is better, 0 is perfect)")
        else:
            print("  Only one cluster found, cannot compute quality metrics")
    
    def visualize_clusters(
        self, 
        embeddings: np.ndarray, 
        labels: np.ndarray,
        save_path: str = None
    ):
        """
        Visualize clusters using PCA dimensionality reduction
        
        Args:
            embeddings: Feature embeddings
            labels: Cluster labels
            save_path: Path to save visualization
        """
        print(f"\n Creating cluster visualization...")
        
        # Reduce to 2D using PCA
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            label_name = f"Cluster {label}" if label >= 0 else "Noise"
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[color],
                label=label_name,
                alpha=0.6,
                s=50
            )
        
        plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        plt.title('Image Clusters Visualization (PCA 2D Projection)')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f" Visualization saved to: {save_path}")
        
        plt.close()
    
    def save_cluster_assignments(
        self,
        image_paths: List[str],
        labels: np.ndarray,
        output_path: str
    ):
        """
        Save cluster assignments to CSV
        
        Args:
            image_paths: List of image file paths
            labels: Cluster labels
            output_path: Path to save CSV
        """
        df = pd.DataFrame({
            'image_path': image_paths,
            'cluster': labels
        })
        
        df.to_csv(output_path, index=False)
        print(f" Cluster assignments saved to: {output_path}")
    
    def print_cluster_summary(self, labels: np.ndarray, image_paths: List[str]):
        """
        Print detailed cluster summary with visual analysis
        
        Args:
            labels: Cluster labels
            image_paths: List of image paths
        """
        print(f"\n CLUSTER SUMMARY REPORT")
        print("=" * 80)
        
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_images = [img for img, mask in zip(image_paths, cluster_mask) if mask]
            cluster_size = len(cluster_images)
            
            print(f"\n  Cluster {label}:")
            print(f"   Size: {cluster_size} images ({cluster_size/len(labels)*100:.1f}%)")
            
            # Analyze sample images from cluster
            sample_size = min(5, cluster_size)
            sample_images = np.random.choice(cluster_images, sample_size, replace=False)
            
            print(f"   Sample images:")
            for img_path in sample_images:
                print(f"     - {img_path}")
            
            # Visual characteristics analysis
            characteristics = self._analyze_cluster_characteristics(sample_images)
            print(f"   Characteristics:")
            print(f"     - Avg brightness: {characteristics['brightness']:.2f}")
            print(f"     - Avg colorfulness: {characteristics['colorfulness']:.2f}")
            print(f"     - Dominant color mode: {characteristics['color_mode']}")
    
    def _analyze_cluster_characteristics(self, image_paths: List[str]) -> dict:
        """Analyze visual characteristics of cluster images"""
        from PIL import Image
        
        brightnesses = []
        colorfulness_values = []
        color_modes = []
        
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img)
                
                # Brightness (average pixel intensity)
                brightness = np.mean(img_array)
                brightnesses.append(brightness)
                
                # Colorfulness (standard deviation of RGB channels)
                colorfulness = np.std(img_array)
                colorfulness_values.append(colorfulness)
                
                color_modes.append('RGB')
                
            except Exception:
                pass
        
        return {
            'brightness': np.mean(brightnesses) if brightnesses else 0,
            'colorfulness': np.mean(colorfulness_values) if colorfulness_values else 0,
            'color_mode': Counter(color_modes).most_common(1)[0][0] if color_modes else 'Unknown'
        }
    
    def get_largest_cluster(self, labels: np.ndarray) -> int:
        """
        Get the ID of the largest cluster
        
        Args:
            labels: Cluster labels
            
        Returns:
            Cluster ID with most images
        """
        cluster_counts = Counter(labels)
        # Filter out noise (-1 label)
        cluster_counts = {k: v for k, v in cluster_counts.items() if k >= 0}
        largest_cluster = max(cluster_counts, key=cluster_counts.get)
        return largest_cluster