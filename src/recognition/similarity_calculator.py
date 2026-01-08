"""
Module tính độ tương đồng giữa các vector đặc trưng
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SimilarityCalculator:
    """
    Class tính độ tương đồng giữa vector query và database
    Hỗ trợ nhiều metric: cosine, euclidean
    """
    
    @staticmethod
    def cosine_similarity(qv: np.ndarray, features: np.ndarray) -> np.ndarray:
        """
        Tính Cosine Similarity giữa 1 query vector và N vectors trong database
        
        Công thức: sim(A, B) = (A · B) / (||A|| × ||B||)
        Giá trị: [0, 1] với features đã normalize
        
        Args:
            qv: Query vector (D,)
            features: Database vectors (N, D)
            
        Returns:
            Similarities (N,) - độ tương đồng với từng vector
        """
        # Chuẩn hóa query vector
        qv_norm = np.linalg.norm(qv)
        if qv_norm == 0:
            logger.warning("Query vector có norm = 0")
            return np.zeros(len(features))
        
        # Chuẩn hóa database vectors
        feats_norm = np.linalg.norm(features, axis=1)
        
        # Tránh chia cho 0
        denom = feats_norm * qv_norm
        denom[denom == 0] = 1e-12
        
        # Tính dot product và chia cho norm
        sims = np.dot(features, qv) / denom
        
        logger.debug(f"Cosine similarity: min={sims.min():.3f}, max={sims.max():.3f}, mean={sims.mean():.3f}")
        
        return sims
    
    @staticmethod
    def euclidean_distance(qv: np.ndarray, features: np.ndarray) -> np.ndarray:
        """
        Tính Euclidean Distance
        
        Args:
            qv: Query vector (D,)
            features: Database vectors (N, D)
            
        Returns:
            Distances (N,) - khoảng cách đến từng vector
        """
        distances = np.linalg.norm(features - qv, axis=1)
        return distances
    
    @staticmethod
    def normalize_similarities(sims: np.ndarray) -> np.ndarray:
        """Chuẩn hóa similarities về [0, 1]"""
        min_sim = sims.min()
        max_sim = sims.max()
        
        if max_sim - min_sim == 0:
            return np.zeros_like(sims)
        
        return (sims - min_sim) / (max_sim - min_sim)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    calc = SimilarityCalculator()
    
    # Test với random vectors
    qv = np.random.rand(100)
    features = np.random.rand(500, 100)
    
    sims = calc.cosine_similarity(qv, features)
    print(f"Similarities shape: {sims.shape}")
    print(f"Min: {sims.min():.3f}, Max: {sims.max():.3f}")
