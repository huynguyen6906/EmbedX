import h5py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from . import text
from . import image

class brute_force_search:
    def __init__(self):
        self.image_paths = None
        self.image_vectors = None

        self.text_paths = None
        self.text_vectors = None

    def read_image_converted_vectors(self, file_path):
        with h5py.File(file_path, "r") as f:
            self.image_paths = np.array([p.decode("utf-8") for p in f["path"][:]])
            self.image_vectors = np.array(f["embeddings"][:])

    def read_text_converted_vectors(self, file_path):
        with h5py.File(file_path, "r") as f:
            self.text_paths = np.array([p.decode("utf-8") for p in f["path"][:]])
            self.text_vectors = np.array(f["embeddings"][:])

    def search_image(self, text):
        """
        Tìm hình ảnh gần nhất với text_vector bằng cosine similarity
        Trả về path ảnh phù hợp nhất
        """
        # Tính cosine similarity giữa text_vector và tất cả image_vectors
        text_vector = image.embed_Text(text)
        sims = cosine_similarity(text_vector.reshape(1, -1), self.image_vectors)[0]
        best_idx = np.argmax(sims)
        return self.image_paths[best_idx], sims[best_idx]

    def search_topK_images(self, text, k=5):
        """
        Trả về top-K ảnh có độ tương đồng cosine cao nhất với text_vector
        """
        text_vector = image.embed_Text(text)
        sims = cosine_similarity(text_vector.reshape(1, -1), self.image_vectors)[0]
        topk_idx = np.argsort(sims)[::-1][:k]
        return [(self.image_paths[i], sims[i]) for i in topk_idx]
