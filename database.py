import json
import os
import pickle
import shutil
import logging
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingDatabase:
    def __init__(self, db_path="embeddings_db"):
        self.db_path = db_path
        self.embeddings_file = os.path.join(db_path, "embeddings.pkl")
        self.metadata_file = os.path.join(db_path, "metadata.json")
        self.embeddings: List[List[float]] = []
        self.metadata: List[Dict] = []
        self._embeddings_array = None  # for faster search
        self._load_database()

    def _load_database(self):
        """Load existing embeddings and metadata"""
        os.makedirs(self.db_path, exist_ok=True)

        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, 'rb') as f:
                self.embeddings = pickle.load(f)

        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)

        self._update_numpy_cache()

    def _save_database(self):
        """Save embeddings and metadata"""
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(self.embeddings, f)

        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        self._update_numpy_cache()

    def _update_numpy_cache(self):
        """Cache embeddings as NumPy array for faster similarity search"""
        if self.embeddings:
            self._embeddings_array = np.array(self.embeddings, dtype=np.float32)
        else:
            self._embeddings_array = None

    def clear_database(self):
        """
        Completely delete all files inside the embeddings_db directory
        and recreate a fresh, empty folder.
        """
        try:
            if os.path.exists(self.db_path):
                shutil.rmtree(self.db_path)
                print(f"🧹 Cleared all files in '{self.db_path}' folder.")
                logging.info(f"Cleared all files in '{self.db_path}' folder.")
            else:
                print(f"No existing folder '{self.db_path}' found — creating a new one.")
                logging.info(f"No existing folder '{self.db_path}' found — creating a new one.")

            # Recreate clean folder
            os.makedirs(self.db_path, exist_ok=True)

            # Reset in-memory structures
            self.embeddings = []
            self.metadata = []
            self._update_numpy_cache()

        except Exception as e:
            print(f"Error while clearing database: {e}")
            logging.error(f"Error while clearing database: {e}")


    def clear(self):
        """for clearing the database, used by chatbot"""
        self.clear_database()

    def add_pdf_embeddings(self, pdf_name: str, embeddings_data: List[Dict]):
        """Add embeddings for a new PDF"""
        self.remove_pdf_embeddings(pdf_name)

        for i, item in enumerate(embeddings_data):
            self.embeddings.append(item['embedding'])
            meta = {
                'pdf_name': pdf_name,
                'chunk_id': i,
                'text': item['text'],
                'embedding_index': len(self.embeddings) - 1
            }
            # Keep extra metadata from main.py (like page_num)
            if 'metadata' in item:
                meta.update(item['metadata'])

            self.metadata.append(meta)

        self._save_database()
        print(f"Added {len(embeddings_data)} embeddings for '{pdf_name}'")

    def remove_pdf_embeddings(self, pdf_name: str):
        """Remove all embeddings for a specific PDF"""
        indices_to_remove = [i for i, meta in enumerate(self.metadata) if meta['pdf_name'] == pdf_name]

        for i in reversed(indices_to_remove):
            del self.embeddings[i]
            del self.metadata[i]

        for i, meta in enumerate(self.metadata):
            meta['embedding_index'] = i

        self._save_database()
        print(f"Removed embeddings for '{pdf_name}'")

    def search_similar_chunks(self, query_embedding: List[float], top_k: int = 5, threshold: float = 0.7) -> List[Dict]:
        """Search for similar chunks based on cosine similarity"""
        if self._embeddings_array is None:
            return []

        query_array = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        similarities = cosine_similarity(query_array, self._embeddings_array)[0]

        results = [
            {
                'metadata': self.metadata[i],
                'similarity': float(similarity),
                'text': self.metadata[i]['text']
            }
            for i, similarity in enumerate(similarities) if similarity >= threshold
        ]

        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

    def get_all_pdf_names(self) -> List[str]:
        """Return a list of all unique PDF names in the database"""
        return list(set(meta['pdf_name'] for meta in self.metadata))

    def get_total_chunks(self) -> int:
        """Return total number of stored chunks"""
        return len(self.embeddings)

    def get_database_stats(self) -> Dict:
        """Optional, can return stats as a single dictionary"""
        return {
            'total_chunks': self.get_total_chunks(),
            'total_pdfs': len(self.get_all_pdf_names()),
            'pdf_names': self.get_all_pdf_names()
        }
