from utils import get_openai_embedding, get_openai_client
import time

client = get_openai_client()
start = time.time()
embedding = get_openai_embedding("Testing the new fast embedding model.", client)
print("Embedding length:", len(embedding))
print("Elapsed time:", round(time.time() - start, 2), "seconds")
