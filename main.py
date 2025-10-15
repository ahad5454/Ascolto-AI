import os
import json
import logging
from database import EmbeddingDatabase
from utils import extract_text_from_pdf, chunk_text, get_openai_embedding

# Configure logger
logging.basicConfig(
    filename="main_log.log",
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger()

def process_pdf(file_path, save_json=True, db_path="embeddings_db"):
    pdf_name = os.path.basename(file_path).replace(".pdf", "")
    logger.info(f"\n--- Processing PDF: {pdf_name} ---")

    pages_text = extract_text_from_pdf(file_path)
    if not pages_text:
        logger.error(f"No text extracted from {file_path}")
        return None

    all_embeddings = []
    for page_num, page_text in pages_text:
        chunks = chunk_text(page_text)
        logger.info(f"Page {page_num}: {len(chunks)} chunks")

        for chunk in chunks:
            try:
                emb = get_openai_embedding(chunk)
                all_embeddings.append({
                    "text": chunk,
                    "embedding": emb,
                    "metadata": {
                        "pdf_name": pdf_name,
                        "page_num": page_num
                    }
                })
            except Exception as e:
                logger.error(f"Error embedding chunk on page {page_num}: {e}")
                continue

    logger.info(f"Total embeddings generated for {pdf_name}: {len(all_embeddings)}")

    if not all_embeddings:
        logger.warning(f"No valid embeddings for {pdf_name}")
        return None

    db = EmbeddingDatabase(db_path)
    db.add_pdf_embeddings(pdf_name, all_embeddings)
    logger.info(f"Stored embeddings in {db_path}")

    if save_json:
        os.makedirs("embeddings_json", exist_ok=True)
        json_path = os.path.join("embeddings_json", f"{pdf_name}_embeddings.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_embeddings, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved embeddings to JSON: {json_path}")

    return db.get_database_stats()


if __name__ == "__main__":
    db_path = "embeddings_db"
    db = EmbeddingDatabase(db_path)

    print(f"Working dir: {os.getcwd()}")
    data_folder = "data"
    print(f"Looking for PDFs in folder: {data_folder}")
    if not os.path.exists(data_folder):
        print("ERROR: data folder does not exist!")
    else:
        print(f"Files in data/: {os.listdir(data_folder)}")


    # Completely clear old embeddings
    db.clear_database()
    logger.info("Old embeddings database cleared.")

    # Ensure embeddings_json folder is clean
    os.makedirs("embeddings_json", exist_ok=True)
    for old_file in os.listdir("embeddings_json"):
        os.remove(os.path.join("embeddings_json", old_file))
    logger.info("Old JSON embeddings cleared.")

    # Process all PDFs in data/ folder
    data_folder = "data"
    pdf_files = [f for f in os.listdir(data_folder) if f.lower().endswith(".pdf")]

    if not pdf_files:
        logger.warning("No PDF files found in data/ folder.")
        print("No PDF files found in data/ folder.")
    else:
        for file_name in pdf_files:
            file_path = os.path.join(data_folder, file_name)
            stats = process_pdf(file_path, save_json=True, db_path=db_path)
            if stats:
                print(f"Processed {file_name}:", stats)
    print("All done!")