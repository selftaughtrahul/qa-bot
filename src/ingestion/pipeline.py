import json
import os
from pathlib import Path
from src.ingestion.document_parser import DocumentParser
from src.ingestion.text_chunker import chunk_documents
from typing import List, Dict


def ingest_document(file_path: str, output_dir: str = "data/processed") -> List[Dict]:
    """Full ingestion pipeline: parse → chunk → save."""
    print(f"📄 Parsing: {file_path}")
    parser = DocumentParser()
    pages = parser.parse_document(file_path)
    print(f"   ✅ Extracted {len(pages)} page(s).")

    print(f"✂️  Chunking...")
    chunks = chunk_documents(pages)
    print(f"   ✅ Created {len(chunks)} chunks.")

    # Save processed chunks
    os.makedirs(output_dir, exist_ok=True)
    doc_name = Path(file_path).stem
    output_path = os.path.join(output_dir, f"{doc_name}_chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved to: {output_path}")

    return chunks


if __name__ == "__main__":
    # Test with a sample PDF
    chunks = ingest_document("data/raw/sample.pdf")
    print(f"\nTotal chunks: {len(chunks)}")
    print("Sample chunk:", chunks[0])