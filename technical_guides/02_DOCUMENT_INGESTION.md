# Technical Guide 02: Document Ingestion Pipeline

## 📋 Overview
This guide covers parsing documents (PDF, DOCX, TXT), splitting them into chunks, and saving the processed output — the first stage of the RAG pipeline.

---

## Step 1: Document Parser (`src/ingestion/document_parser.py`)

```python
import fitz  # PyMuPDF
from docx import Document
from pathlib import Path
from typing import List, Dict


def parse_pdf(file_path: str) -> List[Dict]:
    """Extract text from a PDF file, page by page."""
    doc = fitz.open(file_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append({
                "source": Path(file_path).name,
                "page": page_num,
                "content": text
            })
    doc.close()
    return pages


def parse_docx(file_path: str) -> List[Dict]:
    """Extract text from a DOCX file."""
    doc = Document(file_path)
    full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    return [{"source": Path(file_path).name, "page": 1, "content": full_text}]


def parse_txt(file_path: str) -> List[Dict]:
    """Extract text from a plain text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return [{"source": Path(file_path).name, "page": 1, "content": content}]


def parse_document(file_path: str) -> List[Dict]:
    """Route to the correct parser based on file extension."""
    ext = Path(file_path).suffix.lower()
    parsers = {".pdf": parse_pdf, ".docx": parse_docx, ".txt": parse_txt}
    if ext not in parsers:
        raise ValueError(f"Unsupported file type: {ext}")
    return parsers[ext](file_path)
```

---

## Step 2: Text Chunker (`src/ingestion/text_chunker.py`)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
import os


def chunk_documents(pages: List[Dict], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict]:
    """
    Split page-level text into smaller overlapping chunks.
    Each chunk retains metadata (source, page).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    for page in pages:
        split_texts = splitter.split_text(page["content"])
        for i, text in enumerate(split_texts):
            chunks.append({
                "source": page["source"],
                "page": page["page"],
                "chunk_id": f"{page['source']}_p{page['page']}_c{i}",
                "content": text
            })
    return chunks
```

---

## Step 3: Ingestion Pipeline (`src/ingestion/pipeline.py`)

```python
import json
import os
from pathlib import Path
from document_parser import parse_document
from text_chunker import chunk_documents


def ingest_document(file_path: str, output_dir: str = "data/processed") -> List[Dict]:
    """Full ingestion pipeline: parse → chunk → save."""
    print(f"📄 Parsing: {file_path}")
    pages = parse_document(file_path)
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
```

---

## Step 4: Test the Ingestion Pipeline

```bash
# Place a sample PDF in data/raw/
# Then run:
python src/ingestion/pipeline.py
```

**Expected Output:**
```
📄 Parsing: data/raw/sample.pdf
   ✅ Extracted 12 page(s).
✂️  Chunking...
   ✅ Created 87 chunks.
💾 Saved to: data/processed/sample_chunks.json

Total chunks: 87
Sample chunk: {'source': 'sample.pdf', 'page': 1, 'chunk_id': 'sample.pdf_p1_c0', 'content': '...'}
```

---

## ✅ Checklist
- [ ] `document_parser.py` handles PDF, DOCX, and TXT.
- [ ] `text_chunker.py` splits text with overlap.
- [ ] `pipeline.py` saves chunks to `data/processed/`.
- [ ] Tested with at least one real document.
