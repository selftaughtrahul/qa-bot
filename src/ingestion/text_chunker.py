from langchain_text_splitters import RecursiveCharacterTextSplitter
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