import fitz
from docx import Document
from pathlib import Path
from typing import List, Dict, Callable


class DocumentParser:
    def __init__(self):
        """Initialize the DocumentParser with supported parsers."""
        self.parsers = {
            ".pdf": self.parse_pdf,
            ".docx": self.parse_docx,
            ".txt": self.parse_txt
        }

    def register_parser(self, extension: str, parser_method: Callable[[str], List[Dict]]):
        """
        Register a new parser method for a specific file extension dynamically.
        
        Args:
            extension (str): The file extension, e.g., '.csv'
            parser_method (Callable): The method to parse the file
        """
        if not extension.startswith("."):
            extension = f".{extension}"
        self.parsers[extension.lower()] = parser_method

    def get_supported_formats(self) -> List[str]:
        """Return a list of supported file formats."""
        return list(self.parsers.keys())
        
    def parse_pdf(self, file_path: str) -> List[Dict]:
        """Parse PDF and return extracted text with page numbers."""
        try:
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
            return pages
        except Exception as e:
            raise ValueError(f"Error parsing PDF '{file_path}': {e}")

    def parse_docx(self, file_path: str) -> List[Dict]:
        """Parse DOCX and return extracted text."""
        try:
            doc = Document(file_path)
            full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            if not full_text:
                return []
            return [{"source": Path(file_path).name, "page": 1, "content": full_text}]
        except Exception as e:
            raise ValueError(f"Error parsing DOCX '{file_path}': {e}")
        
    def parse_txt(self, file_path: str) -> List[Dict]:
        """Extract text from a plain text file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                return []
            return [{"source": Path(file_path).name, "page": 1, "content": content}]
        except Exception as e:
            raise ValueError(f"Error parsing TXT '{file_path}': {e}")

    def parse_document(self, file_path: str) -> List[Dict]:
        """Route to the correct parser based on file extension."""
        path_obj = Path(file_path)
        if not path_obj.exists() or not path_obj.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        ext = path_obj.suffix.lower()
        if ext not in self.parsers:
            raise ValueError(f"Unsupported file type '{ext}'. Supported types: {', '.join(self.parsers.keys())}")
            
        parser_method = self.parsers[ext]
        return parser_method(file_path)

    def parse_directory(self, directory_path: str, recursive: bool = False) -> List[Dict]:
        """
        Parse all supported documents in a directory.
        
        Args:
            directory_path (str): The path to the directory.
            recursive (bool): Whether to search subdirectories recursively.
            
        Returns:
            List[Dict]: Combined list of extracted text from all supported documents.
        """
        path_obj = Path(directory_path)
        if not path_obj.exists() or not path_obj.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory_path}")

        extracted_data = []
        
        # Determine whether to use recursive glob (rglob) or standard iterdir
        file_iterator = path_obj.rglob("*") if recursive else path_obj.iterdir()
        
        for file_path in file_iterator:
            if file_path.is_file() and file_path.suffix.lower() in self.parsers:
                try:
                    extracted_data.extend(self.parse_document(str(file_path)))
                except Exception as e:
                    print(f"Warning: Skipping {file_path} due to error: {e}")
                    
        return extracted_data
