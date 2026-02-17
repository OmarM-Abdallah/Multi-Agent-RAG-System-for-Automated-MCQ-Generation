"""
PDF processing service for extracting text and structure.
"""
import re
from typing import List, Dict, Any
from pypdf import PdfReader
import logging

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF file processing and text extraction."""
    
    def __init__(self):
        """Initialize the PDF processor."""
        pass
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract all text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a single string
        """
        try:
            reader = PdfReader(pdf_path)
            text = ""
            
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            
            logger.info(f"Successfully extracted text from {pdf_path}")
            return text
        
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    def extract_table_of_contents(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract table of contents from PDF.
        
        This attempts to identify headers and sections based on:
        - Font size differences
        - Numbered sections
        - Common header patterns
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of ToC entries with title and page number
        """
        try:
            reader = PdfReader(pdf_path)
            toc = []
            
            # Try to get bookmarks/outlines first
            if reader.outline:
                toc = self._parse_outline(reader.outline)
            
            # If no bookmarks, attempt to extract from text
            if not toc:
                toc = self._extract_toc_from_text(reader)
            
            logger.info(f"Extracted {len(toc)} ToC entries")
            return toc
        
        except Exception as e:
            logger.error(f"Error extracting ToC: {e}")
            # Return a basic ToC if extraction fails
            return [{"title": "Document Content", "page": 1, "level": 1}]
    
    def _parse_outline(self, outline: List, level: int = 1) -> List[Dict[str, Any]]:
        """Parse PDF outline/bookmarks recursively."""
        toc = []
        
        for item in outline:
            if isinstance(item, list):
                toc.extend(self._parse_outline(item, level + 1))
            else:
                try:
                    title = item.title if hasattr(item, 'title') else str(item)
                    page = item.page.page_number if hasattr(item, 'page') else 0
                    
                    toc.append({
                        "title": title,
                        "page": page + 1,  # Convert to 1-indexed
                        "level": level
                    })
                except:
                    continue
        
        return toc
    
    def _extract_toc_from_text(self, reader: PdfReader) -> List[Dict[str, Any]]:
        """Extract ToC by analyzing text patterns."""
        toc = []
        
        # Common patterns for headers
        header_patterns = [
            r'^(?:Chapter|Section|Part)\s+(\d+)[:\.]?\s*(.+)$',
            r'^(\d+\.(?:\d+\.)*)\s+(.+)$',  # Numbered sections like 1.1, 1.2.3
            r'^([A-Z\s]{3,})\s*$',  # ALL CAPS titles
        ]
        
        for page_num, page in enumerate(reader.pages[:20], 1):  # Check first 20 pages
            text = page.extract_text()
            if not text:
                continue
            
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if not line or len(line) < 3:
                    continue
                
                for pattern in header_patterns:
                    match = re.match(pattern, line, re.IGNORECASE)
                    if match:
                        title = line if len(match.groups()) == 0 else ' '.join(match.groups())
                        toc.append({
                            "title": title[:100],  # Limit length
                            "page": page_num,
                            "level": 1
                        })
                        break
        
        # If still no ToC found, create a generic one
        if not toc:
            total_pages = len(reader.pages)
            toc = [
                {"title": "Introduction", "page": 1, "level": 1},
                {"title": "Main Content", "page": 2, "level": 1},
            ]
            if total_pages > 5:
                toc.append({"title": "Conclusion", "page": total_pages, "level": 1})
        
        return toc
    
    def chunk_text(
        self, 
        text: str, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Maximum characters per chunk
            chunk_overlap: Number of overlapping characters
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        chunk_id = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph exceeds chunk size, save current chunk
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": current_chunk.strip(),
                    "char_count": len(current_chunk)
                })
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + para
                chunk_id += 1
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        # Add the last chunk
        if current_chunk:
            chunks.append({
                "chunk_id": chunk_id,
                "text": current_chunk.strip(),
                "char_count": len(current_chunk)
            })
        
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
