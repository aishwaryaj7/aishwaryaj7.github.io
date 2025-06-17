"""Multi-modal document content extractor using pyMuPDF and modern libraries."""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import base64
from io import BytesIO
import json

from PIL import Image
import pytesseract
import pymupdf as fitz
import pymupdf4llm

from ..core.config import settings
from ..core.logger import get_logger

logger = get_logger(__name__)


class MultiModalDocumentExtractor:
    """Extract content from multi-modal documents including text, images, and tables."""
    
    def __init__(self):
        """Initialize the document extractor."""
        self.supported_formats = {
            '.pdf': self._extract_pdf_content,
            '.png': self._extract_image_content,
            '.jpg': self._extract_image_content,
            '.jpeg': self._extract_image_content,
            '.tiff': self._extract_image_content,
            '.bmp': self._extract_image_content,
        }
    
    async def extract_content(self, file_path: Path) -> Dict[str, any]:
        """
        Extract content from a document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing extracted content with metadata
        """
        logger.info(f"Extracting content from {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > settings.max_file_size_mb:
            raise ValueError(f"File size ({file_size_mb:.2f} MB) exceeds limit ({settings.max_file_size_mb} MB)")
        
        # Check file format
        file_extension = file_path.suffix.lower()
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Extract content based on file type
        extractor_func = self.supported_formats[file_extension]
        content = await extractor_func(file_path)
        
        # Add metadata
        content.update({
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_size_mb': file_size_mb,
            'file_extension': file_extension,
        })
        
        logger.info(f"Successfully extracted content from {file_path}")
        return content
    
    async def _extract_pdf_content(self, file_path: Path) -> Dict[str, any]:
        """Extract content from PDF files using pyMuPDF."""

        try:
            # Open PDF with pyMuPDF
            doc = fitz.open(str(file_path))

            # Extract structured content using pymupdf4llm
            md_text = pymupdf4llm.to_markdown(doc)

            # Organize content by type
            text_content = []
            tables = []
            images = []
            metadata = []

            # Process each page
            for page_num in range(len(doc)):
                page = doc[page_num]

                # Extract text blocks
                blocks = page.get_text("dict")
                for block in blocks["blocks"]:
                    if "lines" in block:  # Text block
                        block_text = ""
                        for line in block["lines"]:
                            for span in line["spans"]:
                                block_text += span["text"] + " "

                        if block_text.strip():
                            text_content.append({
                                'content': block_text.strip(),
                                'category': 'Text',
                                'page_number': page_num + 1,
                                'bbox': block.get("bbox", []),
                            })

                # Extract tables
                tables_on_page = page.find_tables()
                for table in tables_on_page:
                    table_data = table.extract()
                    if table_data:
                        tables.append({
                            'content': self._format_table_data(table_data),
                            'page_number': page_num + 1,
                            'bbox': table.bbox,
                        })

                # Extract images
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    try:
                        # Extract image data
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)

                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            img_base64 = base64.b64encode(img_data).decode()

                            images.append({
                                'content': f'Image {img_index + 1} from page {page_num + 1}',
                                'page_number': page_num + 1,
                                'image_data': {
                                    'base64': img_base64,
                                    'width': pix.width,
                                    'height': pix.height,
                                    'format': 'PNG',
                                },
                            })

                        pix = None  # Free memory
                    except Exception as e:
                        logger.warning(f"Could not extract image {img_index} from page {page_num + 1}: {e}")

                # Page metadata
                metadata.append({
                    'page_number': page_num + 1,
                    'width': page.rect.width,
                    'height': page.rect.height,
                })

            doc.close()

            return {
                'content_type': 'pdf',
                'text_content': text_content,
                'tables': tables,
                'images': images,
                'total_pages': len(metadata),
                'extraction_metadata': metadata,
                'markdown_content': md_text,  # Full markdown representation
            }

        except Exception as e:
            logger.error(f"Error extracting PDF content with pyMuPDF: {e}")
            # Fallback to basic text extraction
            return await self._extract_pdf_fallback(file_path)
    
    async def _extract_pdf_fallback(self, file_path: Path) -> Dict[str, any]:
        """Fallback PDF extraction method using basic pyMuPDF."""
        try:
            doc = fitz.open(str(file_path))
            text_content = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    text_content.append({
                        'content': text.strip(),
                        'category': 'Text',
                        'page_number': page_num + 1,
                    })

            doc.close()

            return {
                'content_type': 'pdf',
                'text_content': text_content,
                'tables': [],
                'images': [],
                'total_pages': len(text_content),
                'extraction_metadata': [],
            }
        except Exception as e:
            logger.error(f"Fallback PDF extraction failed: {e}")
            return {
                'content_type': 'pdf',
                'text_content': [{'content': 'PDF content extraction failed', 'category': 'Error'}],
                'tables': [],
                'images': [],
                'total_pages': 1,
                'extraction_metadata': [],
            }
    
    async def _extract_image_content(self, file_path: Path) -> Dict[str, any]:
        """Extract content from image files using OCR."""
        
        try:
            # Load image
            image = Image.open(file_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Extract text using OCR
            ocr_text = pytesseract.image_to_string(image, lang='eng')
            
            # Get image metadata
            width, height = image.size
            
            # Convert image to base64 for storage/processing
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            return {
                'content_type': 'image',
                'text_content': [{'content': ocr_text.strip(), 'category': 'OCR_Text'}] if ocr_text.strip() else [],
                'image_data': {
                    'base64': image_base64,
                    'width': width,
                    'height': height,
                    'format': image.format,
                },
                'ocr_confidence': await self._get_ocr_confidence(image),
            }
            
        except Exception as e:
            logger.error(f"Error extracting image content: {e}")
            raise
    
    async def _process_extracted_image(self, element_dict: Dict) -> Optional[Dict]:
        """Process an image extracted from PDF."""
        # This would process images extracted by unstructured
        # For now, return a placeholder
        return {
            'content': 'Extracted image from PDF',
            'metadata': element_dict.get('metadata', {}),
        }
    
    def _format_table_data(self, table_data: List[List[str]]) -> str:
        """Format table data as a readable string."""
        if not table_data:
            return ""

        # Convert table to markdown format
        formatted_rows = []
        for i, row in enumerate(table_data):
            if i == 0:  # Header row
                formatted_rows.append("| " + " | ".join(str(cell) for cell in row) + " |")
                formatted_rows.append("| " + " | ".join("---" for _ in row) + " |")
            else:
                formatted_rows.append("| " + " | ".join(str(cell) for cell in row) + " |")

        return "\n".join(formatted_rows)

    async def _get_ocr_confidence(self, image: Image.Image) -> float:
        """Get OCR confidence score."""
        try:
            # Use pytesseract to get confidence data
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            return sum(confidences) / len(confidences) if confidences else 0.0
        except Exception:
            return 0.0
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return list(self.supported_formats.keys())
    
    async def batch_extract(self, file_paths: List[Path]) -> List[Dict[str, any]]:
        """Extract content from multiple files concurrently."""
        tasks = [self.extract_content(file_path) for file_path in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log errors
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process {file_paths[i]}: {result}")
            else:
                valid_results.append(result)
        
        return valid_results 