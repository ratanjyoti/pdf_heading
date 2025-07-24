# Filename: utils/pdf_extractor.py

import fitz
import re
from typing import List, Dict, Any

class PdfExtractor:
    """
    Final, robust version: Extracts logical text blocks, sorts by reading order,
    and precisely identifies blocks within tables with improved style detection.
    """

    def extract_enriched_blocks(self, pdf_path: str) -> List[Dict[str, Any]]:
        doc = fitz.open(pdf_path)
        all_blocks_data = []

        for page_num, page in enumerate(doc, 1):
            table_bboxes = self._get_table_bboxes_from_lines(page)
            
            blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
            sorted_blocks = sorted(blocks, key=lambda b: (b['bbox'][1], b['bbox'][0]))

            for block in sorted_blocks:
                if block.get('type') == 0 and block.get('lines'):
                    processed_block = self._process_block(block, page, page_num, table_bboxes)
                    if processed_block and processed_block['text']:
                        all_blocks_data.append(processed_block)

        # Note: Merging is removed for now to ensure granular, accurate blocks first.
        # This gives you the most precise data for labeling.
        return self._post_process_spacing(all_blocks_data)

    def _get_table_bboxes_from_lines(self, page: fitz.Page) -> List[fitz.Rect]:
        # This function remains the same
        drawings = page.get_drawings()
        if not drawings: return []
        h_lines = [d['rect'] for d in drawings if d['rect'].width > 30 and d['rect'].height < 3]
        v_lines = [d['rect'] for d in drawings if d['rect'].height > 20 and d['rect'].width < 3]
        if not h_lines or not v_lines: return []
        try:
            min_x0, max_x1 = min(line.x0 for line in v_lines), max(line.x1 for line in v_lines)
            min_y0, max_y1 = min(line.y0 for line in h_lines), max(line.y1 for line in h_lines)
            return [fitz.Rect(min_x0, min_y0, max_x1, max_y1)]
        except ValueError: return []

    def _process_block(self, block: Dict, page: fitz.Page, page_num: int, table_bboxes: List[fitz.Rect]) -> Dict:
        """Processes a single logical block from PyMuPDF with improved style detection."""
        block_bbox = fitz.Rect(block['bbox'])
        
        # Consolidate all spans into a single block representation
        text_parts = []
        font_sizes, font_names = [], []
        
        for line in block.get('lines', []):
            for span in line.get('spans', []):
                text_parts.append(span['text'])
                font_sizes.append(span['size'])
                font_names.append(span['font'])

        if not text_parts:
            return None

        block_text = re.sub(r'\s+', ' ', " ".join(text_parts).replace('\u2019', "'")).strip()
        
        # --- IMPROVED STYLE DETECTION LOGIC ---
        most_common_font = max(set(font_names), key=font_names.count) if font_names else "N/A"
        font_name_lower = most_common_font.lower()

        # Keywords that indicate a bold font weight
        bold_keywords = ['bold', 'black', 'heavy', 'semibold']
        # Keywords that indicate an italic style
        italic_keywords = ['italic', 'oblique']
        
        is_bold = any(keyword in font_name_lower for keyword in bold_keywords)
        is_italic = any(keyword in font_name_lower for keyword in italic_keywords)

        block_center = (block_bbox.tl + block_bbox.br) / 2
        is_in_table = any(block_center in cell_bbox for cell_bbox in table_bboxes)
        column = 1 if block_center.x < page.rect.width / 2 else 2

        return {
            'text': block_text,
            'bbox': {'x0': block_bbox.x0, 'y0': block_bbox.y0, 'x1': block_bbox.x1, 'y1': block_bbox.y1},
            'page_number': page_num,
            'page_width': page.rect.width,
            'page_height': page.rect.height,
            'font_size': max(set(font_sizes), key=font_sizes.count) if font_sizes else 0,
            'font_name': most_common_font,
            'is_bold': is_bold,
            'is_italic': is_italic,
            'word_count': len(block_text.split()),
            'line_count': len(block.get('lines', [])),
            'is_in_table': is_in_table,
            'column': column
        }

    def _post_process_spacing(self, all_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # This function remains the same
        for i, current_block in enumerate(all_blocks):
            if i + 1 < len(all_blocks) and all_blocks[i+1]['page_number'] == current_block['page_number']:
                current_block['vertical_space_after'] = all_blocks[i+1]['bbox']['y0'] - current_block['bbox']['y1']
            else:
                current_block['vertical_space_after'] = 100
            if i > 0 and all_blocks[i-1]['page_number'] == current_block['page_number']:
                 current_block['vertical_space_before'] = current_block['bbox']['y0'] - all_blocks[i-1]['bbox']['y1']
            else:
                 current_block['vertical_space_before'] = 100
        return all_blocks