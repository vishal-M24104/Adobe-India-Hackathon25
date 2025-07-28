import re
from collections import Counter
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
import logging
from pathlib import Path
from contextlib import contextmanager
import os
import json
import sys


class DocumentHeuristics:
    def __init__(self):
        self.baseline_font_size = 12.0
        self.baseline_font_name = ""
        self.font_size_hierarchy = {}
        self.extracted_title = ""


    def establish_baseline(self, all_lines: List[Dict]) -> None:
        font_sizes, font_names = [], []
        for line in all_lines:
            for span in line.get("spans", []):
                size, font = span.get("size"), span.get("font")
                if size and size < 18:
                    font_sizes.append(size)
                if font:
                    font_names.append(font)
        if font_sizes:
            self.baseline_font_size = Counter(font_sizes).most_common(1)[0][0]
            unique_sizes = sorted(set(font_sizes), reverse=True)
            self.font_size_hierarchy = {size: idx for idx, size in enumerate(unique_sizes)}
        if font_names:
            self.baseline_font_name = Counter(font_names).most_common(1)[0][0]


    def calculate_title_score(self, line: Dict, page_width: float) -> float:
        spans = line.get("spans", [])
        if not spans: return 0.0
        text = line.get("text", "").strip()
        if not text or len(text) < 2: return 0.0


        skip_patterns = [
            r'.*@.*\.(com|org|net|edu|gov|info).*',
            r'.*(http|https|www\.|ftp\.).*',
            r'^\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}.*',
            r'.*\d{4}[\-\/\.]\d{1,2}[\-\/\.]\d{1,2}.*',
            r'.*page\s+\d+(\s+(of|\/)\s+\d+)?.*',
            r'.*\d+\s*(of|\/)\s*\d+.*',
            r'^v?\d+\.\d+(\.\d+)?.*',
            r'^\s*\d+\s*$',
            r'^.*\.(pdf|doc|docx|txt|ppt|pptx)$',
            r'^\s*[{}()\[\]<>]+\s*$',
            r'^\s*[•·▪▫‣⁃➤►▶\-\*\+]+\s*$',
            r'^[\s\d\-\+\*\/=<>(){}[\]]+$',
        ]
        for pattern in skip_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return 0.0


        bbox = line.get("bbox", [0, 0, 0, 0])
        size = spans[0].get("size", self.baseline_font_size)
        page_num = line.get("page_num", 0)
        font = spans[0].get("font", "")
        flags = spans[0].get("flags", 0)
        if page_num > 2: return 0.0


        size_ratio = size / self.baseline_font_size if self.baseline_font_size > 0 else 1.0
        size_score = max(size_ratio * 12, 0)
        y0 = bbox[1]
        position_score = 800 if y0 < 100 else 400 if y0 < 200 else 200 if y0 < 400 else max(50 - (y0 - 400) * 0.1, 0)
        centering_score = 0
        if page_width > 0:
            midpoint = (bbox[0] + bbox[2]) / 2
            page_center = page_width / 2
            centering_score = (1 - abs(midpoint - page_center) / max(page_center, 1)) * 4
        is_bold = "bold" in font.lower() or "black" in font.lower() or (flags & 16)
        style_score = 4 if is_bold else 0
        word_count = len(text.split())
        word_score = 6 if 1 <= word_count <= 3 else 4 if word_count <= 8 else 2 if word_count <= 15 else 0 if word_count <= 25 else -word_count * 0.3
        case_score = 2 if text.isupper() and word_count <= 10 else 3 if text.istitle() else 0
        return size_score + position_score + centering_score + style_score + word_score + case_score


    def _clean_and_deduplicate_text(self, text: str) -> str:
        if not text: return ""
        words = re.findall(r'\b\w+\b', text)
        if not words: return text.strip()
        cleaned_words, prev_word = [], None
        for word in words:
            word_lower = word.lower()
            if word_lower != prev_word:
                cleaned_words.append(word)
                prev_word = word_lower
        if len(cleaned_words) < len(words) * 0.3 and len(words) > 10:
            unique_words, seen = [], set()
            for word in words[:15]:
                word_lower = word.lower()
                if word_lower not in seen and len(word) > 1:
                    unique_words.append(word)
                    seen.add(word_lower)
                    if len(unique_words) >= 8:
                        break
            return " ".join(unique_words)
        return " ".join(cleaned_words)


    def _text_similarity(self, text1: str, text2: str) -> float:
        if not text1 or not text2: return 0.0
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2: return 0.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0


    def combine_title_lines(self, first_page_lines: List[Dict]) -> str:
        if not first_page_lines: return "Untitled"
        title_candidates = []
        for line in first_page_lines:
            score = self.calculate_title_score(line, line.get("page_width", 595))
            if score > 25:
                title_candidates.append((score, line))
        if not title_candidates:
            for line in first_page_lines:
                score = self.calculate_title_score(line, line.get("page_width", 595))
                if score > 10: title_candidates.append((score, line))
        if not title_candidates: return "Untitled"
        title_candidates.sort(key=lambda x: (-x[0], x[1].get("bbox", [0, 0, 0, 0])[1]))
        combined_text_parts, reference_y = [], None
        max_candidates = min(3, len(title_candidates))
        for i, (score, line) in enumerate(title_candidates[:max_candidates]):
            text = line.get("text", "").strip()
            bbox = line.get("bbox", [0, 0, 0, 0])
            cleaned_text = self._clean_and_deduplicate_text(text)
            if not cleaned_text or len(cleaned_text) < 2: continue
            if reference_y is None:
                combined_text_parts.append(cleaned_text)
                reference_y = bbox[1]
            else:
                y_distance = abs(bbox[1] - reference_y)
                if y_distance < 150:
                    if not any(self._text_similarity(cleaned_text, existing) > 0.7 for existing in combined_text_parts):
                        combined_text_parts.append(cleaned_text)
                        reference_y = bbox[1]
                else:
                    break
        if not combined_text_parts: return "Untitled"
        final_title = " ".join(combined_text_parts).strip()
        final_title = re.sub(r'\s+', ' ', final_title)
        if not final_title: return "Untitled"
        self.extracted_title = final_title.lower()
        return final_title


    def extract_features(self, line: Dict) -> Dict[str, Any]:
        text = line.get("text", "").strip()
        spans = line.get("spans", [])
        bbox = line.get("bbox", [0, 0, 0, 0])
        page_num = line.get("page_num", 0)
        if not spans or not text: return {}


        # Table detection for structure-based table skip (row/col/serials, e.g. see user screenshot)
        table_like = (
            re.match(r'^\s*\d+\.\s*\w+', text) or
            re.match(r'^[A-Za-z ]+\s*\|\s*[A-Za-z ]+', text) or
            re.match(r'^(S\.?No|Name|Age|Relationship)', text, re.IGNORECASE) or
            re.match(r'^\d+\s*$', text.strip())
        )
        if table_like: return {}
        skip_patterns = [
            r'.*@.*\.(com|org|net|edu|gov|info).*',
            r'.*(http|https|www\.|ftp\.).*',
            r'^\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}.*',
            r'.*\d{4}[\-\/\.]\d{1,2}[\-\/\.]\d{1,2}.*',
            r'.*page\s+\d+(\s+(of|\/)\s+\d+)?.*',
            r'.*\d+\s*(of|\/)\s*\d+.*',
            r'^v?\d+\.\d+(\.\d+)?.*',
            r'^\s*\d+\s*$',
            r'^.*\.(pdf|doc|docx|txt|ppt|pptx)$',
            r'^[\s\d\-\+\*\/=<>(){}[\]]+$',
            r'^[{}()\[\]<>]+\s*[{}()\[\]<>]*\s*$',
            r'^[•·▪▫‣⁃➤►▶\-\*\+]+\s*$',
            r'^[\d\.\s\|-]+$',  # Table borders or numeric grids
            r'^\s*\|.*\|\s*$',  # Pipe-separated table rows
        ]
        for pattern in skip_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return {}
        if self.extracted_title and self._text_similarity(text.lower(), self.extracted_title) > 0.8:
            return {}
        if len(text.strip()) < 3: return {}
        alpha_count = sum(1 for c in text if c.isalpha())
        if len(text) > 0 and alpha_count / len(text) < 0.5: return {}
        punct_count = sum(1 for c in text if not c.isalnum() and c != ' ')
        if len(text) > 0 and punct_count / len(text) > 0.4: return {}
        numeric_count = sum(1 for c in text if c.isdigit())
        if len(text) > 0 and numeric_count / len(text) > 0.4 and len(text.split()) < 5:
            return {}


        span = spans[0]
        size = span.get("size", self.baseline_font_size)
        font = span.get("font", "")
        flags = span.get("flags", 0)
        features = self._detect_section_patterns(text)
        features.update({
            "text": features.get("clean_text", text),
            "original_text": text,
            "page": page_num + 1,
            "relative_font_size": size / self.baseline_font_size if self.baseline_font_size > 0 else 1.0,
            "is_bold": "bold" in font.lower() or "black" in font.lower() or (flags & 16),
            "is_italic": "italic" in font.lower() or (flags & 2),
            "indentation_level": bbox[0],
            "word_count": len(text.split()),
            "text_case": self._analyze_case(text),
            "size": size,
            "font": font,
            "y_position": bbox[1],
            "font_hierarchy_level": self.font_size_hierarchy.get(size, 999)
        })
        return features


    def _detect_section_patterns(self, text: str) -> Dict[str, Any]:
        features = {
            "is_numbered_section": False,
            "is_standard_heading": False,
            "section_level": 0,
            "clean_text": text,
            "numbering_pattern": None
        }
        main_patterns = [
            (r"^\s*(\d+)\.\s*(.+)$", "numeric"),
            (r"^\s*([A-Z])\.\s*(.+)$", "alpha"),
            (r"^\s*([IVX]+)\.\s*(.+)$", "roman"),
            (r"^\s*(Chapter|Part|Section)\s+(\d+|\w+)[:\.]?\s*(.*)$", "named"),
        ]
        sub_patterns = [
            (r"^\s*(\d+\.\d+)[\.\s]\s*(.+)$", "numeric_sub"),
            (r"^\s*([A-Z]\.\d+)[\.\s]\s*(.+)$", "alpha_sub"),
        ]
        subsub_patterns = [
            (r"^\s*(\d+\.\d+\.\d+)[\.\s]\s*(.+)$", "numeric_subsub"),
            (r"^\s*([A-Z]\.\d+\.\d+)[\.\s]\s*(.+)$", "alpha_subsub"),
        ]
        for pattern, pattern_type in main_patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                features["is_numbered_section"] = True
                features["section_level"] = 1
                features["numbering_pattern"] = pattern_type
                if pattern_type == "named" and len(match.groups()) > 2:
                    features["clean_text"] = f"{match.group(1)} {match.group(2)} {match.group(3).strip()}"
                elif len(match.groups()) > 1:
                    features["clean_text"] = f"{match.group(1)}. {match.group(2).strip()}"
                else:
                    features["clean_text"] = match.group(1).strip()
                return features
        for pattern, pattern_type in sub_patterns:
            match = re.match(pattern, text)
            if match:
                features["is_numbered_section"] = True
                features["section_level"] = 2
                features["numbering_pattern"] = pattern_type
                features["clean_text"] = f"{match.group(1)} {match.group(2).strip()}"
                return features
        for pattern, pattern_type in subsub_patterns:
            match = re.match(pattern, text)
            if match:
                features["is_numbered_section"] = True
                features["section_level"] = 3
                features["numbering_pattern"] = pattern_type
                features["clean_text"] = f"{match.group(1)} {match.group(2).strip()}"
                return features


        words = text.split()
        if (3 <= len(words) <= 8 and not text.endswith('.') and not re.search(r'\d{4}', text) and
                not re.search(r'[{}()\[\]<>⇤⇢]', text) and text.istitle()):
            features["is_standard_heading"] = True
        return features


    def _analyze_case(self, text: str) -> str:
        if text.isupper(): return 'upper'
        elif text.istitle(): return 'title'
        elif text.islower(): return 'lower'
        else: return 'mixed'


    def is_heading_candidate(self, features: Dict[str, Any]) -> bool:
        if not features: return False
        text = features.get("text", "")
        relative_size = features.get("relative_font_size", 1.0)
        is_bold = features.get("is_bold", False)
        word_count = features.get("word_count", 0)
        text_case = features.get("text_case", "")
        is_numbered_section = features.get("is_numbered_section", False)
        is_standard_heading = features.get("is_standard_heading", False)
        font_hierarchy_level = features.get("font_hierarchy_level", 999)
        if not text or len(text.strip()) < 3 or word_count > 15:
            return False
        if is_numbered_section or is_standard_heading:
            return True
        criteria_count = 0
        if relative_size > 1.3: criteria_count += 2
        elif relative_size > 1.1: criteria_count += 1
        if font_hierarchy_level <= 1: criteria_count += 1
        if is_bold: criteria_count += 2
        if text_case in ['title', 'upper']: criteria_count += 1
        if 3 <= word_count <= 6: criteria_count += 1
        indentation = features.get("indentation_level", 0)
        if indentation < 30: criteria_count += 1
        return criteria_count >= 4


    def classify_hierarchy(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        outline = []
        for candidate in candidates:
            hierarchy_score = self._calculate_hierarchy_score(candidate)
            if self._is_structural_artifact(candidate.get("text", "")): continue
            level = self._determine_heading_level(candidate, hierarchy_score)
            confidence = min(hierarchy_score / 10.0, 1.0)
            outline.append({
                "level": level,
                "text": candidate.get("text", ""),
                "page": candidate.get("page", 1),
                "_confidence": confidence   # internal only, not used in output
            })
        return outline


    def _is_structural_artifact(self, text: str) -> bool:
        artifact_patterns = [
            r'^[mt\s\*⇤]+$',
            r'^[\d\s\-\+\*\/=<>(){}[\]]+$',
            r'^\d+\s*(end|return)$',
            r'^[xyz]\d*[\s=]*$'
        ]
        text_lower = text.lower().strip()
        for pattern in artifact_patterns:
            if re.match(pattern, text_lower):
                return True
        return False


    def _determine_heading_level(self, candidate: Dict[str, Any], hierarchy_score: float) -> str:
        if candidate.get("is_numbered_section"):
            section_level = candidate.get("section_level", 1)
            return f"H{min(section_level, 3)}"
        if hierarchy_score >= 9: return "H1"
        elif hierarchy_score >= 7: return "H2"
        else: return "H3"


    def _calculate_hierarchy_score(self, candidate: Dict[str, Any]) -> float:
        score = 0.0
        relative_size = candidate.get("relative_font_size", 1.0)
        score += relative_size * 3
        font_hierarchy_level = candidate.get("font_hierarchy_level", 999)
        if font_hierarchy_level == 0: score += 4
        elif font_hierarchy_level == 1: score += 3
        elif font_hierarchy_level == 2: score += 2
        if candidate.get("is_bold", False): score += 2
        text_case = candidate.get("text_case", "")
        if text_case == 'upper': score += 1.5
        elif text_case == 'title': score += 1
        indentation = candidate.get("indentation_level", 0)
        if indentation < 20: score += 2
        elif indentation < 50: score += 1
        word_count = candidate.get("word_count", 0)
        if word_count <= 3: score += 1
        elif word_count <= 6: score += 0.5
        if candidate.get("is_numbered_section", False):
            section_level = candidate.get("section_level", 1)
            if section_level == 1: score += 3
            elif section_level == 2: score += 2
            elif section_level == 3: score += 1
        return score


class PDFOutlineExtractor:
    def __init__(self, max_pages: int = 50, title_search_pages: int = 3, enable_logging: bool = True):
        self.max_pages = max_pages
        self.title_search_pages = title_search_pages
        self.heuristics = DocumentHeuristics()
        if enable_logging:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = None


    def _log(self, level: str, message: str):
        if self.logger:
            getattr(self.logger, level)(message)


    @contextmanager
    def _open_pdf(self, pdf_path: str):
        doc = None
        try:
            doc = fitz.open(pdf_path)
            yield doc
        finally:
            if doc: doc.close()


    def extract_document_outline(self, pdf_path: str) -> Dict[str, Any]:
        if not self._validate_pdf_path(pdf_path):
            raise ValueError(f"Invalid PDF path: {pdf_path}")
        with self._open_pdf(pdf_path) as doc:
            if doc.is_encrypted:
                raise ValueError("PDF is encrypted and cannot be processed")
            if doc.page_count == 0:
                raise ValueError("PDF contains no pages")
            self._log('info', f"Processing PDF with {doc.page_count} pages")
            pages_to_process = min(len(doc), self.max_pages)
            all_lines = self._extract_structured_text_optimized(doc, pages_to_process)
            if not all_lines:
                return { "title": "Untitled", "outline": [] }
            self.heuristics.establish_baseline(all_lines)
            title = self._identify_title(all_lines)
            toc_outline = self._extract_from_toc(all_lines)
            if toc_outline:
                outline = toc_outline
            else:
                outline = self._extract_headings(all_lines)
            return {"title": title, "outline": outline}


    def _validate_pdf_path(self, pdf_path: str) -> bool:
        try:
            path = Path(pdf_path)
            return path.exists() and path.is_file() and path.suffix.lower() == '.pdf'
        except Exception:
            return False


    def _extract_structured_text_optimized(self, doc: fitz.Document, pages_to_process: int) -> List[Dict]:
        all_lines = []
        for page_num in range(pages_to_process):
            try:
                page = doc[page_num]
                page_width = page.rect.width
                page_data = page.get_text("dict")
                is_multi_column = self._detect_multi_column(page_data, page_width)
                page_lines = self._process_page_blocks(page_data, page_width, page_num, is_multi_column)
                all_lines.extend(page_lines)
            except Exception:
                continue
        return all_lines


    def _detect_multi_column(self, page_data: Dict, page_width: float) -> bool:
        left_count = right_count = 0
        half_width = page_width / 2
        for block in page_data.get("blocks", []):
            for line in block.get("lines", []):
                bbox = line.get("bbox", [0, 0, 0, 0])
                if bbox[0] < half_width:
                    left_count += 1
                else:
                    right_count += 1
        return left_count > 5 and right_count > 5


    def _process_page_blocks(self, page_data: Dict, page_width: float, page_num: int, is_multi_column: bool) -> List[Dict]:
        page_lines = []
        prev_bbox = None
        for block in page_data.get("blocks", []):
            if block.get("type") != 0:
                continue
            block_lines = block.get("lines", [])
            # Simple detection: skip table-like blocks (all rows have multiple numbers or pipe/columns per line)
            num_table_lines = sum(self._is_likely_table_line(l.get("text", "")) for l in block_lines)
            if len(block_lines) > 2 and num_table_lines / len(block_lines) > 0.6:
                continue  # Skip the block (is table)
            for line in block_lines:
                line_data = self._process_line(line, page_width, page_num, is_multi_column, prev_bbox)
                if line_data:
                    page_lines.append(line_data)
                    prev_bbox = line.get("bbox")
        return page_lines


    def _is_likely_table_line(self, text: str) -> bool:
        if not text: return False
        text = text.strip()
        if (
            re.match(r'^\s*\d+\.', text) or
            re.match(r'^[A-Za-z ]+\s*\|\s*[A-Za-z ]+', text) or
            re.match(r'^(S\.?No|Name|Age|Relationship)', text, re.IGNORECASE) or
            '---' in text or '____' in text or '.....' in text
        ):
            return True
        numeric_count = sum(1 for c in text if c.isdigit())
        if len(text) > 0 and numeric_count / len(text) > 0.4 and len(text.split()) <= 6:
            return True
        return False


    def _process_line(self, line: Dict, page_width: float, page_num: int, is_multi_column: bool, prev_bbox: Optional[List[float]]) -> Optional[Dict]:
        line_text = "".join(span.get("text", "") for span in line.get("spans", []))
        clean_text = line_text.strip()
        if not clean_text: return None
        if self._is_likely_table_line(clean_text): return None
        if prev_bbox and abs(line["bbox"][1] - prev_bbox[1]) < 5 and clean_text == line.get("text", ""):
            return None
        column_position = "left" if line["bbox"][0] < page_width / 2 else "right"
        return {
            "text": clean_text,
            "spans": line.get("spans", []),
            "bbox": line.get("bbox", [0, 0, 0, 0]),
            "page_width": page_width,
            "page_num": page_num,
            "column_position": column_position
        }


    def _identify_title(self, all_lines: List[Dict]) -> str:
        title_search_lines = [line for line in all_lines if line.get("page_num", 0) < self.title_search_pages]
        if not title_search_lines:
            return "Untitled"
        return self.heuristics.combine_title_lines(title_search_lines) or "Untitled"


    def _extract_from_toc(self, all_lines: List[Dict]) -> List[Dict[str, Any]]:
        toc_patterns = [r"table of contents", r"contents", r"index", r"toc"]
        toc_page = None
        for line in all_lines:
            text = line.get("text", "").strip().lower()
            if any(re.search(pattern, text) for pattern in toc_patterns) and line.get("page_num", 0) < 5:
                toc_page = line.get("page_num", 0)
                break
        if toc_page is None:
            return []
        toc_lines = [line for line in all_lines if line.get("page_num") == toc_page]
        outline = []
        for line in toc_lines:
            text = line.get("text", "").strip()
            if not text or re.search('|'.join(toc_patterns), text.lower()):
                continue
            match = re.match(r"^(.*?)\s*(\.{2,})?\s*(\d+)$", text)
            if match:
                heading_text = match.group(1).strip()
                page_str = match.group(3)
                try:
                    page = int(page_str)
                except ValueError:
                    continue
                features = self.heuristics.extract_features(line)
                if not features or not self.heuristics.is_heading_candidate(features):
                    continue
                # Skip if similar to title
                if self.heuristics._text_similarity(heading_text.lower(), self.heuristics.extracted_title) > 0.8:
                    continue
                hierarchy_score = self.heuristics._calculate_hierarchy_score(features)
                # Only add if confidence > 0.8 (per user requirement); do not output confidence
                confidence = min(hierarchy_score / 10.0, 1.0)
                if confidence >= 0.8:
                    outline.append({
                        "level": self.heuristics._determine_heading_level(features, hierarchy_score),
                        "text": heading_text,
                        "page": page
                    })
        return outline


    def _extract_headings(self, all_lines: List[Dict]) -> List[Dict[str, Any]]:
        heading_candidates = []
        for line in all_lines:
            try:
                features = self.heuristics.extract_features(line)
                if line.get("column_position") == "right":
                    features["indentation_level"] += 100
                if features and self.heuristics.is_heading_candidate(features):
                    heading_candidates.append(features)
            except Exception:
                continue
        heading_candidates.sort(key=lambda x: (x.get("page", 1), x.get("y_position", 0)))
        outline = self.heuristics.classify_hierarchy(heading_candidates)
        # Filter: remove confidence, and only output if confidence >= 0.8
        output_outline = []
        for item in outline:
            if item.get("_confidence", 0.0) >= 0.8:
                d = {k: v for k, v in item.items() if not k.startswith("_")}
                output_outline.append(d)
        return output_outline


def extract_document_outline(pdf_path: str, max_pages: int = 50, title_search_pages: int = 3) -> Dict[str, Any]:
    extractor = PDFOutlineExtractor(max_pages=max_pages, title_search_pages=title_search_pages)
    return extractor.extract_document_outline(pdf_path)


DATA_FOLDER = "input"
RESULTS_FOLDER = "output"


def main():
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    if not os.path.isdir(DATA_FOLDER):
        print(f"Error: Input directory '{DATA_FOLDER}' not found.")
        print("Please create the 'data' folder and place your PDF files there.")
        sys.exit(1)
    pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print("No PDF files found in 'data' folder.")
        return
    print(f"Found {len(pdf_files)} PDF files to process...\n{'='*50}")
    successful_extractions = 0
    for i, filename in enumerate(pdf_files, 1):
        input_path = os.path.join(DATA_FOLDER, filename)
        output_filename = Path(filename).stem + ".json"
        output_path = os.path.join(RESULTS_FOLDER, output_filename)
        print(f"[{i}/{len(pdf_files)}] Processing '{filename}'...")
        try:
            outline_data = extract_document_outline(input_path)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(outline_data, f, ensure_ascii=False, indent=2)
            print(f"    ✓ Success -> '{output_filename}'")
            successful_extractions += 1
        except Exception as e:
            print(f"    ✗ Error: {str(e)}")
            error_data = {
                "error": str(e),
                "file": filename,
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, indent=2)
    print("=" * 50)
    print(f"Processing complete!\n  Total files: {len(pdf_files)}\n  Successful: {successful_extractions}\n  Failed: {len(pdf_files) - successful_extractions}")
    print(f"  JSON files saved to: {RESULTS_FOLDER}/")


if __name__ == "__main__":
    main()
