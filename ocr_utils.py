import os
import re
import cv2
import numpy as np

# RapidOCR
try:
    from rapidocr_onnxruntime import RapidOCR
    HAS_RAPIDOCR = os.environ.get("USE_RAPIDOCR", "True").lower() == "true"
except ImportError:
    HAS_RAPIDOCR = False

# PaddleOCR disabled for now
HAS_PADDLEOCR = False

# EasyOCR disabled for now
HAS_EASYOCR = False

# Google GenAI disabled for now
HAS_GENAI = False


class PoleOCR:
    def __init__(self):
        # 1. Initialize RapidOCR (Tier 1)
        if HAS_RAPIDOCR:
            print("Initializing OCR Engine (RapidOCR)...")
            try:
                self.rapid_reader = RapidOCR()
            except Exception as e:
                print(f"Failed to initialize RapidOCR: {e}")
                self.rapid_reader = None
        else:
            print("RapidOCR not found.")
            self.rapid_reader = None

        # 2. Other OCR engines disabled for now
        self.paddle_reader = None
        self.easy_reader = None
        self.genai_client = None

    def process_pole_tag(self, image, box, prefer_full_image=False):
        """
        Processes a detected pole: Crops, Enhances, and Scans for IDs.
        box: [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = map(int, box)
        h_orig, w_orig = image.shape[:2]

        if prefer_full_image:
            x1, y1, x2, y2 = 0, 0, w_orig, h_orig

        # 1. Generous Box Expansion
        width = x2 - x1
        pad_left = int(1.8 * width)
        
        # Pad other sides slightly for margin
        x1_expanded = max(0, x1 - pad_left)
        y1_expanded = max(0, y1 - 10)
        x2_expanded = min(w_orig, x2 + 10)
        y2_expanded = min(h_orig, y2 + 10)
        
        crop_bgr = image[y1_expanded:y2_expanded, x1_expanded:x2_expanded]
        if crop_bgr.size == 0:
            return "Not Found"
            
        # 2. Black Patch Localization
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        mean_val = float(np.mean(gray))
        thresh_val = min(80, max(20, int(mean_val * 0.55)))
        _, dark_mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological closing to find the bounding box of the dark-painted patch
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Find contours
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        patch_crop = crop_bgr.copy()
        if False and contours:
            # Get largest contour by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            best_cnt = contours[0]
            if cv2.contourArea(best_cnt) > 200:
                x_p, y_p, w_p, h_p = cv2.boundingRect(best_cnt)
                # Crop to the dark patch bounding box with small margin
                margin = 5
                y1_p = max(0, y_p - margin)
                y2_p = min(crop_bgr.shape[0], y_p + h_p + margin)
                x1_p = max(0, x_p - margin)
                x2_p = min(crop_bgr.shape[1], x_p + w_p + margin)
                patch_crop = crop_bgr[y1_p:y2_p, x1_p:x2_p]

        if patch_crop.size == 0:
            patch_crop = crop_bgr
            
        # 3. Contrast Enhancement
        lab = cv2.cvtColor(patch_crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        enhanced_bgr = cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2BGR)
        
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        enhanced = cv2.filter2D(enhanced_bgr, -1, kernel)
        
        # 4. Generate candidate crops for OCR
        line_crops = []
        frac_crops = []
        
        # Fallbacks: also test the raw expanded crop and the raw unexpanded pole bbox crop
        raw_pole_box_crop = image[y1:y2, x1:x2]
        
        candidates = [crop_bgr, raw_pole_box_crop, enhanced] + line_crops + frac_crops
        candidates = [c for c in candidates if c is not None and c.size > 0]
        
        # 5. RapidOCR-only execution
        rapid_texts = []

        if self.rapid_reader:
            for crop in candidates:
                try:
                    res, _ = self.rapid_reader(crop)
                    if res:
                        crop_text = " ".join([line[1] for line in res])
                        if crop_text.strip():
                            rapid_texts.append(crop_text)
                            score = self._pole_score(crop_text)
                            if score > 1800:
                                return self._normalize_pole_id(crop_text)
                except Exception as e:
                    print(f"RapidOCR error during processing: {e}")

        if rapid_texts:
            best_rapid = self._assemble_lines(rapid_texts)
            if best_rapid and best_rapid != "Not Found":
                return best_rapid

        return "Not Found"

    def _extract_text_line_crops(self, img):
        """
        Uses connected components to slice the crop into isolated line-level candidates.
        """
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(thresh)
        
        components = []
        for i in range(1, num_labels):
            x, y, cw, ch, area = stats[i]
            if area < 8 or cw < 2 or ch < 2:
                continue
            components.append((x, y, cw, ch, y + ch/2))
            
        if not components:
            return []
            
        components.sort(key=lambda x: x[4])
        
        rows = []
        for comp in components:
            x, y, cw, ch, _ = comp
            placed = False
            for r in rows:
                r_y1, r_y2 = r['y1'], r['y2']
                overlap_y1 = max(y, r_y1)
                overlap_y2 = min(y + ch, r_y2)
                overlap_h = overlap_y2 - overlap_y1 if overlap_y1 < overlap_y2 else 0
                min_h = min(ch, r_y2 - r_y1)
                if min_h > 0 and (overlap_h / min_h) > 0.4:
                    r['components'].append(comp)
                    r['x1'] = min(r['x1'], x)
                    r['y1'] = min(r['y1'], y)
                    r['x2'] = max(r['x2'], x + cw)
                    r['y2'] = max(r['y2'], y + ch)
                    placed = True
                    break
            if not placed:
                rows.append({
                    'x1': x, 'y1': y, 'x2': x + cw, 'y2': y + ch,
                    'components': [comp]
                })
                
        row_crops = []
        for r in rows:
            rx1, ry1, rx2, ry2 = r['x1'], r['y1'], r['x2'], r['y2']
            pad_h = max(2, int((ry2 - ry1) * 0.1))
            pad_w = max(2, int((rx2 - rx1) * 0.1))
            
            ry1_pad = max(0, ry1 - pad_h)
            ry2_pad = min(h, ry2 + pad_h)
            rx1_pad = max(0, rx1 - pad_w)
            rx2_pad = min(w, rx2 + pad_w)
            
            crop = img[ry1_pad:ry2_pad, rx1_pad:rx2_pad]
            if crop.size > 0:
                row_crops.append(crop)
                
        return row_crops

    def _fractional_line_crops(self, img):
        """
        Fallback slicing by vertical split (top/bottom halves with overlap).
        """
        h, w = img.shape[:2]
        crops = []
        if h > 30:
            split_h = int(h * 0.6)
            crops.append(img[0:split_h, 0:w])
            crops.append(img[h - split_h:h, 0:w])
        return crops

    def _normalize_pole_id(self, text):
        if not text:
            return "Not Found"
            
        text = text.upper().strip()
        
        # 1. Correct common boundary characters
        text = text.replace('{', '6')
        for char in ['}', ']', ')']:
            text = text.replace(char, '7')
            
        # 2. Standardize prefix variations
        text = re.sub(r'\b(R[DCOZS5][S5]{2}|R[DCOZS5][S5]G|ROSS|RCSS|RDSG|RVSS|RZSS|RD5S)\b', 'RDSS', text)
        text = re.sub(r'\bR\s*[. ]*\s*D\s*[. ]*\s*S\s*[. ]*\s*S\b', 'RDSS', text)
        
        # 3. Standardize voltage suffixes
        text = re.sub(r'\b(H[ .]*T[ .]*|H[. ]*T)\b', 'HT', text)
        text = re.sub(r'\b(L[ .]*T[ .]*|L[. ]*T)\b', 'LT', text)
        
        # 4. Clean tokens using substitutions for numeric clusters
        tokens = text.split()
        normalized_tokens = []
        
        for i, token in enumerate(tokens):
            is_numeric = False
            if i > 0 and normalized_tokens[i-1] == "RDSS":
                is_numeric = True
            elif any(c.isdigit() for c in token):
                is_numeric = True
            elif token not in ["RDSS", "HT", "LT"]:
                is_numeric = True
                
            if is_numeric:
                sub_map = {
                    'O': '0', 'Q': '0',
                    'S': '5',
                    'B': '8',
                    'Z': '2',
                    'I': '1', 'L': '1'
                }
                cleaned = []
                for c in token:
                    if c in sub_map:
                        cleaned.append(sub_map[c])
                    else:
                        cleaned.append(c)
                token_val = "".join(cleaned)
                normalized_tokens.append(token_val)
            else:
                normalized_tokens.append(token)
                
        final_text = " ".join(normalized_tokens)
        
        # Format standardized output: RDSS <digits> <suffix>
        rdss_match = re.search(r'\bRDSS\b', final_text)
        digits_match = re.search(r'\b\d+\b', final_text)
        suffix_match = re.search(r'\b(HT|LT)\b', final_text)
        
        parts = []
        if rdss_match:
            parts.append("RDSS")
        if digits_match:
            parts.append(digits_match.group(0))
        if suffix_match:
            parts.append(suffix_match.group(0))
            
        if parts:
            return " ".join(parts)
            
        return final_text if final_text else "Not Found"

    def _pole_score(self, text):
        norm_text = self._normalize_pole_id(text)
        if not norm_text or norm_text == "Not Found":
            return 0
            
        has_rdss = "RDSS" in norm_text
        digits = re.findall(r'\d+', norm_text)
        num_digits = sum(len(d) for d in digits)
        
        score = 1000
        
        if has_rdss and num_digits > 0:
            score += 500
            
        if "HT" in norm_text or "LT" in norm_text:
            score += 200
            
        if num_digits == 0:
            score -= 500
        elif num_digits == 1:
            score -= 300
        elif num_digits >= 2:
            score += num_digits * 100
            
        return score

    def _assemble_lines(self, texts):
        cleaned_texts = [self._normalize_pole_id(t) for t in texts if t]
        cleaned_texts = [t for t in cleaned_texts if t and t != "Not Found"]
        
        if not cleaned_texts:
            return "Not Found"
            
        best_text = "Not Found"
        best_score = 0
        for text in cleaned_texts:
            score = self._pole_score(text)
            if score > best_score:
                best_score = score
                best_text = text
                
        if best_score >= 1500:
            return best_text
            
        has_rdss = any("RDSS" in t for t in cleaned_texts)
        digits = []
        suffixes = []
        
        for t in cleaned_texts:
            for d in re.findall(r'\b\d+\b', t):
                if d not in digits:
                    digits.append(d)
            if "HT" in t and "HT" not in suffixes:
                suffixes.append("HT")
            if "LT" in t and "LT" not in suffixes:
                suffixes.append("LT")
                
        if has_rdss and digits:
            main_digits = sorted(digits, key=len, reverse=True)[0]
            assembled = f"RDSS {main_digits}"
            if suffixes:
                assembled += f" {suffixes[0]}"
            return assembled
            
        return best_text
