import os
import cv2
import numpy as np
import easyocr

class OCREngine:
    def __init__(self, use_gpu=False):
        print("[OCR Engine] Initializing EasyOCR (download_enabled=True)...")
        try:
            self.easyocr_reader = easyocr.Reader(['en'], gpu=use_gpu, download_enabled=True)
            print("[OCR Engine] EasyOCR initialized successfully.")
        except Exception as e:
            raise RuntimeError(f"[OCR Engine] Critical Error: Failed to initialize EasyOCR ({e}).")

    def preprocess_image(self, img_path):
        """Applies contrast enhancement and grayscale conversion."""
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return img, gray, enhanced

    def detect_crossed_out_lines(self, img, bbox):
        """Detects horizontal red/dark strikethrough lines crossing through bounding box ROI."""
        x, y, w, h = bbox
        roi = img[y:y+h, x:x+w]
        if roi.size == 0:
            return False
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        detected_lines = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, horizontal_kernel)
        return float(np.sum(detected_lines > 0)) > 10.0

    def extract_page_ocr(self, img_path, page_num=1, margin_threshold=140):
        """
        Extracts text bounding boxes using EasyOCR and groups them into line elements.
        """
        img, gray, enhanced = self.preprocess_image(img_path)
        img_h, img_w, _ = img.shape

        results = self.easyocr_reader.readtext(img_path)
        if not results:
            raise RuntimeError(f"[OCR Engine] EasyOCR failed to detect any text on page {page_num} ({img_path}).")

        # Group raw box detections into line elements
        raw_boxes = []
        for (pts, text, conf) in results:
            text_str = text.strip()
            if not text_str:
                continue
            pts_arr = np.array(pts, dtype=np.int32)
            x_min = int(np.min(pts_arr[:, 0]))
            y_min = int(np.min(pts_arr[:, 1]))
            x_max = int(np.max(pts_arr[:, 0]))
            y_max = int(np.max(pts_arr[:, 1]))
            w = max(1, x_max - x_min)
            h = max(1, y_max - y_min)
            raw_boxes.append({
                'x': x_min, 'y': y_min, 'w': w, 'h': h,
                'text': text_str, 'conf': float(conf)
            })

        raw_boxes.sort(key=lambda b: (b['y'], b['x']))

        lines = []
        y_tolerance = 18
        for b in raw_boxes:
            matched = False
            for line in lines:
                if abs(b['y'] - line['y_avg']) <= y_tolerance:
                    line['boxes'].append(b)
                    line['y_avg'] = float(np.mean([box['y'] for box in line['boxes']]))
                    matched = True
                    break
            if not matched:
                lines.append({'y_avg': float(b['y']), 'boxes': [b]})

        extracted_elements = []
        for line in lines:
            line['boxes'].sort(key=lambda b: b['x'])
            x_min = min(b['x'] for b in line['boxes'])
            y_min = min(b['y'] for b in line['boxes'])
            x_max = max(b['x'] + b['w'] for b in line['boxes'])
            y_max = max(b['y'] + b['h'] for b in line['boxes'])
            w = max(1, x_max - x_min)
            h = max(1, y_max - y_min)

            combined_text = " ".join(b['text'] for b in line['boxes'])
            avg_conf = float(np.mean([b['conf'] for b in line['boxes']]))
            is_margin = (x_min < margin_threshold)
            is_crossed = self.detect_crossed_out_lines(img, [x_min, y_min, w, h])

            extracted_elements.append({
                "page": page_num,
                "bbox": [x_min, y_min, w, h],
                "text": combined_text,
                "confidence": round(avg_conf, 3),
                "is_margin": is_margin,
                "is_crossed_out": is_crossed
            })

        extracted_elements.sort(key=lambda elem: (elem["bbox"][1], elem["bbox"][0]))
        return extracted_elements, (img_w, img_h)

