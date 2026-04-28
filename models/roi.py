"""
Palm ROI Extractor
==================
MediaPipe Hands 랜드마크 기반 장문 ROI 추출.
MediaPipe 실패 시 컨투어 기반 fallback 적용.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional


class PalmROIExtractor:
    """
    Parameters
    ----------
    output_size : int   — 출력 정사각형 크기 (px)
    grayscale   : bool  — True이면 단채널 그레이스케일 반환
    min_detection_confidence : float — MediaPipe 검출 임계값
    """

    def __init__(
        self,
        output_size: int = 128,
        grayscale: bool = True,
        min_detection_confidence: float = 0.3,
        min_landmark_visibility: float = 0.5,
    ):
        self.output_size = output_size
        self.grayscale = grayscale
        self.min_detection_confidence = min_detection_confidence
        self.min_landmark_visibility = min_landmark_visibility
        self._mp_hands = mp.solutions.hands

    def extract(self, image: np.ndarray) -> Optional[np.ndarray]:
        """BGR 이미지 → ROI (MediaPipe only). 실패 시 None."""
        return self._extract_mediapipe(image)

    def extract_with_fallback(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        BGR 이미지 → ROI.
        1. MediaPipe 시도
        2. 실패 시 컨투어 기반 fallback
        3. 둘 다 실패하면 None
        """
        roi = self._extract_mediapipe(image)
        if roi is None:
            roi = self._extract_contour(image)
        return roi

    # ── MediaPipe 랜드마크 기반 ──────────────────────────────────────
    def _extract_mediapipe(self, image: np.ndarray) -> Optional[np.ndarray]:
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with self._mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=self.min_detection_confidence,
        ) as hands:
            results = hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        lm = results.multi_hand_landmarks[0].landmark

        visibilities = [lm[i].visibility for i in range(21)]
        if max(visibilities) > 0.1:
            if any(v < self.min_landmark_visibility for v in visibilities):
                return None

        def pt(idx) -> np.ndarray:
            return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float32)

        wrist     = pt(0)
        idx_mcp   = pt(5)
        mid_mcp   = pt(9)
        ring_mcp  = pt(13)
        pinky_mcp = pt(17)

        finger_base_mid = (idx_mcp + mid_mcp + ring_mcp + pinky_mcp) / 4.0
        palm_center = finger_base_mid * 0.7 + wrist * 0.3

        palm_width = float(np.linalg.norm(idx_mcp - pinky_mcp))
        crop_size  = int(palm_width * 1.3)

        direction = finger_base_mid - wrist
        angle_deg = float(np.degrees(np.arctan2(direction[0], -direction[1])))

        roi = self._rotate_and_crop(image, palm_center, angle_deg, crop_size)
        if roi is not None and self._quality_score(roi) < 0.35:
            return None
        return roi

    # ── 컨투어 기반 fallback ─────────────────────────────────────────
    def _extract_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

        best_roi, best_score = None, 0.0

        for method, val in [('otsu', None), ('fixed', 40), ('fixed', 70), ('fixed', 100)]:
            if method == 'otsu':
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                _, binary = cv2.threshold(gray, val, 255, cv2.THRESH_BINARY)

            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k)

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            largest   = max(contours, key=cv2.contourArea)
            rect      = cv2.minAreaRect(largest)
            center, (bw, bh), angle = rect
            crop_size = int(min(bw, bh) * 0.75)

            roi = self._rotate_and_crop(
                image, np.array(center, dtype=np.float32), angle, crop_size
            )
            if roi is None:
                continue

            score = self._quality_score(roi)
            if score > best_score:
                best_score = score
                best_roi   = roi

        return best_roi if best_score > 0.3 else None

    # ── 공통 유틸 ────────────────────────────────────────────────────
    def _quality_score(self, roi: np.ndarray) -> float:
        gray = roi if roi.ndim == 2 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray > 10))

    def _rotate_and_crop(
        self,
        image: np.ndarray,
        center: np.ndarray,
        angle_deg: float,
        crop_size: int,
    ) -> Optional[np.ndarray]:
        h, w  = image.shape[:2]
        cx, cy = float(center[0]), float(center[1])

        M       = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        half = crop_size // 2
        x1, y1 = int(cx) - half, int(cy) - half
        x2, y2 = x1 + crop_size, y1 + crop_size

        if x1 < 0: x1, x2 = 0, crop_size
        if y1 < 0: y1, y2 = 0, crop_size
        if x2 > w: x1, x2 = w - crop_size, w
        if y2 > h: y1, y2 = h - crop_size, h
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        roi = rotated[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        roi = cv2.resize(roi, (self.output_size, self.output_size),
                         interpolation=cv2.INTER_AREA)
        if self.grayscale:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return roi
