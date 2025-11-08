# services/autofill/ocr_attendance.py
from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import re
import numpy as np
import cv2
import pytesseract

# 필요시 직접 경로 지정:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------- 공통 유틸 ----------------
HM_RE = re.compile(r"(\d{1,3})\s*[hH]\s*(\d{1,2})\s*[mM]")

def _hm_to_min(hm: str) -> int:
    m = HM_RE.search(hm or "")
    return int(m.group(1))*60 + int(m.group(2)) if m else 0

def _min_to_hm(mn: int) -> str:
    return f"{mn//60}h {mn%60}m"

def _pick_hm(s: str) -> Optional[str]:
    m = HM_RE.search(s or "")
    return f"{int(m.group(1))}h {int(m.group(2))}m" if m else None

def _bgr2rgb(x): return cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

def _ocr_numbers(img: np.ndarray) -> str:
    """
    숫자+h/m을 읽어 문자열로 반환. 여러 PSM 모드를 시도하여 가장 긴 숫자 반환.
    """
    # 여러 PSM 모드 시도 (7: 한 줄, 6: 블록, 13: 원시 라인)
    psm_modes = [7, 6, 13]
    candidates = []

    for psm in psm_modes:
        cfg = f'--psm {psm} --oem 1 -c tessedit_char_whitelist=0123456789hmHM'
        try:
            txt = pytesseract.image_to_string(img, lang="eng", config=cfg)
            txt = txt.replace(" ", "").replace("\n", "")

            # 200h10m / 176h0m 형태든 200h 10m이든 모두 커버
            m = re.search(r"(\d{1,3})[hH]\s*(\d{1,2})[mM]", txt)
            if m:
                result = f"{m.group(1)}h {m.group(2)}m"
                candidates.append((result, int(m.group(1))))
                continue

            # 분이 없는 케이스(176h)
            m2 = re.search(r"(\d{1,3})[hH]", txt)
            if m2:
                result = f"{m2.group(1)}h 0m"
                candidates.append((result, int(m2.group(1))))
        except Exception:
            pass

    # 시간(h) 숫자가 가장 큰 것 반환 (200 > 00)
    if candidates:
        return max(candidates, key=lambda x: x[1])[0]

    return ""

# def _ocr_numbers(img: np.ndarray) -> str:
#     """(기존 함수 사용) 숫자+h/m을 읽어 문자열로 반환."""
#     if img.ndim == 3:
#         g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     else:
#         g = img
#     # 여러 전처리 조합을 간단히 시도
#     variants = []
#     variants.append(g)
#     variants.append(cv2.GaussianBlur(g, (3,3), 0))
#     for s in (1.6, 2.0, 2.4):
#         variants.append(cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR))
#     _, otsu = cv2.threshold(cv2.GaussianBlur(g,(3,3),0),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     variants.append(otsu)
#     variants.append(255-otsu)

#     cfgs = [
#         r'--psm 7 -l eng -c tessedit_char_whitelist=0123456789hmHM classify_bln_numeric_mode=1',
#         r'--psm 6 -l eng -c tessedit_char_whitelist=0123456789hmHM classify_bln_numeric_mode=1',
#     ]
#     best = None
#     for v in variants:
#         for cfg in cfgs:
#             try:
#                 t = pytesseract.image_to_string(v, config=cfg)
#                 t = t.replace(" ", "").replace("\n","")
#                 m = re.search(r"(\d{1,3})[hH]\s*(\d{1,2})[mM]", t)
#                 if m:
#                     cand = f"{int(m.group(1))}h {int(m.group(2))}m"
#                 else:
#                     m2 = re.search(r"(\d{1,3})[hH]", t)
#                     cand = f"{int(m2.group(1))}h 0m" if m2 else ""
#                 if cand:
#                     if best is None or _hm_to_min(cand) > _hm_to_min(best):
#                         best = cand
#             except Exception:
#                 pass
#     return best or ""

def _prep_bin(g: np.ndarray, scale: float = 1.6) -> np.ndarray:
    g = cv2.GaussianBlur(g, (3,3), 0)
    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 35, 11)


def _read_line_value(img, panel_x, y, offset_down=50):
    """
    색 점(또는 라벨)의 y를 기준으로, 그보다 약간 아래(offset_down) 라인에서
    오른쪽 숫자(예: 176h 0m / 6h 10m)를 OCR.
    """
    H, W = img.shape[:2]
    y1 = max(0, y + offset_down - 20)
    y2 = min(H, y + offset_down + 20)
    x1 = panel_x + 40   # 점 오른쪽 여유
    x2 = W - 10

    gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    bin_img = _prep_bin(gray, 1.6)   # 가볍게 확대 + adaptive threshold
    return _ocr_numbers(bin_img)



def _hsv_mask(img, lower, upper):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))


def _find_total_hm(img: np.ndarray) -> str:
    """
    좌측 상단의 TOTAL ROI 영역에서 큰 숫자(예: 200h 10m)를 읽는다.
    여러 전처리 방법을 시도하여 가장 신뢰도 높은 결과를 반환.
    """
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 좌상단 큰 숫자 영역 - x1을 더 왼쪽으로 확장하여 첫 번째 숫자 완전히 포함
    x1, y1, x2, y2 = int(0.015*W), int(0.14*H), int(0.42*W), int(0.40*H)
    roi = gray[y1:y2, x1:x2]

    # 여러 전처리 방법 시도
    candidates = []

    # 1) 다양한 스케일로 확대 (3.0까지 시도)
    for scale in [1.5, 2.0, 2.5]:
        scaled = _prep_bin(roi, scale)
        result = _ocr_numbers(scaled)
        if result:
            candidates.append((result, _hm_to_min(result)))

    # 2) Sharpening 적용 후 스케일
    kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(roi, -1, kernel_sharpen)
    for scale in [2.0, 2.5]:
        scaled = _prep_bin(sharpened, scale)
        result = _ocr_numbers(scaled)
        if result:
            candidates.append((result, _hm_to_min(result)))

    # 3) OTSU 이진화
    _, otsu = cv2.threshold(cv2.GaussianBlur(roi, (3,3), 0), 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 확대 후 OTSU
    for scale in [2.0, 2.5]:
        otsu_scaled = cv2.resize(otsu, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        result = _ocr_numbers(otsu_scaled)
        if result:
            candidates.append((result, _hm_to_min(result)))

    # 4) OTSU 반전
    result = _ocr_numbers(255 - otsu)
    if result:
        candidates.append((result, _hm_to_min(result)))

    # 5) 기본 adaptive threshold
    result = _ocr_numbers(_prep_bin(roi, 1.8))
    if result:
        candidates.append((result, _hm_to_min(result)))

    # 빈도 기반 선택: 가장 많이 나온 값을 선택
    if candidates:
        from collections import Counter
        # 결과 문자열만 추출 (예: "200h 10m")
        results = [c[0] for c in candidates]
        # 빈도 계산
        counter = Counter(results)
        # 가장 많이 나온 값 반환
        most_common = counter.most_common(1)[0][0]
        return most_common

    return ""


def _mask_hsv(img: np.ndarray, lower: Tuple[int,int,int], upper: Tuple[int,int,int]) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))

def _find_dot_lines(img: np.ndarray) -> Dict[str, int]:
    """
    오른쪽 패널의 색 점 y좌표 탐색.
    반환: {'select': y, 'extend': y}  (화면 절대좌표)
    """
    H, W = img.shape[:2]
    x1 = int(0.56 * W)
    panel = img[:, x1:].copy()

    # 초록(선택근무) HSV 범위
    # H: 35~85 정도가 일반적인 녹색 대역
    green1 = _mask_hsv(panel, (35,  40,  40), (85, 255, 255))

    # 빨강(초과근무) HSV는 2개 구간(0~10, 170~180)
    red1 = _mask_hsv(panel, (0,   70,  50), (10, 255, 255))
    red2 = _mask_hsv(panel, (170, 70,  50), (180,255, 255))
    red = cv2.bitwise_or(red1, red2)

    def biggest_blob_y(mask: np.ndarray) -> Optional[int]:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None
        cnt = max(cnts, key=cv2.contourArea)
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        # 너무 작은 노이즈 제외
        if r < 3: return None
        # 절대좌표 y (패널 오프셋 보정)
        return int(cy)

    y_g = biggest_blob_y(green1)
    y_r = biggest_blob_y(red)

    # 절대좌표로 변환 (x는 안 쓰니까 y만)
    return {
        "select": (y_g if y_g is not None else None),
        "extend": (y_r if y_r is not None else None),
        "panel_x": x1
    }

def _read_line_value(img: np.ndarray, panel_x: int, y: int) -> str:
    """
    주어진 y라인에서 우측 값만 OCR.
    여러 전처리 방법을 시도하여 빈도 기반으로 결과 선택.
    """
    H, W = img.shape[:2]
    from collections import Counter
    # 라인 범위를 위쪽으로 이동 (숫자가 점보다 위에 있음)
    y1 = max(0, y - 30)
    y2 = min(H, y + 18)
    # 시간 숫자는 점 오른쪽 한참 떨어져 있으므로 x는 패널 시작+넉넉한 오프셋
    x1 = panel_x + 40
    x2 = W - 10
    gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

    candidates = []
    # 여러 스케일 시도
    for scale in [1.6, 2.0, 2.5]:
        bin_img = _prep_bin(gray, scale)
        result = _ocr_numbers(bin_img)
        if result:
            candidates.append(result)

    # 빈도 기반 선택
    if candidates:
        counter = Counter(candidates)
        return counter.most_common(1)[0][0]

    return ""


# === (추가) 점 주변 offset을 스캔하며 가장 먼저 읽히는 시간 찾기 ===
def _scan_best_hm(img: np.ndarray, panel_x: int, y: int,
                  offset_min: int = -20, offset_max: int = 40, step: int = 4) -> str:
    """
    점의 y 좌표 주변에서 숫자를 스캔.
    offset -20부터 시작하여 점보다 위쪽도 확인 (숫자가 점 위에 있을 수 있음).
    """
    H, W = img.shape[:2]
    from collections import Counter
    candidates = []

    # 여러 offset을 시도하여 후보 수집
    for off in range(offset_min, offset_max + 1, step):
        y_center = y + off
        y1 = max(0, y_center - 22)
        y2 = min(H, y_center + 22)
        x1 = panel_x + 40
        x2 = W - 10

        gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

        # 여러 전처리 방법 시도
        for scale in [1.6, 2.0, 2.5]:
            bin_img = _prep_bin(gray, scale)
            s = _ocr_numbers(bin_img)
            if _pick_hm(s):
                candidates.append(s)

    # 빈도 기반 선택
    if candidates:
        counter = Counter(candidates)
        return counter.most_common(1)[0][0]

    return ""

# === (추가) 색이 안 잡힐 때: 원형(●) 자체를 찾는 폴백 ===
def _find_dot_lines_by_circles(img: np.ndarray) -> Dict[str, int]:
    H, W = img.shape[:2]
    panel_x = int(0.56 * W)
    panel = img[:, panel_x:].copy()
    g = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5,5), 0)
    # 작은 원형들 찾기 (반경은 화면에 맞춰 5~14 정도)
    circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
                               param1=80, param2=18, minRadius=5, maxRadius=14)
    ys = []
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        # 패널 왼쪽 영역(라벨/점이 있는 x<100 근처)만 채택
        for (cx, cy, r) in circles:
            if cx < 100 and r >= 4:
                ys.append(int(cy))
    ys = sorted(ys)
    # 보통 순서: 선택(위) → 초과(중간) → 승인(아래)
    sel_y = ys[0] if len(ys) >= 1 else None
    ext_y = ys[1] if len(ys) >= 2 else None
    return {"panel_x": panel_x, "select": sel_y, "extend": ext_y}

# === (추가) 패널 전체에서 시간 텍스트들을 y순으로 모아 폴백 ===
def _find_times_by_ocr_in_panel(img: np.ndarray) -> Dict[str, str]:
    H, W = img.shape[:2]
    panel_x = int(0.56 * W)
    panel = img[:, panel_x:].copy()
    cfg = "--psm 6 -l eng+kor"
    data = pytesseract.image_to_data(panel, config=cfg, output_type=pytesseract.Output.DICT)
    n = len(data["text"])
    rows = []
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        hm = _pick_hm(txt.replace(" ", ""))
        if hm:
            y = int(data["top"][i])
            rows.append((y, hm))
    rows.sort(key=lambda x: x[1])  # 값은 상관없고…
    rows.sort(key=lambda x: x[0])  # y로 다시 정렬
    sel = rows[0][1] if len(rows) >= 1 else ""
    ext = rows[1][1] if len(rows) >= 2 else ""
    return {"panel_x": panel_x, "select_hm": sel, "extend_hm": ext}



#===main===
def extract_times_from_image(image_bytes: bytes, return_debug: bool=False):
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지 디코딩 실패")

    H, W = img.shape[:2]

    # 1) 총근무
    total_hm = _find_total_hm(img)
    # total_hm = _find_total_hm_stronger(img)

    # 2) 기본: 색 마스크로 점 y 찾기
    lines = _find_dot_lines(img)  # {'panel_x', 'select', 'extend'}

    # 2-1) 색이 안 잡히면: 원형 검출 폴백
    if lines.get("select") is None and lines.get("extend") is None:
        lines = _find_dot_lines_by_circles(img)

    select_hm = ""
    extend_hm = ""

    # 2-2) 점 y가 있으면 offset 스캔으로 숫자 읽기 (위쪽부터 스캔)
    if lines.get("select") is not None:
        select_hm = _scan_best_hm(img, lines["panel_x"], lines["select"], -20, 40, 4)
    if lines.get("extend") is not None:
        extend_hm = _scan_best_hm(img, lines["panel_x"], lines["extend"], -20, 40, 4)

    # 2-3) 그래도 못 읽었으면: 패널 전체 OCR해서 y순으로 선택/초과 픽스
    if (not select_hm) or (not extend_hm):
        alt = _find_times_by_ocr_in_panel(img)
        if not select_hm:
            select_hm = alt["select_hm"]
        if not extend_hm:
            extend_hm = alt["extend_hm"]

    # 3) 보정: 초과가 비었으면 total-select
    if not extend_hm and total_hm and select_hm:
        extend_hm = _min_to_hm(max(0, _hm_to_min(total_hm) - _hm_to_min(select_hm)))

    result = {
        "total_hm": total_hm or "",
        "select_hm": select_hm or "",
        "extend_hm": extend_hm or ""
    }

    if not return_debug:
        return result

    # ---- 디버그 시각화 ----
    overlay = img.copy()
    # TOTAL ROI 좌표를 _find_total_hm()과 동일하게 설정
    tx1, ty1, tx2, ty2 = int(0.005*W), int(0.20*H), int(0.25*W), int(0.42*H)
    cv2.rectangle(overlay, (tx1,ty1), (tx2,ty2), (0,0,255), 2)
    cv2.putText(overlay, "TOTAL ROI", (tx1, max(ty1-8, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    px = lines.get("panel_x", int(0.56*W))
    y_offset = int(0.05 * H)  # 약간 위로 올리기
    cv2.rectangle(overlay, (px, 0+y_offset), (W, H-y_offset), (255,0,0), 2)
    cv2.putText(overlay, "RIGHT PANEL", (px+6, 24+y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    def draw_read_box(y, color=(0,255,0), off=50):
        if y is None: return
        
        y1 = max(0, y + off - 20); y2 = min(H, y + off + 16)
        x1 = px + 40; x2 = W - 10
        cv2.line(overlay, (px, y), (W-1, y), color, 2)
        cv2.rectangle(overlay, (x1+100, y1+12), (x2, y2+12), color, 2)

    draw_read_box(lines.get("select"), (0,255,0), 5)
    draw_read_box(lines.get("extend"), (0,0,255), 5)

    panel = img[:, px:].copy()
    gmask = _mask_hsv(panel, (35,40,40), (85,255,255))
    rmask = cv2.bitwise_or(
        _mask_hsv(panel, (0,70,50), (10,255,255)),
        _mask_hsv(panel, (170,70,50), (180,255,255))
    )

    debug = {
        "orig": _bgr2rgb(img),
        "overlay": _bgr2rgb(overlay),
        "green_mask": gmask,
        "red_mask": rmask,
    }
    return result, debug

