import io, requests
from utils.settings import NAVER_CLIENT_ID, NAVER_CLIENT_SECRET

# 간단 텍스트 검색 → 좌표 변환(geocoding)
def geocode(query: str):
    if not (NAVER_CLIENT_ID and NAVER_CLIENT_SECRET):
        return None
    url = "https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode"
    headers = {"X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID, "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET}
    resp = requests.get(url, headers=headers, params={"query": query}, timeout=10)
    resp.raise_for_status()
    items = resp.json().get("addresses", [])
    if not items: return None
    x = float(items[0]["x"]); y=float(items[0]["y"])
    return x, y  # lon, lat

# 정적 지도 이미지 (경로 polyline 없이 핀만; 해커톤 버전)
def static_map(lonlat_list: list[tuple[float,float]], w=700, h=500, level=12):
    if not (NAVER_CLIENT_ID and NAVER_CLIENT_SECRET): return None
    base = "https://naveropenapi.apigw.ntruss.com/map-static/v2/raster"
    headers = {"X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID, "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET}
    # 마커 문자열
    m = []
    for i,(x,y) in enumerate(lonlat_list):
        color = "red" if i==0 else "blue"
        m.append(f"type:t|size:small|pos:{x}%20{y}|color:{color}")
    params = {"w": w, "h": h, "level": level, "markers": "|".join(m)}
    r = requests.get(base, headers=headers, params=params, timeout=10)
    r.raise_for_status()
    return io.BytesIO(r.content)

# (선택) 길찾기 요약(도보/자동차 등 Directions API 사용 - 간략)
def route_summary(origin_xy, dest_xy):
    # Directions5 API 문서 기준 (자동차)
    if not (NAVER_CLIENT_ID and NAVER_CLIENT_SECRET): return None
    url = "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving"
    headers = {"X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID, "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET}
    s = f"{origin_xy[0]},{origin_xy[1]}"
    e = f"{dest_xy[0]},{dest_xy[1]}"
    r = requests.get(url, headers=headers, params={"start": s, "goal": e}, timeout=10)
    r.raise_for_status()
    js = r.json()
    try:
        path = js["route"]["traoptimal"][0]["summary"]
        dist_km = round(path["distance"]/1000, 2)
        dur_min = int(path["duration"]/60000)
        return {"distance_km": dist_km, "duration_min": dur_min}
    except Exception:
        return None
