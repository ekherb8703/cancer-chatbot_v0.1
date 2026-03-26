"""
fetch_cancer_info.py
────────────────────
국립암센터 암정보센터(cancer.go.kr) 웹페이지에서
암 관련 정보를 텍스트로 추출하여 data/ 폴더에 저장하는 스크립트.

사용법:
    python fetch_cancer_info.py

실행하면 data/ 폴더에 .txt 파일들이 생성됩니다.
이후 app.py를 실행하면 자동으로 RAG 벡터 DB가 구축됩니다.
"""

import os
import time
import requests
from bs4 import BeautifulSoup

# ──────────────────────────────────────────────
# 크롤링 대상 페이지 목록
# 국립암센터 암정보센터의 주요 암종별 정보 페이지
# ──────────────────────────────────────────────
CANCER_PAGES = {
    # 암종별 정보 (개요, 위험요인, 증상, 진단, 치료, 예방)
    "위암": [
        "https://www.cancer.go.kr/lay1/S1T639C643/contents.do",
        "https://www.cancer.go.kr/lay1/S1T639C644/contents.do",
        "https://www.cancer.go.kr/lay1/S1T639C645/contents.do",
    ],
    "대장암": [
        "https://www.cancer.go.kr/lay1/S1T641C655/contents.do",
        "https://www.cancer.go.kr/lay1/S1T641C656/contents.do",
        "https://www.cancer.go.kr/lay1/S1T641C657/contents.do",
    ],
    "유방암": [
        "https://www.cancer.go.kr/lay1/S1T645C675/contents.do",
        "https://www.cancer.go.kr/lay1/S1T645C676/contents.do",
        "https://www.cancer.go.kr/lay1/S1T645C677/contents.do",
    ],
    "간암": [
        "https://www.cancer.go.kr/lay1/S1T643C663/contents.do",
        "https://www.cancer.go.kr/lay1/S1T643C664/contents.do",
        "https://www.cancer.go.kr/lay1/S1T643C665/contents.do",
    ],
    "폐암": [
        "https://www.cancer.go.kr/lay1/S1T647C687/contents.do",
        "https://www.cancer.go.kr/lay1/S1T647C688/contents.do",
        "https://www.cancer.go.kr/lay1/S1T647C689/contents.do",
    ],
    "자궁경부암": [
        "https://www.cancer.go.kr/lay1/S1T649C699/contents.do",
        "https://www.cancer.go.kr/lay1/S1T649C700/contents.do",
        "https://www.cancer.go.kr/lay1/S1T649C701/contents.do",
    ],
}

# 검진 안내 페이지
SCREENING_PAGES = {
    "국가암검진_안내": [
        "https://www.cancer.go.kr/lay1/S1T626C628/contents.do",
    ],
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_page_text(url: str) -> str:
    """웹페이지에서 본문 텍스트를 추출합니다."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        resp.encoding = "utf-8"

        soup = BeautifulSoup(resp.text, "html.parser")

        # 국립암센터 사이트의 본문 영역 선택자들
        content_selectors = [
            ".content_area",
            ".cont_area",
            ".sub_content",
            "#content",
            ".board_view",
            "article",
        ]

        content = None
        for selector in content_selectors:
            content = soup.select_one(selector)
            if content:
                break

        if not content:
            content = soup.find("body")

        # 불필요한 태그 제거
        for tag in content.find_all(["script", "style", "nav", "header", "footer"]):
            tag.decompose()

        # 텍스트 추출 & 정리
        text = content.get_text(separator="\n", strip=True)

        # 빈 줄 정리
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return "\n".join(lines)

    except Exception as e:
        print(f"  ⚠️  실패: {url} → {e}")
        return ""


def save_texts(topic: str, texts: list[str]):
    """추출된 텍스트를 파일로 저장합니다."""
    os.makedirs(DATA_DIR, exist_ok=True)

    combined = f"# {topic}\n\n"
    for i, text in enumerate(texts):
        if text:
            combined += f"\n{'='*60}\n"
            combined += text
            combined += "\n"

    filepath = os.path.join(DATA_DIR, f"{topic}.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(combined)

    print(f"  ✅ 저장 완료: {filepath} ({len(combined):,}자)")


def main():
    print("=" * 60)
    print("🏥 국립암센터 암정보센터 자료 수집")
    print("=" * 60)

    all_pages = {**CANCER_PAGES, **SCREENING_PAGES}

    for topic, urls in all_pages.items():
        print(f"\n📋 {topic} ({len(urls)}페이지)")

        texts = []
        for url in urls:
            print(f"  📥 {url}")
            text = fetch_page_text(url)
            if text:
                texts.append(text)
                print(f"     → {len(text):,}자 추출")
            time.sleep(1)  # 서버 부하 방지

        if texts:
            save_texts(topic, texts)
        else:
            print(f"  ⚠️  {topic}: 추출된 내용이 없습니다.")

    print(f"\n{'='*60}")
    print("✅ 완료! data/ 폴더의 파일들이 RAG에 사용됩니다.")
    print("   이제 'streamlit run app.py'를 실행하세요.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
