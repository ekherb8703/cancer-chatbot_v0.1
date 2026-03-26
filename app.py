"""
app.py
──────
암 건강 상담 챗봇 (메인 앱)
- 국립암센터 암정보센터 자료 기반 RAG
- OpenAI → Claude → Gemini 자동 전환
- Streamlit Secrets로 API 키 관리
"""

import streamlit as st
from llm_providers import get_llm_response
from rag_engine import build_vector_store, search, format_context

# ──────────────────────────────────────────────
# 페이지 설정
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="암 건강 상담 챗봇",
    page_icon="🏥",
    layout="centered",
)

# ──────────────────────────────────────────────
# 시스템 프롬프트
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """당신은 암 관련 건강 정보를 제공하는 전문 상담 AI입니다.

## 역할
- 암의 종류, 검진, 예방, 치료, 사후관리 등에 대해 근거 중심의 정보를 제공합니다.
- 한국 국가암검진사업(NCSP)의 검진 권고안을 숙지하고 있습니다.
- 국립암센터 암정보센터의 자료를 기반으로 답변합니다.

## 답변 원칙
1. **근거 기반**: 제공된 참고자료와 공식 가이드라인에 근거하여 답변합니다.
2. **쉬운 설명**: 일반인이 이해할 수 있는 수준으로 설명합니다.
3. **한계 명시**: 개인 진단이나 치료 처방은 하지 않으며, 반드시 의료기관 방문을 권고합니다.
4. **공감적 태도**: 암 관련 질문은 불안을 동반할 수 있으므로, 공감적이고 따뜻한 어조를 유지합니다.
5. **참고자료 활용**: 아래 참고자료가 있으면 이를 우선 활용하고, 출처를 간단히 언급합니다.

## 국가암검진사업(NCSP) 6대 암 검진 권고 요약
- 위암: 만 40세 이상, 2년마다 위내시경 (또는 위장조영검사)
- 대장암: 만 50세 이상, 1년마다 분변잠혈검사 → 양성 시 대장내시경
- 간암: 만 40세 이상 고위험군, 6개월마다 간초음파 + 혈청AFP
- 유방암: 만 40세 이상 여성, 2년마다 유방촬영술
- 자궁경부암: 만 20세 이상 여성, 2년마다 자궁경부세포검사
- 폐암: 만 54~74세 고위험 흡연자, 2년마다 저선량 흉부CT

## 주의사항
- "저는 AI 상담 도우미이며, 의사가 아닙니다"라는 점을 필요시 안내합니다.
- 응급 상황이 의심되면 즉시 119 또는 가까운 응급실 방문을 안내합니다.
"""

SYSTEM_PROMPT_WITH_CONTEXT = SYSTEM_PROMPT + """
## 참고자료 (국립암센터 암정보센터)
아래는 사용자의 질문과 관련된 참고자료입니다. 이 내용을 활용하여 답변하세요.

{context}
"""

# ──────────────────────────────────────────────
# 내부 설정 (사용자에게 노출하지 않음)
# ──────────────────────────────────────────────
TEMPERATURE = 0.3
USE_RAG = True

has_openai = bool(st.secrets.get("OPENAI_API_KEY"))
has_claude = bool(st.secrets.get("ANTHROPIC_API_KEY"))
has_gemini = bool(st.secrets.get("GOOGLE_API_KEY"))

# ──────────────────────────────────────────────
# 벡터 DB 초기화 (앱 시작 시 1회)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def init_vector_store():
    return build_vector_store()

vector_store = init_vector_store()
doc_count = vector_store.count() if vector_store else 0

# ──────────────────────────────────────────────
# 암종별 주요 질문
# ──────────────────────────────────────────────
CANCER_TOPICS = {
    "🫁 위암": [
        "위암 검진은 몇 살부터 받나요?",
        "위내시경 검사 전 준비사항은?",
        "위암의 초기 증상은 무엇인가요?",
    ],
    "🫀 대장암": [
        "대장암 검진은 어떻게 하나요?",
        "분변잠혈검사에서 양성이 나왔어요",
        "대장내시경 전 준비사항이 궁금해요",
    ],
    "🫁 간암": [
        "간암 검진 대상은 누구인가요?",
        "B형간염 보유자인데 검진 주기는?",
        "간암의 위험 요인은 무엇인가요?",
    ],
    "🎀 유방암": [
        "유방암 자가검진은 어떻게 하나요?",
        "유방촬영술(맘모그래피)은 어떤 검사인가요?",
        "치밀유방이란 무엇인가요?",
    ],
    "🩺 자궁경부암": [
        "자궁경부암 검진은 몇 살부터 받나요?",
        "HPV 백신은 언제 맞아야 하나요?",
        "자궁경부세포검사 전 주의사항은?",
    ],
    "💨 폐암": [
        "폐암 검진 대상은 누구인가요?",
        "저선량 흉부CT는 어떤 검사인가요?",
        "갑년(pack-year) 계산은 어떻게 하나요?",
    ],
}

# ──────────────────────────────────────────────
# 메인 UI
# ──────────────────────────────────────────────
st.title("🏥 암 건강 상담 chatbot")
st.caption("국립암센터 암정보센터 자료 기반 · 건강 정보 제공")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 이력 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ──────────────────────────────────────────────
# 초기 화면: 암종별 버튼
# ──────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("#### 궁금한 암종을 선택하세요")

    # 암종 선택 탭
    tabs = st.tabs(list(CANCER_TOPICS.keys()))

    for tab, (cancer_name, questions) in zip(tabs, CANCER_TOPICS.items()):
        with tab:
            for q in questions:
                if st.button(q, key=f"q_{cancer_name}_{q}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": q})
                    st.rerun()

    st.divider()
    st.markdown("#### 💬 또는 직접 질문을 입력하세요")

# ──────────────────────────────────────────────
# 대화 중: 대화 초기화 버튼
# ──────────────────────────────────────────────
if st.session_state.messages:
    col_spacer, col_btn = st.columns([4, 1])
    with col_btn:
        if st.button("🗑️ 초기화", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ──────────────────────────────────────────────
# 사용자 입력
# ──────────────────────────────────────────────
if prompt := st.chat_input("암 관련 질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# ──────────────────────────────────────────────
# LLM 호출
# ──────────────────────────────────────────────
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    if not any([has_openai, has_claude, has_gemini]):
        st.error("⚠️ 현재 서비스 점검 중입니다. 잠시 후 다시 시도해 주세요.")
        st.stop()

    user_query = st.session_state.messages[-1]["content"]

    with st.chat_message("assistant"):
        with st.spinner("답변을 생성하고 있습니다..."):
            # RAG 검색
            context_text = ""
            results = []
            if USE_RAG and doc_count > 0:
                results = search(user_query)
                context_text = format_context(results)

            # 시스템 프롬프트
            if context_text:
                system = SYSTEM_PROMPT_WITH_CONTEXT.format(context=context_text)
            else:
                system = SYSTEM_PROMPT

            # 메시지 구성 (최근 20개 메시지)
            api_messages = [{"role": "system", "content": system}]
            recent = st.session_state.messages[-20:]
            api_messages += [
                {"role": m["role"], "content": m["content"]} for m in recent
            ]

            # LLM 호출
            response_text, model_used = get_llm_response(
                api_messages, temperature=TEMPERATURE
            )

            st.markdown(response_text)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
            })

# ──────────────────────────────────────────────
# 하단 고정 안내
# ──────────────────────────────────────────────
st.divider()
st.caption("ℹ️ 이 챗봇은 의료 정보 제공 목적이며, 전문 의료 상담을 대체하지 않습니다. 정확한 진단 및 치료는 의료기관을 방문하세요.")
