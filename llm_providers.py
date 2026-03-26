"""
llm_providers.py
─────────────────
OpenAI → Claude → Gemini 순서로 자동 전환되는 LLM 클라이언트.
하나의 API가 실패(할당량 초과 등)하면 다음 모델로 넘어갑니다.
"""

import streamlit as st


def _get_openai_response(messages: list, temperature: float) -> str | None:
    """OpenAI GPT 호출 (1순위)"""
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature,
            max_tokens=2048,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.toast(f"⚠️ OpenAI 호출 실패 → Claude로 전환합니다. ({e})", icon="🔄")
        return None


def _get_claude_response(messages: list, temperature: float) -> str | None:
    """Anthropic Claude 호출 (2순위)"""
    api_key = st.secrets.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        # Claude 형식으로 변환: system은 별도, 나머지는 messages
        system_msg = ""
        claude_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                claude_messages.append({"role": m["role"], "content": m["content"]})

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=system_msg,
            messages=claude_messages,
            temperature=temperature,
        )
        return response.content[0].text
    except Exception as e:
        st.toast(f"⚠️ Claude 호출 실패 → Gemini로 전환합니다. ({e})", icon="🔄")
        return None


def _get_gemini_response(messages: list, temperature: float) -> str | None:
    """Google Gemini 호출 (3순위)"""
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Gemini 형식으로 변환
        gemini_history = []
        system_text = ""
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            elif m["role"] == "user":
                gemini_history.append({"role": "user", "parts": [m["content"]]})
            elif m["role"] == "assistant":
                gemini_history.append({"role": "model", "parts": [m["content"]]})

        # 시스템 프롬프트를 첫 번째 user 메시지에 합침
        if system_text and gemini_history:
            first = gemini_history[0]
            first["parts"][0] = f"[시스템 지침]\n{system_text}\n\n[사용자 질문]\n{first['parts'][0]}"

        chat = model.start_chat(history=gemini_history[:-1])
        last_msg = gemini_history[-1]["parts"][0] if gemini_history else ""

        response = chat.send_message(
            last_msg,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=2048,
            ),
        )
        return response.text
    except Exception as e:
        st.toast(f"⚠️ Gemini 호출도 실패했습니다. ({e})", icon="❌")
        return None


def get_llm_response(messages: list, temperature: float = 0.3) -> tuple[str, str]:
    """
    OpenAI → Claude → Gemini 순서로 시도하여 응답을 반환합니다.

    Returns:
        (응답 텍스트, 사용된 모델 이름)
    """
    providers = [
        ("GPT-4o-mini", _get_openai_response),
        ("Claude Sonnet", _get_claude_response),
        ("Gemini Flash", _get_gemini_response),
    ]

    for name, func in providers:
        result = func(messages, temperature)
        if result:
            return result, name

    return (
        "죄송합니다. 현재 모든 AI 서비스에 연결할 수 없습니다. "
        "API 키 설정을 확인하거나 잠시 후 다시 시도해 주세요.",
        "없음",
    )
