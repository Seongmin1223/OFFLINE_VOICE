from config import config


class ConversationManager:
    """
    대화 히스토리 관리 및 llama.cpp용 프롬프트 생성.
    메모리 절약을 위해 최대 히스토리 길이를 제한합니다.
    """

    def __init__(self):
        self.history:      list[dict] = []
        self.max_history:  int        = config.CONVERSATION_MAX_HISTORY
        self.system_prompt: str       = config.LLM_SYSTEM_PROMPT

    def add_user(self, text: str) -> None:
        self.history.append({"role": "user", "content": text.strip()})
        self._trim()

    def add_ai(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text.strip()})
        self._trim()

    def _trim(self) -> None:
        """오래된 대화를 제거해 컨텍스트 윈도우 초과를 방지합니다."""
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-(self.max_history * 2):]

    def reset(self) -> None:
        """대화 초기화."""
        self.history.clear()

    def get_prompt(self) -> str:
        """
        TinyLlama ChatML 형식의 프롬프트를 생성합니다.
        llama_engine 내부에서도 래핑하지만 manager가 히스토리 전체를 관리합니다.
        """
        lines = [f"<|system|>\n{self.system_prompt}</s>"]

        for msg in self.history:
            if msg["role"] == "user":
                lines.append(f"<|user|>\n{msg['content']}</s>")
            else:
                lines.append(f"<|assistant|>\n{msg['content']}</s>")

        lines.append("<|assistant|>")
        return "\n".join(lines)

    def get_plain_prompt(self) -> str:
        """단순 텍스트 형식의 프롬프트 (디버그 용도)."""
        lines = [f"System: {self.system_prompt}"]
        for msg in self.history:
            role = "User" if msg["role"] == "user" else "AI"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)


    def last_user_message(self) -> str:
        for msg in reversed(self.history):
            if msg["role"] == "user":
                return msg["content"]
        return ""

    def last_ai_message(self) -> str:
        for msg in reversed(self.history):
            if msg["role"] == "assistant":
                return msg["content"]
        return ""

    def turn_count(self) -> int:
        return sum(1 for m in self.history if m["role"] == "user")