import asyncio
import requests
import json
from config import config
from domains.llm.models import LLMResponse

class LlamaEngine:
    def __init__(self):
        self.server_url     = config.LLM_SERVER_URL
        self.max_tokens     = config.LLM_MAX_TOKENS
        self.temperature    = config.LLM_TEMPERATURE
        self.top_p          = config.LLM_TOP_P
        self.top_k          = config.LLM_TOP_K
        self.repeat_penalty = config.LLM_REPEAT_PENALTY

    def _check_server(self) -> None:
        try:
            r = requests.get(f"{self.server_url}/health", timeout=3)
            if r.status_code != 200:
                raise RuntimeError()
        except Exception:
            raise RuntimeError("llama-server가 실행되지 않았습니다!")

    def _format_prompt(self, messages: list[dict]) -> str:
        """EXAONE 3.5 공식 템플릿 형식으로 프롬프트를 강제 조립"""
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"[|system|]\n{content}\n"
            elif role == "user":
                prompt += f"[|user|]\n{content}\n"
            elif role == "assistant":
                prompt += f"[|assistant|]\n{content}\n"
        
        # 마지막에 AI가 대답할 차례임을 알리는 태그 추가
        prompt += "[|assistant|]\n"
        return prompt

    def generate_sync(self, messages: list[dict]) -> LLMResponse:
        self._check_server()
        
        # 1. messages 배열 대신 문자열 프롬프트 직접 조립
        raw_prompt = self._format_prompt(messages)
        
        payload = {
            "prompt":         raw_prompt,
            "max_tokens":     self.max_tokens,
            "temperature":    self.temperature,
            "top_p":          self.top_p,
            "top_k":          self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "stream":         False,
            # 모델이 혼자 질문하고 대답하는 환각(Hallucination) 방지용 제동 장치
            "stop":           ["[|user|]", "[|system|]"],
        }
        
        print("[LLM] 응답 생성 중...")
        response = requests.post(
            f"{self.server_url}/v1/completions", # /chat/completions -> /completions 로 변경
            json=payload,
            timeout=120,
        )
        if response.status_code != 200:
            raise RuntimeError(f"llama-server 오류: {response.text}")
        data = response.json()
        text = data["choices"][0]["text"].strip() # "message"]["content"] -> "text" 로 변경
        print(f"[LLM] 응답: {text[:80]}{'...' if len(text) > 80 else ''}")
        return LLMResponse(text=text)

    def stream_sync(self, messages: list[dict], callback):
        self._check_server()
        
        # 1. messages 배열 대신 문자열 프롬프트 직접 조립
        raw_prompt = self._format_prompt(messages)
        
        payload = {
            "prompt":         raw_prompt,
            "max_tokens":     self.max_tokens,
            "temperature":    self.temperature,
            "top_p":          self.top_p,
            "top_k":          self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "stream":         True,
            "stop":           ["[|user|]", "[|system|]"],
        }

        buffer = ""
        sentence_endings = (".", "!", "?", "~", "。", "！", "？", "\n")

        with requests.post(
            f"{self.server_url}/v1/completions", # 엔드포인트 변경
            json=payload,
            stream=True,
            timeout=120,
        ) as resp:
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    line = line[6:]
                if line == "[DONE]":
                    break
                try:
                    data  = json.loads(line)
                    # "choices"][0]["delta"]["content"] -> "choices"][0]["text"] 로 구조 변경
                    delta = data["choices"][0].get("text", "") 
                    if not delta:
                        continue
                    buffer += delta
                    print(delta, end="", flush=True)

                    for ending in sentence_endings:
                        if ending in buffer:
                            parts = buffer.split(ending)
                            for part in parts[:-1]:
                                sentence = part.strip()
                                if sentence:
                                    callback(sentence + ending)
                            buffer = parts[-1]
                except Exception:
                    continue

        if buffer.strip():
            callback(buffer.strip())
        print()

    async def generate(self, messages: list[dict]) -> str:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.generate_sync, messages)
        return result.text

    async def stream(self, messages: list[dict], callback):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.stream_sync, messages, callback)