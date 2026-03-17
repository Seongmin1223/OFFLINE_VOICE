import asyncio
import requests
from config import config
from domains.llm.models import LLMRequest, LLMResponse


class LlamaEngine:

    def __init__(self):
        self.server_url    = config.LLM_SERVER_URL
        self.max_tokens    = config.LLM_MAX_TOKENS
        self.temperature   = config.LLM_TEMPERATURE
        self.system_prompt = config.LLM_SYSTEM_PROMPT

    def _check_server(self) -> None:
        try:
            r = requests.get(f"{self.server_url}/health", timeout=3)
            if r.status_code != 200:
                raise RuntimeError()
        except Exception:
            raise RuntimeError(
                "llama-server가 실행되지 않았습니다!\n"
                f"{config.LLAMA_BIN} "
                f"-m {config.LLAMA_MODEL} "
                f"-c {config.LLM_CONTEXT_SIZE} -t {config.LLM_THREADS} --port {config.LLM_SERVER_URL.split(':')[-1]}"
            )

    def generate_sync(self, request: LLMRequest) -> LLMResponse:
        self._check_server()
        payload = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": request.prompt},
            ],
            "max_tokens":  self.max_tokens,
            "temperature": self.temperature,
            "stream":      False,
        }
        print("[LLM] 응답 생성 중...")
        response = requests.post(
            f"{self.server_url}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        if response.status_code != 200:
            raise RuntimeError(f"llama-server 오류: {response.text}")
        data = response.json()
        text = data["choices"][0]["message"]["content"].strip()
        print(f"[LLM] 응답: {text[:80]}{'...' if len(text) > 80 else ''}")
        return LLMResponse(text=text)

    def stream_sync(self, prompt: str, callback):
        """
        스트리밍으로 토큰을 받아 문장 단위로 callback 호출.
        callback(sentence: str) 형태로 호출됩니다.
        """
        self._check_server()
        payload = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": prompt},
            ],
            "max_tokens":  self.max_tokens,
            "temperature": self.temperature,
            "stream":      True,
        }

        buffer = ""
        sentence_endings = (".", "!", "?", "~", "。", "！", "？", "\n")

        with requests.post(
            f"{self.server_url}/v1/chat/completions",
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
                    import json
                    data  = json.loads(line)
                    delta = data["choices"][0]["delta"].get("content", "")
                    if not delta:
                        continue
                    buffer += delta
                    print(delta, end="", flush=True)

                    # 문장 끝나면 바로 TTS 콜백
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

        # 남은 버퍼 처리
        if buffer.strip():
            callback(buffer.strip())
        print()

    async def generate(self, prompt: str) -> str:
        loop    = asyncio.get_event_loop()
        request = LLMRequest(prompt=prompt)
        result  = await loop.run_in_executor(None, self.generate_sync, request)
        return result.text

    async def stream(self, prompt: str, callback):
        """비동기 스트리밍."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.stream_sync, prompt, callback)