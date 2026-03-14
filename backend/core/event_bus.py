import asyncio
import logging
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)


class EventBus:
    """
    경량 비동기 이벤트 버스.
    파이프라인 단계별 결과를 WebSocket이나 로거에 전달할 때 사용합니다.
    """

    def __init__(self):
        self.subscribers: dict[str, list[Callable]] = {}

    def subscribe(self, event_type: str, handler: Callable) -> None:
        """핸들러 등록. 동기/비동기 함수 모두 지원합니다."""
        self.subscribers.setdefault(event_type, []).append(handler)

    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        handlers = self.subscribers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    async def publish(self, event_type: str, data: dict) -> None:
        """등록된 모든 핸들러를 호출합니다."""
        handlers = self.subscribers.get(event_type, [])
        for handler in handlers:
            try:
                result = handler(data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"[EventBus] '{event_type}' 핸들러 오류: {e}")

    def subscribe_all(self, handler: Callable) -> None:
        """모든 이벤트를 수신하는 와일드카드 핸들러."""
        self.subscribe("*", handler)

    async def publish(self, event_type: str, data: dict) -> None:
        all_handlers = (
            self.subscribers.get(event_type, []) +
            self.subscribers.get("*", [])
        )
        for handler in all_handlers:
            try:
                result = handler(data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"[EventBus] '{event_type}' 핸들러 오류: {e}")