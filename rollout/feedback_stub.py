"""测试专用：按脚本返回预设回复的假 LLM，不联网、不花钱。

放在 rollout/ 下而不是 tests/ 下，是因为 P3 合成轨迹时也可能想用它做 dry-run。
"""

from __future__ import annotations


class StubLLM:
    """``responses`` 里的元素是字符串或异常实例，按调用顺序消费。

    用完预设后重复最后一个字符串（模拟"模型一直给同样答案"）。
    """

    def __init__(self, responses: list):
        self._responses = list(responses)
        self.calls: list[list[dict]] = []
        self.model = "stub"
        self.base_url = "stub://"

    def chat(self, messages: list[dict], **kw) -> str:  # noqa: ANN003
        self.calls.append([dict(m) for m in messages])
        if not self._responses:
            return "```sql\nSELECT 1\n```"
        item = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        if isinstance(item, Exception):
            raise item
        return item
