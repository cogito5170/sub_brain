"""claude -p 호출 + 토큰 장부. 호출마다 usage 를 그대로 남긴다(추정하지 않는다)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time


class ClaudeP:
    def __init__(self, model: str, timeout: int = 240, retries: int = 2, thinking: int | None = None):
        """thinking=0 이면 생각(extended thinking)을 끈다 -- '추론이 약한 모델' 조건."""
        self.exe = shutil.which("claude")
        self.env = dict(os.environ, MAX_THINKING_TOKENS=str(thinking)) if thinking is not None else None
        self.model, self.timeout, self.retries = model, timeout, retries
        self.calls: list[dict] = []

    def __call__(self, system: str, user: str) -> str:
        cmd = [self.exe, "-p", "--model", self.model, "--system-prompt", system, "--tools", "",
               "--output-format", "json", "--exclude-dynamic-system-prompt-sections",
               "--setting-sources", "project", "--disable-slash-commands"]
        err = ""
        for attempt in range(self.retries + 1):
            t0 = time.time()
            try:
                r = subprocess.run(cmd, input=user, capture_output=True, text=True,
                                   timeout=self.timeout, cwd="/tmp", env=self.env)
                d = json.loads(r.stdout)
                if d.get("is_error"):
                    raise RuntimeError(str(d.get("result"))[:300])
                u = d.get("usage", {})
                self.calls.append({
                    "in": u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                          + u.get("cache_creation_input_tokens", 0),
                    "in_uncached": u.get("input_tokens", 0),
                    "out": u.get("output_tokens", 0),
                    "thinking": (u.get("output_tokens_details") or {}).get("thinking_tokens", 0),
                    "cost": d.get("total_cost_usd", 0.0), "sec": round(time.time() - t0, 1),
                    "chars_in": len(system) + len(user), "chars_out": len(d.get("result", "")),
                    "user": user[-3000:], "reply": d.get("result", "")[:4000]})
                return d.get("result", "")
            except Exception as e:                      # 실패한 시도는 장부에 안 넣는다 -- 대신 세어 둔다
                err = f"{type(e).__name__}: {e}"
                time.sleep(3 * (attempt + 1))
        self.calls.append({"in": 0, "in_uncached": 0, "out": 0, "thinking": 0, "cost": 0.0, "sec": 0,
                           "chars_in": 0, "chars_out": 0, "error": err[:300]})
        return ""

    def totals(self) -> dict:
        keys = ("in", "in_uncached", "out", "thinking", "cost", "sec", "chars_in", "chars_out")
        t = {k: round(sum(c[k] for c in self.calls), 6) for k in keys}
        t["calls"] = len(self.calls)
        t["errors"] = sum(1 for c in self.calls if c.get("error"))
        return t
