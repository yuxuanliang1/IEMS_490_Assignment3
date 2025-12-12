import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from openai import OpenAI


JUDGE_SYSTEM = '''You are an impartial judge for comparing two assistant responses to the same user prompt.
Focus on helpfulness, correctness, completeness, clarity, and following the user\'s instructions.
Do NOT favor verbosity by default. If both are bad, choose the less-bad one. If they are equivalent, choose tie.
Return a JSON object with keys: winner (A|B|tie) and rationale (1-3 sentences).'''

JUDGE_USER_TEMPLATE = '''User prompt:
{prompt}

Response A:
{a}

Response B:
{b}

Decide which response is better overall.
Return JSON only.'''


@dataclass
class JudgeConfig:
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    temperature: float = 0.0
    max_retries: int = 6
    sleep_base: float = 1.5


class DeepSeekJudge:
    def __init__(self, cfg: JudgeConfig, cache_path: Optional[Path] = None):
        api_key = os.environ.get(cfg.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing env var {cfg.api_key_env}. Please set it before running.")
        self.client = OpenAI(api_key=api_key, base_url=cfg.base_url)
        self.cfg = cfg
        self.cache_path = cache_path
        self.cache: Dict[str, Dict] = {}
        if cache_path and cache_path.exists():
            self.cache = json.loads(cache_path.read_text(encoding="utf-8"))

    def _key(self, prompt: str, a: str, b: str) -> str:
        return str(abs(hash((prompt, a, b))))

    def judge(self, prompt: str, a: str, b: str) -> Dict:
        k = self._key(prompt, a, b)
        if k in self.cache:
            return self.cache[k]

        msg = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER_TEMPLATE.format(prompt=prompt, a=a, b=b)},
        ]

        last_err = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.cfg.model,
                    messages=msg,
                    temperature=self.cfg.temperature,
                    stream=False,
                )
                content = resp.choices[0].message.content
                out = json.loads(content)
                w = str(out.get("winner", "tie")).strip().lower()
                if w in ("a", "b"):
                    out["winner"] = w.upper()
                else:
                    out["winner"] = "tie"
                self.cache[k] = out
                if self.cache_path:
                    self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                    self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")
                return out
            except Exception as e:
                last_err = e
                time.sleep(self.cfg.sleep_base * (2 ** attempt))

        raise RuntimeError(f"Judge failed after retries: {last_err}")
