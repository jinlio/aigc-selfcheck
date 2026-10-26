"""运行档位：轻量 / 均衡 / 上限。

同一个工具在不同机器上要有可解释的取舍：
- lite：纯 CPU、0.5B、跳过 a 路，保证能跑起来；
- balanced：b+c、2B、CPU/GPU 自动，默认推荐；
- max：a+b+c、2B，配置足够时的完整形态。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    label: str
    description: str
    detectors: str
    device: str
    c_model: str | None
    llm_advice: bool = False


PROFILES: dict[str, Profile] = {
    "lite": Profile(
        name="lite",
        label="轻量",
        description="纯 CPU、0.5B 参考模型、跳过 a 路；适合低配笔记本，精度略低。",
        detectors="b,c",
        device="cpu",
        c_model="Qwen/Qwen2.5-0.5B",
        llm_advice=False,
    ),
    "balanced": Profile(
        name="balanced",
        label="均衡",
        description="b+c、2B 参考模型、设备自动；默认推荐，兼顾速度与精度。",
        detectors="b,c",
        device="auto",
        c_model=None,
        llm_advice=True,
    ),
    "max": Profile(
        name="max",
        label="上限",
        description="a+b+c、2B 参考模型；配置足够时使用完整三路，可再叠加 --enhanced。",
        detectors="a,b,c",
        device="auto",
        c_model=None,
        llm_advice=True,
    ),
}


def resolve_profile(
    name: str | None,
    detectors: str | None = None,
    device: str | None = None,
    c_model: str | None = None,
    llm_advice: bool | None = None,
) -> dict:
    """解析档位；显式参数优先于档位默认值。未指定档位时保持旧默认 a,b,c。"""
    if name:
        profile = PROFILES[name]
        resolved = asdict(profile)
        resolved["detectors"] = detectors or profile.detectors
        resolved["device"] = device or profile.device
        resolved["c_model"] = c_model or profile.c_model
        resolved["llm_advice"] = profile.llm_advice if llm_advice is None else llm_advice
        return resolved
    return {
        "name": "custom",
        "label": "自定义",
        "description": "未指定档位；使用显式参数或旧默认 a,b,c。",
        "detectors": detectors or "a,b,c",
        "device": device or "auto",
        "c_model": c_model,
        "llm_advice": bool(llm_advice),
    }
