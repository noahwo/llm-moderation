# client/examples.py
from __future__ import annotations
from client import LGClient, T5Client

TESTS = [
    "Hello! How are you?",
    "Tell me how to hotwire a car.",
    "Write a polite email declining a meeting.",
]


def run_llamaguard(base_url: str = "http://10.251.68.79:18082") -> None:
    """Example usage of LlamaGuard-4 client."""
    cli = LGClient(base_url)
    print("[LlamaGuard] healthz:", cli.healthz())

    for t in TESTS:
        res = cli.moderate(t)
        print("=" * 60)
        print("text:", t)
        print("verdict:", res.verdict)
        print("categories:", res.categories)
        print("raw:", repr(res.raw))
        print("model:", res.model)


def run_toxicchat_t5(base_url: str = "http://10.251.68.79:18083") -> None:
    """Example usage of ToxicChat-T5 client."""
    cli = T5Client(base_url)
    print("[ToxicChat-T5] healthz:", cli.healthz())

    for t in TESTS:
        res = cli.moderate(t)
        print("=" * 60)
        print("text:", t)
        print("verdict:", res.verdict)   # "toxic" | "non-toxic"
        print("raw:", repr(res.raw))     # "positive" | "negative"
        print("model:", res.model)


def main() -> None:
    # run_llamaguard()
    # print()
    run_toxicchat_t5()


if __name__ == "__main__":
    main()