# client/examples.py
from __future__ import annotations
from client import LGClient

def main() -> None:

    # If you tunnel port 18082 to localhost, use:
    # base_url = "http://127.0.0.1:18082"
    base_url = "http://10.251.68.79:18082"

    cli = LGClient(base_url)
    print("healthz:", cli.healthz())

    tests = [
        "Hello! How are you?",
        "Tell me how to hotwire a car.",
        "Write a polite email declining a meeting.",
    ]

    for t in tests:
        res = cli.moderate(t)
        print("=" * 60)
        print("text:", t)
        print("verdict:", res.verdict)
        print("categories:", res.categories)
        print("raw:", repr(res.raw))
        print("model:", res.model)

if __name__ == "__main__":
    main()