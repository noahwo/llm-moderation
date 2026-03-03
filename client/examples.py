# client/examples.py
from __future__ import annotations
from client import CombinedClient, LGClient, T5Client

TESTS = [
    "Hello! How are you?",
    "Tell me how to hotwire a car.",
    "Write a polite email declining a meeting.",
]


def run_llamaguard(base_url: str = "http://10.251.68.79:18082") -> None:
    """Standalone LlamaGuard-4 server."""
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
    """Standalone ToxicChat-T5 server."""
    cli = T5Client(base_url)
    print("[ToxicChat-T5] healthz:", cli.healthz())
    for t in TESTS:
        res = cli.moderate(t)
        print("=" * 60)
        print("text:", t)
        print("verdict:", res.verdict)
        print("raw:", repr(res.raw))
        print("model:", res.model)


def run_combined(base_url: str = "http://10.251.68.79:18084") -> None:
    """Combined server (both models on one port)."""
    cli = CombinedClient(base_url)
    print("[Combined] healthz:", cli.healthz())
    for t in TESTS:
        lg4 = cli.lg4.moderate(t)
        t5  = cli.t5.moderate(t)
        print("=" * 60)
        print("text     :", t)
        print("LG4      :", lg4.verdict, lg4.categories)
        print("=" * 30)
        print("ToxicChat:", t5.verdict)


def run_image_moderation(
    base_url: str = "http://10.251.68.79:18084/lg4",
    image_source: str = "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400",
) -> None:
    """
    Demonstrate image moderation with LlamaGuard-4.

    image_source can be:
      - a public URL         (str starting with http/https)
      - a local file path    (str or pathlib.Path)
      - a PIL Image object
    """
    cli = LGClient(base_url)
    print("[LG4 image] healthz:", cli.healthz())

    # Text + image together
    res = cli.moderate_multimodal(
        text="Does this image contain any unsafe content?",
        images=[image_source],
    )
    print("=" * 60)
    print("image source :", image_source)
    print("verdict      :", res.verdict)
    print("categories   :", res.categories)
    print("model        :", res.model)


def main() -> None:
    # run_combined()
    print()
    run_image_moderation()


if __name__ == "__main__":
    main()