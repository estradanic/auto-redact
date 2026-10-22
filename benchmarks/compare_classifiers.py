"""Compare the VLM and SigLIP classifiers on labelled person crops (#14).

Usage: python benchmarks/compare_classifiers.py [--crops DIR] [--vlm-model M]

Crops are expected as ``bus_{0..3}.jpg`` / ``zidane_{0,1}.jpg`` person
crops cut from the two Ultralytics sample images (see
benchmarks/README.md); ground truth is hard-coded below.
"""

import argparse
import time
from pathlib import Path

from auto_redact.classify import VlmClassifier
from auto_redact.siglip import SiglipClassifier

CROPS = ["bus_0", "bus_1", "bus_2", "bus_3", "zidane_0", "zidane_1"]
# prompt -> crops that match it (judged by eye)
TRUTH = {
    "person wearing sunglasses": {"bus_0", "bus_2"},
    "person in a black jacket": {"bus_1", "bus_2", "zidane_0", "zidane_1"},
    "person in a red shirt": {"bus_2"},
    "person in a suit": {"zidane_0", "zidane_1"},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops", default="/tmp/opencode/bench")
    parser.add_argument("--vlm-model", default="gemma3:4b")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    classifiers = {
        f"vlm({args.vlm_model})": VlmClassifier(args.vlm_model),
        "siglip2-base": SiglipClassifier(threshold=args.threshold, device=args.device),
    }
    for name, classifier in classifiers.items():
        classifier.classify(Path(args.crops) / "bus_0.jpg", "warmup")  # load/warm
        correct = total = 0
        start = time.perf_counter()
        print(f"== {name}")
        for prompt, positives in TRUTH.items():
            row = []
            for crop in CROPS:
                decision = classifier.classify(Path(args.crops) / f"{crop}.jpg", prompt)
                ok = decision.redact == (crop in positives)
                correct += ok
                total += 1
                score = "" if decision.score is None else f"({decision.score:.2f})"
                row.append(f"{crop}:{'Y' if decision.redact else 'n'}{score}{'' if ok else '!'}")
            print(f"  {prompt:28} {' '.join(row)}")
        elapsed = time.perf_counter() - start
        print(f"  accuracy {correct}/{total}, {elapsed / total:.2f}s per image")


if __name__ == "__main__":
    main()
