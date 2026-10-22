# Classifier benchmark (#14)

`compare_classifiers.py` runs both classifiers over six person crops cut by
YOLO11n-pose from Ultralytics' `bus.jpg` and `zidane.jpg` samples
(`bus_0..3`, `zidane_0..1`) for four prompts, with hand-labelled ground truth.

```sh
env -u LD_LIBRARY_PATH PYTHONPATH=. .venv/bin/python benchmarks/compare_classifiers.py \
    --crops DIR --vlm-model gemma3:4b --device cpu
```

## Results (24 image/prompt pairs)

VLM = `gemma3:4b` via ollama (ollama's Moondream was ruled out: it returns empty replies to every yes/no question), on the GPU;
SigLIP = `google/siglip2-base-patch16-224`, threshold 0.5, on CPU.

| Classifier   | Correct | Time/image | Misses |
|--------------|---------|------------|--------|
| VLM gemma3:4b | 23/24  | ~27.7 s (incl. GPU contention) | 1 false positive (black jacket on a partial crop) |
| SigLIP2 base  | 19/24  | ~1.7 s (CPU) | 5 false negatives |

| Prompt | SigLIP score on true matches | Highest score on non-matches |
|--------|------------------------------|------------------------------|
| wearing sunglasses | 0.00, 0.40 | 0.05 |
| black jacket | 0.19, 0.31, 0.70, 0.73 | 0.02 |
| red shirt | 0.05 | 0.02 |
| suit | 0.91, 0.87 | 0.10 |

Takeaways: SigLIP is ~15x faster even on CPU and its scores usually rank the
matches above non-matches, but the softmax-over-negatives score is poorly
calibrated for small/occluded attributes (sunglasses, a red shirt under a
coat), so the default 0.5 threshold misses; a lower `--threshold` (~0.15-0.3)
would fix most misses on this tiny set but is not tuned. The VLM is more
accurate on subtle attributes but far slower. Tiny sample: treat as indicative.
