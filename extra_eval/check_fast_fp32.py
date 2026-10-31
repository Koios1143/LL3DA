#!/usr/bin/env python3
"""Logic check of ``infer.py --fast``: with the LLM in float32 (rounding no longer flips beam
near-ties), the fast path must reproduce the repo's own generation exactly. Runs ScanQA and
dense captioning on the first ``--scenes`` ScanQA val scenes with both paths (``--fast`` with one
and with 32 sequences per beam search) and prints the share of identical outputs.
Takes the same path arguments as ``infer.py``.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import infer as I  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for a in ["--ckpt", "--vocab", "--qformer_vocab", "--scannet_data", "--scanqa_dir"]:
        ap.add_argument(a, required=True)
    ap.add_argument("--scenes", type=int, default=3)
    args = ap.parse_args()
    args.batch, args.max_des_len, args.grid_size_3d, args.max_prompts = 8, 256, 255, 16
    inp = I.Inputs(args)
    model = I.build_model(args, inp)
    model.captioner.transformer.float()
    model.captioner.dtype = torch.float32
    by = {}
    for r in I.load_task("scanqa", args):
        by.setdefault(r["scene_id"], []).append(r)
    qa = {"up": [], "fast_b1": [], "fast_b32": []}
    caps = {"up": [], "fast": []}
    for scene in sorted(by)[:args.scenes]:
        sn = inp.scene(args.scannet_data, scene)
        si = {k: torch.from_numpy(sn[k][None]).cuda() for k in I.SCENE_KEYS}
        qa["up"] += I.upstream_qa(model, si, by[scene], inp, args)
        for gb in (1, 32):
            args.gen_batch = gb
            qa[f"fast_b{gb}"] += I.fast_qa(model, si, by[scene], inp, args)
        for k, f in [("up", I.upstream_densecap), ("fast", I.fast_densecap)]:
            out = f(model, si, inp, args)
            caps[k] += [p["caption"] for p in I.densecap_records(out, sn, si, inp.tokenizer, scene)["proposals"]]
    n = len(qa["up"])
    for k in ("fast_b1", "fast_b32"):
        print(f"QA {k}: {sum(a == b for a, b in zip(qa['up'], qa[k]))}/{n} identical")
    print(f"captions: {sum(a == b for a, b in zip(caps['up'], caps['fast']))}/{len(caps['up'])} identical")


if __name__ == "__main__":
    main()
