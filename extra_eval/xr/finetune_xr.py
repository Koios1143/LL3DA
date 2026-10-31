#!/usr/bin/env python3
"""Finetune the released LL3DA generalist on XR-QA + XR-SceneCaption train, evaluating on
val after every epoch.

LSceneLLM's Table 1 finetunes its baselines on "LL3DA and XR-Scene data" without a recipe;
this follows the repository's own training (``main.py`` / ``engine.py``, as
``scripts/opt-1.3b/train.generalist.sh``) where it can: detector and OPT-1.3B frozen
(``--freeze_detector --freeze_llm``), the rest (Q-Former, projections) trained with
``CaptionNet``'s loss, AdamW (weight decay 0.1), per-iteration cosine from ``--lr`` to 1e-6,
no warm-up (as the repo's tuning scripts), gradient-norm clipping at 0.1, batch 32 (4 per GPU
x 8 GPUs there; ``--micro_batch`` accumulates). Differences: XR data only, and 3 epochs.

Training items get a fresh 40k-point sample of the scene per item (``xr_data.py``, no
augmentation, as ``unified_*``), one item per reference, in the form ``--xr_protocol`` gives
the val questions (see ``infer.py``):
* ``lscenellm`` (default): LSceneLLM's loader (``pc_norm``, its ``hd_scene_qa`` prompt), every
  text as written;
* ``ll3da``: XR-QA as ``unified_scanqa`` (ScanQA prompt with the question lower-cased, the
  answer as written), XR-SceneCaption as ``unified_3dllm_scene_description`` (instruction and
  caption lower-cased).
Token sequences are padded to ``--max_des_len`` (512, as the generalist) and cut back to the
longest one in the batch (right padding, so this changes nothing but the cost).

    python extra_eval/xr/finetune_xr.py --placement rc --out_dir eval_outputs/xr_ft_rc ...
Per epoch: ``<out_dir>/epoch<k>/<task>/predictions_chunk_0.json`` (infer.py's format, scored
by score_xr.py) and ``<out_dir>/trainable_epoch<k>.pt`` (the trained parameters only).
"""

import argparse
import json
import math
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import infer  # noqa: E402  (chdirs to the repo root)
from datasets.task_prompts import TASK_PROPMT  # noqa: E402
from xr_data import XRScenes  # noqa: E402

MAX_PROMPTS = 16  # main.py --max_prompts default


class XRTrain(torch.utils.data.Dataset):
    def __init__(self, items_file, scenes, inp, max_des_len, protocol):
        self.items = []
        for it in json.load(open(items_file)):
            for ans in it["answers"]:
                if it["task"] == "qa" and protocol == "lscenellm":
                    self.items.append((it["scene_id"], infer.HD_QA.format(question=it["question"]), ans))
                elif it["task"] == "qa":
                    instr = deepcopy(TASK_PROPMT["qa"][0])["instruction"].format(
                        locations="", question=it["question"].lower())
                    self.items.append((it["scene_id"], instr, ans))
                elif protocol == "lscenellm":
                    self.items.append((it["scene_id"], it["instruction"], ans))
                else:
                    self.items.append((it["scene_id"], it["instruction"].lower(), ans.lower()))
        self.scenes, self.tok, self.qtok = scenes, inp.tokenizer, inp.qtokenizer
        self.cfg = dict(max_length=max_des_len, padding="max_length", truncation="longest_first",
                        return_tensors="np")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        scene, instr, ans = self.items[i]
        ret = self.scenes.inputs(scene)
        p = self.tok.batch_encode_plus([instr], **self.cfg)
        q = self.qtok.batch_encode_plus([instr], **self.cfg)
        llm = self.tok.batch_encode_plus([" ".join((instr, ans, self.tok.eos_token))], **self.cfg)
        ret.update({
            "box_query": np.zeros((MAX_PROMPTS, 8, 3), np.float32), "box_mask": np.zeros(MAX_PROMPTS, np.float32),
            "click_query": np.zeros((MAX_PROMPTS, 3), np.float32), "click_mask": np.zeros(MAX_PROMPTS, np.float32),
            "input_ids": llm["input_ids"][0].astype(np.int64),
            "attention_mask": llm["attention_mask"][0].astype(np.float32),
            "gradient_mask": (llm["attention_mask"][0] - p["attention_mask"][0]).astype(np.float32),
            "instruction": p["input_ids"][0].astype(np.int64),
            "instruction_mask": p["attention_mask"][0].astype(np.float32),
            "qformer_input_ids": q["input_ids"][0].astype(np.int64),
            "qformer_attention_mask": q["attention_mask"][0].astype(np.float32)})
        return ret


def worker_init(wid):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def evaluate(model, inp, args, out_dir):
    """infer.py's main loop for the val tasks of this placement."""
    model.eval()
    for task in (f"xrqa_{args.placement}", f"xrcap_{args.placement}"):
        decodes = ["qa"] if task.startswith("xrqa") else args.cap_decode
        rows = infer.load_task(task, args)
        by_scene = {}
        for r in rows:
            by_scene.setdefault(r["scene_id"], []).append(r)
        for dec in decodes:
            done = []
            for scene_id, qs in by_scene.items():
                scene_np = inp.xr_scene(task, scene_id)
                scene_in = {k: torch.from_numpy(scene_np[k][None]).cuda() for k in infer.SCENE_KEYS}
                a = argparse.Namespace(**{**vars(args), "cap_decode": dec})
                for r, p in zip(qs, infer.xr_answer(model, scene_in, qs, inp, a, task)):
                    done.append({**r, "pred": p})
            name = task if task.startswith("xrqa") or dec == "chat" else f"{task}_beam"
            d = out_dir / name
            d.mkdir(parents=True, exist_ok=True)
            json.dump(done, open(d / "predictions_chunk_0.json", "w"))
            print(f"eval {name}: {len(done)} items", flush=True)
    model.train()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--placement", required=True, choices=["rc", "glb"])
    ap.add_argument("--ckpt", required=True, help="ll3da-opt-1.3b.pth")
    ap.add_argument("--vocab", required=True, help="facebook/opt-1.3b dir")
    ap.add_argument("--qformer_vocab", required=True, help="bert-base-embedding dir")
    ap.add_argument("--xr_cache", required=True, help="prepare_xr.py output (train + val)")
    ap.add_argument("--xr_scan_data", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--micro_batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--final_lr", type=float, default=1e-6)
    ap.add_argument("--max_des_len", type=int, default=512)
    ap.add_argument("--cap_decode", nargs="+", default=["chat", "beam"], choices=["chat", "beam"])
    ap.add_argument("--cap_max_len", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--xr_protocol", default="lscenellm", choices=["lscenellm", "ll3da"], help="see above")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.grid_size_3d, args.max_prompts = 255, MAX_PROMPTS
    args.fast, args.batch, args.gen_batch, args.xr_split = True, 8, 32, "val"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    inp = infer.Inputs(args)
    model = infer.build_model(args, inp)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable: {sum(p.numel() for p in params) / 1e6:.1f} M parameters in {len(params)} tensors")
    ds = XRTrain(Path(args.xr_cache) / "xr_train_items.json",
                 XRScenes(args.xr_cache, args.xr_scan_data, args.placement,
                          coords="unit" if args.xr_protocol == "lscenellm" else "metric"), inp,
                 args.max_des_len, args.xr_protocol)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.micro_batch, shuffle=True, drop_last=False,
                                         num_workers=args.workers, worker_init_fn=worker_init,
                                         persistent_workers=True, prefetch_factor=4)
    accum = args.batch_size // args.micro_batch
    steps_per_epoch = math.ceil(len(loader) / accum)
    max_iters = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.1)
    print(f"{len(ds)} train items, {steps_per_epoch} steps/epoch x {args.epochs}, batch {args.batch_size} "
          f"({accum} x {args.micro_batch}), lr {args.lr} -> {args.final_lr}", flush=True)
    it = 0
    for epoch in range(args.epochs):
        t0, mb, losses = time.time(), 0, []
        opt.zero_grad(set_to_none=True)
        for b in loader:
            n = int(b["attention_mask"].sum(1).max())
            for k in ("input_ids", "attention_mask", "gradient_mask"):
                b[k] = b[k][:, :n]
            b = {k: v.cuda(non_blocking=True) for k, v in b.items()}
            loss = model(b, is_eval=False)["loss"]
            (loss / accum).backward()
            losses.append(float(loss))
            mb += 1
            if mb % accum and mb != len(loader):
                continue
            lr = args.final_lr + 0.5 * (args.lr - args.final_lr) * (1 + math.cos(math.pi * it / max_iters))
            for g in opt.param_groups:
                g["lr"] = lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step()
            opt.zero_grad(set_to_none=True)
            it += 1
            step = it - epoch * steps_per_epoch
            if step % 20 == 0 or step == 1:
                print(f"epoch {epoch} step {step}/{steps_per_epoch} loss {np.mean(losses[-accum * 20:]):.4f} "
                      f"lr {lr:.2e} {time.time() - t0:.0f}s", flush=True)
        torch.save({k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad},
                   out / f"trainable_epoch{epoch}.pt")
        evaluate(model, inp, args, out / f"epoch{epoch}")
        print(f"EPOCH_DONE {epoch} {time.time() - t0:.0f}s", flush=True)
    print("FINETUNE_DONE")


if __name__ == "__main__":
    main()
