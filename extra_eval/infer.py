#!/usr/bin/env python3
"""LL3DA (released generalist ``ll3da-opt-1.3b.pth``) inference on Scan2Cap, ScanQA, SQA3D,
Real-3DQA and Beacon3D QA, one process per GPU (``--num_chunks`` / ``--chunk_idx``, whole
scenes per chunk).

Inputs are built as the repo's val datasets build them (``datasets/unified_scanqa.py``,
``datasets/unified_densecap_scanrefer.py``): the scene's ``_aligned_vert.npy`` (xyz, rgb,
normal, + height) randomly subsampled to 40k points, the first prompt of
``TASK_PROPMT['qa']`` / ``['densecap']``, instructions padded to ``--max_des_len`` (256, as
``scripts/opt-1.3b/eval.*.sh``). The one change: the 40k-point subsample is seeded by the scene
name (``crc32``), so every question of a scene sees the same points and both paths below see
identical inputs (upstream draws a new subsample per item from an unseeded worker RNG).

Two paths with identical prompts, inputs and decoding (beam 4; ``qa`` 16 tokens,
``dense-cap`` 48 tokens, as ``captioner.forward``):

* default: the repo's own ``CaptionNet.forward(..., is_eval=True)`` per batch of
  ``--batch`` items (``predict_answer`` / ``predict_densecap``: one beam search per question,
  each step re-running the LLM over the whole sequence).
* ``--fast``: the detector runs once per scene; the Q-Former runs batched over the scene's
  questions (dense captioning: over its proposals); and the LLM decodes ``--gen_batch``
  sequences at a time with ``beam_search_kv``, the repo's ``beam_search_decode`` step for step
  (same scores, beam bookkeeping and heap of finished beams) but feeding only the newest token
  per step against a reordered KV cache, with left padding + attention mask for questions of
  different lengths.

Tasks:
* ``scan2cap``: dense captioning on the ScanRefer val scenes (``ScanRefer_filtered_val``);
  per scene, every proposal kept by objectness + NMS is written with its caption, matched
  ScanNet object and box IoU; ``score.py`` applies the repo's ``evaluate_densecap`` matching
  at IoU 0.25 and 0.5.
* ``scanqa`` (val), ``sqa3d_test``, ``real3dqa_<split>``, ``beacon3d_<domain>_qa``: the
  question (SQA3D-style tasks: ``<situation> <question>``) lower-cased into the ScanQA prompt.
* ``xrqa_<placement>``, ``xrcap_<placement>`` (``rc`` | ``glb``): LSceneLLM's XR-QA /
  XR-SceneCaption val (``extra_eval/xr/``; scenes from ``xr_data.py``, 40k points seeded by the
  scene alias). ``--xr_protocol``:
  - ``lscenellm`` (default): as LSceneLLM runs its baselines ("keeping all other settings the
    same as ours"): its loader's ``pc_norm`` (scene in the unit sphere), its XR-QA prompt
    (``TASK_PROMPT['hd_scene_qa'][0]``), questions and captions' instructions as written;
  - ``ll3da``: as this repository's own data: metric coordinates, the ScanQA prompt with the
    question lower-cased, the caption instruction lower-cased (``unified_3dllm_scene_description``).
  Captions use the dataset's instruction (``### human: Describe the room. ### assistant:``, the
  form of LL3DA's scene-description data), decoded as ``--cap_decode``: ``chat`` is the repo's
  scene-description evaluation (``predict_chat``: top-k / top-p sampling, up to 512 tokens;
  seeded per scene), ``beam`` the QA beam search (beam 4) up to ``--cap_max_len`` tokens.

Writes ``<output_dir>/<task>/predictions_chunk_<i>.json`` (resumes per scene) and
``<output_dir>/stats_chunk_<i>.json``.
"""

import argparse
import heapq
import json
import os
import re
import sys
import time
import zlib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)  # DatasetConfig reads ./data/scannet/meta_data
import datasets.scannet_base_dataset as sbd  # noqa: E402
from datasets.task_prompts import TASK_PROPMT  # noqa: E402
from models.model_general import CaptionNet  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from utils.box_util import box3d_iou_batch_tensor  # noqa: E402
from utils.proposal_parser import parse_predictions  # noqa: E402

REAL3DQA_SPLITS = ["debiased_test", "rotation_0", "rotation_90", "rotation_180", "rotation_270"]
QA_SUFFIX = " Answer the question using a single word or phrase."
SQA_TYPE = {"what": 0, "is": 1, "how": 2, "can": 3, "which": 4}
HD_QA = '### human: based on the 3D scene with multiple rooms, answer the question: "{question}" ### assistant:'
MAX_LEN = {"qa": 16, "dense-cap": 48}  # captioner.forward response_config


# ---------------------------------------------------------------- annotations
def xr_task(task):
    """'xrqa_rc' -> ('qa', 'rc'); None for other tasks."""
    m = re.fullmatch(r"xr(qa|cap)_(rc|glb)", task)
    return m.groups() if m else None


def load_task(task, args):
    """-> list of {qid, scene_id, question, answers, ...} (scan2cap: one row per scene)."""
    if xr_task(task):
        kind, _ = xr_task(task)
        rows = json.load(open(os.path.join(args.xr_cache, f"xr_{args.xr_split}_items.json")))
        return [{"qid": r["idx"], **{k: r[k] for k in ("idx", "scene_id", "question", "answers", "key", "task")},
                 **({"instruction": r["instruction"]} if kind == "cap" else {})}
                for r in rows if r["task"] == kind]
    if task == "scan2cap":
        scenes = open(os.path.join(args.scanrefer_dir, "ScanRefer_filtered_val.txt")).read().splitlines()
        return [{"qid": s, "scene_id": s} for s in scenes]
    if task == "scanqa":
        rows = json.load(open(os.path.join(args.scanqa_dir, "ScanQA_v1.0_val.json")))
        return [{"qid": r["question_id"], "scene_id": r["scene_id"], "question": r["question"],
                 "answers": r["answers"]} for r in rows]
    if task == "sqa3d_test":  # Chat-Scene's sqa3d_test.json: prompt = situation question + suffix
        rows = json.load(open(args.sqa3d_test))
        out = []
        for i, r in enumerate(rows):
            assert r["prompt"].endswith(QA_SUFFIX), r["prompt"]
            out.append({"qid": i, "scene_id": r["scene_id"], "question": r["prompt"][:-len(QA_SUFFIX)],
                        "answers": r["ref_captions"], "sqa_type": r["sqa_type"]})
        return out
    if task.startswith("real3dqa_"):
        out = []
        for line in open(os.path.join(args.real3dqa_dir, task[len("real3dqa_"):] + ".jsonl")):
            x = json.loads(line)
            q = x["question"].strip()
            out.append({"qid": int(x["question_id"]), "scene_id": x["scene_id"],
                        "question": f'{x["situation"].strip()} {q}', "answers": [x["answer"]],
                        "sqa_type": SQA_TYPE.get(q.split(" ")[0].lower(), 5),
                        "question_type": x["question_type"]})
        return out
    if task.startswith("beacon3d_") and task.endswith("_qa"):
        domain = task[len("beacon3d_"):-len("_qa")]
        rows = json.load(open(os.path.join(args.beacon3d_dir, "data", domain, f"{domain}_qa.json")))
        return [{"qid": i, "scene_id": r["scene_id"], "question": r["question"], "answers": r["answers"],
                 "question_id": r["question_id"]} for i, r in enumerate(rows)]
    raise ValueError(task)


def scene_dir(task, args):
    if task.startswith("beacon3d_") and not task.startswith("beacon3d_scannet"):
        return os.path.join(args.beacon3d_scans, task[len("beacon3d_"):-len("_qa")])
    return args.scannet_data


# ---------------------------------------------------------------- inputs
class Inputs:
    """The val datasets' ``__getitem__`` for one scene, without the per-item RNG."""

    def __init__(self, args):
        self.args = args
        self.cfg = sbd.DatasetConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(args.vocab, add_bos_token=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.qtokenizer = AutoTokenizer.from_pretrained(args.qformer_vocab)
        self.qtokenizer.pad_token = self.tokenizer.eos_token
        self.qtokenizer.padding_side = "right"
        self.tok_cfg = dict(max_length=args.max_des_len, padding="max_length", truncation="longest_first",
                            return_tensors="np")
        self.bases = {}
        self.xr = {}

    def base(self, data_dir):
        if data_dir not in self.bases:
            sbd.DATASET_ROOT_DIR = data_dir
            self.bases[data_dir] = sbd.ScanNetBaseDataset(
                self.args, self.cfg, split_set="all", num_points=40000, use_color=True, use_normal=True,
                use_multiview=False, use_height=True, augment=False, use_random_cuboid=False)
        return self.bases[data_dir]

    def scene(self, data_dir, scene_id):
        np.random.seed(zlib.crc32(scene_id.encode()))  # pc_util.random_sampling draws from np.random
        return self.base(data_dir)._get_scan_data(scene_id)

    def xr_scene(self, task, scene_id):
        sys.path.insert(0, str(REPO / "extra_eval" / "xr"))
        from xr_data import XRScenes
        pl = xr_task(task)[1]
        if pl not in self.xr:
            self.xr[pl] = XRScenes(self.args.xr_cache, self.args.xr_scan_data, pl,
                                   coords="unit" if self.args.xr_protocol == "lscenellm" else "metric")
        np.random.seed(zlib.crc32(scene_id.encode()))
        return self.xr[pl].inputs(scene_id)

    def instruction(self, kind, question=None):
        if kind == "raw":  # the instruction as given (XR-SceneCaption), lower-cased
            text = question.lower()
        elif kind == "raw_cased":  # the instruction as given
            text = question
        elif kind == "hd_qa":  # LSceneLLM's own XR-QA prompt (TASK_PROMPT['hd_scene_qa'][0])
            text = HD_QA.format(question=question)
        else:
            prompt = deepcopy(TASK_PROPMT[kind][0])
            text = prompt["instruction"].format(locations="", question=question.lower() if question else "")
        p = self.tokenizer.batch_encode_plus([text], **self.tok_cfg)
        q = self.qtokenizer.batch_encode_plus([text], **self.tok_cfg)
        return {"instruction": p["input_ids"][0].astype(np.int64),
                "instruction_mask": p["attention_mask"][0].astype(np.float32),
                "qformer_input_ids": q["input_ids"][0].astype(np.int64),
                "qformer_attention_mask": q["attention_mask"][0].astype(np.float32)}


def to_batch(dicts, device):
    return {k: torch.from_numpy(np.stack([d[k] for d in dicts])).to(device) for k in dicts[0]}


SCENE_KEYS = ["point_clouds", "point_cloud_dims_min", "point_cloud_dims_max"]


# ---------------------------------------------------------------- KV-cached beam search
@torch.no_grad()
def beam_search_kv(transformer, inputs_embeds, attention_mask, max_length, num_beams, eos_token_id, **_):
    """``generation_utils.beam_search_decode`` with a KV cache (see module docstring)."""
    nbeams = num_beams
    batch, prefix_length, channel = inputs_embeds.shape
    device = inputs_embeds.device
    embedding_layer = transformer.get_input_embeddings()
    output_scores = torch.zeros(batch, nbeams).to(device)
    output_ids = torch.ones(batch, nbeams, max_length).to(device) * eos_token_id
    batch_beam_results = {b: [[float("-inf"), (float("-inf"), float("-inf")), None, None]
                              for _ in range(nbeams)] for b in range(batch)}
    mask = attention_mask.long()
    for word_id in range(max_length):
        if word_id == 0:
            out = transformer(inputs_embeds=inputs_embeds, attention_mask=mask, use_cache=True)
            topk_scores, topk_inds = out.logits[:, -1, :].topk(k=nbeams, largest=True, dim=-1)
            output_ids[..., word_id] = topk_inds
            output_scores += torch.log_softmax(topk_scores, dim=-1)
            # every beam starts from the same prefix
            past = tuple(tuple(t.repeat_interleave(nbeams, dim=0) for t in layer) for layer in out.past_key_values)
            mask = mask.repeat_interleave(nbeams, dim=0)
            continue
        last = output_ids[..., word_id - 1].reshape(batch * nbeams, 1).long()
        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
        out = transformer(inputs_embeds=embedding_layer(last), attention_mask=mask, past_key_values=past,
                          use_cache=True)
        past = out.past_key_values
        last_word_logits = out.logits[:, -1, :].reshape(batch, nbeams, -1)
        if word_id != max_length - 1:
            beam_scores = output_scores.unsqueeze(-1) + torch.log_softmax(last_word_logits, dim=-1)
            output_scores, select_inds = beam_scores.reshape(batch, -1).topk(k=nbeams, largest=True, dim=-1)
            select_beam_id = select_inds // last_word_logits.shape[-1]
            select_word_id = select_inds % last_word_logits.shape[-1]
        else:
            last_word_probs = torch.log_softmax(last_word_logits, dim=-1)
            output_scores += last_word_probs[..., eos_token_id]
            select_beam_id = torch.arange(nbeams).to(device).unsqueeze(0).repeat(batch, 1)
            select_word_id = torch.ones_like(output_ids[..., -1]) * eos_token_id
        output_ids = torch.gather(output_ids, 1, select_beam_id.unsqueeze(-1).repeat(1, 1, max_length))
        output_ids[..., word_id] = select_word_id
        # the cache follows the beams that survived
        flat = (select_beam_id + torch.arange(batch, device=device).unsqueeze(1) * nbeams).reshape(-1)
        past = tuple(tuple(t.index_select(0, flat) for t in layer) for layer in past)
        sentence_log_prob = output_scores / (word_id + 1)
        finished_batch, finished_beams = torch.where(select_word_id == eos_token_id)
        for batch_id, beam_id in zip(finished_batch.cpu().tolist(), finished_beams.cpu().tolist()):
            sentence = [sentence_log_prob[batch_id, beam_id].cpu().tolist(), (word_id, beam_id),
                        output_ids[batch_id, beam_id], sentence_log_prob[batch_id, [beam_id]]]
            heapq.heappushpop(batch_beam_results[batch_id], sentence)
        output_scores[select_word_id == eos_token_id] = -float("inf")
    output_ids = torch.cat([torch.cat([s.unsqueeze(0) for _, _, s, _ in batch_beam_results[b]], 0).unsqueeze(0)
                            for b in range(batch)], 0)
    output_scores = torch.cat([torch.cat([p.unsqueeze(0) for _, _, _, p in batch_beam_results[b]], 0).unsqueeze(0)
                               for b in range(batch)], 0).squeeze(-1)
    return torch.gather(output_ids.long(), 1,
                        output_scores.argmax(-1, keepdim=True).unsqueeze(1).repeat(1, 1, max_length)).squeeze(1)


def expand(t, n):
    return t.expand(n, *t.shape[1:])


def qformer_prefix(captioner, det_out, scene_in, instr, n, box_query=None):
    """``captioner._get_instruction_response`` for n questions/proposals of one scene."""
    det = {k: expand(det_out[k], n) for k in ("enc_features", "sem_cls_logits", "box_corners")}
    det["prop_features"] = [expand(det_out["prop_features"][-1], n)]  # only [-1] is read
    inputs = {k: expand(scene_in[k], n) for k in SCENE_KEYS}
    inputs.update(instr)
    return captioner._get_instruction_response(det, inputs, box_query=box_query)


@torch.no_grad()
def fast_qa(model, scene_in, rows, inp, args, kind="qa", max_len=MAX_LEN["qa"]):
    captioner = model.captioner
    det_out = model.detector(scene_in, is_eval=True)
    emb = captioner.transformer.get_input_embeddings()
    preds = []
    for s in range(0, len(rows), args.gen_batch):
        chunk = rows[s:s + args.gen_batch]
        instr = to_batch([inp.instruction(kind, r["instruction"] if kind.startswith("raw") else r["question"])
                          for r in chunk],
                         scene_in["point_clouds"].device)
        prefix = qformer_prefix(captioner, det_out, scene_in, instr, len(chunk)).to(captioner.dtype)
        # left-pad the instructions: [pad .. pad | 32 query tokens | instruction]
        lens = instr["instruction_mask"].sum(1).long()
        L = int(lens.max())
        ids = torch.full((len(chunk), L), inp.tokenizer.pad_token_id, dtype=torch.long, device=prefix.device)
        m = torch.zeros(len(chunk), L, dtype=torch.long, device=prefix.device)
        for i in range(len(chunk)):
            ids[i, L - lens[i]:] = instr["instruction"][i, :lens[i]]
            m[i, L - lens[i]:] = 1
        tok = emb(ids)
        pad = L - lens  # move the pads in front of the query tokens
        embeds = torch.zeros(len(chunk), prefix.shape[1] + L, prefix.shape[2], dtype=prefix.dtype, device=prefix.device)
        mask = torch.zeros(len(chunk), prefix.shape[1] + L, dtype=torch.long, device=prefix.device)
        for i in range(len(chunk)):
            p = int(pad[i])
            embeds[i, p:p + prefix.shape[1]] = prefix[i]
            embeds[i, p + prefix.shape[1]:] = tok[i, p:]
            mask[i, p:] = 1
        out = beam_search_kv(captioner.transformer, embeds, mask, max_len, **captioner.caption_config)
        preds += decode(inp.tokenizer, out)
    return preds


@torch.no_grad()
def fast_densecap(model, scene_in, inp, args):
    captioner = model.captioner
    det_out = model.detector(scene_in, is_eval=True)
    emb = captioner.transformer.get_input_embeddings()
    instr1 = to_batch([inp.instruction("densecap")], scene_in["point_clouds"].device)
    instr_ids = instr1["instruction"][0][instr1["instruction_mask"][0] == 1]
    sem = det_out["sem_cls_logits"]
    keep = (sem.argmax(-1) != sem.shape[-1] - 1)[0].nonzero().squeeze(1).tolist()
    nprop = det_out["box_corners"].shape[1]
    output_ids = torch.full((nprop, MAX_LEN["dense-cap"]), inp.tokenizer.eos_token_id, dtype=torch.long,
                            device=sem.device)
    for s in range(0, len(keep), args.gen_batch):
        props = keep[s:s + args.gen_batch]
        n = len(props)
        instr = {k: expand(v, n) for k, v in instr1.items()}
        box_query = det_out["box_corners"][0, props].unsqueeze(1)  # n x 1 x 8 x 3
        prefix = qformer_prefix(captioner, det_out, scene_in, instr, n, box_query=box_query)
        embeds = torch.cat([prefix, emb(instr_ids)[None].expand(n, -1, -1)], 1).to(captioner.dtype)
        mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)
        output_ids[props] = beam_search_kv(captioner.transformer, embeds, mask, MAX_LEN["dense-cap"],
                                           **captioner.caption_config)
    det_out["output_ids"] = output_ids.unsqueeze(0)
    return det_out


def decode(tokenizer, output_ids):
    texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [" ".join(filter(lambda w: w, t.split(" "))) for t in texts]  # evaluate_qa.py


# ---------------------------------------------------------------- upstream path
@torch.no_grad()
def upstream_qa(model, scene_in, rows, inp, args, kind="qa", max_len=MAX_LEN["qa"], task_name="qa"):
    """``task_name`` 'qa': the repo's beam search (``predict_answer``) up to ``max_len`` tokens;
    'chat': its ``predict_chat`` (sampling, 512 tokens)."""
    preds = []
    for s in range(0, len(rows), args.batch):
        chunk = rows[s:s + args.batch]
        b = to_batch([inp.instruction(kind, r["instruction"] if kind.startswith("raw") else r["question"])
                      for r in chunk],
                     scene_in["point_clouds"].device)
        b.update({k: expand(scene_in[k], len(chunk)).contiguous() for k in SCENE_KEYS})
        if task_name == "qa":
            det = model.detector(b, is_eval=True)
            out = model.captioner.predict_answer(det, b, max_gen_length=max_len)
        else:
            out = model(b, is_eval=True, task_name=task_name)
        preds += decode(inp.tokenizer, out["output_ids"])
    return preds


XR_KINDS = {"lscenellm": ("hd_qa", "raw_cased"), "ll3da": ("qa", "raw")}  # QA, caption instruction


def xr_answer(model, scene_in, rows, inp, args, task):
    qa_kind, cap_kind = XR_KINDS[args.xr_protocol]
    if xr_task(task)[0] == "qa":
        return (fast_qa if args.fast else upstream_qa)(model, scene_in, rows, inp, args, kind=qa_kind)
    if args.cap_decode == "chat":
        torch.manual_seed(zlib.crc32(rows[0]["scene_id"].encode()))
        return upstream_qa(model, scene_in, rows, inp, args, kind=cap_kind, task_name="chat")
    return (fast_qa if args.fast else upstream_qa)(model, scene_in, rows, inp, args, kind=cap_kind,
                                                   max_len=args.cap_max_len)


@torch.no_grad()
def upstream_densecap(model, scene_in, inp, args):
    b = to_batch([inp.instruction("densecap")], scene_in["point_clouds"].device)
    b.update(scene_in)
    return model(b, is_eval=True, task_name="dense-cap")


# ---------------------------------------------------------------- scan2cap records
def densecap_records(out, scene_np, scene_in, tokenizer, scene_id):
    """Per kept proposal: caption, matched ScanNet object id and its box IoU, as
    ``eval_utils/evaluate_densecap.py`` computes them (before its ``test_min_iou`` cut)."""
    gt_corners = torch.from_numpy(scene_np["gt_box_corners"]).to(out["box_corners"].device)[None]
    gt_ids = torch.from_numpy(scene_np["gt_object_ids"]).to(out["box_corners"].device)[None]
    _, nq, _, _ = out["box_corners"].shape
    nobj = gt_corners.shape[1]
    ious = box3d_iou_batch_tensor(out["box_corners"].unsqueeze(2).repeat(1, 1, nobj, 1, 1).view(-1, 8, 3),
                                  gt_corners.unsqueeze(1).repeat(1, nq, 1, 1, 1).view(-1, 8, 3)).view(1, nq, nobj)
    ious, idx = ious.max(-1)
    obj = torch.gather(gt_ids, 1, idx)
    good = out["sem_cls_logits"].argmax(-1) != out["sem_cls_logits"].shape[-1] - 1
    nms = parse_predictions(out["box_corners"], out["sem_cls_prob"], out["objectness_prob"],
                            scene_in["point_clouds"])
    good &= torch.from_numpy(nms).to(good.device) == 1
    caps = tokenizer.batch_decode(out["output_ids"][0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    recs = []
    for p in good[0].nonzero().squeeze(1).tolist():
        recs.append({"prop": p, "obj_id": int(obj[0, p]), "iou": float(ious[0, p]),
                     "caption": ("sos " + caps[p] + " eos").replace("  ", " ")})
    return {"qid": scene_id, "scene_id": scene_id, "proposals": recs}


# ---------------------------------------------------------------- main
def shard(scene_counts, n, i):
    """Longest-processing-time assignment of whole scenes to n chunks."""
    loads, owner = [0] * n, {}
    for s, c in sorted(scene_counts.items(), key=lambda x: (-x[1], x[0])):
        k = min(range(n), key=lambda j: loads[j])
        owner[s] = k
        loads[k] += c
    return {s for s, k in owner.items() if k == i}


def build_model(args, inp):
    margs = SimpleNamespace(detector="detector_Vote2Cap_DETR", captioner="ll3da", freeze_detector=True,
                            freeze_llm=True, use_beam_search=True, vocab=args.vocab,
                            qformer_vocab=args.qformer_vocab, use_color=True, use_normal=True,
                            use_height=True, use_multiview=False)
    model = CaptionNet(margs, inp.cfg, None)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    bad = [k for k in missing if not k.startswith("captioner.transformer.")]
    assert not bad and not unexpected, (bad[:5], unexpected[:5])
    return model.cuda().eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", required=True, help="space-separated task tags")
    ap.add_argument("--ckpt", required=True, help="ll3da-opt-1.3b.pth")
    ap.add_argument("--vocab", required=True, help="facebook/opt-1.3b dir")
    ap.add_argument("--qformer_vocab", required=True, help="bert-base-embedding dir")
    ap.add_argument("--scannet_data", required=True, help="LL3DA scannet_data dir")
    ap.add_argument("--scanrefer_dir")
    ap.add_argument("--scanqa_dir")
    ap.add_argument("--sqa3d_test")
    ap.add_argument("--real3dqa_dir")
    ap.add_argument("--beacon3d_dir", help="vendored Beacon3D (data/<domain>/<domain>_qa.json)")
    ap.add_argument("--beacon3d_scans", help="prepare_scans.py output (3RScan / MultiScan)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_chunks", type=int, default=1)
    ap.add_argument("--chunk_idx", type=int, default=0)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--batch", type=int, default=8, help="default path: items per forward (eval.*.sh: 8)")
    ap.add_argument("--gen_batch", type=int, default=32, help="--fast: sequences per beam search")
    ap.add_argument("--max_des_len", type=int, default=256)
    ap.add_argument("--max_scenes", type=int, default=0, help="debug: first n scenes per task")
    ap.add_argument("--xr_cache", help="prepare_xr.py output (xr_<split>_items.json, regions/)")
    ap.add_argument("--xr_scan_data", help="SceneVerse HM3D scan_data (room_center/ for placement rc)")
    ap.add_argument("--xr_split", default="val")
    ap.add_argument("--cap_decode", default="chat", choices=["chat", "beam"], help="xrcap_*: see above")
    ap.add_argument("--cap_max_len", type=int, default=256, help="xrcap_* with --cap_decode beam")
    ap.add_argument("--xr_protocol", default="lscenellm", choices=["lscenellm", "ll3da"], help="xr*: see above")
    args = ap.parse_args()
    args.grid_size_3d, args.max_prompts = 255, 16

    inp = Inputs(args)
    model = build_model(args, inp)
    t0, stats = time.time(), {"chunk_idx": args.chunk_idx, "fast": args.fast, "tasks": {}}
    for task in args.tasks.split():
        rows = load_task(task, args)
        by_scene = {}
        for r in rows:
            by_scene.setdefault(r["scene_id"], []).append(r)
        weight = {s: (60 if task == "scan2cap" else len(v)) for s, v in by_scene.items()}
        mine = sorted(shard(weight, args.num_chunks, args.chunk_idx))
        if args.max_scenes:
            mine = mine[:args.max_scenes]
        out_dir = Path(args.output_dir) / task
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"predictions_chunk_{args.chunk_idx}.json"
        done = json.load(open(out_file)) if out_file.exists() else []
        have = {d["scene_id"] for d in done}
        data_dir = None if xr_task(task) else scene_dir(task, args)
        tt, n = time.time(), 0
        for si, scene_id in enumerate(mine):
            if scene_id in have:
                continue
            scene_np = inp.xr_scene(task, scene_id) if xr_task(task) else inp.scene(data_dir, scene_id)
            scene_in = {k: torch.from_numpy(scene_np[k][None]).cuda() for k in SCENE_KEYS}
            if task == "scan2cap":
                out = (fast_densecap if args.fast else upstream_densecap)(model, scene_in, inp, args)
                done.append(densecap_records(out, scene_np, scene_in, inp.tokenizer, scene_id))
                n += 1
            else:
                qs = by_scene[scene_id]
                if xr_task(task):
                    preds = xr_answer(model, scene_in, qs, inp, args, task)
                else:
                    preds = (fast_qa if args.fast else upstream_qa)(model, scene_in, qs, inp, args)
                for r, p in zip(qs, preds):
                    done.append({**r, "pred": p})
                n += len(qs)
            if si % 10 == 0 or si == len(mine) - 1:
                json.dump(done, open(out_file, "w"))
                print(f"[{task} {si + 1}/{len(mine)}] {scene_id}: {n} items, {time.time() - tt:.0f}s", flush=True)
        json.dump(done, open(out_file, "w"))
        stats["tasks"][task] = {"items": n, "seconds": round(time.time() - tt, 1)}
    stats["seconds"] = round(time.time() - t0, 1)
    stats["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)
    json.dump(stats, open(Path(args.output_dir) / f"stats_chunk_{args.chunk_idx}.json", "w"), indent=1)
    print("INFER_DONE", json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
