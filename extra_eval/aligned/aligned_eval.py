"""Aligned scoring: one text normalisation for every baseline (and just3d).

Decided with the user in the 2026-11 scoring audit. Every repo vendors an identical copy of
this file (``extra_eval/aligned/aligned_eval.py``); the per-repo ``score_aligned.py`` only
loads that repo's saved predictions, strips the model's own special tokens (no truncation)
and calls the functions here.

Modes (applied symmetrically to prediction and references):

* ``raw``        strings as-is (whitespace collapsed); LSceneLLM's XR protocol.
* ``ptb``        PTB tokenizer (lower-case, punctuation split off and dropped).
* ``clean_ptb``  ``clean_answer`` then PTB.  QA-style text (ScanQA, XR-QA).
* ``s2c``        ``sos <text> eos`` then PTB.  Scan2Cap (references come with sos/eos).

Exact match: ``clean_answer`` on both sides, then ``answer_match`` that first looks for an
exact hit over *all* references and only then for a substring hit (the official
Real-3DQA/Beacon3D matcher returns at the first substring hit and can miss a later exact
one).  An empty cleaned prediction scores 0 / 0.

Caption metrics come from pycocoevalcap 1.2 (BLEU-1..4, METEOR, ROUGE-L, CIDEr, x100);
PTB and METEOR need ``java`` on PATH.
"""

from __future__ import annotations

import re
from collections import OrderedDict

__all__ = ["clean_answer", "strip_ws", "answer_match", "em_scores", "normalise",
           "caption_metrics", "score_captions", "score_qa"]


def clean_answer(data):
    """LEO / SQA3D / Real-3DQA ``clean_answer`` (same rules in every repo)."""
    if isinstance(data, list):
        return [clean_answer(x) for x in data]
    data = data.lower()
    data = re.sub(r"[ ]+$", "", data)
    data = re.sub(r"^[ ]+", "", data)
    data = re.sub(r" {2,}", " ", data)
    data = re.sub(r"\.[ ]{2,}", ". ", data)
    data = re.sub(r"[^a-zA-Z0-9,'\s\-:]+", "", data)
    data = re.sub(r"ç", "c", data)
    data = re.sub(r"’", "'", data)
    data = re.sub(r"\bletf\b", "left", data)
    data = re.sub(r"\blet\b", "left", data)
    data = re.sub(r"\btehre\b", "there", data)
    data = re.sub(r"\brigth\b", "right", data)
    data = re.sub(r"\brght\b", "right", data)
    data = re.sub(r"\bbehine\b", "behind", data)
    data = re.sub(r"\btv\b", "TV", data)
    data = re.sub(r"\bchai\b", "chair", data)
    data = re.sub(r"\bwasing\b", "washing", data)
    data = re.sub(r"\bwaslked\b", "walked", data)
    data = re.sub(r"\boclock\b", "o'clock", data)
    data = re.sub(r"\bo'[ ]+clock\b", "o'clock", data)
    for d, w in (("0", "zero"), ("none", "zero"), ("1", "one"), ("2", "two"), ("3", "three"),
                 ("4", "four"), ("5", "five"), ("6", "six"), ("7", "seven"), ("8", "eight"),
                 ("9", "nine"), ("10", "ten"), ("11", "eleven"), ("12", "twelve"),
                 ("13", "thirteen"), ("14", "fourteen"), ("15", "fifteen"), ("16", "sixteen"),
                 ("17", "seventeen"), ("18", "eighteen"), ("19", "nineteen"), ("20", "twenty"),
                 ("23", "twenty-three")):
        data = re.sub(r"\b%s\b" % d, w, data)
    data = re.sub(r"\b([a-zA-Z]+)([0-9])\b", r"\g<1>", data)
    data = re.sub(r"\ba\b ([a-zA-Z]+)", r"\g<1>", data)
    data = re.sub(r"\ban\b ([a-zA-Z]+)", r"\g<1>", data)
    data = re.sub(r"\bthe\b ([a-zA-Z]+)", r"\g<1>", data)
    data = re.sub(r"\bbackwards\b", "backward", data)
    return data


def strip_ws(text: str) -> str:
    """Collapse whitespace (newlines would desync METEOR's line protocol) and drop ``|``."""
    return " ".join((text or "").replace("|", " ").split())


# --------------------------------------------------------------------------- exact match
def answer_match(pred: str, gts: list[str]) -> tuple[int, int]:
    """(exact, relaxed) on already-cleaned strings; exact over all refs first."""
    if pred in gts:
        return 1, 1
    p = "".join(pred.split())
    for gt in gts:
        g = "".join(gt.split())
        if p in g or g in p:
            return 0, 1
    return 0, 0


def em_scores(preds: dict, refs: dict) -> dict:
    """EM / EM-R in percent over ``preds`` {k: str} and ``refs`` {k: [str]}."""
    em = emr = 0
    for k, p in preds.items():
        p = clean_answer(p or "").strip()
        if not p:
            continue
        e, r = answer_match(p, [g.strip() for g in clean_answer(list(refs[k]))])
        em += e
        emr += r
    n = max(len(preds), 1)
    return {"EM": 100.0 * em / n, "EM-R": 100.0 * emr / n}


# --------------------------------------------------------------------------- caption metrics
def _ptb(d: dict) -> dict:
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    return PTBTokenizer().tokenize({k: [{"caption": c} for c in v] for k, v in d.items()})


def normalise(refs: dict, preds: dict, mode: str) -> tuple[dict, dict]:
    """-> (gts {k: [str]}, res {k: [str]}) after ``mode`` on both sides."""
    keys = list(preds)
    gts = OrderedDict((k, [strip_ws(r) for r in refs[k]]) for k in keys)
    res = OrderedDict((k, [strip_ws(preds[k])]) for k in keys)
    if mode == "raw":
        return gts, res
    if mode == "clean_ptb":
        gts = OrderedDict((k, clean_answer(v)) for k, v in gts.items())
        res = OrderedDict((k, clean_answer(v)) for k, v in res.items())
    elif mode == "s2c":
        def wrap(s):
            s = s.strip()
            if not s.startswith("sos "):
                s = "sos " + s if s != "sos" else s
            if not s.endswith(" eos") and s != "sos eos":
                s = s + " eos"
            return s
        gts = OrderedDict((k, [wrap(r) for r in v]) for k, v in gts.items())
        res = OrderedDict((k, [wrap(r) for r in v]) for k, v in res.items())
    elif mode != "ptb":
        raise ValueError(mode)
    return _ptb(gts), _ptb(res)


def caption_metrics(gts: dict, res: dict) -> dict:
    """pycocoevalcap BLEU-1..4 / METEOR / ROUGE-L / CIDEr, x100, on prepared dicts."""
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    # an empty hypothesis breaks METEOR's java protocol on some inputs; a lone token scores ~0
    res = OrderedDict((k, [v[0] if v and v[0].strip() else "<empty>"]) for k, v in res.items())
    b, _ = Bleu(4).compute_score(gts, res)
    m, _ = Meteor().compute_score(gts, res)
    r, _ = Rouge().compute_score(gts, res)
    c, _ = Cider().compute_score(gts, res)
    return {"CIDEr": 100 * c, "BLEU-1": 100 * b[0], "BLEU-4": 100 * b[3],
            "METEOR": 100 * m, "ROUGE-L": 100 * r}


def score_captions(refs: dict, preds: dict, modes=("ptb", "raw")) -> dict:
    """{mode: metrics} for caption-style benchmarks."""
    return {mode: caption_metrics(*normalise(refs, preds, mode)) for mode in modes}


def score_qa(refs: dict, preds: dict, modes=("clean_ptb", "raw")) -> dict:
    """{mode: metrics + EM/EM-R} for QA-style benchmarks (EM is always clean_answer based)."""
    em = em_scores(preds, refs)
    return {mode: {**caption_metrics(*normalise(refs, preds, mode)), **em} for mode in modes}
