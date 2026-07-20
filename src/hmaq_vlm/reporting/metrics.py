from __future__ import annotations

from statistics import mean


def evaluate_captions(references: dict[str, list[str]], hypotheses: dict[str, str]) -> dict[str, float]:
    """Run COCO caption metrics, including Java-backed METEOR/SPICE when installed."""
    if set(references) != set(hypotheses):
        raise ValueError("reference and hypothesis image ids must match")
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.spice.spice import Spice
    except ImportError as error:
        raise RuntimeError("caption evaluation requires pycocoevalcap and Java for METEOR/SPICE") from error
    res = {key: [value] for key, value in hypotheses.items()}
    bleu, _ = Bleu(4).compute_score(references, res)
    cider, _ = Cider().compute_score(references, res)
    meteor, _ = Meteor().compute_score(references, res)
    rouge, _ = Rouge().compute_score(references, res)
    spice, _ = Spice().compute_score(references, res)
    lengths = [len(caption.split()) for caption in hypotheses.values()]
    return {"cider": float(cider), "bleu1": float(bleu[0]), "bleu4": float(bleu[3]), "meteor": float(meteor), "rouge_l": float(rouge), "spice": float(spice), "caption_length": mean(lengths) if lengths else 0.0, "empty_generation_rate": sum(length == 0 for length in lengths) / len(lengths) if lengths else 0.0}
