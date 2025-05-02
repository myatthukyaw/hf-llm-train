import evaluate
import numpy as np


class EvalMetrics:
    def compute_rouge(self, preds: list, refs: list) -> dict:
        """Evaluate summaries using ROUGE metric."""
        rouge = evaluate.load("rouge")
        rouge_results = rouge.compute(
            predictions=preds, 
            references=refs,
            use_stemmer=True,
            use_aggregator=True
        )
        return rouge_results

    def compute_bleu(self, preds: list, refs: list) -> dict:
        """Evaluate summaries using BLEU metric."""
        bleu = evaluate.load("bleu")
        references_for_bleu = [[ref] for ref in refs]
        bleu_results = bleu.compute(
            predictions=preds, references=references_for_bleu
        )
        return bleu_results