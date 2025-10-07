"""
callbacks.py

Contains the EvalCallback class for evaluating a model at the end of each epoch,
saving best models, metrics, and sample outputs.
"""

import re
import pandas as pd
from transformers import TrainerCallback, GenerationConfig

class EvalCallback(TrainerCallback):
    """
    Callback for evaluating a model on a validation dataset at the end of each epoch.

    Attributes:
        val_dataset: The validation dataset.
        tokenizer: Tokenizer to decode model outputs.
        model: The model being trained.
        scores_dir: Directory to save evaluation metrics and outputs.
        model_name: Name used to save the best model.
        best_rouge: Highest rouge2_f_measure seen so far.
        best_epoch: Epoch number corresponding to best_rouge.
        scores: List of metric dictionaries for each epoch.
        sample_outputs: List of last predicted outputs per epoch.
    """

    def __init__(self, val_dataset, tokenizer, model, scores_dir, model_name):
        """
        Initialize the EvalCallback.
        """
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.model = model
        self.scores_dir = scores_dir
        self.model_name = model_name

        self.best_rouge = -1
        self.best_epoch = 0
        self.scores = []
        self.sample_outputs = []

    def _generate_predictions(self):
        """
        Generate gold and predicted texts for the validation dataset.

        Returns:
            tuple: (list of gold texts, list of predicted texts)
        """
        generation_config = GenerationConfig(do_sample=False, max_new_tokens=1000)
        gold, preds = [], []

        for sample in self.val_dataset:
            input_ids = self.tokenizer.encode(sample["input"], return_tensors="pt").cuda()
            output_ids = self.model.generate(input_ids=input_ids, generation_config=generation_config)
            decoded = self.tokenizer.decode(output_ids[0])
            preds.append(re.split(r"\[EOG\]|\[/INST\]", decoded)[1])
            gold.append(sample["output"])

        return gold, preds

    def _evaluate_predictions(self, gold, preds, test_set):
        """
        Compute evaluation metrics for predictions.

        Args:
            gold: List of reference outputs.
            preds: List of generated outputs.
            test_set: Name of the test set (used to select metric function).

        Returns:
            dict: Evaluation metrics.
        """
        from summary_metrics import compute_metrics, compute_metrics_argessay

        metrics_fn = compute_metrics_argessay if test_set == "argessays" else compute_metrics
        return metrics_fn(predictions=preds, references=gold)

    def _update_best_model(self, metrics, trainer):
        """
        Save the model if the current epoch's rouge2_f_measure is the best.

        Args:
            metrics: Current epoch metrics dictionary.
            trainer: Trainer instance to save the model.
        """
        if metrics["rouge2_f_measure"] > self.best_rouge:
            self.best_rouge = metrics["rouge2_f_measure"]
            self.best_epoch = metrics["epoch"]
            trainer.model.save_pretrained(f"models_finetuned/{self.model_name}")
        metrics["best_epoch"] = self.best_epoch

    def _save_epoch_results(self, metrics, preds):
        """
        Save metrics and sample predictions to disk.

        Args:
            metrics: Metrics dictionary for current epoch.
            preds: Generated outputs for current epoch.
        """
        self.scores.append(metrics)
        pd.DataFrame(self.scores).to_csv(f"{self.scores_dir}/llama_results.csv", index=False)

        with open(f"{self.scores_dir}/sample_output.txt", "w") as f:
            for i, text in enumerate(preds, 1):
                f.write(f"sample {i}\n{text}\n\n")

        self.sample_outputs.append(preds[-1])

    def on_epoch_end(self, args, state, control, **kwargs):
        """
        Hook called at the end of each epoch.
        Performs generation, evaluation, updates best model, and saves metrics.
        """
        self.model.eval()

        gold, preds = self._generate_predictions()
        metrics = self._evaluate_predictions(gold, preds, test_set=args.test_set)
        metrics["epoch"] = len(self.scores) + 1

        self._update_best_model(metrics, kwargs["trainer"])
        self._save_epoch_results(metrics, preds)

        self.model.train()
