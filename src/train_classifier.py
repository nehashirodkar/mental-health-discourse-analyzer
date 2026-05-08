"""Fine-tune RoBERTa for distress-severity classification.

Weighted CrossEntropyLoss handles class imbalance: severe-class
errors carry more weight than majority-class errors, so the model
cannot win by predicting the majority class everywhere.
"""
from __future__ import annotations

import json

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src import config


def load_splits():
    train = pd.read_csv(config.TRAIN_CSV)
    val = pd.read_csv(config.VAL_CSV)
    test = pd.read_csv(config.TEST_CSV)
    return train, val, test


def to_hf(df: pd.DataFrame, tokenizer) -> Dataset:
    ds = Dataset.from_pandas(df[["text", "severity"]].rename(columns={"severity": "labels"}))

    def tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=config.MAX_LEN,
            padding=False,
        )
    ds = ds.map(tok, batched=True, remove_columns=["text"])
    return ds


class WeightedTrainer(Trainer):
    """Trainer with class-weighted CrossEntropy loss for imbalance handling."""

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fn(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_precision": p,
        "macro_recall": r,
        "macro_f1": f1,
    }


def main():
    train_df, val_df, test_df = load_splits()
    print(f"[train] sizes: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    train_ds = to_hf(train_df, tokenizer)
    val_ds = to_hf(val_df, tokenizer)
    test_ds = to_hf(test_df, tokenizer)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(config.LABELS)),
        y=train_df["severity"].values,
    )
    class_weights = torch.tensor(weights, dtype=torch.float)
    print(f"[train] class weights: {dict(zip(config.LABELS, weights.round(3)))}")

    model = AutoModelForSequenceClassification.from_pretrained(
        config.BASE_MODEL,
        num_labels=len(config.LABELS),
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=str(config.CLASSIFIER_DIR),
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE * 2,
        learning_rate=config.LR,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        seed=config.SEED,
        report_to="none",
    )

    config.MLRUNS_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(config.MLRUNS_DIR.as_uri())
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="roberta-distress") as run:
        mlflow.log_params({
            "base_model": config.BASE_MODEL,
            "max_len": config.MAX_LEN,
            "batch_size": config.BATCH_SIZE,
            "epochs": config.EPOCHS,
            "lr": config.LR,
            "weight_decay": config.WEIGHT_DECAY,
            "warmup_ratio": config.WARMUP_RATIO,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
        })
        mlflow.log_dict(
            {l: float(w) for l, w in zip(config.LABELS, weights)},
            "class_weights.json",
        )

        trainer = WeightedTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
            class_weights=class_weights,
        )

        trainer.train()

        for epoch_log in trainer.state.log_history:
            step = epoch_log.get("step", 0)
            for k, v in epoch_log.items():
                if isinstance(v, (int, float)) and k not in ("epoch", "step"):
                    mlflow.log_metric(k, v, step=step)

        test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
        print("[train] test metrics:", test_metrics)
        mlflow.log_metrics({k: v for k, v in test_metrics.items() if isinstance(v, (int, float))})

        preds = trainer.predict(test_ds)
        y_true = preds.label_ids
        y_pred = preds.predictions.argmax(-1)
        report = classification_report(
            y_true, y_pred, target_names=config.LABELS, zero_division=0, output_dict=True
        )
        with open(config.MODELS_DIR / "test_report.json", "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(str(config.MODELS_DIR / "test_report.json"))
        print(classification_report(y_true, y_pred, target_names=config.LABELS, zero_division=0))

        trainer.save_model(str(config.CLASSIFIER_DIR))
        tokenizer.save_pretrained(str(config.CLASSIFIER_DIR))
        mlflow.log_artifacts(str(config.CLASSIFIER_DIR), artifact_path="model")
        print(f"[train] saved model -> {config.CLASSIFIER_DIR}")
        print(f"[train] mlflow run id: {run.info.run_id}")


if __name__ == "__main__":
    main()
