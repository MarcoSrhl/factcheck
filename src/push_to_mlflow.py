"""Push the trained fact-checker model to DagsHub MLflow.

Usage
-----
    python -m src.push_to_mlflow
    python -m src.push_to_mlflow --model models/fact_checker --run-name my_run
    python -m src.push_to_mlflow --accuracy 0.52 --notes "after pipeline fixes"
"""

import argparse
import logging

import dagshub
import mlflow
import mlflow.transformers
import torch
import transformers
from transformers import BertForSequenceClassification, BertTokenizer, pipeline

from factcheck.model import NUM_LABELS

logger = logging.getLogger(__name__)

DAGSHUB_REPO = "NLP-Fact-checking"
DAGSHUB_OWNER = "MarcoSrhl"


def push(
    model_path: str = "models/fact_checker",
    run_name: str = "fact_checker_bert",
    experiment_name: str = "fact-checker",
    accuracy: float | None = None,
    notes: str | None = None,
) -> str:
    """Load a local model and push it to DagsHub MLflow.

    Returns the MLflow run ID.
    """
    # 1. Auth + init DagsHub tracking
    dagshub.init(DAGSHUB_REPO, DAGSHUB_OWNER, mlflow=True)
    logger.info("Tracking URI: %s", mlflow.get_tracking_uri())

    # 2. Load model and tokenizer
    model = BertForSequenceClassification.from_pretrained(
        model_path, num_labels=NUM_LABELS
    )
    tokenizer = BertTokenizer.from_pretrained(model_path)
    hf_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
    logger.info("Loaded model from %s", model_path)

    # 3. Log to MLflow
    pip_reqs = [
        f"transformers=={transformers.__version__}",
        f"torch=={torch.__version__}",
    ]

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.transformers.log_model(
            transformers_model=hf_pipeline,
            artifact_path="model",
            task="text-classification",
            pip_requirements=pip_reqs,
        )

        mlflow.log_params({
            "model_base": "bert-base-uncased",
            "num_labels": NUM_LABELS,
            "max_length": 128,
            "source_path": model_path,
        })

        if accuracy is not None:
            mlflow.log_metrics({"val_accuracy": accuracy})

        if notes:
            mlflow.set_tag("mlflow.note.content", notes)

        run_id = run.info.run_id
        logger.info("Pushed to MLflow. Run ID: %s", run_id)

    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Push model to DagsHub MLflow.")
    parser.add_argument(
        "--model", type=str, default="models/fact_checker",
        help="Path to the local model directory.",
    )
    parser.add_argument(
        "--run-name", type=str, default="fact_checker_bert",
        help="Name for the MLflow run.",
    )
    parser.add_argument(
        "--experiment", type=str, default="fact-checker",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--accuracy", type=float, default=None,
        help="Validation accuracy to log.",
    )
    parser.add_argument(
        "--notes", type=str, default=None,
        help="Free-text notes to attach to the run.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    run_id = push(
        model_path=args.model,
        run_name=args.run_name,
        experiment_name=args.experiment,
        accuracy=args.accuracy,
        notes=args.notes,
    )
    print(f"Run ID: {run_id}")


if __name__ == "__main__":
    main()
