"""Repackage existing MLflow GAN runs into registerable MLflow Models.

Converts runs that only contain raw .pt artifacts (logged via
``mlflow.log_artifacts``) into proper MLflow PyTorch models that can be
registered and promoted to Staging / Production.

Usage
-----
    # Repackage the latest GAN run
    python -m src.repackage_mlflow

    # Repackage a specific run
    python -m src.repackage_mlflow --run-id abc123

    # Repackage all GAN runs
    python -m src.repackage_mlflow --all
"""

import argparse
import logging
import os
import tempfile

import dagshub
import mlflow
import mlflow.pytorch
import torch

from factcheck.gan_model import BERTDiscriminator, FactGAN

logger = logging.getLogger(__name__)

DAGSHUB_REPO = "NLP-Fact-checking"
DAGSHUB_OWNER = "MarcoSrhl"
EXPERIMENT = "fact-checker"
MODEL_TYPE = "bert-gan-swap"
REGISTRY_NAME = "fact-checker-gan"


def _init_mlflow() -> mlflow.tracking.MlflowClient:
    dagshub.init(DAGSHUB_REPO, DAGSHUB_OWNER, mlflow=True)
    return mlflow.tracking.MlflowClient()


def find_gan_runs(
    client: mlflow.tracking.MlflowClient,
    run_id: str | None = None,
    all_runs: bool = False,
) -> list[mlflow.entities.Run]:
    """Find GAN runs to repackage."""
    exp = client.get_experiment_by_name(EXPERIMENT)
    if not exp:
        raise RuntimeError(f"Experiment '{EXPERIMENT}' not found")

    if run_id:
        run = client.get_run(run_id)
        return [run]

    max_results = 100 if all_runs else 1
    runs = client.search_runs(
        exp.experiment_id,
        filter_string=f"params.model_type = '{MODEL_TYPE}'",
        max_results=max_results,
        order_by=["start_time DESC"],
    )
    if not runs:
        raise RuntimeError("No GAN runs found")

    # Skip runs that are already packaging runs (no gan_model artifact)
    original_runs = [
        r for r in runs
        if not (r.info.run_name or "").startswith("package_")
        and "source_run_id" not in r.data.params
    ]
    return original_runs


def repackage_run(
    client: mlflow.tracking.MlflowClient,
    run: mlflow.entities.Run,
    registry_name: str = REGISTRY_NAME,
) -> str:
    """Repackage a single run into a registerable MLflow Model.

    1. Download .pt artifacts
    2. Recreate BERTDiscriminator and load weights
    3. Log as proper MLflow PyTorch model in a new run
    4. Register in the Model Registry

    Returns the new packaging run ID.
    """
    original_id = run.info.run_id
    original_name = run.info.run_name or original_id
    original_params = run.data.params
    original_metrics = run.data.metrics

    print(f"\n--- Repackaging: {original_name} (id={original_id}) ---")

    # 1. Download artifacts to temp dir
    with tempfile.TemporaryDirectory(prefix="repackage_") as tmp:
        artifact_dir = client.download_artifacts(original_id, "gan_model", tmp)
        print(f"  Downloaded artifacts to {artifact_dir}")

        disc_path = os.path.join(artifact_dir, "discriminator.pt")
        meta_path = os.path.join(artifact_dir, "gan_meta.pt")

        if not os.path.isfile(disc_path):
            print(f"  SKIP: no discriminator.pt found")
            return ""

        # 2. Recreate model and load weights
        meta = {}
        if os.path.isfile(meta_path):
            meta = torch.load(meta_path, map_location="cpu", weights_only=True)

        discriminator = BERTDiscriminator(device=torch.device("cpu"))
        discriminator.load_state_dict(
            torch.load(disc_path, map_location="cpu", weights_only=True)
        )
        discriminator.eval()
        print(f"  Loaded discriminator ({sum(p.numel() for p in discriminator.parameters()):,} params)")

        # 3. Log as proper MLflow model in a new packaging run
        mlflow.set_experiment(EXPERIMENT)
        new_run_name = f"package_{original_id[:8]}"

        with mlflow.start_run(run_name=new_run_name) as new_run:
            # Log the discriminator as a proper PyTorch model
            mlflow.pytorch.log_model(
                pytorch_model=discriminator,
                artifact_path="model",
                pip_requirements=[
                    f"torch=={torch.__version__}",
                    "transformers",
                ],
            )

            # Copy original params + metrics
            mlflow.log_params({
                "model_type": MODEL_TYPE,
                "source_run_id": original_id,
                "source_run_name": original_name,
                **{k: v for k, v in original_params.items() if k != "model_type"},
            })
            if original_metrics:
                mlflow.log_metrics(original_metrics)

            mlflow.set_tag("mlflow.note.content",
                           f"Repackaged from run {original_name} ({original_id})")

            new_run_id = new_run.info.run_id
            print(f"  New run: {new_run_name} (id={new_run_id})")

        # 4. Register model via client API (more reliable on DagsHub)
        model_uri = f"runs:/{new_run_id}/model"
        try:
            client.create_registered_model(registry_name)
        except mlflow.exceptions.MlflowException:
            pass  # Already exists
        mv = client.create_model_version(
            name=registry_name,
            source=model_uri,
            run_id=new_run_id,
        )
        print(f"  Registered: {registry_name} v{mv.version}")

        return new_run_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repackage MLflow GAN runs into registerable models."
    )
    parser.add_argument("--run-id", type=str, default=None,
                        help="Specific run ID to repackage")
    parser.add_argument("--all", action="store_true",
                        help="Repackage all GAN runs (default: latest only)")
    parser.add_argument("--registry-name", type=str, default=REGISTRY_NAME,
                        help=f"Model Registry name (default: {REGISTRY_NAME})")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    registry_name = args.registry_name

    client = _init_mlflow()
    runs = find_gan_runs(client, run_id=args.run_id, all_runs=args.all)
    print(f"Found {len(runs)} run(s) to repackage")

    results = []
    for run in runs:
        new_id = repackage_run(client, run, registry_name=registry_name)
        if new_id:
            results.append((run.info.run_id, new_id))

    print(f"\n=== Done: {len(results)}/{len(runs)} repackaged ===")
    for orig, new in results:
        print(f"  {orig[:8]} -> {new[:8]}")


if __name__ == "__main__":
    main()
