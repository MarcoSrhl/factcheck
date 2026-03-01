"""Score triplets with the GAN discriminator downloaded from MLflow.

Usage
-----
    python -m src.infer_gan "Paris" "is capital of" "France"
    python -m src.infer_gan "London" "is capital of" "France"
    python -m src.infer_gan --phrase "Paris is the capital of France"
"""

import argparse
import os
import tempfile

import dagshub
import mlflow
import torch

from factcheck.gan_model import FactGAN

DAGSHUB_REPO = "NLP-Fact-checking"
DAGSHUB_OWNER = "MarcoSrhl"
EXPERIMENT = "fact-checker"
MODEL_TYPE = "bert-gan-swap"


def download_gan_from_mlflow(local_dir: str | None = None) -> str:
    """Download the latest GAN artifacts from MLflow.

    Returns the local directory containing discriminator.pt etc.
    """
    dagshub.init(DAGSHUB_REPO, DAGSHUB_OWNER, mlflow=True)
    client = mlflow.tracking.MlflowClient()

    exp = client.get_experiment_by_name(EXPERIMENT)
    if not exp:
        raise RuntimeError(f"Experiment '{EXPERIMENT}' not found on MLflow")

    runs = client.search_runs(
        exp.experiment_id,
        filter_string=f"params.model_type = '{MODEL_TYPE}'",
        max_results=1,
        order_by=["start_time DESC"],
    )
    if not runs:
        raise RuntimeError("No GAN run found on MLflow")

    run = runs[0]
    run_id = run.info.run_id
    print(f"Found MLflow run: {run.info.run_name} (id={run_id})")

    dst = local_dir or tempfile.mkdtemp(prefix="gan_mlflow_")
    client.download_artifacts(run_id, "gan_model", dst)
    model_dir = os.path.join(dst, "gan_model")
    print(f"Downloaded to: {model_dir}")
    return model_dir


def load_gan(model_dir: str) -> FactGAN:
    """Load a FactGAN from a local directory."""
    gan = FactGAN()
    gan.load(model_dir)
    gan.discriminator.eval()
    return gan


def score_triplets(gan: FactGAN, triplets: list[tuple[str, str, str]]) -> list[dict]:
    """Score triplets and return results."""
    scores = gan.discriminate_triplets(triplets)
    results = []
    for triplet, score in zip(triplets, scores.squeeze(-1).tolist()):
        verdict = "REAL" if score > 0.5 else "FAKE"
        results.append({
            "triplet": triplet,
            "score": score,
            "verdict": verdict,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Score triplets with GAN from MLflow")
    parser.add_argument("subject", nargs="?", help="Subject entity")
    parser.add_argument("predicate", nargs="?", help="Predicate/relation")
    parser.add_argument("object", nargs="?", help="Object entity")
    parser.add_argument("--phrase", type=str, help="Free-text claim (auto-extracts triplet)")
    parser.add_argument("--local", type=str, default=None,
                        help="Use local model dir instead of downloading from MLflow")
    args = parser.parse_args()

    # Load model
    if args.local:
        model_dir = args.local
        print(f"Using local model: {model_dir}")
    else:
        model_dir = download_gan_from_mlflow()

    gan = load_gan(model_dir)

    # Build triplets
    if args.phrase:
        from factcheck.triplet_extractor import TripletExtractor
        extractor = TripletExtractor()
        triplets = extractor.extract(args.phrase)
        if not triplets:
            print(f"Could not extract triplets from: {args.phrase}")
            return
        print(f"Extracted triplet: ({triplets[0][0]}, {triplets[0][1]}, {triplets[0][2]})")
    elif args.subject and args.predicate and args.object:
        triplets = [(args.subject, args.predicate, args.object)]
    else:
        parser.error("Provide either subject/predicate/object or --phrase")

    # Score
    results = score_triplets(gan, triplets)
    print()
    for r in results:
        s, p, o = r["triplet"]
        print(f"  ({s}, {p}, {o})")
        print(f"  Score: {r['score']:.4f} -> {r['verdict']}")
        print()


if __name__ == "__main__":
    main()
