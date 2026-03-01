"""Mini training run to test the full flow: Train → Neon DB → MLflow.

Trains on a tiny dataset for 2 epochs, saves data to Neon DB,
pushes the model to MLflow, and links the two via mlflow_run_id.

Usage
-----
    python -m src.mini_train
"""

import shutil
import logging

from dotenv import load_dotenv

load_dotenv()

from src.train import train
from src.database import NeonDB
from src.push_to_mlflow import push

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

MINI_OUTPUT_DIR = "models/mini_test"


def main():
    print("=" * 60)
    print("MINI TRAIN — End-to-end flow test")
    print("=" * 60)

    # --- Step 1: Train (small dataset, 2 epochs, save to DB) ---
    print("\n[1/4] Training with 2 epochs on synthetic data...")
    output_dir = train(
        data_path=None,  # use synthetic data
        output_dir=MINI_OUTPUT_DIR,
        epochs=2,
        batch_size=4,
        save_to_db=True,
    )

    # --- Step 2: Get the DB run_id we just created ---
    print("\n[2/4] Retrieving DB run_id...")
    db = NeonDB()
    # Get the latest run_id (the one we just created)
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT run_id, status, num_training_examples FROM training_runs ORDER BY run_id DESC LIMIT 1"
        )
        row = cursor.fetchone()
    db_run_id = row[0]
    print(f"  DB run_id: {db_run_id} | status: {row[1]} | examples: {row[2]}")

    # --- Step 3: Push model to MLflow ---
    print("\n[3/4] Pushing model to MLflow...")
    mlflow_run_id = push(
        model_path=MINI_OUTPUT_DIR,
        run_name=f"mini_test_run_{db_run_id}",
        experiment_name="fact-checker",
        notes=f"Mini test run — db_run_id={db_run_id}",
    )
    print(f"  MLflow run_id: {mlflow_run_id}")

    # --- Step 4: Link MLflow run_id in DB ---
    print("\n[4/4] Linking MLflow run_id to DB entry...")
    db.update_training_run(
        run_id=db_run_id,
        mlflow_run_id=mlflow_run_id,
    )
    print(f"  DB run_id {db_run_id} → mlflow_run_id {mlflow_run_id}")

    # --- Summary ---
    run_info = db.get_training_run(db_run_id)
    data_stats = db.get_training_data_stats(db_run_id)

    print("\n" + "=" * 60)
    print("FLOW TEST COMPLETE")
    print("=" * 60)
    print(f"  DB run_id:       {db_run_id}")
    print(f"  MLflow run_id:   {mlflow_run_id}")
    print(f"  Status:          {run_info['status']}")
    print(f"  Metrics:         {run_info['metrics']}")
    print(f"  Data stats:      {data_stats}")
    print(f"  Model saved to:  {MINI_OUTPUT_DIR}")
    print("\nA person checking the DB can find the training data via run_id")
    print("and the model version via mlflow_run_id.")

    # --- Cleanup temp model ---
    print(f"\nCleaning up temporary model at {MINI_OUTPUT_DIR}...")
    shutil.rmtree(MINI_OUTPUT_DIR, ignore_errors=True)
    print("Done.")


if __name__ == "__main__":
    main()
