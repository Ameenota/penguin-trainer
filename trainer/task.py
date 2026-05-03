import pandas as pd
import xgboost as xgb
from google.cloud import bigquery, aiplatform
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from google.cloud import storage
from datetime import datetime
import logging
import os

# Configure logging for Cloud Logging compatibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
BUCKET_NAME = "pipelines1341"
PROJECT_ID = "conspiracy-493120"
REGION = "us-central1"
MODEL_DISPLAY_NAME = "penguin-classifier"
# The specific Model Resource ID to use for versioning
TARGET_MODEL_ID = "2914109334373793792"

def upload_to_gcs(local_path, bucket_name, gcs_path):
    """Uploads a file to Google Cloud Storage."""
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        logger.info(f"Successfully uploaded {local_path} to gs://{bucket_name}/{gcs_path}")
    except Exception as e:
        logger.error(f"Failed to upload to GCS: {e}")
        raise

def train_and_register():
    # 1. Initialize Vertex AI
    aiplatform.init(experiment="penguin-classification-exp", project=PROJECT_ID, location=REGION)

    # 2. Load Data from BigQuery
    logger.info("Fetching data from BigQuery...")
    client = bigquery.Client(project=PROJECT_ID)
    query = "SELECT * FROM `bigquery-public-data.ml_datasets.penguins`"
    df = client.query(query).to_dataframe().dropna()

    # Prepare features and labels
    X = df[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']]
    y = df['species'].astype('category').cat.codes
    labels = df['species'].astype('category').cat.categories

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Start Experiment Run
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    run_name = f"xgboost-penguin-run-{timestamp}"
    
    # We capture the 'run' object here to use it for explicit lineage logging
    with aiplatform.start_run(run_name) as run:
        logger.info(f"Starting experiment run: {run_name}")
        
        # Log basic metadata
        aiplatform.log_metrics({"xgboost_version": xgb.__version__})
        params = {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.05}
        aiplatform.log_params(params)
        
        # Train Model
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        logger.info("Training complete.")
        
        # Evaluation
        predictions = model.predict(X_test)
        report = classification_report(y_test, predictions, target_names=labels, output_dict=True)
        
        # Log key metrics
        aiplatform.log_metrics({
            "accuracy": report["accuracy"],
            "macro_avg_f1": report["macro avg"]["f1-score"],
        })

        # 4. Save and Upload Model Artifact
        local_model_path = "model.bst"
        model.save_model(local_model_path)
        upload_to_gcs(local_model_path, BUCKET_NAME, "models/model.bst")

        # 5. Register Model in Registry with Versioning
        parent_model_path = f"projects/{PROJECT_ID}/locations/{REGION}/models/{TARGET_MODEL_ID}"

        model_uploaded = aiplatform.Model.upload(
            display_name=MODEL_DISPLAY_NAME,
            artifact_uri=f"gs://{BUCKET_NAME}/models/",
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.2-1:latest",
            parent_model=parent_model_path 
        )
        
        # 6. EXPLICIT LINEAGE LINKING
        # Fixed positional argument for the model object per SDK docs
        artifact_name = f"penguin-model-{timestamp}"
        
        run.log_model(
            model,
            display_name=artifact_name,
            # Linking the Experiment Artifact to the Registry Model Resource
            uri=model_uploaded.uri 
        )
        
        aiplatform.log_params({"model_artifact_name": artifact_name})
        logger.info(f"Model and Lineage logged successfully for {artifact_name}")

if __name__ == "__main__":
    train_and_register()