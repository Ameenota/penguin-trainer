
rm trainer.zip
zip -r trainer.zip . -x "*.DS_Store" ".git/*"
gcloud storage cp trainer.zip gs://pipelines1341/trainer.zip

gcloud ai custom-jobs create \
  --display-name=penguin-trainer-v8 \
  --region=us-central1 \
  --project=conspiracy-493120 \
  --service-account=myservice@conspiracy-493120.iam.gserviceaccount.com \
  --python-package-uris=gs://pipelines1341/trainer.zip\
  --worker-pool-spec=machine-type=e2-standard-4,replica-count=1,executor-image-uri=us-docker.pkg.dev/vertex-ai/training/xgboost-cpu.2-1:latest,python-module=trainer.task \
  --args="--model-dir=gs://pipelines1341/models/"