# netflix

## Steps for running in the cloud

```sh
docker build -t my-python-app .
```

```sh
gcloud builds submit --tag gcr.io/tindart-8c83b/my-python-app:latest .
```

```sh
gcloud run jobs deploy my-python-job \ 
  --image gcr.io/tindart-8c83b/my-python-app:latest \
  --region us-central1 \
  --max-retries 0 --memory 1Gi
```

```sh
gcloud run jobs execute my-python-job \
  --region us-central1
```
