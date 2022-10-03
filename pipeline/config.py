import os

PIPELINE_NAME = ''

#GCS_BUCKET_NAME = ''

#GCP_PROJECT_ID = ''
#GCP_REGION = ''

#PREPROCESSING_FN = ''
#RUN_FN = ''
# RUN_FN = ''

TRAIN_NUM_STEPS = 100
EVAL_NUM_STEPS = 100

EVAL_ACCURACY_THRESHOLD = 0.6


DATAFLOW_BEAM_PIPELINE_ARGS = [
  '--project=' + GCP_PROJECT_ID,
  '--runner=DataflowRunner',
  '--temp_location=' + os.path.join('gs://', GCS_BUCKET_NAME, 'tmp'),
  '--region=' + GCP_REGION,
  '--experiments=shuffle_mode=auto',
  '--disk_size_gb=50',
]

GCP_AI_PLATFORM_TRAINING_ARGS = {
  'project': GCP_PROJECT_ID,
  'region': GCP_REGION,
  'masterConfig': {
    'imageUri': 'gcr.io/' + GCP_PROJECT_ID + ''
  }
}

GCP_AI_PLATFORM_SERVING_ARGS = {
  'model_name': PIPELINE_NAME,
  'project_id': GCP_PROJECT_ID,
  'regions': [GCP_REGION],
}
