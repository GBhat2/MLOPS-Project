import os
from absl import logging

from tfx import v1 as tfx
from tfx.orchestration.data_types import RuntimeParameter
from pipeline import configs
from pipeline import local_pipeline



def run():
    """Define a pipeline."""

    tfx.orchestration.LocalDagRunner().run(
        _create_pipeline(
      pipeline_name=PIPELINE_NAME,
      pipeline_root=_taxi_root,
      data_root=_data_root,
      module_file=_taxi_trainer_module_file,
      serving_model_dir=_serving_model_dir,
      metadata_path=METADATA_PATH))


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    run()
