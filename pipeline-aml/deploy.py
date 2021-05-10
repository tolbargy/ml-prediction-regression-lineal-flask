import os
import azureml.core
from azureml.core import Workspace, Experiment, Dataset, RunConfiguration
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import Dataset
from azureml.pipeline.core import PipelineEndpoint

# Verificamos version del Azure ML SDK
print("Azure ML SDK version:", azureml.core.VERSION)

# Cargamos el workspace de AML
ws = Workspace.from_config()
print(f'WS name: {ws.name}\nRegion: {ws.location}\nSubscription id: {ws.subscription_id}\nResource group: {ws.resource_group}')

# Creacion del compute cluster si es que no existe
aml_compute_target = "cpu-cluster"
try:
    aml_compute = AmlCompute(ws, aml_compute_target)
except ComputeTargetException:
    config = AmlCompute.provisioning_configuration(vm_size = "STANDARD_D2_V2", min_nodes = 0, max_nodes = 1,
                                                   idle_seconds_before_scaledown=3600)
    aml_compute = ComputeTarget.create(ws, aml_compute_target, config)
    aml_compute.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

# Obtenemos el dataset y lo subimos al worskpace
name_dataset='compensation-dataset-train'
datastore = ws.get_default_datastore()
datastore.upload(src_dir='../dataset', target_path=name_dataset, overwrite=True)
ds = Dataset.File.from_files(path=[(datastore, name_dataset)])
ds.register(ws, name=name_dataset, description='Dataset compensacion salarial', create_new_version=True)

# Algo del dataset que aun no se que será
training_dataset = Dataset.get_by_name(ws, name_dataset)
# Download dataset to compute node - we can also use .as_mount() if the dataset does not fit the machine
training_dataset_consumption = DatasetConsumptionConfig("training_dataset", training_dataset).as_download()


# Configuracion del pipeline
runconfig = RunConfiguration.load("runconfig.yml")

train_step = PythonScriptStep(name="train-step",
                        source_directory="./main",
                        script_name="train.py",
                        arguments=['--data-path', training_dataset_consumption],
                        inputs=[training_dataset_consumption],
                        runconfig=runconfig,
                        allow_reuse=False)

steps = [train_step]

# Creamos el objeto de pipeline y lo validamos
pipeline = Pipeline(workspace=ws, steps=steps)
pipeline.validate()

# Enviar el pipeline frente a un experimento
pipeline_run = Experiment(ws, 'pipeline-regresion').submit(pipeline)
pipeline_run.wait_for_completion()

# Publicamos el pipeline
published_pipeline = pipeline.publish('pipeline-regresion')
published_pipeline


# Publicar el pipeline como endpoint
endpoint_name = "pipeline-regresion-endpoint"
try:
   pipeline_endpoint = PipelineEndpoint.get(workspace=ws, name=endpoint_name)
   # Add new default endpoint - only works from PublishedPipeline
   pipeline_endpoint.add_default(published_pipeline)
except Exception:
    pipeline_endpoint = PipelineEndpoint.publish(workspace=ws,
                                            name=endpoint_name,
                                            pipeline=pipeline,
                                            description="New Training Pipeline Endpoint")