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

# 1 === Verificamos version del Azure ML SDK
print("Azure ML SDK version:", azureml.core.VERSION)

# 2 === Cargamos el workspace de AML
ws = Workspace.from_config()
print(f'WS name: {ws.name}\nRegion: {ws.location}\nSubscription id: {ws.subscription_id}\nResource group: {ws.resource_group}')

# 3 === Creacion del compute cluster si es que no existe
aml_compute_target = "cpu-cluster"
try:
    aml_compute = AmlCompute(ws, aml_compute_target)
except ComputeTargetException:
    config = AmlCompute.provisioning_configuration(vm_size = "STANDARD_D2_V2", min_nodes = 0, max_nodes = 1,
                                                   idle_seconds_before_scaledown=3600)
    aml_compute = ComputeTarget.create(ws, aml_compute_target, config)
    aml_compute.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

# 4 === Obtenemos el dataset y lo subimos al worskpace
name_dataset='compensation-dataset-train'
datastore = ws.get_default_datastore()
datastore.upload(src_dir='../dataset', target_path=name_dataset, overwrite=True)
ds = Dataset.File.from_files(path=[(datastore, name_dataset)])
ds.register(ws, name=name_dataset, description='Dataset compensacion salarial', create_new_version=True)

# 5 === Algo del dataset que aun no se que será
training_dataset = Dataset.get_by_name(ws, name_dataset)

# 6 === Download dataset to compute node - we can also use .as_mount() if the dataset does not fit the machine
training_dataset_consumption = DatasetConsumptionConfig("training_dataset", training_dataset).as_download()

# 7 === Step 1: Preparar datos
prepared_data = PipelineData("prepared_data",datastore=datastore)
prepare_runconfig = RunConfiguration.load("./1-prepare/runconfig.yml")
prepare_step = PythonScriptStep(name="Preparar datos",
                        runconfig=prepare_runconfig,
                        source_directory="./1-prepare",
                        script_name=prepare_runconfig.script,
                        arguments=['--data-input-path', training_dataset_consumption,
                                   '--data-output-path', prepared_data],
                        inputs=[training_dataset_consumption],
                        outputs=[prepared_data],
                        allow_reuse=False)

# 8 === Step 2: Entrenar modelo
train_runconfig = RunConfiguration.load("./2-train/runconfig.yml")
train_step = PythonScriptStep(name="Entrenar modelo",
                        runconfig=train_runconfig,
                        source_directory="./2-train",
                        script_name=train_runconfig.script,
                        arguments=['--data-input-path', prepared_data],
                        inputs=[prepared_data],
                        allow_reuse=False)
                        
# 9 === Step 3: Empaquetar modelo


# 10 === Step 4: Registrar modelo


# 11 === Step 5: Validar modelo


# 12 === Configuracion del pipeline
steps = [prepare_step, train_step]
name = 'pipeline-regresion'
pipeline = Pipeline(workspace=ws, steps=steps)
pipeline.validate()

# 13 === Enviar el pipeline frente a un experimento
pipeline_run = Experiment(ws, name).submit(pipeline)
pipeline_run.wait_for_completion()

# 14 === Publicamos el pipeline
published_pipeline = pipeline.publish(name)
published_pipeline