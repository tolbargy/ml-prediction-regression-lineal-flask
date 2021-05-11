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

# 3 === Variables
aml_compute_target = "cpu-cluster"
aml_compute_target_size = "STANDARD_D2_V2"
name_dataset='compensation-dataset-train'
location_local_dataset='../dataset'
name_pipeline = 'pipeline-regresion'
name_model = 'salary-compensation'

# 4 === Creacion del compute cluster para el entrenamiento
try:
    aml_compute = AmlCompute(ws, aml_compute_target)
except ComputeTargetException:
    config = AmlCompute.provisioning_configuration(vm_size = aml_compute_target_size, min_nodes = 0, max_nodes = 1,
                                                   idle_seconds_before_scaledown=3600)
    aml_compute = ComputeTarget.create(ws, aml_compute_target, config)
    aml_compute.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

# 5 === Obtenemos el dataset y lo subimos al worskpace
datastore = ws.get_default_datastore()
datastore.upload(src_dir=location_local_dataset, target_path=name_dataset, overwrite=True)
ds = Dataset.File.from_files(path=[(datastore, name_dataset)])
ds.register(ws, name=name_dataset, description='Dataset compensacion salarial', create_new_version=True)

# 6 === Obtener el dataset subido
dataset = Dataset.get_by_name(ws, name_dataset)

# 7 === Descargar el dataset
dataset_consumption = DatasetConsumptionConfig("dataset", dataset).as_download()

# 8 === Step 1: Preparar datos
prepared_data = PipelineData("prepared_data",datastore=datastore)
prepare_runconfig = RunConfiguration.load("./1-prepare/runconfig.yml")
prepare_step = PythonScriptStep(name="Preparar datos",
                        runconfig=prepare_runconfig,
                        source_directory="./1-prepare",
                        script_name=prepare_runconfig.script,
                        arguments=['--dataset_path', dataset_consumption,
                                   '--prepared_data_path', prepared_data],
                        inputs=[dataset_consumption],
                        outputs=[prepared_data],
                        allow_reuse=False)

# 9 === Step 2: Entrenar modelo
model_path = PipelineData("model",datastore=datastore)
train_runconfig = RunConfiguration.load("./2-train/runconfig.yml")
train_step = PythonScriptStep(name="Entrenar modelo",
                        runconfig=train_runconfig,
                        source_directory="./2-train",
                        script_name=train_runconfig.script,
                        arguments=['--prepared_data_path', prepared_data,
                                   '--name_model', name_model,
                                   '--model_path', model_path],
                        inputs=[prepared_data],
                        outputs=[model_path],
                        allow_reuse=False)
 
# 10 === Step 3: Registrar modelo
register_runconfig = RunConfiguration.load("./3-register/runconfig.yml")
register_step = PythonScriptStep(name="Registrar modelo",
                        runconfig=register_runconfig,
                        source_directory="./3-register",
                        script_name=register_runconfig.script,
                        arguments=['--model_path', model_path],
                        allow_reuse=False)

# 11 === Step 4: Validar modelo
validate_runconfig = RunConfiguration.load("./4-validate/runconfig.yml")
validate_step = PythonScriptStep(name="Validar modelo",
                        runconfig=validate_runconfig,
                        source_directory="./4-validate",
                        script_name=validate_runconfig.script,
                        arguments=['--name_model', name_model],
                        allow_reuse=False)


# 12 === Configuracion del pipeline
steps = [prepare_step, train_step, register_step, validate_step]
pipeline = Pipeline(workspace=ws, steps=steps)
pipeline.validate()

# 13 === Enviar el pipeline frente a un experimento
#pipeline_run = Experiment(ws, name_pipeline).submit(pipeline)
#pipeline_run.wait_for_completion()

# 14 === Publicamos el pipeline
published_pipeline = pipeline.publish(name_pipeline)
published_pipeline