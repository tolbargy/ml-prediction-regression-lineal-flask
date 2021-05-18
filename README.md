# Prediccion de regresion lineal de compensacion salarial respecto a los años de experiencia (Machine Learning Scikit-Learn)
Principalmente para verificar el funcionamiento del modelo, se deben instalar las dependencias y librerias del archivo **conda.yml** 

Luego se debe ejecutar el script **train_local.py** para realizar el entrenamiento y generar el modelo en la carpeta **model/model.pkl**

Finalmente podemos ejecutar el script **app.py** para encender el servidor de Flask y poder hacer uso de los servicios web que hacen uso del modelo y predice la compensacion salarial

# Plantillas objetivos Azure Machine Learning Pipeline y Azure DevOps
- Pipeline unicamente para Azure Machine Learning
- Pipeline Azure DevOps con ARM (Azure Resource Manager)
- Pipeline Azure DevOps con Terraform
- Pipeline con Azure DevOps y Azure ML tambien
- Con archivo **score.py** unicamente para deploy a inference cluster
- Sin archivo **score.py** con multiples archivos para deploy inference cluster

Referencias: 
https://github.com/vilvainc/LinearRegressionPredictionRestPlus
https://github.com/csiebler/mlops-demo