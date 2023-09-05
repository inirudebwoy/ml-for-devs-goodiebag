import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    CodeConfiguration,
    Environment,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    Model,
)
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

# enter details of your Azure Machine Learning workspace
subscription_id = os.environ.get(
    "SUBSCRIPTION_ID", "11111111-1111-1111-1111-111111111111"
)
resource_group = os.environ.get("RESOURCE_GROUP", "my-ml-rg")
workspace = os.environ.get("WORKSPACE", "my-ml-workspace")

# get a handle to the workspace
ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace
)

# # Define an endpoint name
endpoint_name = "ml-presentation-endpoint-netguru"


# create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name, description="this is a sample endpoint", auth_mode="key"
)

model = Model(
    path="../models/vit-base-patch16-224", description="this is a sample model"
)
env = Environment(
    conda_file="./environment/conda.yaml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
)

blue_deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=model,
    environment=env,
    code_configuration=CodeConfiguration(
        code="./onlinescoring", scoring_script="score.py"
    ),
    instance_type="Standard_DS3_v2",
    instance_count=1,
)

ml_client.online_endpoints.begin_create_or_update(endpoint, local=True)
ml_client.online_deployments.begin_create_or_update(
    deployment=blue_deployment, local=True
)


print(
    ml_client.online_endpoints.invoke(
        endpoint_name=endpoint_name,
        request_file="./image-request.json",
        local=True,
    )
)
print(
    ml_client.online_deployments.get_logs(
        name="blue", endpoint_name=endpoint_name, lines=250, local=True
    )
)
