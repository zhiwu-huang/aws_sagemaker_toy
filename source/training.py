import time
import sys
import sagemaker

from sagemaker.pytorch import PyTorch
from torchvision import datasets, transforms


prefix = "sagemaker/pytorch-mnist"
sagemaker_session = sagemaker.Session()

role = sys.argv[1]
bucket = sys.argv[2]
stack_name = sys.argv[3]
commit_id = sys.argv[4]
commit_id = commit_id[0:7]

timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
job_name = stack_name + "-" + commit_id + "-" + timestamp

# Getting the data
datasets.MNIST(
    "data",
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)

# Uploading the data to S3
inputs = sagemaker_session.upload_data(
    path="data", bucket=bucket, key_prefix=prefix + "/MNIST"
)
print(f"input spec (in this case, just an S3 path): {inputs}")


estimator = PyTorch(
    entry_point="code/mnist.py",
    # source_dir="code",
    role=role,
    framework_version="1.4.0",
    instance_count=2,
    # instance_type="ml.p3.2xlarge",
    instance_type="ml.m4.xlarge",
    py_version="py3",
    # use_spot_instances=True,  # Use a spot instance
    # max_run=300,  # Max training time
    # max_wait=600,  # Max training time + spot waiting time
    hyperparameters={"epochs": 10, "backend": "gloo"},
)

print(f"Training job name: {job_name}")

estimator.fit({"training": "s3://" + bucket + "/" + prefix}, job_name=job_name)

# Deploy the model
endpoint_name = f"{stack_name}-{commit_id[:7]}"
predictor = estimator.deploy(
    initial_instance_count=1, instance_type="ml.m4.xlarge", endpoint_name=endpoint_name
)
