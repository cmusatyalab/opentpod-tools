# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google.cloud import storage
from google.cloud import automl

def create_bucket(bucket_name):
    """Creates a new bucket."""
    # bucket_name = "your-new-bucket-name"
    storage_client = storage.Client()
    bucket = storage_client.create_bucket(bucket_name)
    print("Bucket {} created".format(bucket.name))

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    # logger.info(
    #     "File {} uploaded to {}.".format(
    #         source_file_name, destination_blob_name
    #     )
    # )

def create_dataset(project_id, display_name):
    client = automl.AutoMlClient()

    # A resource that represents Google Cloud Platform location.
    project_location = f"projects/{project_id}/locations/us-central1"
    metadata = automl.ImageObjectDetectionDatasetMetadata()
    dataset = automl.Dataset(
        display_name=display_name,
        image_object_detection_dataset_metadata=metadata,
    )

    # Create a dataset with the dataset metadata in the region.
    response = client.create_dataset(parent=project_location, dataset=dataset)

    created_dataset = response.result()

    # Display the dataset information
    print("Dataset name: {}".format(created_dataset.name))
    print("Dataset id: {}".format(created_dataset.name.split("/")[-1]))

    return created_dataset.name.split("/")[-1]

def import_data(project_id, dataset_id, path):
    client = automl.AutoMlClient()
    # Get the full path of the dataset.
    dataset_full_id = client.dataset_path(
        project_id, "us-central1", dataset_id
    )
    # Get the multiple Google Cloud Storage URIs
    input_uris = path.split(",")
    gcs_source = automl.GcsSource(input_uris=input_uris)
    input_config = automl.InputConfig(gcs_source=gcs_source)
    # Import data from the input URI
    response = client.import_data(name=dataset_full_id, input_config=input_config)
    print("Processing import...")
    print("Data imported. {}".format(response.result()))

def train_model(project_id, dataset_id, display_name):
    client = automl.AutoMlClient()
    # A resource that represents Google Cloud Platform location.
    project_location = f"projects/{project_id}/locations/us-central1"
    # Leave model unset to use the default base model provided by Google
    # train_budget_milli_node_hours: The actual train_cost will be equal or
    # less than this value.
    # https://cloud.google.com/automl/docs/reference/rpc/google.cloud.automl.v1#imageobjectdetectionmodelmetadata
    metadata = automl.ImageObjectDetectionModelMetadata(
        train_budget_milli_node_hours=24000
    )
    model = automl.Model(
        display_name=display_name,
        dataset_id=dataset_id,
        image_object_detection_model_metadata=metadata,
    )
    # Create a model with the model metadata in the region.
    response = client.create_model(parent=project_location, model=model)
    print("Training operation name: {}".format(response.operation.name))
    print("Training started...")

def create_dataset_class(project_id, display_name):
    client = automl.AutoMlClient()

    # A resource that represents Google Cloud Platform location.
    project_location = f"projects/{project_id}/locations/us-central1"
    metadata = automl.ImageClassificationDatasetMetadata(
    classification_type=automl.ClassificationType.MULTICLASS
    )
    dataset = automl.Dataset(
        display_name=display_name,
        image_classification_dataset_metadata=metadata,
    )

    # Create a dataset with the dataset metadata in the region.
    response = client.create_dataset(parent=project_location, dataset=dataset)

    created_dataset = response.result()

    # Display the dataset information
    print("Dataset name: {}".format(created_dataset.name))
    print("Dataset id: {}".format(created_dataset.name.split("/")[-1]))
    return created_dataset.name.split("/")[-1]

def train_model_class(project_id, dataset_id, display_name):
    client = automl.AutoMlClient()
    # A resource that represents Google Cloud Platform location.
    project_location = f"projects/{project_id}/locations/us-central1"
    # Leave model unset to use the default base model provided by Google
    # train_budget_milli_node_hours: The actual train_cost will be equal or
    # less than this value.
    # https://cloud.google.com/automl/docs/reference/rpc/google.cloud.automl.v1#imageobjectdetectionmodelmetadata
    metadata = automl.ImageClassificationModelMetadata(
        train_budget_milli_node_hours=24000
    )
    model = automl.Model(
        display_name=display_name,
        dataset_id=dataset_id,
        image_classification_model_metadata=metadata,
    )
    # Create a model with the model metadata in the region.
    response = client.create_model(parent=project_location, model=model)
    print("Training operation name: {}".format(response.operation.name))
    print("Training started...")

    