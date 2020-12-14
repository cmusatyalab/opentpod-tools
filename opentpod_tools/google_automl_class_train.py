import argparse
from .google_cloud_helper import create_dataset_class, import_data, train_model_class

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-p", "--project", required=True, help="Google cloud project id")
	parser.add_argument("-n", "--name", required=True, help="Model name")
	parser.add_argument("-c", "--csv", required=True, help="CSV file path")

	args = parser.parse_args()
	datasetid = create_dataset_class(args.project, args.name)
	import_data(args.project, datasetid, args.csv)
	train_model_class(args.project, datasetid, args.name)
# gs://bucket_1213_2/info.csv
if __name__ == "__main__":
    main()
