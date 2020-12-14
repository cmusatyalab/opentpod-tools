import argparse
from .google_cloud_helper import create_dataset, import_data, train_model

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-p", "--project", required=True, help="Google cloud project id")
	parser.add_argument("-n", "--name", required=True, help="Model name")
	parser.add_argument("-c", "--csv", required=True, help="CSV file path")

	args = parser.parse_args()
	datasetid = create_dataset(args.project, args.name)
	import_data(args.project, datasetid, args.csv)
	train_model(args.project, datasetid, args.name)

if __name__ == "__main__":
    main()
