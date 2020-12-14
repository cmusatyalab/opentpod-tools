import os
import shutil
import argparse
import time
from .google_cloud_helper import upload_blob

def prepareOutput(outputpath):
	if os.path.exists(outputpath):
		shutil.rmtree(outputpath)
	os.mkdir(outputpath)
	# datapath = os.path.join(outputpath, 'data')
	# os.mkdir(datapath)

def setInfo(imgpath, item, name, outputpath, bucketpath):
	store_name = item + name
	# dst = os.path.join(outputpath, 'data', store_name)
	# shutil.copyfile(imgpath, dst)
	prefix = 'UNASSIGNED' + ',' + 'gs://' + os.path.join(bucketpath, store_name)
	upload_blob(bucketpath, imgpath, store_name)
	return prefix + ',' + item + '\n'

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-b", "--bucket", required=True, help="Google cloud bucket name (please DO NOT include gs://)")
	parser.add_argument("-p", "--path", required=True, help="Input dataset path")
	args = parser.parse_args()
	temppath = 'temp' + str(int(time.time()))
	prepareOutput(temppath)
	info = ""
	for subdir, dirs, files in os.walk(args.path):
		for file in files:
			if file.endswith('.jpg') or file.endswith('.jpeg'):
				img = os.path.join(subdir, file)
				# print(subdir)
				# print(subdir.split('/')[-1])
				# print(img)
				info += setInfo(img, subdir.split('/')[-1], file, 'result', args.bucket)
				
	outputcsv = open(os.path.join(temppath, 'info.csv'), 'w+')
	outputcsv.write(info)
	outputcsv.close()
	upload_blob(args.bucket, os.path.join(temppath, 'info.csv'), 'info.csv')
	print('CSV file path: ')
	print('gs://' + os.path.join(args.bucket, 'info.csv'))
	shutil.rmtree(temppath)

if __name__ == "__main__":
    main()
