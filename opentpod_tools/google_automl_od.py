import argparse
import json
import os
import shutil
from PIL import Image

def prepareOutput(outputpath='result'):
	if os.path.exists(outputpath):
		shutil.rmtree(outputpath)
	os.mkdir(outputpath)
	datapath = os.path.join(outputpath, 'data')
	os.mkdir(datapath)

def obtainData(jsonpath, outputpath, busketpath):
	dic = json.loads(open(jsonpath).read())
	item2label = {}
	counter = 0
	for i in dic['categories']['label']['labels']:
		item2label[counter] = i['name']
		counter += 1

	uniqueid = 0
	info = ""
	for i in dic['items']:
		imagepath = i['image']['path']
		path_piece = imagepath.split('/')[-1]
		store_name = str(uniqueid) + path_piece
		uniqueid += 1
		height = i['image']['size'][0]
		width = i['image']['size'][1]
		dst = os.path.join(outputpath, 'data', store_name)
		shutil.copyfile(imagepath, dst)
		prefix = 'UNASSIGNED' + ',' + os.path.join(busketpath, 'data', store_name)
		for j in i['annotations']:
			lid = j['label_id']
			itemname = item2label[lid]
			xmin = float(j['bbox'][0]) / width
			ymin = float(j['bbox'][1]) / height
			xmax = float(j['bbox'][0] + j['bbox'][2]) / width
			ymax = float(j['bbox'][1] + j['bbox'][3]) / height
			iteminfo = ',' + itemname + ',' + str(xmin) + ',' + str(ymin) + ',,,' + str(xmax) + ',' + str(ymax) + ',,\n'
			info = info + prefix + iteminfo

	outputcsv = open(os.path.join(outputpath, 'info.csv'), 'w+')
	outputcsv.write(info)
	outputcsv.close()

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-b", "--busket", required=True, help="Google cloud busket name (please include gs://)")
	parser.add_argument("-p", "--path", required=True, help="Input dataset path")
	args = parser.parse_args()
	prepareOutput()
	jsonpath = os.path.join(args.path, 'dataset', 'annotations', 'default.json')
	obtainData(jsonpath, 'result', args.busket)


if __name__ == "__main__":
    main()
