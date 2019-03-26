# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np

import darknet.python.darknet as dn

from os.path 				import splitext, basename
from glob					import glob
from darknet.python.darknet import detect


if __name__ == '__main__':
	
	input_dir  = sys.argv[1]
	output_dir = input_dir

	ocr_threshold = .4

	ocr_weights = '/home/wyz/workspace/wyz/git_project/darknet/lp_test/ocr-net_final.weights'
	ocr_netcfg  = '/home/wyz/workspace/wyz/git_project/darknet/lp_test/ocr-net.cfg'
	ocr_dataset = '/home/wyz/workspace/wyz/git_project/darknet/lp_test/ocr-net.data'# data/ocr/ocr-net.data

	ocr_net  = dn.load_net(ocr_netcfg, ocr_weights, 0)
	ocr_meta = dn.load_meta(ocr_dataset)

	imgs_paths = glob('%s/*lp.png' % output_dir)

	print 'Performing OCR...'

	for i,img_path in enumerate(imgs_paths):

		print '\tScanning %s' % img_path
		#basename() remove the directory, and return file name
		#splitext() 分离扩展名
		#get the name of the orginal image 
		bname = basename(splitext(img_path)[0])
		#R is [], including [ name, prob, (b.x, b.y, b.w, b.h) ]
		R = detect(ocr_net, ocr_meta, img_path ,thresh=ocr_threshold)

		if len(R):

			R.sort(key=lambda x: x[2][0])#sort by the 3rd dimension
			#cocatenation of the strings.The separator between elements is " "
			lp_str = ''.join([r[0] for r in R])


			with open('%s/%s_str.txt' % (output_dir,bname),'w') as f:
				f.write(lp_str + '\n')

			print '\t\tLP: %s' % lp_str

		else:

			print 'No characters found'
