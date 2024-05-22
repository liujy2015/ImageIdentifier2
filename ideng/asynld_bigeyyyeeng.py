#! /usr/bin/env python3  
import requests
import shutil
import os
from os import walk
from os import listdir
from os.path import isfile, isdir, join
from bs4 import BeautifulSoup
from PIL import Image
import re
import time
from datetime import datetime,date,timedelta
import cv2
import numpy as np
import json
import traceback


# for testing
#linkdir='../readytoanalyse'
#ansdir='../tanalysed'
#backupfile='../bblink_bk'
#IPcamimg='tipcamimg'
#IPcamimg_bk='bblink_bk'

linkdir='../svtmpdir1'
ansdir='../analysed'
backupfile='../tmpfolder'
IPcamimg='ipcamimg'
IPcamimg_bk='tmpfolder'

tmpfolder='/home/bigeye/tmpfolder/'
maybefolder='../maybegarbage'
maybegarbage='maybegarbage'


# write T_0 file to analysed flag  
#1 write T_0 to file.   0 don't write to file, instead of creating file 
t0flagwriteflag = 1

net = cv2.dnn.readNetFromDarknet("yolov4_pc.cfg","yolov4_pc.weights")
#net = cv2.dnn.readNetFromDarknet("yolov4_peoldn.cfg","yolov4_5000peoldn.weights")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

classes = [line.strip() for line in open("coco.names")]
#classes = [line.strip() for line in open("obj_final_peoldn.names")]
colors = [(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0)]
# blur level smaller factor then bluring larger 
factor = 3.0 

# bflag :1  blur whole image. 2 blur found boxes
def yolo_detect(frame, bflag, mlist):
	# forward propogation
	t1 = datetime.now()
	img = cv2.resize(frame, None, fx=1., fy=1.)
	height, width, channels = img.shape # 0.00392
	# set the block area
	orgimg = img.copy()
	if isinstance(mlist,list):
		print('mlist is a list:',mlist)
		img = putonmask(img,mlist,0)
	t2 = datetime.now()
	# 320,416,608
	blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
	net.setInput(blob)
	outs = net.forward(output_layers)
	t3 = datetime.now()
	# get detection boxes
	class_ids = []
	confidences = []
	boxes = []
	ifound = []
	#print(' type outs.len :',len(outs))
	counter1 = 0
	for out in outs:
		counter1 +=1
		#print(' len out:',len(out))
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			# 20201108  only people count
			if class_id > 0 :
				continue
			confidence = scores[class_id]
			#print('whynotthis objconfidence:',detection[4], ' confidence=',confidence)
			if confidence > 0.1:
				#print('counter1:',counter1,'class_id:',class_id)                
				# Object detected
				center_x = int(detection[0] * width)
				center_y = int(detection[1] * height)
				w = int(detection[2] * width)
				h = int(detection[3] * height)
				
				# Rectangle coordinates
				x = int(center_x - w / 2)
				y = int(center_y - h / 2)
				if x < 0 :
					x = 0
				if y < 0 :
					y = 0
				boxes.append([x, y, w, h])
				#print('boxes.append:x,y,w,h:',[x, y, w, h])
				confidences.append(float(confidence))
				class_ids.append(class_id)
	#print('counter1=',counter1,'\n--boxes:',boxes,'\n--confidences:', confidences,'\n-----class_ids:', class_ids)
	t4 = datetime.now()
	#my_detects['boxes']=str(len(boxes))
	#my_detects['class_ids']=str(len(class_ids))
    # draw boxes   boxes,confidences,conf_threshold , nms_threshold
	indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.3)
	font = cv2.FONT_HERSHEY_PLAIN
	counter0 = 0
	counter1 = 0
	counter2 = 0
	# use orgimg to draw rectangle block or blur
	
	if bflag == 2 or bflag == 3:
		orgimg = allbluring(orgimg,100.0)
		print(' bflag:',bflag,' call allbluring')
	if isinstance(mlist,list):
		print('mlist is a list call putonmask:',mlist)
		orgimg = putonmask(orgimg,mlist,1)
	
	#print('indexes:',indexes)
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			# 20201108 count people only
			# if class_ids[i] > 2 :
			if class_ids[i] > 0 :
				continue
			label = str(classes[class_ids[i]])
			color = colors[0] # just fixed the color jyliu
			if class_ids[i] == 0:
				counter0 += 1
			if class_ids[i] == 1:
				counter1 += 1
				color = colors[1]
			if class_ids[i] == 2:
				counter2 += 1
				color = colors[2]
			#color = colors[class_ids[i]]
			#color = colors[0] # just fixed the color jyliu
			cv2.rectangle(orgimg, (x, y), (x + w, y + h), color, 2)
			#print('boxes i:',i,boxes[i])
			#print(' x,y:',(x,y),' x+w,y+h:',(x + w, y + h) ,'w,h:',(w,h))
			if bflag ==1 or bflag == 3 :
				orgimg = mybluring(orgimg,factor,(x, y), (x + w, y + h) )
			##cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
			#cv2.putText(orgimg, label, ( int(xx/8),int( yy - yy/8) ), font, 2, color, 3)
			cv2.putText(orgimg, label, (x, y + 30), font, 2, color, 3)
			cv2.putText(orgimg, "%.4f"%confidences[i], (x, y + 56), font, 2, color, 2)
			print('x:',x,'y:',y,' confidences[i]:',confidences[i])
			#ifound.append(label)
			#if label not in ifound :
			#  ifound.append(label)
        
	my_detects = {} 
	my_detects['T'] = str(counter0+counter1+counter2)		
	my_detects['P']=str(counter0)
	my_detects['S']=str(counter1)
	my_detects['L']=str(counter2)
	my_detects['W']='0'
	#print('my_detects a:',my_detects)
	if counter2 > 0 :
		my_detects['W']='0'  # this version warning flag nerver be on 
		print('my_detects b:',my_detects)
	print('my_detects c:',my_detects)
	yy,xx,zz = orgimg.shape
	yh = int(yy*0.5)
	xw = int(xx*0.5)
	if yh%2 == 1 :
		yh -=1
	if xw%2 == 1 :
		xw -=1
	cv2.putText(orgimg,my_detects['T'],( int(xx/8),int( yy - yy/8) ),font,3,colors[1],3)
	#cv2.putText(orgimg,my_detects['T'],(150,150),font,3,colors[1],3)
	#orgimg = cv2.resize(orgimg, None, fx=0.5, fy=0.5)
	orgimg = cv2.resize(orgimg, (xw,yh) )
	t5 = datetime.now()
	print('t2-t1=',t2 - t1,' t3-t2=',t3 - t2, ' t4-t3=',t4 - t3,' t5-t4=',t5 - t4) 
	return orgimg , my_detects

def putonmask(img,mlist,flag):
	#print(' in putonmask flag:',flag)
	if flag == 0 :
		#print(' flag==0 ')
		for (x,y,x1,y1) in mlist:
			img[y:y1,x:x1] = 0   # 0 black, 255 white
			#print('mask : x,y,x1,y1', x,y,x1,y1)
	if flag == 1 :
		for (x,y,x1,y1) in mlist:
			cv2.rectangle(img, (x, y), (x1, y1), (0,0,0) , 1)
			#print('draw rectangle:',x,y,x1,y1)
	
	return img

def allbluring(image,factor):
	y,x,channel = image.shape
	kw = int(x/factor)
	kh = int(y/factor)
	if kw %2 == 0:
		kw -= 1
	if kh %2 == 0:
		kh -= 1
	blur = cv2.GaussianBlur(image,(kw,kh),0)
	return blur

def mybluring(image,factor,topleft, bottomright):
	#print('in mybluring toleft:', topleft, '  bottomright:',bottomright)
	x,y = topleft[0], topleft[1]
	if x < 0 :
		x=0
	if y < 0 :
		y=0
	w,h = bottomright[0] - topleft[0], bottomright[1] - topleft[1]
	#print('x:',x,' y:',y,' w:',w,' h:',h )
	#region of interest
	ROI = image[y:y+h,x:x+w]
	kw = int(w/factor)
	kh = int(h/factor)
	#print('kw:',kw," kh:",kh)
	if kw %2 == 0:
		kw -= 1
	if kh %2 == 0:
		kh -= 1
	#print('final kw',kw,' final kh',kh )
	blur = cv2.GaussianBlur(ROI,(kw,kh),0)
	# insert ROI back into image
	image[y:y+h,x:x+w] = blur
	
	return image

def writetojson(respresult,mm):
	with open(respresult,'w') as json_file:
		json.dump(mm,json_file)
	
	return

def preparefile(root, file,myfound):
	camid = os.path.split(root)[-1]
	print('camid:',camid)
	#print('fullpath:',fullpath,'---pathroot: [', os.path.splitext(root),']','split[',os.path.split(root),']')
	extension = os.path.splitext(file)[-1]
	#filename = camid + '_' + f.split('_')[2]+ '_' + extension
	filename = file
	print('  filename:[',filename,']',' extension:[',extension,']')
	num = '_'.join('_'.join((key,val)) for (key,val) in myfound.items() )
    #20201021 result = camid+'_'+filename+'_'+num+'_'+extension
    #result = camid+'_'+filename+extension+'_'+ num +'_'+ extension
	result = camid+'_' + filename +'_' + num + '_' + extension
	analysedfull = os.path.join(ansdir,result)
	#bakfullpath = os.path.join(bakpath, filename)
	#return analysedfull , bakfullpath
	return analysedfull 

def readconf( bmfile ):
	dd = {}
	with open(bmfile,'r') as fd:
		for line in fd:
			if line[0] == '#':
				continue
			mlist=''
			(confcamid,val,mlist) = line.split()
			print(confcamid,val,mlist)
			if mlist != "0" and mlist != '':
				print('mlist is not 0 and empty ')
				mlist=[[int(a) for a in i.split(',')] for i in mlist.split('|')]
			else :
				print(' mlist is 0 or empty.', mlist)
			dd[confcamid] = (int(val),mlist)	
	
	return dd

# smallfiles control
smallfiles = {}
#blur and mask control
bmconfig = {}
bmconfigfile = './conf/allmask.conf'
bmflag = 0
bmcounter = 0

if os.path.exists(bmconfigfile):
	filesize = os.stat(bmconfigfile).st_size
	if filesize > 5 :
		bmflag=1
		bmconfig = readconf(bmconfigfile)

print('bmflag: ',bmflag, ' all bmconfig:',bmconfig)

print('beginning -------- ',datetime.now )
while True :

	#print("....sleep 1.5 seconds.")
	if not os.path.exists(linkdir):
		print(linkdir,' not exist')
		exit(1)
	if not os.path.exists(ansdir):
		os.mkdir(ansdir)
		print(ansdir, ' just has been created ')
	filelist = os.listdir(linkdir)    
	
	# should be marked
	camdirs = listdir(linkdir)
	for f in camdirs:
		somecamid = join(backupfile,f)
		print('somecamid',somecamid)
		if not os.path.isfile(f) :
			if not os.path.exists(somecamid) :
				os.mkdir(somecamid)
				print('i create this folder ',somecamid)
				continue
	my_detects={}
	counter0 = 0
	counter1 = 0
	counter2 = 0	
	
	starttime = datetime.now()
	startstr = starttime.strftime('_%Y%m%d_%H%M%S')
	print('--------------starttime :',startstr)
	
	for root,dirs,files  in walk(linkdir):
		#print('root:',root,' files:',files,'  dirs:',dirs)
		counterf = 0
		for f in files:
			print('---------------------------- f:',f)
			if not f.endswith(".jpg") :
				print(' f is not jpg',f )
				continue
			counterf +=1
			if counterf > 500 :
				print('counterf more than 500, go to other cam folder ') 
				break
			fullpath = join(root,f)
			try:
				filesize = os.stat(fullpath).st_size
			except:
				print('filesize os.stat.st_size some error happened ')
				continue
			print(' filesize:',filesize)
			if filesize < 100:
				print( fullpath, filesize, ' this file size should be larger than 100k so continue ')
				if f not in smallfiles :
					smallfiles[f] = filesize
					print(f,' appended wait for next time smallfiles:',smallfiles,filesize)
					continue
				else:
					if smallfiles[f] == filesize :
						print(f, ' is fixed, so go to process ',smallfiles,' and delete in smallfiles')
						del smallfiles[f]
						if filesize == 0 :
							print(' this file is strange ', fullpath, ' so i remove it to /tmp')
							shutil.move(fullpath,tmpfolder)
							continue
					else :
						print(f, ' is in smallfiles, and filesize is different. smallfiles:',smallfiles)
						smallfiles[f]=filesize
						continue
			else:
				if f in smallfiles:
					del smallfiles[f]
			
			#bakimgpath = fullpath.replace(IPcamimg,IPcamimg_bk) 
			starttimecv2 = datetime.now()
			img = cv2.imread(fullpath)
			endtimecv2 = datetime.now()
			try:
				if bmflag == 1 :
					camid = os.path.split(root)[-1]
					if camid in bmconfig :
						blurflag = bmconfig[camid][0]
						masklist = bmconfig[camid][1]
						print(camid ,' has been found in bmconfig:',bmconfig[camid])
					else :
						blurflag = 0
						masklist = '0'
				img2, rtdetect = yolo_detect(img,blurflag,masklist)
			except :
				#print('Error happened in yolo_detect , move this file to other place')
				traceback.print_exc()
				camid = os.path.split(root)[-1]
				garbagefullpath = join( maybefolder,f+startstr+camid+'sv1.jpg')
				#print(error,'and remove to ',garbagefullpath)
				print('yolo_detect error move fullpath:', fullpath ,' garbagefull:', garbagefullpath)
				shutil.move(fullpath,garbagefullpath)
				continue
				
			endtimeyo = datetime.now()
			#analysedfile , bakimgpath = preparefile(root,f,rtdetect)
			analysedfile  = preparefile(root,f,rtdetect)
			
			print('iiiiiiiiiii analysedfile:',analysedfile)
			#print('iiiiiiiiiii bakimgpath:',bakimgpath)
			try:
				if rtdetect['T'] == '0' and t0flagwriteflag == 0 :
					with  open(analysedfile,'a') as bb:	# just create a empty file 
						print('this file Total == 0 ',analysedfile)
				else:
					cv2.imwrite(analysedfile,img2)
				#cv2.imwrite(analysedfile,img2)
			except OSError as error:
				print(error,' and change name to ',analysedfile+startstr)
				cv2.imwrite(analysedfile+startstr+'.jpg',img2)
			#writetojson(respresult,mm)
			endcv2write = datetime.now()
			print('cv2readt=',endtimecv2 - starttimecv2,' y_detect time=',endtimeyo - endtimecv2,'cv2writet=',endcv2write - endtimeyo )
			print('analysedfile:',analysedfile)
			try:
				#os.rename(fullpath,bakimgpath)
				#shutil.move(fullpath,bakimgpath)
				os.remove(fullpath)
			except OSError as error :
				#print(error,'and remove to ',bakimgpath+startstr)
				garbagefullpath = join( maybefolder,f+startstr)
				print(error,'and remove to ',garbagefullpath)
				#os.rename(fullpath,bakimgpath+startstr)
				shutil.move(fullpath,garbagefullpath)
	
	endtime = datetime.now()
	endtimestr=endtime.strftime('_%Y%m%d_%H%M%S')
	print('endtime-------------------:',endtimestr)
	timeinterval = endtime - starttime
	print('endtime-starttime',timeinterval, type(timeinterval))
	if timeinterval <  timedelta(seconds=1) :
		print('go to sleep for a while')
		time.sleep(4.2)
		
	bmcounter +=1
	if bmcounter > 700 :
		bmcounter = 0
		if os.path.exists(bmconfigfile):
			filesize = os.stat(bmconfigfile).st_size
			if filesize > 5 :
				bmflag=1
				bmconfig = readconf(bmconfigfile)
		print('reread the bmconfigfile:',bmconfig)
	





