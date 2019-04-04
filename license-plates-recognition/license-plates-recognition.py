import cv2
import numpy as np
import glob
import argparse
import csv
import os
from matplotlib import pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--image", required = True,
	help = "Path to the directory of images")
ap.add_argument("-r", "--result", required = True,
	help = "Path to the directory of images")
args = vars(ap.parse_args())
# �Ķ���ͷ� ���丮�� �о���� ���ؼ�

result = args["result"] # result�� �Ķ���Ϳ��� �Է��� �������� �����ϱ����� ���丮�� result�� �����Ѵ�.

f = open('result.csv','w',encoding='utf-8') # ������ �̹����� ����� ������ x,y ��ǥ, �����ʾƷ� x,y ��ǥ�� ���� ���� �����ϴ� result.csv ���Ͽ� �������ؼ� ������ ����.

for path in glob.glob(args["image"] + "/*.jpg"): # image ���丮�� .jpg�̹����� �ϳ��� �о���� ���ؼ�
	filename = path[path.rfind("/") + 1:] # filename���� ����� �������κ��� ���ϸ�.jpg �� �������ش�.
	filename_f, file_extension = os.path.splitext(filename) #[1]�� �����ؼ� filename �߿����� result.csv���Ͽ��� .jpg�� �� ���ϸ� �����ؾ������� .jpg �κ��� �и��Ͽ� Ȯ���ڿ� ���ϸ��� ��������
	img = cv2.imread(path)

	gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY) # gray-scale �̹����� ��ȯ

	blurred = cv2.GaussianBlur(gray, (11,11) ,0) # GaussianBlur ����

	thresh = cv2.adaptiveThreshold(blurred, 255,
cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11 ,1) # adaptiveThreshold  ����

	canny = cv2.Canny(thresh,50,150) # canny ����

	kernel = np.array([[1, 1, 1], [1, 1, 1],[1, 1, 1]])
	dilated = cv2.dilate(canny, kernel) # dilated ����

# �ܰ躰�� ���� ����� ���� ���� �ڵ� �ʿ�� �ּ�����
	#titles = ['gray','GaussianBlur','adaptiveThreshold','canny','dilated'] 
	#images = [gray,blurred,thresh,canny,dilated]

	#for i in range(5):
	#	plt.subplot(2,3,i+1), plt.imshow(images[i],'gray')
	#	plt.title(titles[i])
	#	plt.xticks([]),plt.yticks([])
	#plt.show() 

	(_,contours, hierarchy) = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# dilated ���� ���� �̹����� ���� ����

	copy_img =img.copy()
	cv2.drawContours(copy_img, contours, -1, (0, 255, 0), 3)
	contours_image = copy_img # �׳� ������ ���������� �������� �̹���

	p_contours = [] # ���� ���ɼ����ִ� ���� �����ϱ����� �迭
	
	for c in contours:
		epsilon = 0.06*cv2.arcLength(c, True) # Contour Approximation ����
		approx = cv2.approxPolyDP(c, epsilon, True)
		if len(approx) == 4: # �簢�� ���
			x, y, w, h = cv2.boundingRect(approx)
			if (w / float(h)) >= 1.2:
				p_contours.append(approx)
	areas = [cv2.contourArea(c) for c in p_contours]
	max_index = np.argmax(areas)
	plate_contours = p_contours[max_index]
	# [4]�� �����Ͽ� ���� �������߿��� �ʺ�/ ���̰� 1.2 �̻��� �͵��� �ִ��� ũ�⸦ ������ �κи��� plate_contours�� ����	

	copy_img =img.copy()
	cv2.drawContours(copy_img, [plate_contours], -1, (0, 255, 0), 3)
	result_image = copy_img # �ִ� ������ ���� ������ �̹���

	wr = csv.writer(f)
	wr.writerow(["%s        %d     %d     %d     %d"%(filename_f,plate_contours[0][0][0] ,plate_contours[0][0][1] ,plate_contours[2][0][0],plate_contours[2][0][1])]) # [2]�����Ͽ� result.csv ���Ͽ� ���������� ��ǥ�� �̿��ؼ� ���� ��x,y��ǥ, ������ �Ʒ�x,y��ǥ, ���ϸ��� �Է��Ѵ�.

	cv2.imwrite('%s/%s.jpg' % (result,filename_f),result_image) # [3] �����Ͽ� ���Ĺ����� result���丮�ȿ� result_image�� �������ش�.

# ������ �δܰ� �����̹�����, ����̹����� �����ֱ� ���� �ڵ� �ʿ�� �ּ�����
	#titles = ['contours_image','result'] 
	#images = [contours_image,result_image]

	#for i in range(2):
	#	plt.subplot(2,1,i+1), plt.imshow(images[i],'gray')
	#	plt.title(titles[i])
	#	plt.xticks([]),plt.yticks([])
	#plt.show() 

f.close() # result.csv���Ͽ� ���� �Է��̵Ǹ� ������ close ���ݴϴ�.