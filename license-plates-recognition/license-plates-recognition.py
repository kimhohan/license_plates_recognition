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
# 파라미터로 디렉토리를 읽어오기 위해서

result = args["result"] # result에 파라미터에서 입력한 검출결과를 저장하기위한 디렉토리인 result를 저장한다.

f = open('result.csv','w',encoding='utf-8') # 각각의 이미지의 결과인 왼쪽위 x,y 좌표, 오른쪽아래 x,y 좌표와 파일 명을 저장하는 result.csv 파일에 쓰기위해서 파일을 연다.

for path in glob.glob(args["image"] + "/*.jpg"): # image 디렉토리에 .jpg이미지를 하나씩 읽어오기 위해서
	filename = path[path.rfind("/") + 1:] # filename에는 경로중 마지막부분인 파일명.jpg 만 저장해준다.
	filename_f, file_extension = os.path.splitext(filename) #[1]을 참고해서 filename 중에서도 result.csv파일에는 .jpg을 뗀 파일명만 저장해야함으로 .jpg 로부터 분리하여 확장자와 파일명을 따로저장
	img = cv2.imread(path)

	gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY) # gray-scale 이미지로 변환

	blurred = cv2.GaussianBlur(gray, (11,11) ,0) # GaussianBlur 적용

	thresh = cv2.adaptiveThreshold(blurred, 255,
cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11 ,1) # adaptiveThreshold  적용

	canny = cv2.Canny(thresh,50,150) # canny 적용

	kernel = np.array([[1, 1, 1], [1, 1, 1],[1, 1, 1]])
	dilated = cv2.dilate(canny, kernel) # dilated 적용

# 단계별로 적용 결과를 보기 위한 코드 필요시 주석제거
	#titles = ['gray','GaussianBlur','adaptiveThreshold','canny','dilated'] 
	#images = [gray,blurred,thresh,canny,dilated]

	#for i in range(5):
	#	plt.subplot(2,3,i+1), plt.imshow(images[i],'gray')
	#	plt.title(titles[i])
	#	plt.xticks([]),plt.yticks([])
	#plt.show() 

	(_,contours, hierarchy) = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# dilated 까지 끝난 이미지의 윤곽 검출

	copy_img =img.copy()
	cv2.drawContours(copy_img, contours, -1, (0, 255, 0), 3)
	contours_image = copy_img # 그냥 윤곽들 검출했을때 보기위한 이미지

	p_contours = [] # 가장 가능성이있는 곳을 지정하기위한 배열
	
	for c in contours:
		epsilon = 0.06*cv2.arcLength(c, True) # Contour Approximation 적용
		approx = cv2.approxPolyDP(c, epsilon, True)
		if len(approx) == 4: # 사각형 모양
			x, y, w, h = cv2.boundingRect(approx)
			if (w / float(h)) >= 1.2:
				p_contours.append(approx)
	areas = [cv2.contourArea(c) for c in p_contours]
	max_index = np.argmax(areas)
	plate_contours = p_contours[max_index]
	# [4]을 참고하여 많은 윤곽들중에서 너비/ 높이가 1.2 이상인 것들중 최대의 크기를 가지는 부분만을 plate_contours에 저장	

	copy_img =img.copy()
	cv2.drawContours(copy_img, [plate_contours], -1, (0, 255, 0), 3)
	result_image = copy_img # 최대 윤곽을 구해 적용한 이미지

	wr = csv.writer(f)
	wr.writerow(["%s        %d     %d     %d     %d"%(filename_f,plate_contours[0][0][0] ,plate_contours[0][0][1] ,plate_contours[2][0][0],plate_contours[2][0][1])]) # [2]참고하여 result.csv 파일에 위에서구한 죄표를 이용해서 왼쪽 위x,y좌표, 오른쪽 아래x,y좌표, 파일명을 입력한다.

	cv2.imwrite('%s/%s.jpg' % (result,filename_f),result_image) # [3] 참고하여 파파미터중 result디렉토리안에 result_image를 저장해준다.

# 마지막 두단계 윤곽이미지와, 결과이미지를 보여주기 위한 코드 필요시 주석제거
	#titles = ['contours_image','result'] 
	#images = [contours_image,result_image]

	#for i in range(2):
	#	plt.subplot(2,1,i+1), plt.imshow(images[i],'gray')
	#	plt.title(titles[i])
	#	plt.xticks([]),plt.yticks([])
	#plt.show() 

f.close() # result.csv파일에 전부 입력이되면 파일은 close 해줍니다.