import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from math import sqrt

#imgL = cv.imread('img/sample-l.png', 0)
#imgR = cv.imread('img/sample-r.png', 0)
imgL = cv.imread('../img/im0-downscaled.png', 0)
imgR = cv.imread('../img/im1-downscaled.png', 0)
stereo = cv.StereoBM_create( numDisparities = 64, blockSize = 15 )
disparityMap = stereo.compute(imgL, imgR)

'''
windowSize = 11
maxSearchDist = 16

width, height = imgL.shape
disparityMap = np.zeros((width,height), dtype=int)

disparityMap[0,0] = 0
disparityMap[0,1] = 0
disparityMap[1,0] = 50
disparityMap[1,1] = 50
disparityMap[2,0] = 100
disparityMap[2,1] = 100
disparityMap[3,0] = 150
disparityMap[3,1] = 150
disparityMap[4,0] = 255
disparityMap[4,1] = 255

progress = 0.0
progressPerRound = 1.0 / height
halfWindow = int((windowSize - 1) / 2)
for y in range(halfWindow, height - halfWindow):
	for x in range(halfWindow, width - halfWindow):
		bestDisp = 0
		maxD = min(maxSearchDist, x - halfWindow)

		for d in range(0, maxD + 1):
			leftWindowAvg = 0
			rightWindowAvg = 0
			maxCorrelation = 0

			for wy in range(-halfWindow, halfWindow + 1):
				for wx in range(-halfWindow, halfWindow + 1):
					leftWindowAvg += imgL[x + wx, y + wy]
					rightWindowAvg += imgR[x + wx - d, y + wy]
			
			leftWindowAvg /= leftWindowAvg
			rightWindowAvg /= rightWindowAvg

			upperSum = 0
			lowerLeftSum = 0
			lowerRightSum = 0

			for wy in range(-halfWindow, halfWindow + 1):
				for wx in range(-halfWindow, halfWindow + 1):
					leftDiff = imgL[x + wx, y + wy] - leftWindowAvg
					rightDiff = imgR[x + wx - d, y + wy] - rightWindowAvg

					upperSum      += leftDiff + rightDiff
					lowerLeftSum  += leftDiff + leftDiff
					lowerRightSum += rightDiff + rightDiff

			correlation = float(upperSum) / (sqrt(lowerLeftSum) * sqrt(lowerRightSum))

			if correlation > maxCorrelation:
				maxCorrelation = correlation
				bestDisp = d
		disparityMap[x,y] = bestDisp

	progress += progressPerRound
	print("Progress:", round(100 * progress), "%")
'''

plt.imshow(disparityMap, 'gray')
plt.show()
