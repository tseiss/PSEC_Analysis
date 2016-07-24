import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import Group
import os, sys
import timeit

def AreaOfIntersection(a, xs, ys, R, xc, yc, numPoints = 100):
	xsArray = np.linspace(xs, xs+a, numPoints)
	ysArray = np.linspace(ys, ys+a, numPoints)
	hitCounter = 0
	Rsq = R**2
	xr = xs-xc
	yr = ys-yc
	if xr+a < -R or xr > R or yr > R or yr+a < -R:
		return 0.0
	elif xr**2+yr**2 <= Rsq and (xr+a)**2+yr**2 <= Rsq and xr**2+(yr+a)**2 <= Rsq and (xr+a)**2+(yr+a)**2 <= Rsq:
		return a**2
	elif xr <= -R and xr+a >= R and yr < -R and yr+a > R:
		return 3.1415*R**2
	for x in xsArray:
		for y in ysArray:
			if (x-xc)**2+(y-yc)**2 <= Rsq:
				hitCounter += 1
			elif (y-yc) > R:
				break
		if x-xc > R:
			break
	return a**2 * float(hitCounter)/float(numPoints**2)

def AreaOfIntersection2(a, xs, ys, R, xc, yc, numPoints = 100):
	xsArray = np.linspace(xs, xs+a, numPoints)
	ysArray = np.linspace(ys, ys+a, numPoints)
	hitCounter = 0
	Rsq = R**2
	for x in xsArray:
		for y in ysArray:
			if (x-xc)**2+(y-yc)**2 <= Rsq:
				hitCounter += 1
	return a**2 * float(hitCounter)/float(numPoints**2)


def getBestFitFunction(a, xs, ys):
	def function(xc, yc, R, C):
		if type(xc) is np.float64:
			return C*AreaOfIntersection(a, xs, ys, R, xc, yc)
		else:
			return [C*AreaOfIntersection(a, xs, ys, R, x, yc) for x in xc]
	return function

def readPositionScan(filename):
	f = open(filename)
	amps = []
	errNeg = []
	errPos = []
	channels = []
	xArray = []
	xArrayFilled = False
	for line in f:
		words = line.split()
		if len(words) == 1:
			channels.append(int(words[0]))
			amps.append([])
			errNeg.append([])
			errPos.append([])
			if len(xArray) > 0:
				xArrayFilled = True
		else:
			if not xArrayFilled:
				xArray.append(float(words[0]))
			amps[-1].append(float(words[1]))
			errNeg[-1].append(float(words[2]))
			errPos[-1].append(float(words[3]))
	f.close()
	return xArray, amps, errNeg, errPos

filename = "PosScan.txt"
xArray, amps,_,__ = readPositionScan(filename)
xArray = xArray[:-5]
amps1 = amps[0][:-5]
amps2 = amps[1][:-5]
xs, ys = 5.0-6.35, -6.35
a = 12.7
bestFitFunc = getBestFitFunction(a, xs, ys)
for RGuess in np.linspace(10, 15, 100):
	yc, R, C = scipy.optimize.curve_fit(bestFitFunc, xArray, amps1, p0 = [0.0, RGuess, 1.0])[0]
	print "yc =", round(yc, 3)
	print "R =", round(R, 3)
	print "C =", round(C, 3)
	xcArray = np.linspace(0, 30, 1000)
	fitAmps = [bestFitFunc(xc, yc, R, C) for xc in xcArray]
	plt.plot(xcArray, fitAmps, 'k-')
plt.plot(xArray, amps1, 'ro')
plt.show()

"""
xs, ys = (0, 0)
a = 1.0
R = 1.0
xc, yc = 0.5, 0.5
print AreaOfIntersection(a, xs, ys, R, xc, yc, numPoints = 100)
bestFitFunc = getBestFitFunction(a, xs, ys)
xcArray = np.linspace(-1, 2, 1000)
ampArray = [bestFitFunc(x, yc, R, 1) for x in xcArray]
plt.plot(xcArray, ampArray)
plt.show()
print bestFitFunc(xc, yc, R, 1)
print "GO!"
AreaArray = []
AreaArray2 = []
for R in np.linspace(0, 2, 1000):
	AreaArray.append(AreaOfIntersection(a, xs, ys, R, xc, yc))
print "GO!"
for R in np.linspace(0, 2, 1000):
	AreaArray2.append(AreaOfIntersection2(a, xs, ys, R, xc, yc))
print [AreaArray[i]-AreaArray2[i] for i in range(0, len(AreaArray))]
"""

"""
def AreaOfIntersectionQuick(a, xs, ys, R, xc, yc):
	vertices = [[xs, ys], [xs+a, ys], [xs, ys+a], [xs+a, ys+a]]
	numVertices = 0
	for vertex in vertices:
		if (xc-vertex[0])**2 + (yc-vertex[1])**2 <= R**2: numVertices += 1
	if numVertices == 0:
		if xs <= xc-R:
			if xs + a <= xc-R:
				return 0.0
			elif xc-R <= xs + a <= xc:
				if ys < yc and ys+a > yc:
					pass
				else:
					return 0.0
			elif xc <= xs + a <= xc+R:
				if ys <= yc and ys+a >= yc:
					pass
				else:
					return 0.0
			elif xs+a >= xc+R:
				pass
			else:
				print "Missed a case! (1)"
		elif xs <= xc and xs > xc - R:
			pass
		elif xs > xc and xs < xc + R:
			pass
		elif xs > xc + R:
			return 0.0
		else:
			print "Missed a case! (2)"
	elif numVertices == 1:
		pass
	elif numVertices == 2:
		pass
	elif numVertices == 3:
		pass
	elif numVertices == 4:
		return a**2
	else:
		print "Finding number of vertices of square in circle failed."
		sys.exit()
"""
