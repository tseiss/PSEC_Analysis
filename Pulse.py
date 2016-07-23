import numpy as np
import matplotlib.pyplot as plt
import sys

class Pulse:
	#Takes array of numbers that is the waveform, and the time step between each number
	#Give timestep in nanoseconds and waveform in millivolts
	def __init__(self, waveform, timestep):
		self.waveform = []
		self.timestep = timestep
		for voltage in waveform:
			self.waveform.append(voltage)
		self.integral = None
		self.rawWaveform = [x for x in self.waveform]
		
	#Returns the waveform as a string
	def __str__(self):
		return str(self.waveform) + "\n"
	
	#Adds two pulses together. Checks for same number of samples. If one is zero, returns other pulse
	def __add__(self, pulse2):
		if self.getNumSamples() == 0:
			return pulse2
		elif pulse2.getNumSamples() == 0:
			return self
		elif self.getNumSamples() != pulse2.getNumSamples():
			print "Cannot add two pulses with different numbers of points."
			print "Pulse1 number of samples:", self.getNumSamples()
			print "Pulse2 number of samples:", pulse2.getNumSamples()
			sys.exit()
		else:
			waveformSum = [self.waveform[i]+pulse2.waveform[i] for i in range(0, self.getNumSamples())]
			return Pulse(waveformSum, self.timestep)
				
		
	#Multiplies a waveform by the number num. Allows for scaling. Num must be second
	def __mul__(self, num):
		newWaveform = [x*num for x in self.waveform]
		return Pulse(newWaveform, self.timestep)
	
	#Returns the number of samples in a pulse
	def getNumSamples(self):
		return len(self.waveform)
	
	#Returns waveform array of the pulse
	def getWaveform(self):
		return self.waveform
	
	#Does the integral of the pulse and returns the value
	def integrate(self):
		if self.integral:
			return self.integral
		else:
			tot = 0
			for samp in self.waveform:
				tot += samp
			self.integral = tot*self.timestep
			return self.integral
			
	#Finds the average value between sample min and sample max
	#includes min but does not include max
	def average(self, min, max):
		return np.average(self.waveform[min:max])
	
	#Plots the pulse
	def plot(self, show = True, fmt = None, ax = None ):
		x = [n*self.timestep for n in range(0, len(self.waveform))]
		if fmt and ax:
			handle, = ax.plot(x, self.waveform, fmt)
		elif fmt and not ax:
			handle, = plt.plot(x, self.waveform, fmt)
		elif not fmt and ax:
			handle, = ax.plot(x, self.waveform)
		else:
			handle, = plt.plot(x, self.waveform)
		plt.xlabel("Time (ns)", fontsize = 18)
		plt.ylabel("Pulse Amplitude (mV)", fontsize = 18)
		if show:
			plt.show()
		return handle
	
	#Finds the max and returns [sampleNumMax, max]
	#SampleNumMax is an array of all places where the max is reached
	#Searches in window between sample numbers (lowerLoc, upperLoc)
	#If one (or both) not given, searches to end of waveform in that direction
	def findMax(self, lowerLoc = None, upperLoc = None):
		if lowerLoc is None:
			lowerLoc = 0
		if upperLoc is None:
			upperLoc = len(self.waveform)
		maxAmp = max(self.waveform[lowerLoc:upperLoc])
		sampleNumMax = []
		for i in range(lowerLoc, upperLoc):
			if self.waveform[i] == maxAmp:
				sampleNumMax.append(i)
		return [sampleNumMax, maxAmp]
	#Finds the min and returns [sampleNumMin, min]
	#SampleNumMin is an array of all places where the min is reached		
	def findMin(self, lowerLoc = None, upperLoc = None):
		if lowerLoc is None:
			lowerLoc = 0
		if upperLoc is None:
			upperLoc = len(self.waveform)
		minAmp = min(self.waveform[lowerLoc:upperLoc])
		sampleNumMin = []
		for i in range(0, len(self.waveform)):
			if self.waveform[i] == minAmp:
				sampleNumMin.append(i)
		return [sampleNumMin, minAmp]

	#Returns the last sample numbers where the waveform drops beneath a fraction of the minimum (assuming negative peak)
	#Returns None if the pulse has two equal minima
	def getFracMinPoints(self, frac):
		if frac < 0 or frac > 1:
			print "Amplitude fraction must be between 0 and 1."
			return None
		sampleNumMin, minAmp = self.findMin()
		if len(sampleNumMin) > 1:
			return None
		sampleNumMin = sampleNumMin[0]
		backIndices = range(0, sampleNumMin)[::-1] #Count backward from sampleNumMin
		forwardIndices = range(sampleNumMin, len(self.waveform))
		backHalfPoint = None
		forwardHalfPoint = None
		fracAmp = minAmp*frac
		for i in backIndices[1:]:
			if self.waveform[i] >= fracAmp and self.waveform[i+1] < fracAmp:
				backHalfPoint = i
		for i in forwardIndices[1:]:
			if self.waveform[i] >= fracAmp and self.waveform[i-1] < fracAmp:
				forwardHalfPoint = i
		return (backHalfPoint, forwardHalfPoint)

	#Finds rise point of pulse, assuming negative polarity
	#Returns (sample number of the rise point, voltage of rise point)
	#Searches for max slope in slopeSearchWindow samples before the max
	#When the steepest tangent line on the rise is more than two 
	#samples away from the pulse, mark as the rise point.
	def findRisePoint(self, slopeSearchWindow, sepLimit, plotting = False):
		minSamples = self.findMin()[0]
		if len(minSamples) > 1:
			print "Finding rise point failed."
			self.plot()
			sys.exit()
		minSample = minSamples[0]
		if minSample == (slopeSearchWindow-1): #Handles an off-by-one error that sometimes occurs
			rise = self.waveform[minSample-slopeSearchWindow+1:minSample]	
		else:
			rise = self.waveform[minSample-slopeSearchWindow:minSample]
		minSlope = float('inf')
		minSlopeIndex = None
		for i in range(0, len(rise)-1):
			slope = rise[i+1]-rise[i]
			if slope < minSlope and rise[i]-rise[i-1] < 0 and rise[i-1]-rise[i-2] < 0:
				minSlope = slope
				minSlopeIndex = i + minSample-slopeSearchWindow
		if minSlopeIndex is None:
			return None
		dif = 0
		j = minSlopeIndex
		x0 = minSlopeIndex
		y0 = self.waveform[minSlopeIndex]
		y1 = self.waveform[minSlopeIndex+1]
		if y1 - y0 == 0:
			return None
		while dif < sepLimit:
			j -= 1
			dif = x0 + float(self.waveform[j] - y0)/(y1-y0) - j #From the equation for a line
		risePoint = j
		if not plotting:
			return (risePoint, self.waveform[risePoint])
		elif plotting:
			risePoint = (j, self.waveform[j])
			steepPoint = (minSlopeIndex, y0)
			linePoints_x = range(j, minSample-6)
			linePoints_y = [(y1-y0)*(x-x0) + y0 for x in linePoints_x]
			linePoints_x = [x*self.timestep for x in linePoints_x]
			handle = self.plot(show = False, fmt = 'k')
			plt.plot(linePoints_x, linePoints_y, 'r')
			plt.plot(risePoint[0]*self.timestep, risePoint[1], 'ro')
			plt.plot(steepPoint[0]*self.timestep, steepPoint[1], 'ro')
			return handle
	
	#Just like findRisePoint, but also returns the steepest point and the
	#points in the line through it
	def plotWithRisePoint(self, slopeSearchWindow, sepLimit):
		minSamples = self.findMin()[0]
		if len(minSamples) > 1:
			print "Finding rise point failed."
			self.plot()
			sys.exit()
		minSample = minSamples[0]
		rise = self.waveform[minSample-slopeSearchWindow:minSample]
		minSlope = float('inf')
		for i in range(0, len(rise)-1):
			slope = rise[i+1]-rise[i]
			if slope < minSlope and rise[i]-rise[i-1] < 0 and rise[i-1]-rise[i-2] < 0:
				minSlope = slope
				minSlopeIndex = i + minSample-slopeSearchWindow
		dif = 0
		j = minSlopeIndex
		x0 = minSlopeIndex
		y0 = self.waveform[minSlopeIndex]
		y1 = self.waveform[minSlopeIndex+1]
		while dif < sepLimit:
			j -= 1
			dif = x0 + float(self.waveform[j] - y0)/(y1-y0) - j #Worked out in notebook
		
	#Shifts every value in the pulse by amount voltage
	def shiftPulse(self, voltage):
		for i in range(0, len(self.waveform)):
			self.waveform[i] += voltage
	
	#Cuts the waveform down to the window between sample numbers min and max
	#Returns false if min or max is out of range. Returns true otherwise
	def cut(self, minSamp, maxSamp):
		if minSamp < 0 or minSamp >= self.getNumSamples:
			return False
		elif maxSamp > self.getNumSamples() or maxSamp < 0:
			return False
		else:
			self.waveform = self.waveform[minSamp:maxSamp]
			return True
	
	#Returns the rise time in samples between rise point and max
	def getRiseTime(self, estRiseTime, risePointTrig):
		maxSamples, maxVoltage = self.findMin()
		if len(maxSamples) > 1:
			print "Attempted to find rise time of pulse with more than one min location."
			sys.exit()
		else:
			maxSample = maxSamples[0]
		risePoint = self.findRisePoint(estRiseTime, risePointTrig)[0]
		if not risePoint:
			return None
		return (maxSample - risePoint)
			
	#Returns the 10-90 rise time, taking the last oscillation below 10% and the first below 90%,
	#iterating backwards from the maximum
	#if symmetrizing, take first threshold crossing on both channels (if iterating forwards)
	def getMeasRiseTime(self, discrete = False, symmetrizing = True):
		maxSamples, maxVoltage = self.findMin()
		if len(maxSamples) > 1:
			print "Attempted to find rise time of pulse with more than one min location."
			sys.exit()
		else:
			maxSample = maxSamples[0]
		iArray = range(0, maxSample)[::-1] #Reverses order of array
		point10 = None
		point90 = None

		#Evan implemented: 
		#this is the number of samples after which the algorithm says:
		#"I have found the point10 for real, no need to double
		#check in the case of a noise spike"
		schlopSamples = 5 
		schlopCount = -1
		if discrete:
			for i in iArray:
				if self.waveform[i] >= 0.9*maxVoltage and not point90:
					point90 = i+1
				if self.waveform[i] >= 0.1*maxVoltage and not point10:
					point10 = i
				#If come back above 10%, reset p10 until drop back beneath
				if self.waveform[i] < 0.1*maxVoltage and point10:
					point10 = None
		#print "After discrete check, types are: " + str(type(point10)) + "\t" + str(type(point90))
		else: #Find exact crossing point on line between them
			for i in iArray:
				if schlopCount >= 0:
					schlopCount += 1
				print "i = " + str(i) + "\t" + str(self.waveform[i])
				if self.waveform[i] >= 0.9*maxVoltage and not point90:
					if self.waveform[i] == 0.9*maxVoltage: 
						point90 = i
					else:
						x1 = float(i)
						x2 = float(i+1)
						y1 = self.waveform[i]
						y2 = self.waveform[i+1]
						point90 = (0.9*maxVoltage*(x2-x1)+y2*x1-y1*x2)/(y2-y1)
				if symmetrizing:
					if self.waveform[i] < 0.9*maxVoltage and point90:
						point90 = None
				if self.waveform[i] >= 0.1*maxVoltage and not point10:
					schlopCount = 0 #initialize schlop counting
					if self.waveform[i] == 0.1*maxVoltage: 
						point10 = i
					else:
						x1 = float(i)
						x2 = float(i+1)
						y1 = self.waveform[i]
						y2 = self.waveform[i+1]
						point10 = (0.1*maxVoltage*(x2-x1)+y2*x1-y1*x2)/(y2-y1)
				#If come back above 10%, reset p10 until drop back beneath
				if self.waveform[i] < 0.1*maxVoltage and point10 and schlopCount < schlopSamples:
					schlopSamples = -1 #turn schlop counting off
					point10 = None

		return point90 - point10

	#Computes the integral of the square of the waveform
	def getPower(self):
		tot = 0
		for x in self.waveform:
			tot += x**2
		tot *= self.timestep
		return tot
	
	#Computes the FFT of the waveform (if not already computed) and stores it
	def getfft(self):
		freq = []
		fft = []
		n = len(self.waveform)
		d = self.timestep*(1.0e-9)
		freq = np.fft.fftfreq(n, d)
		fft = np.absolute(np.fft.fft(self.waveform))
		freq = [x*1e-9 for x in freq if x > 0]
		fft = fft[:len(freq)]
		return (freq, fft)
	
	#Plots the FFT of the pulse
	def plotfft(self, show = True):
		(freq, fft) = self.getfft()
		plt.semilogy(freq, fft, 'k')
		plt.xlim(0, max(freq))
		plt.xlabel("Frequency (GHz)")
		plt.ylabel("Power Spectrum")
		if show:
			plt.show()
