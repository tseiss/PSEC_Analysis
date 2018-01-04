import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import scipy.interpolate as inter 

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

	def getTimestep(self):
		return self.timestep
	
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

	#Plots the pulse and spline fit
	def plotSpline(self, show = True, fmt = None, ax = None):
		spline_wave = self.getSplineWaveform(10000)
		x = [n*self.timestep for n in np.linspace(0, len(self.waveform), len(spline_wave))]
		if fmt and ax:
			handle, = ax.plot(x, spline_wave, fmt)
		elif fmt and not ax:
			handle, = plt.plot(x, spline_wave, fmt)
		elif not fmt and ax:
			handle, = ax.plot(x, spline_wave)
		else:
			handle, = plt.plot(x, spline_wave)
		plt.xlabel("Time (ns)", fontsize = 18)
		plt.ylabel("Pulse Amplitude (mV)", fontsize = 18)
		if show:
			plt.show()
		return handle

	#Plots the pulse with a butterworth filter on it
	def plotFiltered(self, buttFreq, show = True, fmt = None, ax = None ):
		filtwave = self.butterworthFilter(buttFreq)
		x = [n*self.timestep for n in range(0, len(filtwave))]

		if fmt and ax:
			handle, = ax.plot(x, filtwave, fmt)
		elif fmt and not ax:
			handle, = plt.plot(x, filtwave, fmt)
		elif not fmt and ax:
			handle, = ax.plot(x, filtwave)
		else:
			handle, = plt.plot(x, filtwave)
		plt.xlabel("Time (ns)", fontsize = 18)
		plt.ylabel("Pulse Amplitude (mV)", fontsize = 18)
		if show:
			plt.show()
		return handle
	
	#Finds the max and returns [sampleNumMax, max]
	#SampleNumMax is an array of all places where the max is reached
	#Searches in window between sample numbers (min_sample, max_sample)
	#If one (or both) not given, searches to end of waveform in that direction
	def findMax(self, min_sample = None, max_sample = None):
		if min_sample is None:
			min_sample = 0
		if max_sample is None:
			max_sample = len(self.waveform)
		maxAmp = max(self.waveform[min_sample:max_sample])
		sampleNumMax = []
		for i in range(min_sample, max_sample):
			if self.waveform[i] == maxAmp:
				sampleNumMax.append(i)
		return [sampleNumMax, maxAmp]

	#Finds the min and returns [sampleNumMin, min]
	#SampleNumMin is an array of all places where the min is reached		
	def findMin(self, min_sample = None, max_sample = None):
		if min_sample is None:
			min_sample = 0
		if max_sample is None:
			max_sample = len(self.waveform)
		min_amp = min(self.waveform[min_sample:max_sample])
		min_sample_number = []
		for i in range(len(self.waveform)):
			if self.waveform[i] == min_amp:
				min_sample_number.append(i)
		return [min_sample_number, min_amp]

	#Find the value and sample of the "peak" of the pulse
	#this function is blind to polarity and thus accepts both
	#polarities
	def findPeak(self, min_sample = None, max_sample = None):
		if(min_sample is None):
			min_sample = 0
		if(max_sample is None):
			max_sample = len(self.waveform)

		windowedwave = self.waveform[min_sample:max_sample]
		abswave = [abs(_) for _ in windowedwave]
		maxidx = abswave.index(max(abswave))
		maxV = windowedwave[maxidx] #this value is negative or positive, reflecting polarity
		return (maxidx, maxV)

	#Returns the last sample numbers where the waveform drops beneath a fraction of the minimum (assuming negative peak)
	#Returns None if the pulse has two equal minima
	def getFracMinPoints(self, frac):
		if frac < 0 or frac > 1:
			print "Amplitude fraction must be between 0 and 1."
			return None
		sampleNumMin, minAmp = self.findMin()


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


	#Returns the 10-90 rise time, by looking at the
	#peak and iterating until it falls below 90% and 10%
	#No baseline subtraction done at this point
	def getMeasRiseTime(self, peak_sample, peak_amp):
		polarity = np.sign(peak_amp)

		samp_count = peak_sample
		thresh90 = polarity*0.9*peak_amp
		thresh10 = polarity*0.1*peak_amp
		sam90 = None
		sam10 = None
		while True: 
			samp_count -= 1
			if(samp_count == -1):
				break
			if(polarity*self.waveform[samp_count] <= thresh90 and sam90 is None):
				#found 90 point. Interpolate
				if(polarity*self.waveform[samp_count] == thresh90):
					sam90 = samp_count
				else:
					i = samp_count
					x1 = float(i)
					x2 = float(i+1)
					y1 = abs(self.waveform[i])
					y2 = abs(self.waveform[i+1])
					m = (y2 - y1)/(x2 - x1)
					b = y2 - m*x2
					if(m == 0):
						return (None)

					sam90 = (polarity*thresh90 - b)/m
				continue
			elif(polarity*self.waveform[samp_count] <= thresh10 and sam10 is None):
				#found 10 point. Interpolate
				if(polarity*self.waveform[samp_count] == thresh10):
					sam10 = samp_count
				else:
					i = samp_count
					x1 = float(i)
					x2 = float(i+1)
					y1 = abs(self.waveform[i])
					y2 = abs(self.waveform[i+1])
					m = (y2 - y1)/(x2 - x1)
					b = y2 - m*x2
					if(m == 0):
						return (None)
					sam10 = (polarity*thresh10 - b)/m
				continue

			elif(sam90 is not None and sam10 is not None):
				#we're done here
				break
			else:
				continue

		if(sam10 is None or sam90 is None):
			return (None)

		else:
			if(abs(sam90 - sam10) > 1000):
				self.plot()
				#sys.exit()
			return abs(sam90 - sam10)



	#Computes the integral of the square of the waveform
	def getPower(self):
		tot = 0
		for x in self.waveform:
			tot += x**2
		tot *= self.timestep
		return tot
	
	#Computes the power spectral density of the waveform
	def getPSD(self):
		H = np.fft.fft(self.waveform)
		freq = np.fft.fftfreq(len(self.waveform), d=self.timestep)
		#power spectral density
		pH = [(m.real**2 + m.imag**2) for m in H]
		hznorm = 50*np.sqrt(0.5*abs(min(freq) - max(freq))) #i don't understand the 50
		psdH = pH/hznorm

		#fold the frequency to include only positive
		#frequencies, adds negative to positive
		folded = []
		newf = []
		lost_index = len(freq)/2
		folded.append(2*H[0])
		newf.append(freq[0])
		for i in range(1, lost_index):
			folded.append(np.sqrt((H[i] + H[-i])**2))
			newf.append(freq[i])

		folded.append(H[lost_index])
		newf.append(-1*freq[lost_index])
		return (newf, folded)
	

	#return a list of all sample voltages
	def getAllSampleVoltages(self):
		return self.waveform

	def getWaveform(self):
		return ([n*self.timestep for n in range(0, len(self.waveform))], self.waveform)

	#returns the waveform with a butterworth low pass
	#filter applied
	def butterworthFilter(self, buttFreq):
		filter_order = 5
		#buttFreq assumed in GHz
		nyq = np.pi/(self.timestep) #for the scipy package, they define nyquist this way in rads/sec
		rad_buttFreq = buttFreq #now in radian units
		b, a = butter(filter_order,rad_buttFreq/nyq)
		hfilt = lfilter(b, a, self.waveform)
		return hfilt

	#counts the number of pulses above threshold
	#after a butterowrth filter
	#assumes threshold is the actual value (like -3mv)
	#and assumes the pulse is negative polar
	def countPulses(self, threshold, buttFreq):
		filt_wave = self.butterworthFilter(buttFreq)
		npulses = 0
		#requires this many samples above threshold to be counted as a pulse
		sample_buffer_count = 5
		below_thresh = False
		sample_latch = None
		for i, v in enumerate(filt_wave):
			if(v < threshold and not below_thresh):
				below_thresh = True
				sample_latch = i
				continue

			if(v > threshold and below_thresh):
				samples_under_thresh = abs(i - sample_latch)
				if(samples_under_thresh >= sample_buffer_count):
					#that is a true pulse, add it to n
					npulses += 1

				#if the above if statement doesnt pass
				#it was noise, forget about it, no increment to npulses
				below_thresh = False
				sample_latch = None
				continue

		#if the last sample never goes above threshold
		#then we should count it as a pulse
		if(below_thresh):
			npulses += 1

		return npulses


	#Does a butterworth filter on the waveform and 
	#then finds all pulses under threshold. 
	#Return the sample numbers for which this peak
	#is above threshold
	def findPulseSamplebounds(self, threshold, buttFreq):
		filt_wave = self.butterworthFilter(buttFreq)
		sample_bound_list = []
		#requires this many samples above threshold to be counted as a pulse
		sample_buffer_count = 5
		below_thresh = False
		sample_latch = None
		for i, v in enumerate(filt_wave):
			if(v < threshold and not below_thresh):
				below_thresh = True
				sample_latch = i
				continue

			if(v > threshold and below_thresh):
				samples_under_thresh = abs(i - sample_latch)
				if(samples_under_thresh >= sample_buffer_count):
					#that is a true pulse, add its sample bounds
					#to the list
					bounds = [sample_latch, i]
					sample_bound_list.append(bounds)

				#if the above if statement doesnt pass
				#it was noise, forget about it, no increment to npulses
				below_thresh = False
				sample_latch = None
				continue

		#if the last sample never goes above threshold
		#then we should count it as a pulse
		if(below_thresh):
			bounds = [sample_latch, len(filt_wave) - 1]
			sample_bound_list.append(bounds)

		return sample_bound_list


	#smoothing is a least squares error, so 0.1 allows
	#less error in the fit than 1.5. ranges from 0 to inf
	def getSplineWaveform(self, smoothing = None):
		oldx = range(len(self.waveform))
		newx = np.linspace(0, max(oldx), len(self.waveform)*10)

		if(smoothing is None):
			s1 = inter.InterpolatedUnivariateSpline(oldx, self.waveform)
		else:
			s1 = inter.UnivariateSpline(oldx, self.waveform, s=smoothing)

		return s1(newx)








