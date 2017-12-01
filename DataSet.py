import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy.optimize import curve_fit
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import sys
import Pulse
import Event

class DataSet:
	#Creates an class filled with all the events out of filename
	def __init__(self, filename, numSamplesPerEvent = 256, numChannels = 6, timestep = 9.7e-2, polarity = +1):
		self.numSamplesPerEvent = numSamplesPerEvent
		self.numChannels = numChannels
		self.timestep = timestep #nanoseconds
		
		self.eventArray = []
		self.readFile(filename, polarity)
		self.averageEvent = None
		
		self.iterIndex = 0 #Keeps track of where we are when iterating through
		self.numEvents = len(self.eventArray)
		self.primCh = None
	
	#Instructions for how to cast a DataSet as a string. Prints number of events,
	#then prints the first event
	def __str__(self):
		out = "Number of Events: " + str(len(self.eventArray)) + "\n"
		if len(self.eventArray) > 0:
			out += "Printing first event:\n"
			out += str(self.eventArray[0])
		return out
	
	#Allows for iteration through the DataSet
	#Can do things like: for event in DataSet: print event
	def __iter__(self):
		return self	
	def next(self):
		if self.iterIndex >= self.numEvents:
			self.iterIndex = 0
			raise StopIteration
		else:
			self.iterIndex += 1
			return self.eventArray[self.iterIndex-1]
			
	#Fills the dataset with events from the file filename
	#Setting polarity = -1 negates all voltages
	def readFile(self, filename, polarity = +1):
		if not (polarity == 1 or polarity == -1):
			print "Voltage polarity must be either +1 or -1"
			sys.exit()
		print "Reading file...",
		if len(self.eventArray) > 0:
			print "Appending new file's data to existing DataSet."
		f = open(filename)
		eventCounter = 0
		sampleCounter = 0
		waveformArray = [[] for _ in range(0, self.numChannels)]
		for line in f:
			words = line.split()
			if words[0] == "#" or words[0][0] == "#":
				try:
					if words[1] == "NotRawData":
						self.numSamplesPerEvent = int(words[2])
				except IndexError:
					continue
				continue
			if sampleCounter == self.numSamplesPerEvent:
				sampleCounter = 0
				eventCounter += 1
				pulseArray = []
				for waveform in waveformArray:
					pulseArray.append(Pulse.Pulse(waveform, self.timestep))
				waveformArray = [[] for _ in range(0, self.numChannels)]
				self.eventArray.append(Event.Event(pulseArray))
			sampleCounter += 1
			for i in range(0, self.numChannels):
				waveformArray[i].append(polarity*1000*float(words[i]))
		pulseArray = []
		for waveform in waveformArray:
			pulseArray.append(Pulse.Pulse(waveform, self.timestep))
		self.eventArray.append(Event.Event(pulseArray))
		print "Done."

	#Writes the events to filename, overwriting filename
	def write(self, filename):
		f = open(filename, 'w')
		numSamplesPerEvent = len(self.eventArray[0].pulseArray[0].waveform)
		f.write("# NotRawData " + str(numSamplesPerEvent) + "\n")
		self.eventArray[0].plotPulses(show = False)
		for event in self:
			for i in range(0, numSamplesPerEvent):
				for j in range(0, self.numChannels):
					try:
						f.write(str(event.pulseArray[j].waveform[i]/1000.0) + " ")
					except:
						print i, j
						print event.pulseArray[j].waveform
						print len(event.pulseArray[j].waveform)
						event.plotPulses()
						sys.exit()
				f.write("\n")
		f.close()
						
	
	#Returns the number of events in the DataSet
	def getNumEvents(self):
		return len(self.eventArray)
		
	#Returns event number eventNum in the data set
	def getEvent(self, eventNum):
		return self.eventArray(eventNum)
		
	#Computes and stores the average pulse shape for each channel
	def averagePulses(self):
		if self.averageEvent:
			return self.averageEvent
		else:
			self.averageEvent = Event.Event([])
			for event in self.eventArray:
				self.averageEvent += event
			self.averageEvent = self.averageEvent*(1/float(len(self.eventArray)))
			return self.averageEvent
			
	#First makes sure peak is not too close to left wall
	#Then cuts data to baseline+pulseWidth samples long and aligns rise points
	#Then averages the points before the rise point, and subtracts that value from everything,
	#	event by event (assumes same pedastal for all channels)
	#baselineLength is the number of samples to save before the rise point
	#pulseWidth is the number of samples to keep after the rise point 
	#estRiseTime is the estimated number of samples to work with behind the peak for finding rise points
	#risePointTrig is the trigger distance for the rise point finder algorithm (can be fraction)
	#minAmplitude is the small amplitude pulse allowed
	#minPeakLoc is the number of samples the peak must be from the left wall. If not given,
	#	computes it from baselineLength+estRiseTime
	#maxPeakLoc is the number of samples the peak must be from the right wall. If not given,
	#	computes it from PulseWidth/2
	def cleanEvents(self, baselineLength, PulseWidth, estRiseTime, risePointTrig, minAmplitude, minPeakLoc = None, maxPeakLoc = None):
		if minPeakLoc is None:
			minPeakLoc = baselineLength + estRiseTime #Require peak to be at least this far from the wall
		if maxPeakLoc is None:
			maxPeakLoc = PulseWidth/2
		removeCounter = 0
		imax = len(self.eventArray)
		i = 0
		while i < imax:
			goodLocation = self.eventArray[i].checkLocation(minPeakLoc, maxPeakLoc)
			if not goodLocation:
				self.eventArray.pop(i)
				imax -= 1
				i -= 1
				removeCounter += 1
			else:
				couldCut = self.eventArray[i].cut(baselineLength, PulseWidth, estRiseTime, risePointTrig)
				if not couldCut:
					self.eventArray.pop(i)
					imax -= 1
					i -= 1
					removeCounter += 1
				else:
					if self.eventArray[i].getAmplitudes()[0] > minAmplitude:
						self.eventArray.pop(i)
						imax -= 1
						i -= 1
						removeCounter += 1
			i += 1
		self.numEvents -= removeCounter
		print self.numEvents, "events remaining"
		for event in self:
			baseline = event.findBaseline(estRiseTime, risePointTrig)
			event.shiftVoltage(-baseline)
		
	#Gets the differences between rise point and max.
	def getRiseTimes(self, estRiseTime, risePointTrig):
		riseTimeArray = []
		for event in self:
			riseTimeArray.append(event.getRiseTime(estRiseTime, risePointTrig))
		return riseTimeArray
			
	#histograms the rise times * does a Gaussian fit
	def plotRiseTimes(self, estRiseTime, risePointTrig):
		riseTimeArray = self.getRiseTimes(estRiseTime, risePointTrig)
		minRiseTime = min(riseTimeArray)
		maxRiseTime = max(riseTimeArray)
		hist = [0 for _ in range(minRiseTime, maxRiseTime+1)]
		x = [i for i in range(minRiseTime, maxRiseTime+1)]
		for val in riseTimeArray:
			hist[val-min(riseTimeArray)]+= 1
		def Gaussian(x, mu, sigma, A):
			return A*np.exp(-(x-mu)**2/(2*sigma**2))
		(mu, sigma, A) = curve_fit(Gaussian, x, hist)[0]
		print "Average Rise Time:", round(mu*self.timestep, 3), "ns"
		print "Standard Dev:", round(sigma*self.timestep, 3), "ns"
		y = [Gaussian(i, mu, sigma, A) for i in x]
		x = [i*self.timestep for i in x]
		plt.semilogy(x, y, 'k')
		plt.semilogy(x, hist, 'ko')
		plt.show()
		
	#gets the 10%-90% rise time array. Assumes pedastal already subtracted.
	#Discrete rounds to nearest sample time. Not discrete interpolates line between
	#samples and finds when it crosses the threshold on the line
	def getMeasRiseTimes(self, discrete = False):
		riseTimes = []
		for event in self:
			riseTimes.append(event.getMeasRiseTime(discrete))
		return riseTimes

	#Computes the average rise time and standard deviation
	def getAvRiseTime(self):
		riseTimes = self.getMeasRiseTimes()
		return np.average(riseTimes)*self.timestep, np.std(riseTimes)*self.timestep
		
	#Plots events with a risetime above the threshold. Plots at most maxPlots events
	def plotEventsAboveThresholdRiseTime(self, threshold, maxPlots, discrete = False):
		eventsToPlot = []
		for i in range(0, len(self.eventArray)):
			riseTime = self.timestep*self.eventArray[i].getMeasRiseTime(discrete)
			if riseTime > threshold:
				eventsToPlot.append(i)
		if len(eventsToPlot) > maxPlots:
			i = 0
			while i < maxPlots:
				print "Plotting event:", eventsToPlot[i]
				eventToPlot = self.eventArray[eventsToPlot[i]]
				eventToPlot.plotPulses([eventToPlot.channelOrder[0]])
				i += 1
		else:
			for i in eventsToPlot:
				print "Plotting event:", i
				self.eventArray[i].plotPulses([self.eventArray[i].channelOrder[0]])
		
	#Plots the histogram of the measured rise times, and fits a Gaussian to it
	def plotMeasRiseTimes(self, ax = None, discrete = False, show = True):
		riseTimeArray = self.getMeasRiseTimes(discrete)
		if not discrete:
			riseTimeArray = [i*self.timestep for i in riseTimeArray]
			print "Average Rise Time:", round(np.average(riseTimeArray),3), "ns"
			print "Standard Dev:", round(np.std(riseTimeArray), 3), "ns"
			if not ax:
				fig, ax = plt.subplots()
			plt.hist(riseTimeArray, bins = 50, alpha=0.5)
			plt.xlabel("Rise Time (ns)", fontsize = 14)
			plt.ylabel("Counts", fontsize = 14)
			majorLocator = MultipleLocator(0.1)
			minorLocator = MultipleLocator(0.01)
			ax.xaxis.set_major_locator(majorLocator)
			ax.xaxis.set_minor_locator(minorLocator)
			majorLocator = MultipleLocator(50)
			ax.yaxis.set_major_locator(majorLocator)
			minorLocator = MultipleLocator(10)
			ax.yaxis.set_minor_locator(minorLocator)
			if show:
				plt.show()
		elif discrete:
			print "Doing discrete plot."
			minRiseTime = min(riseTimeArray)
			maxRiseTime = max(riseTimeArray)
			hist = [0 for _ in range(minRiseTime, maxRiseTime+1)]
			x = [i for i in range(minRiseTime, maxRiseTime+1)]
			for val in riseTimeArray:
				hist[val-min(riseTimeArray)]+= 1
			print "Average Rise Time:", round(np.average(riseTimeArray)*self.timestep,3), "ns"
			print "Standard Dev:", round(np.std(riseTimeArray)*self.timestep, 3), "ns"
			x = [i*self.timestep for i in x]
			plt.plot(x, hist, 'ko')
			plt.xlabel("Rise Time (ns)", fontsize = 14)
			plt.ylabel("Counts", fontsize = 14)
			if show:
				plt.show()
	
	#returns an array of the amplitudes, all 6 amplitudes per event
	def getAmplitudeDist(self,frac, ordered = True):
		amplitudeDist = []
		for event in self:
			amplitudeDist.append(event.getAmpsNearMax(frac, ordered))
		return amplitudeDist

	#Returns the average amplitude and stat. err. on every channel: [[av1, std1], [av2, std2], ...]
	def getAvAmplitudes(self, frac = 0.6, ordered = False):
		amplitudeDist = self.getAmplitudeDist(frac, ordered)
		amps = np.average(amplitudeDist, axis = 0) #Average along the correct axis
		stds = np.std(amplitudeDist, axis = 0)
		returnArray = []
		for i in range(0, len(amps)):
			returnArray.append([amps[i], stds[i]])
		return returnArray

	#Plots a histogram of the primary amplitudes
	def plotAmplitudeDist(self):
		amplitudeDist = [amps[0] for amps in self.getAmplitudeDist()]
		amplitudeDist = [-x for x in amplitudeDist]
		print "Average Amplitude:", round(np.average(amplitudeDist),3), "mV"
		print "Standard Dev:", round(np.std(amplitudeDist), 3), "mV"
		fig, ax = plt.subplots()
		plt.hist(amplitudeDist, bins = 50, alpha=0.75)
		plt.xlabel("Peak Amplitude (mV)", fontsize = 14)
		plt.ylabel("Counts", fontsize = 14)
		majorLocator = MultipleLocator(10)
		minorLocator = MultipleLocator(2)
		ax.xaxis.set_major_locator(majorLocator)
		ax.xaxis.set_minor_locator(minorLocator)
		majorLocator = MultipleLocator(50)
		ax.yaxis.set_major_locator(majorLocator)
		minorLocator = MultipleLocator(10)
		ax.yaxis.set_minor_locator(minorLocator)
		plt.show()

	#return a list of all voltages from all samples
	def getListOfSampleVoltages(self):
		allVoltages = []
		for event in self.eventArray[:200]:
			eventVolts = event.getAllSampleVoltages()
			for v in eventVolts:
				allVoltages.append(v)

		return allVoltages


	#Gets the average power in each channel. Averaged over all events in self
	def getAvPower(self):
		powerArray = [[] for i in range(0, self.numChannels)]
		for event in self:
			powers = event.getPowers()
			for i in range(0, self.numChannels):
				powerArray[i].append(powers[i])
		avPowers = []
		for channel in powerArray:
			avPowers.append(np.average(channel))
		return avPowers

	#Returns an array of the number of events with 
	def getPrimaryChannelArray(self):	
		primChArray = [0 for _ in range(0, self.numChannels)]
		for event in self.eventArray:
			primChArray[event.channelOrder[0]] += 1
		return primChArray
	
	#Plots numEvents random events from the dataset
	def plotRandomEvents(self, numEvents, RisePoint = False):
		if self.numEvents == 0:
			print "No events in data set to plot."
			return
		if not RisePoint:
			allChannels = [0, 1, 2, 3, 4, 5]
			for i in range(0, numEvents):
				num =np.random.randint(0, self.numEvents)
				self.plotEvent(num, allChannels, show = False)
				plt.legend([1, 2, 3, 4, 5, 6])
				plt.title("Event number: " + str(num))
				plt.show()
		elif RisePoint:
			allChannels = [0, 1, 2, 3, 4, 5]
			for i in range(0, numEvents):
				num = np.random.randint(0, self.numEvents)
				(xRise, yRise) = self.eventArray[num].getRisePoint(primCh)
				xRise *= self.timestep
				plt.plot(xRise, yRise, 'ro')
				self.plotEvent(num, allChannels, show = False)
				plt.legend([1, 2, 3, 4, 5, 6])
				#self.plotEvent(num, [0], show = False)
				plt.title("Event number: " + str(num))
				plt.show()
	
	#Plots a given channel of all the events
	def plotAllEvents(self, channel):
		print "Plotting all events..."
		for event in self:
			event.plotPulses([channel], show = False, fmtArray = ['k'])
		plt.show()
	
	#Plots the pulses of the given event number, starting at 0. 
	#Plots the channels listed in the array channelNums (ie = [1, 2, 4])
	def plotEvent(self, eventNum, channelNums, show = True, fmtArray = None, ax = None):
		self.eventArray[eventNum].plotPulses(channelNums, show, fmtArray, ax = None)

	#returns the waveform and timing array 
	def getEventWaveform(self, eventNum, channelNum):
		times, wave = self.eventArray[eventNum].getPulseWaveform(channelNum)
		return (times, wave)
