import numpy as np
import matplotlib.pyplot as plt
import sys
import Pulse

#Each event contains some number of pulses equal to the number of 
#readout channels. This corresponds to a single trigger
class Event():
	#Prepares array of pulses that will add to with addPulse()
	def __init__(self, pulseArray):
		self.pulseArray = pulseArray #The array of pulses that make up the event
		self.iterIndex = 0 #Keeps track of iteration through event
		self.channelOrder = [] #The order of the channels from largest to smallest negative peak
		self._setChannelOrder() #Sets the channel order
	
	#Prints all pulses in the event
	def __str__(self):
		out =  "Number of channels:" + str(len(self.pulseArray)) + "\n"
		out += "All waveforms:\n"
		for pulse in self.pulseArray:
			out += str(pulse)
			out += "\n"
		return out
	
	#Adds two events, channel by channel. If one event has no channels, returns other event
	def __add__(self, event2):
		eventSum = Event([])
		if self.getNumChannels() == 0:
			return event2
		elif event2.getNumChannels() == 0:
			return self
		elif self.getNumChannels() != event2.getNumChannels():
			print "Cannot add events with different number of channels"
			print self
			print event2
			sys.exit()
		else:
			for i in range(0, self.getNumChannels()):
				eventSum.addPulse(self.pulseArray[i]+event2.pulseArray[i])
		return eventSum
		
	#Multiplies all the waveforms in the event by num
	def __mul__(self, num):
		newEvent = Event([])
		for pulse in self.pulseArray:
			newEvent.addPulse(pulse*num)
		return newEvent
	
	#Allows for iteration through the DataSet
	#Can do things like: for event in DataSet: print event
	def __iter__(self):
		return self	
	def next(self):
		if self.iterIndex >= len(self.pulseArray):
			self.iterIndex = 0
			raise StopIteration
		else:
			self.iterIndex += 1
			return self.pulseArray[self.iterIndex-1]
		
	#Orders the pulses from lowest minimum to highest mimimum, filling self.channelOrder
	def _setChannelOrder(self):
		minList = []
		for i in range(0, len(self.pulseArray)):
			minList.append([i, self.pulseArray[i].findMin()[1]])
		minList = sorted(minList, key = lambda entry: entry[1])	
		self.channelOrder = []
		for entry in minList:
			self.channelOrder.append(entry[0])
	
	#Appends a pulse to the pulse array
	def addPulse(self, Pulse):
		self.pulseArray.append(Pulse)
		self._setChannelOrder()
	
	#Returns the number of channels (pulses) in the event	
	def getNumChannels(self):
		return len(self.pulseArray)
	
	#Returns the number of samples per pulse in the event, assuming
	#each channel has the same number of samples
	def getNumSamples(self):
		return len(self.pulseArray[0].getWaveform())
		
	#Returns channel number pulseNum in the event
	def getPulse(self, pulseNum):
		return self.pulseArray[pulseNum]
		
	#Integrates the pulses corresponding to the array ChannelNums
	#Returns an array of integrals
	def integrate(self, ChannelNums = None):
		integralArray = []
		if ChannelNums:
			for i in ChannelNums:
				integralArray.append(self.pulseArray[i].integrate())
		else:
			for i in range(0, len(self.pulseArray)):
				integralArray.append(self.pulseArray[i].integrate())
		return integralArray
		
	#channels = [1, 3, 4, 5] plots channels 1, 3, 4, 6 (indexing from 0)
	def plotPulses(self, channelNums = None, show = True, fmtArray = None, ax = None):
		if not channelNums:
			channelNums = range(0, self.getNumChannels())
		if max(channelNums) > len(self.pulseArray):
			print "Channel number too large in waveform plot"
			sys.exit()
		if fmtArray:
			for i in range(0, len(channelNums)):
				self.pulseArray[channelNums[i]].plot(show = False, fmt = fmtArray[i], ax = None)
		else:
			for i in channelNums:
				self.pulseArray[i].plot(show = False, ax = None)
		if show:
			plt.show()
		else:
			return
	
	def plotPulsesWithRisepoint(self, primCh, estRiseTime, triggerDiff):
		channels = [i for i in range(0, len(self.pulseArray))]
		channels.remove(primCh)
		for i in channels:
			handleSecondary = self.pulseArray[i].plot(show = False, fmt = 'k--')
		handlePrimary = self.pulseArray[primCh].findRisePoint(estRiseTime, triggerDiff, plotting = True)
		if not handlePrimary:
			print "Could not find risePoint"
			return
		plt.legend([handlePrimary, handleSecondary], ['Primary Pad', 'Adjacent Pads'], loc = 4)
		plt.show()

	#Finds the min of each channel. 
	#Returns array of [sampleNumMin, min], one for each channel
	def findMins(self, channelNums):
		outArray = []
		for i in channelNums:
			outArray.append(self.pulseArray[i].findMin())
		return outArray	
	#Finds the max of each channel. 
	#Returns array of [sampleNumMax, max], one for each channel		
	def findMaxes(self, channelNums):
		outArray = []
		for i in channelNums:
			outArray.append(self.pulseArray[i].findMax())
		return outArray
	
	#Given the primary channel of the event, the estimated rise time (search window), 
	#and the trigger difference between the tangent line and the pulse,
	#returns (risePointSampleNum, risePointVoltage). estRiseTime and triggerDiff
	#are measured in # of samples
	def getRisePoint(self, estRiseTime, triggerDiff):
		primCh = self.channelOrder[0]
		return self.pulseArray[primCh].findRisePoint(estRiseTime, triggerDiff)
		
	#Computes the rise time from the rise point to the max
	def getRiseTime(self, estRiseTime, risePointTrig):
		primCh = self.channelOrder[0]
		return self.pulseArray[primCh].getRiseTime(estRiseTime, risePointTrig)
	
	#Computes the 10%-20% rise time on the primary channel
	def getMeasRiseTime(self, discrete = False):
		primCh = self.channelOrder[0]
		return self.pulseArray[primCh].getMeasRiseTime(discrete)
		
	#Checks whether the peak of the primary channel in the event is too close to walls.
	#Returns True if ok, False if too close or if there are two peaks of equal height on primCh.
	def checkLocation(self, minPeakLoc, maxPeakLoc):
		primCh = self.channelOrder[0]
		peakLocs = self.pulseArray[primCh].findMin()[0]
		if len(peakLocs) > 1:
			return False
		if peakLocs[0] < minPeakLoc:
			return False
		elif peakLocs[0] + maxPeakLoc > self.pulseArray[primCh].getNumSamples():
			return False
		else:
			return True
		
	#Cut the event so that the first entry is baselineLength behind the rise point
	#and the last entry is rise point + pulseWidth in front
	#update rise point sample location
	def cut(self, baselineLength, pulseWidth, estRiseTime, risePointTrig):
		primCh = self.channelOrder[0]
		risePoint = self.pulseArray[primCh].findRisePoint(estRiseTime, risePointTrig)
		if not risePoint:
			return False
		else:
			risePoint = risePoint[0]
		for pulse in self:
			couldCut = pulse.cut(risePoint - baselineLength, risePoint+pulseWidth)
			if not couldCut:
				return False
		return True
		
	#Averages all samples in the event before the rise point
	#Finds the baseline voltage
	#Assumes event has already been cut
	def findBaseline(self, estRiseTime, triggerDiff):
		primCh = self.channelOrder[0]
		risePoint = self.pulseArray[primCh].findRisePoint(estRiseTime, triggerDiff)
		if risePoint is None:
			print estRiseTime
			print triggerDiff
			self.plotPulses()
			raise TypeError
		else:
			risePoint = risePoint[0]
		baselineV = self.pulseArray[primCh].average(0, risePoint)
		return baselineV
		
	#shifts all pulses in the event by amount voltage
	def shiftVoltage(self, voltage):
		for pulse in self:
			pulse.shiftPulse(voltage)
	
	#Returns the array of amplitudes in order from largest to smallest
	def getAmplitudes(self, ordered = True):
		if ordered:
			return [self.pulseArray[i].findMin()[1] for i in self.channelOrder]
		else:
			return [self.pulseArray[i].findMin()[1] for i in range(0, len(self.pulseArray))]

	#Returns an array of the integrals of the squares of the waveforms
	def getPowers(self):
		powerArray = []
		for pulse in self:
			powerArray.append(pulse.getPower())
		return powerArray

	#Find largest min. Search on other channels for mins/maxes there	
	#Signal min/max in a given region, channel by channel
	#return the the mins and maxes of every channel within the window [[min0, max0], [min1, max1], ...]
	def getAmpsInWindow(self, lowerLoc, upperLoc):
		amps = []
		for pulse in self:
			amps.append([])
			amps[-1].append(pulse.findMin(lowerLoc = lowerLoc, upperLoc = upperLoc)[1])
			amps[-1].append(pulse.findMax(lowerLoc = lowerLoc, upperLoc = upperLoc)[1])
		return amps

	#Search for min and max (neg and pos) amps between frac of main negative peak
	def getAmpsNearMax(self, frac):
		primCh = self.channelOrder[0]
		lowerPoint, upperPoint = self.pulseArray[primCh].getFracMinPoints(frac)
		amps = self.getAmpsInWindow(lowerPoint, upperPoint)
		return amps
		"""
		print amps
		self.plotPulses()
		print lowerPoint
		print upperPoint
		self.plotPulses(show = False)
		timestep = self.pulseArray[0].timestep
		plt.plot(lowerPoint*timestep, self.pulseArray[primCh].waveform[lowerPoint], 'ko')
		plt.plot(upperPoint*timestep, self.pulseArray[primCh].waveform[upperPoint], 'ko')
		plt.show()
		"""
