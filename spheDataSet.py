import DataSet, Event, Pulse
import sys, os, gc
import numpy as np
import matplotlib.pyplot as plt

class spheEvent(Event.Event):
	def __init__(self, pulseArray = None, event = None):
		if isinstance(pulseArray, Pulse.Pulse):
			Event.Event.__init__(pulseArray)
			self.channelOrder = None
		elif isinstance(event, Event.Event):
			self.pulseArray = event.pulseArray
			self.iterIndex = event.iterIndex 
		else:
			print "spheEvent not properly initialized. Need either pulse array (can be empty) or an event to convert."

	#Cuts all the pulses in the event based on the trigger channel
	def cut(self, trigCh, lowerCut, upperCut, alignmentVoltage):
		alignmentPoint = None
		primWaveform = self.pulseArray[trigCh].waveform
		for i in range(0, len(primWaveform)):
			if primWaveform[i] > alignmentVoltage and primWaveform[i-1] < alignmentVoltage:
				if alignmentPoint is None:
					alignmentPoint = i
				elif alignmentPoint: #If two points at which it rises above alignment point, throw out event
					return False
		if alignmentPoint is None:
			return False
		for pulse in self:
			couldCut = pulse.cut(alignmentPoint-lowerCut, alignmentPoint+upperCut)
			if not couldCut:
				return False
		return True

class spheDataSet(DataSet.DataSet):
	def __init__(self, filename):
		DataSet.DataSet.__init__(self, filename)	
		#Convert all events to spheEvents
		newEventArray = []
		for event in self:
			newEventArray.append(spheEvent(event = event))
		self.eventArray = newEventArray
		gc.collect() #Ensure newEventArray is cleared when __init__() returns

	#Override the cleanEvents to do it in a sphe way
	#alignmentVoltage = the voltage on the leading edge at which all waveforms aligned
	#lowerCut = number of sample before the alignmentPoint to keep
	#upperCut = number of samples after the alignmentPoint to keep
	def cleanEvents(self, alignmentVoltage = 50, lowerCut = 100, upperCut = -10):
		print "Cleaning Events...",
		#Find trigger channel
		numEventsToCheck = 100
		triggerVoltage = 50
		maxChannel = [0 for _ in range(0, self.numChannels)]
		for i in range(0, numEventsToCheck):
			maxesArray = self.eventArray[i].findMaxes()
			if max(maxesArray, key = lambda x: x[1]) > triggerVoltage:
				maxChannel[sorted([maxesArray[i]+[i] for i in range(0, len(maxesArray))], key = lambda x: x[1])[-1][-1]] += 1
		trigChannel = maxChannel.index(max(maxChannel))
		
		#The voltage on the leading edge at which all waveforms aligned
		#number of samples before the alignmentPoint to keep
		#number of samples after the alignmentPoint to keep
		imax = len(self.eventArray)
		i = 0
		removeCounter = 0
		while i < imax:
			couldCut = self.eventArray[i].cut(trigChannel, lowerCut, upperCut, alignmentVoltage)
			if not couldCut:
				self.eventArray.pop(i)
				i -= 1
				imax -= 1
				removeCounter += 1
			i += 1
		self.numEvents -= removeCounter
		print "Done."
		print self.numEvents, "events remaining"
