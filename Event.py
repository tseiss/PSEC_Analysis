import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import peakutils
import sys
import copy
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

	def plotPulsesSpline(self, channelNums = None, show = True, fmtArray = None, ax = None):
		if not channelNums:
			channelNums = range(0, self.getNumChannels())
		if max(channelNums) > len(self.pulseArray):
			print "Channel number too large in waveform plot"
			sys.exit()
		if fmtArray:
			for i in range(0, len(channelNums)):
				self.pulseArray[channelNums[i]].plot(show = False, fmt = fmtArray[i], ax = None)
				self.pulseArray[channelNums[i]].plotSpline(show = False, fmt = fmtArray[i], ax = None)
		else:
			for i in channelNums:
				self.pulseArray[i].plot(show = False, ax = None)
				self.pulseArray[i].plotSpline(show = False, ax = None)
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
	def findMins(self, channelNums = None):
		outArray = []
		if channelNums is None:
			channelNums = range(0, len(self.pulseArray))
		for i in channelNums:
			outArray.append(self.pulseArray[i].findMin())
		return outArray	
	#Finds the max of each channel. 
	#Returns array of [sampleNumMax, max], one for each channel		
	def findMaxes(self, channelNums = None):
		outArray = []
		if channelNums is None:
			channelNums = range(0, len(self.pulseArray))
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
	

	def getMeasRiseTimes(self, noise):
		spline_order = noise*2000 #found to be good for spline smoothing
		pulse_threshold = noise*2 #requirement for there to be any pulse in this waveform
		transmission_velocity_fraction = 0.5
		max_bounce_time_diff = 1.6/transmission_velocity_fraction #ns is the longest time difference between reflected pulses

		rtdict = {_:[] for _ in self.channelOrder}
		for ch in range(len(self.pulseArray)):
			pulse = self.pulseArray[ch]
			max_idx, max_v = pulse.findPeak()
			polarity = np.sign(max_v)
			pulse_wfm = pulse.getWaveform()[1]
			pulse_wfm = [_*polarity for _ in pulse_wfm]

			#spline fit the baseline subtracted wfm
			spline_wave = pulse.getSplineWaveform(spline_order)
			spline_wave = [_*polarity for _ in spline_wave]
			indexes = peakutils.indexes(np.array(spline_wave), thres=0.4, min_dist = 50)
			peaks = [spline_wave[_] for _ in indexes]
			if(len(indexes) == 0):
				#no peaks found
				continue

			if(max(peaks) < pulse_threshold):
				#nothing in this channel, move on
				continue

			#baseline subtract assuming the first 3ns 
			#are baseline
			ns_to_spline_samples = 10.0/pulse.getTimestep()
			baseline_width = 3 #ns
			baseline_samples = int(ns_to_spline_samples*baseline_width)
			if(min(indexes) < baseline_samples):
				#the first pulse is too close to the beginning
				#of the event. Toss it
				continue
			else:
				const_baseline = np.mean(spline_wave[0:baseline_samples])

			#find the relevant pulses that 
			#are the first pulse in a group of 2, 
			#or just an isolated pulse
			indexes = sorted(indexes)
			rise_indexes = [] #indexes of first pulses in groups of 2
			rise_amps = [] #amplitudes of such indexes
			sep_samples = ns_to_spline_samples*max_bounce_time_diff
			for i, ind in enumerate(indexes):
				if(i == 0):
					rise_indexes.append(ind)
					rise_amps.append(pulse_wfm[int(ind/10.0)])
					continue
				if(indexes[i] - indexes[i-1] < sep_samples):
					#this is the second in a bounce pair
					continue
				else:
					rise_indexes.append(ind)
					rise_amps.append(pulse_wfm[int(ind/10.0)])
			"""
			fig, ax = plt.subplots()
			ax.plot(range(len(spline_wave)), spline_wave)
			ax.plot(indexes, peaks, 'ro')
			ax.plot(rise_indexes, rise_amps, 'bo')
			"""

			#now for each rise index, find the rise
			#time associated with 90%/10% of peak amp
			#of the UNSMOOTHED PULSE at that index
			for i, ra in enumerate(rise_amps):
				samp_count = int(rise_indexes[i]/10.0)
				peak_amp = ra
				thresh90 = 0.9*peak_amp - const_baseline
				thresh10 = 0.1*peak_amp - const_baseline
				sam90 = None
				sam10 = None
				while True: 
					samp_count -= 1
					if(samp_count == -1):
						break
					if(pulse_wfm[samp_count] <= thresh90 and sam90 is None):
						#found 90 point. Interpolate
						if(pulse_wfm[samp_count] == thresh90):
							sam90 = samp_count
						else:
							i = samp_count
							x1 = float(i)
							x2 = float(i+1)
							y1 = abs(pulse_wfm[i])
							y2 = abs(pulse_wfm[i+1])
							m = (y2 - y1)/(x2 - x1)
							b = y2 - m*x2
							if(m < 1e-5):
								break

							sam90 = (thresh90 - b)/m
						continue
					elif(pulse_wfm[samp_count] <= thresh10 and sam10 is None):
						#found 10 point. Interpolate
						if(pulse_wfm[samp_count] == thresh10):
							sam10 = samp_count
						else:
							i = samp_count
							x1 = float(i)
							x2 = float(i+1)
							y1 = abs(pulse_wfm[i])
							y2 = abs(pulse_wfm[i+1])
							m = (y2 - y1)/(x2 - x1)
							b = y2 - m*x2
							if(m < 1e-5):
								break
							sam10 = (thresh10 - b)/m
						continue

					elif(sam90 is not None and sam10 is not None):
						#we're done here
						break
					else:
						continue

				if(sam10 is None or sam90 is None):
					continue

				else:
					#ax.plot(sam90*10, thresh90, 'go')
					#ax.plot(sam10*10, thresh10, 'go')
					samdiff = abs(sam90 - sam10)
					risetime = samdiff*pulse.getTimestep() #ns
					if(risetime > 5):
						continue
					rtdict[ch].append(risetime)
					continue
					
			"""
			ax.plot([_*10 for _ in range(len(pulse_wfm))], pulse_wfm)
			ax.plot()
			plt.show()
			"""

		return rtdict

	def testPeakutils(self):
		for ch in range(len(self.pulseArray)):
			pulse = self.pulseArray[ch]
			timestep = pulse.getTimestep()
			mindist_ns = 2 #ns
			mindist_samples = int(mindist_ns/timestep)
			max_idx, max_v = pulse.findPeak()
			polarity = np.sign(max_v)
			pwave = pulse.getWaveform()[1]
			pwave = [_*polarity for _ in pwave]
			indexes = peakutils.indexes(np.array(pwave), thres=0.5, min_dist=mindist_samples)
			y = [pwave[_] for _ in indexes]
			fig, ax = plt.subplots()
			ax.plot(range(len(pwave)), pwave)
			ax.plot(indexes, y, 'ro')
			plt.show()





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

	#returns a list of all sample voltages on all channels
	def getAllSampleVoltages(self):
		allVoltages = []
		for ch in self.channelOrder:
			oneChWave = self.pulseArray[ch].getAllSampleVoltages()
			for v in oneChWave:
				allVoltages.append(v)
		return allVoltages

	#Find largest min. Search on other channels for mins/maxes there	
	#Signal min/max in a given region, channel by channel
	#return the the mins and maxes of every channel within the window [[min0, max0], [min1, max1], ...]
	def getAmpsInWindow(self, min_sample, max_sample):
		amps = []
		for pulse in self:
			amps.append([])
			amps[-1].append(pulse.findMin(min_sample = min_sample, max_sample = max_sample)[1])
			amps[-1].append(pulse.findMax(min_sample = min_sample, max_sample = max_sample)[1])
		return amps


	#return the waveforms inside a window defined by
	#where the primary channel goes "frac" below its
	#max amplitude
	def getWavesNearMax(self, frac, ordered = False):
		primCh = self.channelOrder[0]
		lower_sample, upper_sample = self.pulseArray[primCh].getFracMinPoints(frac)
		waves = [_.getWaveform()[1] for _ in self.pulseArray]
		return [_[lower_sample:upper_sample] for _ in waves]

	#Search for min and max (neg and pos) amps between frac of main negative peak
	def getAmpsNearMax(self, frac, ordered = True):
		primCh = self.channelOrder[0]
		lowerPoint, upperPoint = self.pulseArray[primCh].getFracMinPoints(frac)
		amps = self.getAmpsInWindow(lowerPoint, upperPoint)
		#return only the absolute maximum amplitude found
		max_amps = []
		for a in amps:
			if(abs(a[0]) > abs(a[1])):
				max_amps.append(a[0])
			else:
				max_amps.append(a[1])

		#currently, amps are unordered
		if ordered:
			return [max_amps[i] for i in self.channelOrder]
		else:
			return max_amps
			
		'''
		print amps
		self.plotPulses()
		print lowerPoint
		print upperPoint
		self.plotPulses(show = False)
		timestep = self.pulseArray[0].timestep
		plt.plot(lowerPoint*timestep, self.pulseArray[primCh].waveform[lowerPoint], 'ko')
		plt.plot(upperPoint*timestep, self.pulseArray[primCh].waveform[upperPoint], 'ko')
		plt.show()
		'''

	def getPulseWaveform(self, channelNum):
		times, wave = self.pulseArray[channelNum].getWaveform()
		return (times,wave)

	#get all channel's power spectral density functions
	def getAllPSD(self):
		psd_list = [_.getPSD() for _ in self.pulseArray]
		return psd_list

	def plotFilterComparison(self, buttFreq, channelNums = None, show = True, ax = None):
		if not channelNums:
			channelNums = range(0, self.getNumChannels())
		if max(channelNums) > len(self.pulseArray):
			print "Channel number too large in waveform plot"
			sys.exit()
		
		for i in channelNums:
			self.pulseArray[i].plot(show = False, ax = None, fmt = '-')
			self.pulseArray[i].plotFiltered(buttFreq, show = False, ax = None, fmt = '--')

		if show:
			plt.show()
		else:
			return
		

	#returns an integer number of pulses that reach
	#above the "threshold" voltage after being filtered
	#by a lowpass butterworth filter at freq "buttFreq"
	#uses the main channel only
	def getNumberOfPulses(self, threshold, buttFreq):
		#i know that there are 3 adjacent channels for
		#the setup of tile 21. This can be rewritten 
		#for other applications
		adjacent_channels = self.channelOrder[:-1] 
		ns = [self.pulseArray[ch].countPulses(threshold, buttFreq) for ch in adjacent_channels]

		#if the numbers disagree by more than 1 in any combination, 
		#then toss the event and say n = 0
		disagreement = 1
		ncombs = [_ for _ in combinations(ns, 2)]
		for c in ncombs:
			if(abs(c[0] - c[1]) > disagreement):
				return 0


		#print return number extracted from primary channel
		return ns[0]

	#returns an integer number of pulses that reach
	#above the "threshold" voltage after being filtered
	#by a lowpass butterworth filter at freq "buttFreq"
	#Looks on all channels, doesn't need to be coincident. 
	#If the pulses are coincident across channels, it is called
	#"1" pulses. If they are separate in time by a pulse width, 
	#it is called "2". If no coincidence, it is called "1"
	def getNumberOfPulsesChannelwise(self, threshold, buttFreq):
		#get the pulse count on all channels separately
		ns = [self.pulseArray[ch].countPulses(threshold, buttFreq) for ch in self.channelOrder]

		if(sum(ns) == 0):
			#if there is no pulse, return 0
			return 0
		elif(sum(ns) == 1):
			#if there is only 1, return 1
			return 1
		elif(sum(ns) > 1):
			#if there is more than 1 pulses across 
			#all channels, check for coincidence

			#get channels with non-zero pulse count
			pulse_chans = []
			for i in range(len(ns)):
				if(ns[i] != 0):
					pulse_chans.append(self.channelOrder[i])

			#get all of the sample bounds for 
			#all pulses on the channels with pulses
			#in them. 
			samplebound_list = self.get_all_pulse_samplebounds(threshold, buttFreq, chans = pulse_chans)

			#remove duplicates
			for sb in samplebound_list:
				for sbb in samplebound_list:
					if(sb == sbb):
						samplebound_list.remove(sbb)


			#separate the sample bounds into two groups:
			#those that have overlap, and those that do not
			coincidence_groups = []
			coinc_found = False
			for b in samplebound_list:
				if(len(coincidence_groups) == 0):
					coinc_found = False
					bcomplist = copy.copy(samplebound_list)
					bcomplist.remove(b)
					for bb in bcomplist:
						if(min(bb) <= b[0] <= max(bb) or min(bb) <= b[1] <= max(bb) \
							or min(b) <= bb[0] <= max(b) or min(b) <= bb[1] <= max(b)):
							#this is a coincidence
							coincidence_groups.append([b, bb])
							samplebound_list.remove(b)
							samplebound_list.remove(bb)
							coinc_found = True

					if(coinc_found == False):
						#add it by itself, it is a lone wolf
						coincidence_groups.append([b])

				else:
					#check the already established coincidence groups
					coinc_found = False
					bcomplist = copy.copy(samplebound_list)
					bcomplist.remove(b)
					for i,gp in enumerate(coincidence_groups):
						for gp_b in gp:
							bcomplist.remove(gp_b)
							if(min(gp_b) <= b[0] <= max(gp_b) or min(gp_b) <= b[1] <= max(gp_b) \
								or min(b) <= gp_b[0] <= max(b) or min(b) <= gp_b[1] <= max(b)):
								#this is a coincidence
								coincidence_groups[i].append(b)
								coinc_found = True
					if(coinc_found):
						continue

					#if not already belonging to existing coinc
					#group, check to see if it forms its own with
					#another bound in the list
					for bb in bcomplist:
						if(min(bb) <= b[0] <= max(bb) or min(bb) <= b[1] <= max(bb) \
							or min(b) <= bb[0] <= max(b) or min(b) <= bb[1] <= max(b)):
								#this is a coincidence
								coincidence_groups[i].append([b, bb])
								coinc_found = True

					if(coinc_found == False):
						#add it by itself, it is a lone wolf
						coincidence_groups.append([b])

			return len(coincidence_groups)


	#gets all of the timebounds for which pulses 
	#are below threshold value. Assumes negative polarity
	def get_all_pulse_samplebounds(self, pulse_threshold, buttFreq, chans = None):
		if(chans is None):
			channels = self.channelOrder
		else:
			channels = chans

		all_samplebounds = []
		for ch in channels:
			samplebound_list = self.pulseArray[ch].findPulseSamplebounds(pulse_threshold, buttFreq)
			if(len(samplebound_list) == 0):
				#something went wrong, no pulses found in the list
				#this should never happen if the threshold and buttFreq
				#parameters are the same as for when one counted pulses before
				continue
			else:
				for sb in samplebound_list:
					all_samplebounds.append(sb)

		return all_samplebounds


