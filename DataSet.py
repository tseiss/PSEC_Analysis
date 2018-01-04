import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy.optimize import curve_fit
import scipy.io as spio
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from operator import add
from scipy.misc import factorial
import sys
import Pulse
import Event
import cPickle as pickle 

class DataSet:
	#Creates an class filled with all the events out of filename
	def __init__(self, filename, numSamplesPerEvent = 256, numChannels = 4, timestep = 9.7e-2, polarity = +1):
		self.numSamplesPerEvent = numSamplesPerEvent
		self.numChannels = numChannels
		self.timestep = timestep #nanoseconds
		
		self.eventArray = []
		
		#read's file in the ascii format
		#outputted by the psec4 eval cards
		#self.readFile(filename, polarity)

		#read's file that is in ".mat" format
		#outputted from L0_conversion_BL from the
		#tektronix 3.5GHz scope
		self.readMatlab(filename, polarity)


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


	def readMatlab(self, filename, polarity):
		if not (polarity == 1 or polarity == -1):
			print "Voltage polarity must be either +1 or -1"
			sys.exit()
		print "Reading file...",
		if len(self.eventArray) > 0:
			print "Appending new file's data to existing DataSet."
		
		#load matlab file
		mat = spio.loadmat(filename, squeeze_me=True)
		#grab it's data structure
		ch = mat['CH']

		#re-initialize metadata parameters
		self.numChannels = len(ch)

		#times are the same for all
		#events and all channels
		times = ch[0]['time']
		self.timestep = abs(times[0] - times[1])*10**(9) #puts in units of ns
		self.numSamplesPerEvent = len(ch[0]['data'][0])

		#structure datach[channel-1][event number][sample number]
		datach = []
		for i in range(self.numChannels):
			datach.append(ch[i]['data'])

		#loop over event number
		for evt in range(len(datach[0])):
			#list of length NChannels that holds each channel's pulse for this event
			pulseArray = [] 
			for chn in range(self.numChannels):
				#now datach[chn][evt] is the waveform in Volts
				#converting to mV:
				mvwaveform = [_*1000*polarity for _ in datach[chn][evt]]
				pulseArray.append(Pulse.Pulse(mvwaveform, self.timestep))

			#push this event to the event list
			self.eventArray.append(Event.Event(pulseArray))

		print "Done."


	def getTimestep(self):
		return self.timestep

	#Writes the dataset object to a pickle file
	def writePickle(self, filename):
		#open hdf5 file and create if it doesnt exist
		pickle.dump(self, open(filename, 'wb'))
				
	
	#Returns the number of events in the DataSet
	def getNumEvents(self):
		return len(self.eventArray)
		
	def getNumSamples(self):
		return self.numSamplesPerEvent
		
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
	def getMeasRiseTimes(self, noise):
		riseTimes = {_:[] for _ in range(self.numChannels)}
		for event in self:
			rts = event.getMeasRiseTimes(noise)
			for key, val in rts.items():
				for v in val:
					riseTimes[key].append(v)


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
	#noise is the noise rms in mv
	def plotMeasRiseTimes(self, ax, noise = 4):
		riseTimeArray = self.getMeasRiseTimes(noise)
		colors = ['g','b','m','r']
		for ch in range(self.numChannels):
			rts_ns = riseTimeArray[ch]
			mean = round(np.average(rts_ns),3)
			std = round(np.std(rts_ns), 3)

			n, bins, patches = ax.hist(rts_ns, 100, alpha=0.5, facecolor=colors[ch], label='ch ' + str(ch))

			def Gaussian(x, mu, sigma, A):
				return A*np.exp(-(x-mu)**2/(2*sigma**2))

			fitx = np.linspace(min(bins), max(bins), 500)
			y = Gaussian(fitx, mean, std, max(n))
			ax.plot(fitx, y, colors[ch]+'--', label="mean = " + str(mean))

			plt.xlabel("Rise Time (ns)", fontsize = 14)
			plt.ylabel("Counts", fontsize = 14)
			ax.legend()

	
	#returns an array of the amplitudes, all 6 amplitudes per event
	def getAmplitudeDist(self,frac=0.6, ordered = False):
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
	def plotAmplitudeDist(self, ax):
		amplitudeDist = [amps[0] for amps in self.getAmplitudeDist()]
		amplitudeDist = [-x for x in amplitudeDist]
		print "Average Amplitude:", round(np.average(amplitudeDist),3), "mV"
		print "Standard Dev:", round(np.std(amplitudeDist), 3), "mV"
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

	#Plots a histogram of amplitudes for all channels
	def plotAllAmplitudeDist(self, ax):
		ampDists = self.getAmplitudeDist()
		colors = ['g','b','m','r']

		for ch in range(self.numChannels):
			channel_amps = [_[ch] for _ in ampDists]
			channel_amps = [-1*_ for _ in channel_amps]
			ax.hist(channel_amps, bins=50, alpha=0.75, label='ch ' + str(ch + 1), facecolor=colors[ch])


		plt.xlabel("Peak Amplitude (mV)", fontsize = 14)
		plt.ylabel("Counts", fontsize = 14)
		ax.legend()

	#return a list of all voltages from all samples
	def getListOfSampleVoltages(self):
		allVoltages = []
		for event in self.eventArray[:200]:
			eventVolts = event.getAllSampleVoltages()
			for v in eventVolts:
				allVoltages.append(v)

		return allVoltages


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
			allChannels = range(self.numChannels)
			for i in range(numEvents):
				num =np.random.randint(0, self.numEvents)
				self.plotEvent(num, allChannels, show = False)
				plt.legend(range(1, self.numChannels + 1))
				plt.title("Event number: " + str(num))
				plt.show()
		elif RisePoint:
			allChannels = range(self.numChannels)
			for i in range(numEvents):
				num = np.random.randint(0, self.numEvents)
				(xRise, yRise) = self.eventArray[num].getRisePoint(primCh)
				xRise *= self.timestep
				plt.plot(xRise, yRise, 'ro')
				self.plotEvent(num, allChannels, show = False)
				plt.legend(range(1, self.numChannels + 1))
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

	#sums all of the pulse power spectral densities
	#for all events, keeps channels separate
	def getSummedPSD(self):
		summed_chs = []
		freqs = None 	#assumes all frequency arrays, like the time arrays, are the same
		for ev in self.eventArray:
			psdchs = ev.getAllPSD()
			if(freqs is None):
				freqs = psdchs[0][0]

			for ch in range(self.numChannels):
				this_psd = psdchs[ch][1]
				if(len(summed_chs) != self.numChannels):
					summed_chs.append(this_psd)
					continue
				else:
					map(add, summed_chs[ch], this_psd)

		return (freqs, summed_chs)

	#plots the output of the "getSummedPSD" function
	def plotSummedPSD(self, ax):
		freqs, summed = self.getSummedPSD()
		colors = ['g','b','m','r']
		for ch in range(self.numChannels):
			ax.plot(freqs, summed[ch], colors[ch], label='ch ' + str(ch))

		ax.set_xlabel("freq. (GHz)")
		ax.set_ylabel("power spectral density (arbitrary units)")
		ax.legend()

	#plots random events on one channel, with 
	#two traces: one the raw trace, and one with 
	#a lowpass butterworth filter applied at "buttFreq"
	#critical frequency in GHz
	def plotRandomFilterComparison(self, nevents, channel, buttFreq):
		if self.numEvents == 0:
			print "No events in data set to plot."
			return

		for i in range(nevents):
			num = np.random.randint(0, self.numEvents)
			self.eventArray[num].plotFilterComparison(buttFreq, [channel], show = False)
			plt.title("Event number: " + str(num))
			plt.show()

	#this function is used to get an estimate
	#of dark rate given that one is triggering on
	#a dark pulse and that some fraction of events
	#have multiple pulses in one event window. This
	#is poisson and the rate is extracted 
	#"threshold" is a voltage threshold on negative polar
	#pulses after a butterworth filter

	#this has been found to be moot. It seems like 
	#there is a high probability for there to be multiple pulses
	#in an event window if there is one big pulse. It isn't poisson
	#and it produces "dark rates" on the order of 10MHz
	def multiplePulseDarkRateMeasure(self, threshold, buttFreq, ax):
		#a dictionary holding the frequency of events
		#that have "n" pulses in one pulse window where
		#"n" is the key of the dictionary

		#get event time window for one event, 
		#assuming it is the same for the dataset
		window = self.numSamplesPerEvent*self.timestep*10**(-9) #in units of seconds


		pulse_count_max = 20
		counts = {_:0 for _ in range(pulse_count_max)}
		count_hist = []
		for ev in self.eventArray:
			#returns the number of pulses in the event
			#using the main channel on the event
			n_pulses = ev.getNumberOfPulses(threshold, buttFreq)
			if(n_pulses == 0):
				#this event was classified as 
				#un countable, take an efficiency hit
				continue
			else:
				if(n_pulses < pulse_count_max):
					#the minus 1 comes from the fact that
					#this dataset was triggered on at least 
					#one pulse. I am counting coincidences
					counts[n_pulses - 1] += 1
					count_hist.append(n_pulses - 1)



		def poisson(k, lamb):
			return (lamb**k/factorial(k))*np.exp(-lamb)

		#make a hist and fit it
		n, bin_edges, patches = ax.hist(count_hist, bins=max(count_hist), range=[-0.5, 0.5+max(count_hist)], normed=True)
		bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])

		#fit
		params, cov_matrix = curve_fit(poisson, bin_centers, n)
		newx = np.linspace(0, max(count_hist), 100)
		ax.plot(newx, poisson(newx, *params), 'r-', lw=2, label="Poisson fit:\nlambda = " + str(params[0]) + "\nDark Rate = " + str(params[0]/window) + "Hz")
		ax.set_xlabel("Number of coincident pulses in the trigger window")
		ax.set_ylabel("Normalized frequency")
		ax.legend()


	#counts all of the pulses found on all channels
	#and all events
	def countAllPulses(self, threshold, buttFreq):
		nev = self.numEvents
		n_pulses_tot = 0
		for ev in self.eventArray:
			n_pulses = ev.getNumberOfPulsesChannelwise(threshold, buttFreq)
			n_pulses_tot += n_pulses
			if(n_pulses != 0):
				#plot the event for diagnostic
				print str(n_pulses) + " pulses found" 
				ev.plotPulses([0,1,2,3], True)


		return (n_pulses_tot, nev)












