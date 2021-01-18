# -*- coding: utf-8 -*-
"""
@author: Salma Elnagar

A script for creating the Events Table to prepare the behavioural data for analysis 
by the Grid CAT toolbox.
We extract relevant information about the individual events within an fMRI scanning run.
The outcome is .txt files for each participant containing:
    1) the event name (long, short or passive)
    2) the event onset
    3) the event duration
    4) the angle
    
The data was extracted mostly from the preprocessed data 
and also some were extracted from the raw data

Note: preprocessed data = (RawData.mat), and raw data = (Subject_run_trainingXX_cnd_X_XXXX_X_track.txt)

How to order the data for this script:
1) place this script in the folder containing all participants folders
2) place the Indices folder in the same place as this script
3) use the unzipper.py script to unzip the raw data with the appropriate outpath
  
"""
#%%
# importing modules

from scipy.io import loadmat
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pdb
from scipy.ndimage import gaussian_filter


# specifying important paths. The outpath is where we want to save the .txt files
# and the indices path is where the indices files that tell us the orders of the conditions
outpath = r'/Users/SalmaElnagar/Documents/Uni/ResearchProject/EventTables/EventTablesFinal/'
indices_path = r'/Users/SalmaElnagar/Documents/Uni/ResearchProject/Data/subjects/Indices/'


# defining the main variables, according to the order in the RawData.mat files
struct_name = 'RAW'
runs = ['mrt01', 'mrt02']
conditions = ['LONG', 'SHORT', 'PASSIVE']
trials = ['trial1', 'trial2', 'trial3', 'trial4', 'trial5', 'trial6']
features = ['Track']
DECIMALS = 4 # what we will round our data to
THRESHOLD_SPEED = 10 # in degrees per second
THRESHOLD_DURATION = 0.5 # optional for now
SMOOTHNESS = 2 # smoothness of the gaussian filter 


#%%
# defining functions

def get_active_data(subject, run, condition, start_time=0):
    
    ''' a function for accessing the time and angle (in "track") in the LONG and SHORT conditions 
    (from the preprocessed data)'''
    
    times = []
    angles = []
    folder = subject+r'/data/preprocessed/' # folder where preprocessed data is
    File = loadmat(folder + [name for name in os.listdir(folder) if name.endswith('RawData.mat')][0]) 
    
    # navigating to data
    subfolder = File['RAW'][run].item()[condition].item()['concated'].item()['Track'].item()
    times = subfolder[:, 0] # first column
    angles = subfolder[:, 4]# fifth column
    return np.array(times) + start_time, np.array(angles)
        


def get_passive_data(subject, *args, start_time=0): # run and condition aren't 
# used here so *args is used as a place holder
    
    ''' a function for accessing the Time and Angle (i.e. Track) in PASSIVE condition
    (from the RAW data)'''
    
    active_folder = subject+r'/data/preprocessed/'
    passive_folder = subject + r'/data/raw/Zip/'
    indices_file = [file for file in os.listdir(indices_path) # the files for the indices for each participant
                    if file.startswith(subject) 
                    and file.endswith('index_passive.txt')][0]
    parameters = parse_indices(indices_path+indices_file) # extracting the parameters
    # from the indices file, e.g. training06_LONG_5. The parse indices and parse functions are defined below.
    times = []
    angles = []    
    
    for params in parameters: # a loop for getting the information for passive trials which used training data
        if params[0].startswith('training'): # if the passive cond comes from training data
            for file in os.listdir(passive_folder):
                if (all([(p in file) for p in params]) and # for each parameter in 
                    #the indices file, is it also there in the raw data file
                    params[-1] == file.split('_')[-2] and # is the last parameter 
                    #(i.e. trial number) in the indices == the trial number in the name of the raw file?
                    file.endswith('track.txt')): # and the end of the raw file name is track
                    
                    f = np.loadtxt(passive_folder+file)
                    more_times = f[:, 0]# new times plus the last time from prev trial (will make time continue)
                    more_angles = f[:, 4]
        else: # if the passive cond comes from active data (LONG/ SHORT) in the actual fMRI runs, and not training
            File = loadmat(active_folder + [name for name in os.listdir(active_folder) if name.endswith('RawData.mat')][0])  
            subfolder = File[struct_name][params[0]].item()[params[1]].item()['trial'+params[2]].item()['Track'].item()
            more_times = subfolder[:, 0]
            more_angles = subfolder[:, 4]

        times.extend(more_times - more_times[0] + start_time) 
        # to make sure that the times of trials are in order, and are added to previous trial times
        angles.extend(more_angles)   
        start_time += more_times[-1] - more_times[0]            
    return np.array(times), np.array(angles) 
   


    
def data_to_event(times, angles, condition, start_time=None, passive=False):
   
    ''' a function for taking times and angles and converts them into events 
    with event name, onset, duration and angle '''

    d_angles = gaussian_filter(np.gradient(np.abs(angles), times), SMOOTHNESS) # applying a gaussian filter to smoothen the data;
    # we take the derivative of the angles to define an event with 
    # using absolute value of angles bec we don't care about the direction for now
    # with gradients < threshold, there's no movement of the joystick, which is a way to define events
    # the gaussian filter is to smoothen the signal and make sure the little twitches don't count as an event
    container = []
    onset = times[0] # onset of event--> the first time point of the trial
    prev_angle = angles[0]
    at_zero =  abs(d_angles[0]) < THRESHOLD_SPEED # if the previous data point was below threshold
    for i, (time, angle, d_angle) in enumerate(zip(times, angles, d_angles)):
        duration = time - onset
        if (abs(d_angle) > THRESHOLD_SPEED and # if the new event is greater than the threshold
            duration>THRESHOLD_DURATION and 
            at_zero) or i == len(times) -1: # and the previous data point was below threshold
        
            container.append([condition, # wrapping up the first event (which was already opened by the onset)
                              round(onset, DECIMALS),
                              round(duration, DECIMALS), 
                              prev_angle])
        
            onset = time # create new event, by saying that the onset = current time
           
        at_zero = abs(d_angle) < THRESHOLD_SPEED
        prev_angle = angle
      
    # some plots to make the visualisation of data easy (commented out for now)
    #plt.figure()
    #plt.plot(times, d_angles, markersize= 1, color='tab:blue')
    #print(times)
    
    #for event in container:
        #plt.axvline(event[1], color='k', alpha=0.4, linestyle='--')
    #plt.xlim(0,30)
    #if len(plt.get_fignums())>5:raise Exception
    
    return container




def append_to_txt(eventfile, subject, run, condition, start_time=0):
    
    ''' a function for adding events to the .txt file for each condition, subject and run'''
    
    # we use the previously defined functions to get the appropriate data
    get_data = get_passive_data if condition == 'PASSIVE' else get_active_data 
    events = data_to_event(*get_data(subject, run, condition, start_time=start_time), # this creates the list of our events
                           condition)
    efile = open(eventfile, 'a') # opening .txt file to write the list events in
    for e in events:
        efile.write(';'.join(map(str, e))) # convert our list of events into text
        efile.write('\n') # new line 
    return events[-1][1] + events[-1][2] # returning the last onsetb+ the last duration
                                    



def parse_indices(file):
    
    ''' function for reading .txt files and converts them to indices '''
    
    raw_passives = []
    f = open(file)
    for line in f:
        if line.startswith('#'): # to ignore the comments in the files
            continue
        raw_passives.extend(line.split()) # split the line according to spaces
    return [raw.split('_') for raw in raw_passives]




def get_ordered_conditions(subject, run):
    
    ''' function for ordering the data according to the index_pulses.mat files. 
    note that the order is identified by the numbers of each condition (lowest = first and vice versa'''

    folder = r'Indices/'
    f = loadmat(folder + [name for name in os.listdir(folder) if name.endswith('index_pulses.mat')][0]) 
    # loadig the file; the 0 makes sure we get the actual file within the list not the list itself
    
    group = f['PULSEINFO'][run].item()
    
    return sorted(conditions, key=lambda x: group[x][0][0][0][0]) # lambda is an anon function that allows one expression only


#%%
# main script: calling all the functions we defined

if __name__ == '__main__': # to be able to use the same functions in other scripts through importing them

    for subject in tqdm(os.listdir()):
        if subject[0] in ['P', 'S'] and subject[-1].isdigit() and subject != 'S09': # choosing subject folders
            print(subject)
            for run in runs:            
                start_time = 0
                outputfile = outpath + f'eventTable_run{run[-1]}_{subject}.txt'
                open(outputfile, 'w') # delete the old file and make a new one (so that we wouldn't add to it everytime we run the code)
                ordcond = get_ordered_conditions(subject, run) 
                for condition in ordcond:
                    
                    start_time = append_to_txt(outputfile,
                                               subject,
                                               run,
                                               condition,
                                               start_time=start_time) +20.
               


