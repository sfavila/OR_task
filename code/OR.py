from psychopy import core, event, clock, monitors
from psychopy import visual as vis
import argparse
import os 
import sys
from glob import glob
import os.path as op
import pandas as pd
import numpy as np
import math
import copy
import pdb

try:
    import pylink
except:
    print('Could not import pylink')


def main(arglist):

    ##---Parse arguments---##
    parser = argparse.ArgumentParser()
    parser.add_argument('-subject', required=True, help='subject ID')
    parser.add_argument('-session', required=True, type=int, help='session number [1, 2]')
    parser.add_argument('-block', type=int, default=1, help='block number')
    parser.add_argument('-monitor', help='override monitor')
    parser.add_argument('-viewdist', type=float, help='override viewing distance')
    parser.add_argument('-eyetrack', action='store_true', help='do eye tracking')
    parser.add_argument('-screentest', action='store_true', help='test calibration')
    p = parser.parse_args()


    ##---Display and device set up---##

    # Define monitor params for each experiment phase
    monitor_info = {'hp957c':   {'units':'deg', 'fullscr':True, 'screen':0,
                                 'size':[1680, 1050]},
                    'propixx':  {'units':'deg', 'fullscr':True, 'screen':0,
                                 'size':[1920, 1080]}}
    default_monitor = 'propixx' 

    # Set up monitor
    if p.monitor is None:
        p.monitor = default_monitor
    mon = monitors.Monitor(p.monitor)

    if p.viewdist is not None:
        mon.setDistance(p.viewdist)

    # Connect to eyelink and calibrate
    if p.eyetrack:
        p.el = setup_eyelink(monitor_info[p.monitor]['size'])

    # Set up window (must be called after eyelink calibration)
    win = vis.Window(monitor=mon, **monitor_info[p.monitor])

    # Set quit key and turn off cursor
    try:
        event.globalKeys.add(key='q', func=core.quit)
    except:
        print('No quit key set')
    event.Mouse(visible=False)


    ##---Stimulus setup---##    

    # Store reference, fixation, and path to images
    stim = dict()
    stim['ref'] = makeref(win)
    stim['fix'] = vis.Circle(win, radius=.08, fillColor='#cb181d', lineColor='#cb181d')  
    stim['img_tmp'] = '../stim/%s.png'

    # Optionally display dartboard for testing screen
    if p.screentest:
        dartboard(win, stim)

    # Load experimental stimulus info for this subject
    stim_info = pd.read_csv(op.join('../design', p.subject, 'stim_info.csv'))


    ##---Run Experiment---##

    # Load design info for this session
    design_file = op.join('../design', p.subject, 'design_ses-%02d.csv') 
    design = pd.read_csv(design_file %p.session)

    # Figure out starting block and num blocks
    if p.block is not None:
        design = design.query("block>=@p.block")
    nblocks = design['block'].max()

    # Make data directory
    data_dir = op.join('../data', p.subject)
    if not op.exists(data_dir):
        os.makedirs(data_dir)

    # Iterate through blocks and execute correct block
    for b, b_design in design.groupby('block'):

        btype = b_design['block_type'].iloc[0]

        # Execute study, recognition test, or location test block
        if btype == 'study':
            data = study(win, p, stim, b_design, nblocks)
        elif btype == 'recog':
            data = recog(win, p, stim, b_design, nblocks)
        elif btype == 'loc':
            data = loc(win, p, stim, b_design, nblocks)

        # Write data to csv
        data = pd.concat(data, sort=False).reset_index(drop=True)
        data_file = op.join(data_dir, 'sub-%s_ses-%02d_task-%s_run-%02d_behav.csv')
        data.to_csv(data_file %(p.subject, p.session, btype, b), index=False)

        # Stop eyetracker and get edf
        if p.eyetrack:
            edf_file = op.join(data_dir, 'et_block%02d.EDF' %p.block)
            p.el.stopRecording()
            p.el.closeDataFile()
            p.el.receiveDataFile('temp.EDF', edf_file)
            p.el.close();

    # Quit
    event.clearEvents()
    win.close()
    core.quit()


def setup_eyelink(win_size):
    """Set up eyetracking with Eyelink"""

    # Connect to eyelink
    el = pylink.EyeLink('192.168.1.5')
    pylink.openGraphics()

    # Set contents of edf file
    el.sendCommand('link_sample_data=LEFT,RIGHT,GAZE,AREA')
    el.sendCommand('file_sample_data=LEFT,RIGHT,GAZE,AREA,GAZERES,STATUS')
    el.sendCommand('file_event_filter=LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON')

    # Set coords
    el.sendCommand('screen_pixel_coords=0 0 {} {}'.format(*win_size))
    el.sendMessage('DISPLAY_COORDS 0 0 {} {}'.format(*win_size))

    # Calibrate
    el.setCalibrationType('HV9')
    el.doTrackerSetup(win_size)
    pylink.closeGraphics()

    return el


def study(win, p, stim, b_design, nblocks):
    """Execute study block"""

    # Make text stimuli for this phase
    text = {'start' : ['Press space to start!', 2], 
            'study1': ['Fixate on the red dot.', 1],
            'study2': ['Judge whether each object is smaller or larger than a shoebox.', 0],
            'skeys' : ['1 = smaller     2 = larger    3 = unrecognizable', -1],
            'next'  : ['Block finished!', 0],
            }

    for (key, (t, pos)) in text.items():
        stim[key] = vis.TextStim(win, text=t, height=.5, pos=[0, pos])

    # Show instructions and load image stimuli
    bnum = b_design['block'].iloc[0]
    btext = vis.TextStim(win, text='Loading stimuli for Block %d!' %bnum, height=.5, pos=[0, 2])
    btext.draw()
    stim['study1'].draw()
    stim['study2'].draw()
    stim['skeys'].draw()
    win.flip()

    img_stim = dict()
    for s_i, s in b_design.iterrows():   # stimuli
        img_stim[s['stim_id']] = vis.ImageStim(win, image=stim['img_tmp'] %s['stim_id'],
                                               pos=[s['stim_xpos'], s['stim_ypos']],
                                               size=2.5)
    core.wait(3)

    # Timing parameters
    stim_dur = 2.0
    leadout_dur = 4.0

    # Trigger and response keys
    trigger_key = '5'
    resp_keys = np.arange(1,10).astype(str)
    resp_keys = resp_keys[resp_keys!=trigger_key]

    # Initialize data structure
    data = []

    # Lead in fixation
    [r.draw() for r in stim['ref']]
    stim['fix'].draw()
    win.flip()
    event.waitKeys(keyList=[trigger_key])

    # Start eyetracker recording
    if p.eyetrack:
        p.el.openDataFile('temp.EDF')
        pylink.flushGetkeyQueue()
        p.el.startRecording(1, 1, 1, 1)

    # Start stimulus/trial timing clock
    timer = clock.MonotonicClock()

    # Iterate through trials
    for t_i, t in b_design.iterrows():

        # Display trial image
        [r.draw() for r in stim['ref']]
        stim['fix'].draw()
        img_stim[t['stim_id']].draw()
        stim_ts = timer.getTime()
        win.flip()

        # Record trial onset in edf
        if p.eyetrack: 
            p.el.sendMessage("TRIALID %02d %s" %(t_i+1, t['stim_id']))

        # Listen for subject button press 
        rt_clock = core.MonotonicClock()
        resp = collect_keys(timer, rt_clock, resp_keys, stim_dur, t)

        # Start ISI fixation and record responses for first sec
        [r.draw() for r in stim['ref']]
        stim['fix'].draw()
        win.flip()
        resp += collect_keys(timer, rt_clock, resp_keys, stim_dur+2, t)

        # Start rest of ISI
        isi = core.StaticPeriod(screenHz=60)
        isi.start(t['isi']-2) 

        # Record trial data
        rdata = record_response(resp, t)
        d = t.to_frame().transpose().join(rdata)
        d['stim_timestamp'] = stim_ts
        data.append(d)

        # Finish ISI fixation
        isi.complete()


    # Start lead out time
    leadout = core.StaticPeriod(screenHz=60)
    leadout.start(leadout_dur) 
    leadout.complete()
    print(timer.getTime())

    # Wait for response to proceed with next block or finish
    stim['next'].draw()
    win.flip()
    core.wait(2.0)

    return data


def recog(win, p, stim, b_design, nblocks):
    """Execute recognition test block"""

    # Make text stimuli for this phase
    text = {'start' : ['Get ready!', -1], 
            'recog' : ['Fixate on the red dot. Judge whether each object is old or new.', 1],
            'rkeys' : ['1 = old    2 = new', 0],
            'next'  : ['Block finished!', 0],
            }
    for (key, (t, pos)) in text.items():
        stim[key] = vis.TextStim(win, text=t, height=.5, pos=[0, pos])

    # Show instructions and load image stimuli
    bnum = b_design['block'].iloc[0]
    btext = vis.TextStim(win, text='Loading stimuli for Block %d!' %bnum, height=.5, pos=[0, 2])
    btext.draw()
    stim['recog'].pos = [0, 1.0]
    stim['recog'].draw()
    stim['rkeys'].draw()
    stim['start'].pos = [0, -1.0]
    stim['start'].draw()
    win.flip()

    img_stim = dict()
    for s_i, s in b_design.iterrows():   # stimuli
        img_stim[s['stim_id']] = vis.ImageStim(win, image=stim['img_tmp'] %s['stim_id'],
                                               pos=[s['stim_xpos'], s['stim_ypos']],
                                               size=1.5)
    core.wait(3)

    # Timing parameters
    stim_dur = 2.0
    leadout_dur = 4.0

    # Trigger and response keys
    trigger_key = '5'
    resp_keys = np.arange(1,10).astype(str)
    resp_keys = resp_keys[resp_keys!=trigger_key]

    # Initialize data structure
    data = []

    # Lead in fixation
    [r.draw() for r in stim['ref']]
    stim['fix'].draw()
    win.flip()
    event.waitKeys(keyList=[trigger_key])

    # Start eyetracker recording
    if p.eyetrack:
        p.el.openDataFile('temp.EDF')
        pylink.flushGetkeyQueue()
        p.el.startRecording(1, 1, 1, 1)

    # Start stimulus/trial timing clock
    timer = clock.MonotonicClock()

    # Iterate through trials
    for t_i, t in b_design.iterrows():

        # Display trial image
        [r.draw() for r in stim['ref']]
        img_stim[t['stim_id']].pos = [0, 0]
        img_stim[t['stim_id']].draw()
        stim['fix'].draw()
        stim_ts = timer.getTime()
        win.flip()

        # Record trial onset in edf
        if p.eyetrack: 
            p.el.sendMessage("TRIALID %02d %s" %(t_i+1, t['stim_id']))

        # Listen for subject button press 
        rt_clock = core.MonotonicClock()
        resp = collect_keys(timer, rt_clock, resp_keys, stim_dur, t)

        # Start ISI fixation and record responses for first two sec
        [r.draw() for r in stim['ref']]
        stim['fix'].draw()
        win.flip()
        resp += collect_keys(timer, rt_clock, resp_keys, stim_dur+2, t)

        # Start rest of ISI
        isi = core.StaticPeriod(screenHz=60)
        isi.start(t['isi']-2) 

        # Record trial data
        rdata = record_response(resp, t)
        d = t.to_frame().transpose().join(rdata)
        d['stim_timestamp'] = stim_ts
        data.append(d)

        # Finish ISI fixation
        isi.complete()


    # Start lead out time
    leadout = core.StaticPeriod(screenHz=60)
    leadout.start(leadout_dur) 
    leadout.complete()
    print(timer.getTime())

    # Wait for response to proceed with next block or finish
    stim['next'].draw()
    win.flip()
    core.wait(2.0)

    return data


def loc(win, p, stim, b_design, nblocks):
    """Execute location test block"""

    # Make text stimuli for this phase
    text = {'start': ['Press space to start!', 2], 
            'loc1' : ['Fixate on the red dot.', 1],
            'loc2' : ['Where did you see this object in the first phase?', 0],
            'next' : ['Block finished!', 0],
            }
    for (key, (t, pos)) in text.items():
        stim[key] = vis.TextStim(win, text=t, height=.5, pos=[0, pos])

    # Show instructions and load image stimuli
    bnum = b_design['block'].iloc[0]
    btext = vis.TextStim(win, text='Loading stimuli for Block %d!' %bnum, height=.5, pos=[0, 2])
    btext.draw()
    stim['loc1'].draw()
    stim['loc2'].draw()
    source_cue(win, stim, start=(0, -1.5), length=1, txtheight=.5, fix=False)


    img_stim = dict()
    for s_i, s in b_design.iterrows():   # stimuli
        img_stim[s['stim_id']] = vis.ImageStim(win, image=stim['img_tmp'] %s['stim_id'],
                                               pos=[s['stim_xpos'], s['stim_ypos']],
                                               size=1.5)
    core.wait(5)
    
    # Timing parameters
    stim_dur = 2.0
    leadout_dur = 4.0

    # Trigger and response keys
    trigger_key = '5'
    resp_keys = np.arange(1,10).astype(str)
    resp_keys = resp_keys[resp_keys!=trigger_key]

    # Initialize data structure
    data = []

    # Lead in fixation
    [r.draw() for r in stim['ref']]
    stim['fix'].draw()
    win.flip()
    event.waitKeys(keyList=[trigger_key])

    # Start eyetracker recording
    if p.eyetrack:
        p.el.openDataFile('temp.EDF')
        pylink.flushGetkeyQueue()
        p.el.startRecording(1, 1, 1, 1)

    # Start stimulus/trial timing clock
    timer = clock.MonotonicClock()

    # Iterate through trials
    for t_i, t in b_design.iterrows():

        # Display trial image
        [r.draw() for r in stim['ref']]
        img_stim[t['stim_id']].pos = [0, 0]
        img_stim[t['stim_id']].draw()
        stim['fix'].draw()
        stim_ts = timer.getTime()
        win.flip()

        # Record trial onset in edf
        if p.eyetrack: 
            p.el.sendMessage("TRIALID %02d %s" %(t_i+1, t['stim_id']))

        # Listen for subject button press 
        rt_clock = core.MonotonicClock()
        resp = collect_keys(timer, rt_clock, resp_keys, stim_dur, t)

        # Start ISI fixation and record responses for first two sec
        [r.draw() for r in stim['ref']]
        stim['fix'].draw()
        win.flip()
        resp += collect_keys(timer, rt_clock, resp_keys, stim_dur+2, t)

        # Start rest of ISI
        isi = core.StaticPeriod(screenHz=60)
        isi.start(t['isi']-2) 

        # Record trial data
        rdata = record_response(resp, t)
        d = t.to_frame().transpose().join(rdata)
        d['stim_timestamp'] = stim_ts
        data.append(d)

        # Finish ISI fixation
        isi.complete()

    # Start lead out time
    leadout = core.StaticPeriod(screenHz=60)
    leadout.start(leadout_dur) 
    leadout.complete()
    print(timer.getTime())

    # Wait for response to proceed with next block or finish
    stim['next'].draw()
    win.flip()
    core.wait(2.0)

    return data


def collect_keys(timer, rt_clock, resp_keys, resp_dur, t):
    """Listen for key presses"""

    resp = []
    event.clearEvents()
    while timer.getTime() < t['stim_onset'] + resp_dur:
        resp += event.getKeys(keyList=resp_keys, timeStamped=rt_clock)

    return resp


def record_response(r, t):
    """Code key press meaning"""
    
    # Define responses for each phase and trial type
    resp_map = {'study': {'1':'smaller', '2':'larger', '3':'unrecognizable'},
                'recog': {'1': 'old',    '2': 'new'},
                'loc':   {'1': 1.0, '2':2.0, '3':3.0, '4':4.0}}             
    btype = t['block_type']

    # Record response if one was made
    if r:
        made_resp = True
        key, rt = r[0]
        resp = resp_map[btype][key]

        if btype == 'study':
            acc = np.nan
        elif btype == 'recog':
            acc = int(resp == t['stim_cond'])
        elif btype == 'loc':
            acc = int(resp == t['stim_cond_source'])

    else:
        made_resp = False
        key = np.nan
        rt = np.nan
        resp = np.nan
        acc = np.nan

    # Save response data
    rdata = dict(made_resp=made_resp, key=key, rt=rt, resp=resp, acc=acc)
    rdata = pd.DataFrame(rdata, index=[t.name])
    rdata = rdata[['made_resp', 'key', 'rt', 'resp', 'acc']]

    return rdata


def makeref(win):
    """"Define reference image"""

    ref_radii = [2, 4, 6, 8, 10]
    circ = lambda r, n: [(math.cos(2*math.pi/n*x)*r, 
                          math.sin(2*math.pi/n*x)*r) 
                         for x in np.arange(0, n+1)]
    pts = zip(circ(0, 4), circ(ref_radii[-1], 4))

    ref = []
    for r in ref_radii:
        ref.append(vis.Circle(win, radius=r, lineWidth=.25, edges=100, 
                              lineColor='white'))
    for (p1, p2) in pts:
        ref.append(vis.Line(win, start=p1, end=p2, lineWidth=.25, 
                            lineColor='white'))

    return ref


def dartboard(win, stim):
    """Display dartboard to test stimulus display"""

    # Concentric circles to 12 degrees
    dart_radii = [12, 10, 8, 6, 4, 2, .1]
    dart_colors = ['red','orange','yellow','green','blue','purple', 'white']

    dart = []
    for r, c in zip(dart_radii, dart_colors):
        d = vis.Circle(win, radius=r, lineWidth=2, edges=100, fillColor=c)
        d.draw()

    # Horizontal and vertical meridian
    l1 = vis.Line(win, start=[-25, 0], end=[25, 0], lineWidth=3, lineColor='black')
    l2 = vis.Line(win, start=[0, 25], end=[0, -25], lineWidth=3, lineColor='black')
    l1.draw()
    l2.draw()

    win.flip() 
    event.waitKeys()


def source_cue(win, stim, start=(0, 0), length=.25, txtheight=.3, fix=True):
    """Draw quadrant definition instructions for source/location task"""

    nums = [1, 2, 3, 4]
    ang = [45, 135, 225, 315]

    getx = lambda a, l: start[0] + l * np.cos(np.deg2rad(90-a))
    gety = lambda a, l: start[1] + length * np.sin(np.deg2rad(90-a))

    endpts = zip([getx(a, length) for a in ang], [gety(a, length) for a in ang])
    textpts = zip([getx(a, length * 1.3) for a in ang], [gety(a, length * 1.3) for a in ang])
    for (e, t, n) in zip(endpts, textpts, nums):
        l = vis.Line(win, start=start, end=e, lineWidth=3, lineColor='white')
        l.draw()
        n = vis.TextStim(win, n, pos=t, height=txtheight)
        n.draw()

    if fix:
        stim['fix'].draw()
    win.flip()


if __name__ == "__main__":
    main(sys.argv[1:])