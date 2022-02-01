import argparse
import sys
import os 
import os.path as op
import numpy as np
import pandas as pd
import itertools
from scipy import stats
import pdb

def main(arglist):

	##---Set up---##

	# Parse command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-subject', required=True, help='Subject ID')
	p = parser.parse_args()

	# Set reproducible random seed for this subject
	seed = int(42*(int(p.subject[-3:])/.0087))
	rnd = np.random.RandomState(seed)
	
	# Read in stimulus names and define stimulus and eye positions on screen 
	stim_id = pd.read_csv('../design/stim.csv')
	n_stim = len(stim_id)
	eye_pos = [0, 0]	# degrees of visual angle 
	stim_ang = [45, 135, -135, -45]  # polar angle where 0 is vertical meridian
	stim_eccen = 2.0

	##---Stimulus/condition assignment---##
	
	# Randomize old/new status for each stimulus
	n_target = n_stim/2
	stim_cond = np.repeat(['old', 'new'], n_target)
	rnd.shuffle(stim_cond)

	# Randomize stimulus position on retina during study
	angles = np.repeat(stim_ang, n_target/len(stim_ang))
	rnd.shuffle(angles)

	# Make stimulus info dataframe
	stim_info = pd.DataFrame({'stim_id': stim_id['imname'],
						      'stim_cond':stim_cond,
						      'stim_ang':np.full(n_stim, np.nan),
						      'stim_eccen': stim_eccen})
	
	# Add position info for study trials (old items only)
	old_stim = stim_info['stim_cond']=='old'
	stim_info.loc[old_stim, 'stim_ang'] = angles

	# Rotate angle to physics standard to get Cartesian coords
	getx = lambda s: s['stim_eccen'] * np.cos(np.deg2rad(90-s['stim_ang'])) 
	gety = lambda s: s['stim_eccen'] * np.sin(np.deg2rad(90-s['stim_ang']))
	stim_info.loc[old_stim, 'stim_xpos'] = stim_info.apply(getx, axis=1)
	stim_info.loc[old_stim, 'stim_ypos'] = stim_info.apply(gety, axis=1)
	

	# Update stim cond with source info (visual quarterfield)
	get_quad = lambda x: np.nan if np.isnan(x) else stim_ang.index(x) + 1
	scr_quad = stim_info['stim_ang'].apply(get_quad)
	stim_info.loc[old_stim, 'stim_cond_source'] = scr_quad


	# Order columns
	stim_info = stim_info[['stim_id', 'stim_cond', 'stim_cond_source', 
						   'stim_ang', 'stim_eccen', 'stim_xpos', 'stim_ypos']]
	stim_info_sort = stim_info.sort_values(by=['stim_id'])

	# Write stim info 
	subj_dir = op.join('../design/', p.subject)
	if not op.exists(subj_dir):
		os.makedirs(subj_dir)
	stim_info_sort.to_csv(op.join(subj_dir, 'stim_info.csv'), index=False)

	##---Shared variables---##
	n_blocks = 10
	stim_dur = 2.0
	isi_range = np.arange(4, 8) 
	isi_probs = stats.geom.pmf(np.arange(1,10), .55, isi_range[0])	

	##---Study task design---##

	# Define task variables
	sn_trials = n_target/n_blocks 	# per block 

	# Make study design (stim assigned to old condition only)
	s_design = stim_info.query("stim_cond=='old'")
	s_design = s_design.sample(frac=1, random_state=rnd).reset_index(drop=True)
	s_design.loc[:, 'block_type'] = 'study'
	s_design.loc[:, 'block'] = np.repeat(np.arange(1, n_blocks + 1), sn_trials)
	s_design.loc[:, 'trial'] = np.tile(np.arange(1, sn_trials + 1), n_blocks)

	# Assign trial timing
	isis, onsets = assign_isis(isi_range, isi_probs, sn_trials, n_blocks, stim_dur, rnd)
	s_design['isi'] = isis
	s_design['stim_onset'] = onsets
	s_design['stim_dur'] = stim_dur


	##---Recognition task design---##
	
	# Define task variables
	rn_trials = n_stim/n_blocks 	# per block 

	# Make recognition task design by copying study design and adding new stim to 
	# each block
	new_stim = stim_info.query("stim_cond=='new'")
	new_stim = np.vsplit(new_stim, n_blocks)

	r_design = []
	for b_i, (__, b) in enumerate(s_design.groupby('block')):
		r = pd.concat([b, new_stim[b_i]], sort=False)
		r = r.sample(frac=1, random_state=rnd).reset_index(drop=True)
		r_design.append(r)
	r_design = pd.concat(r_design)

	r_design.loc[:, 'block_type'] = 'recog'
	r_design.loc[:, 'block'] = np.repeat(np.arange(1, n_blocks + 1), rn_trials)
	r_design.loc[:, 'trial'] = np.tile(np.arange(1, rn_trials + 1), n_blocks)

	# Assign trial timing
	isis, onsets = assign_isis(isi_range, isi_probs, rn_trials, n_blocks, stim_dur, rnd)
	r_design['isi'] = isis
	r_design['stim_onset'] = onsets
	r_design['stim_dur'] = stim_dur


	##---Location/source task design---##
	
	# Define task variables
	ln_trials = sn_trials

	# Make location source task design (old trials only)
	l_design = []
	for b_i, (__, b) in enumerate(s_design.groupby('block')):
		b = b.sample(frac=1, random_state=rnd).reset_index(drop=True)
		l_design.append(b)
	l_design = pd.concat(l_design)

	l_design.loc[:, 'block_type'] = 'loc'
	l_design.loc[:, 'block'] = np.repeat(np.arange(1, n_blocks + 1), ln_trials)
	l_design.loc[:, 'trial'] = np.tile(np.arange(1, ln_trials + 1), n_blocks)

	# Assign trial timing
	isis, onsets = assign_isis(isi_range, isi_probs, ln_trials, n_blocks, stim_dur, rnd)
	l_design['isi'] = isis
	l_design['stim_onset'] = onsets
	l_design['stim_dur'] = stim_dur

	##---Final interleaved task design---##

	# Interleave tasks and split into two sessions
	_, s_blocks = zip(*list(s_design.groupby('block')))
	_, r_blocks = zip(*list(r_design.groupby('block')))
	_, l_blocks = zip(*list(l_design.groupby('block')))
	dblocks = [val for pair in zip(s_blocks, r_blocks, l_blocks) for val in pair]
	dses = [dblocks[:len(dblocks)//2], dblocks[len(dblocks)//2:]]

	# Finish each sessions design by adding correct block, sesion info
	for s_i, ds in enumerate(dses):

		bnums = np.arange(1, len(ds)+1)

		# Assign correct session and block numbers
		design = []
		for b_i, d  in zip(bnums, ds):
			d.loc[:, 'session'] = s_i + 1
			d.loc[:, 'block'] = b_i
			design.append(d)
		design = pd.concat(design).reset_index(drop=True)

		# Write to csv
		design = design[['session', 'block', 'trial', 'block_type'] + 
						 stim_info.columns.tolist() +
						['stim_onset', 'stim_dur', 'isi']]
		fname = 'design_ses-%02d.csv' %(s_i+1)
		design.to_csv(op.join(subj_dir, fname), index=False)


def assign_isis(isi_range, isi_probs, n_trials, n_blocks, stim_dur, rnd):

	# Get the number of trials with each isi in a block
	isi_nt = np.ceil(isi_probs[isi_range] * n_trials).astype(int)
	if np.sum(isi_nt)!=n_trials:
		diff = n_trials - np.sum(isi_nt)
		isi_nt[0] = isi_nt[0] + diff

	# Randomize the order for each block	
	isis_block = np.concatenate([[i]*n for (i, n) in zip(isi_range, isi_nt)])
	isis = [pd.Series(isis_block).sample(random_state=rnd, frac=1).values 
		    for _ in np.arange(n_blocks)]
	
	# Compute the trial onsets
	onsets = [[0] + np.cumsum(i + stim_dur)[:-1].tolist() for i in isis]
	
	# Concatenate across blocks
	isis = np.concatenate(isis).astype(float)
	onsets = np.concatenate(onsets)

	return isis, onsets


if __name__ == "__main__":
    main(sys.argv[1:])