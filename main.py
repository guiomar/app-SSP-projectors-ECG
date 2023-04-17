
# Copyright (c) 2021 brainlife.io
#
# This file is a MNE python-based brainlife.io App
#


# set up environment
import os
import json
import mne
import helper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load brainlife config.json
with open('config.json','r') as config_f:
    config = helper.convert_parameters_to_None(json.load(config_f))

# == LOAD DATA ==
fname = config['mne']
raw = mne.io.read_raw_fif(fname, verbose=False)

# parameters:
# raw_event is unused
ecg_projs, ecg_events = mne.preprocessing.compute_proj_ecg(raw, raw_event=None, tmin=config['tmin'], tmax=config['tmax'], n_grad=config['n_grad'],
            n_mag=config['n_mag'], n_eeg=config['n_eeg'], l_freq=config['l_freq'], h_freq=config['h_freq'], average=config['average'], filter_length=config['filter_length'], 
            n_jobs=-1, ch_name=config['ch_name'], reject=None, flat=None, bads=[],
            avg_ref=config['avg_ref'], no_proj=config['no_proj'], event_id=config['event_id'], ecg_l_freq=config['ecg_l_freq'], ecg_h_freq=config['ecg_h_freq'], tstart=config['tstart'],
            qrs_threshold=config['qrs_threshold'], filter_method=config['filter_method'], iir_params=None, copy=False, return_drop_log=True, meg=config['meg'])

mne.write_proj('out_dir/proj.fif', ecg_projs, overwrite=True)

ecg_projs = ecg_projs[3:]

# == FIGURES ==
plt.figure(1)
fig_ep = mne.viz.plot_projs_topomap(ecg_projs, info=raw.info)
fig_ep.savefig(os.path.join('out_figs','ecg_projectors.png'))

ecg_evoked = mne.preprocessing.create_ecg_epochs(raw).average()
ecg_evoked.apply_baseline((None, None))

plt.figure(2)
e=ecg_evoked.plot_joint(picks='mag')
e.savefig(os.path.join('out_figs','meg.png'))

plt.figure(3)
e=ecg_evoked.plot_joint(picks='grad')
e.savefig(os.path.join('out_figs','grad.png'))

plt.figure(4)
e=ecg_evoked.plot_joint(picks='eeg')
e.savefig(os.path.join('out_figs','eeg.png'))





