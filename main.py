
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

#workaround for -- _tkinter.TclError: invalid command name ".!canvas"
# so execution won't hang when figures are shown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Load brainlife config.json
with open('config.json','r') as config_f:
    config = helper.convert_parameters_to_None(json.load(config_f))

# == LOAD DATA ==
fname = config['mne']
raw = mne.io.read_raw_fif(fname, verbose=False)

ecg_projs, ecg_events = mne.preprocessing.compute_proj_ecg(raw, raw_event=None, tmin=config['tmin'], tmax=config['tmax'], n_grad=config['n_grad'],
            n_mag=config['n_mag'], n_eeg=config['n_eeg'], l_freq=config['l_freq'], h_freq=config['h_freq'], average=config['average'], 
            filter_length=config['filter_length'], 
            n_jobs=-1, ch_name=config['ch_name'], reject=None, flat=None, bads=[],
            avg_ref=config['avg_ref'], no_proj=config['no_proj'], event_id=config['event_id'], ecg_l_freq=config['ecg_l_freq'], ecg_h_freq=config['ecg_h_freq'],
            tstart=config['tstart'],
            qrs_threshold=config['qrs_threshold'], filter_method=config['filter_method'], iir_params=config['iir_params'], copy=True, return_drop_log=False,
            meg=config['meg'])

mne.write_proj('out_dir/proj.fif', ecg_projs, overwrite=True)

# == FIGURES ==
fig_ep = mne.viz.plot_projs_topomap(ecg_projs, info=raw.info)
fig_ep.savefig(os.path.join('out_figs','ecg_projectors.png'))

ecg_evoked = mne.preprocessing.create_ecg_epochs(raw).average()
ecg_evoked.apply_baseline((None, None))

f = ecg_evoked.plot_joint()
[fig.savefig(os.path.join('out_figs',f'ecg_{i}.png')) for i, fig in enumerate(f)]

report = mne.Report(title='SSP ECG Projectors')
report.add_projs(info=raw.info, projs=ecg_projs, title = 'SSP ECG Projectors')

report.save('out_report/report_ssp.html', overwrite=True)




