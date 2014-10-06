# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd

def crosscorel(trajs, col, window):
    data = trajs[col].unstack()
    times = data.index.values

    corrs = pd.DataFrame(np.zeros(times.size*2).reshape((times.shape[0], 2))*np.nan,
                         index=pd.Index(times, name='t_stamp'),
                         columns=('corr_{}'.format(col), 'corr_{}_stm'.format(col)))
    for t in times[window//2:-window//2]:
        try:
            sub_data = data.loc[t-window//2:t+window//2].dropna(axis=1).T
            if not sub_data.shape[0] > 1:
                corrs.loc[t] = np.nan, np.nan
                continue
            corr = np.corrcoef(sub_data)
            corr = np.ma.masked_equal(np.triu(corr, k=1), 0).ravel()
            corrs.loc[t] = corr.mean(), corr.std()/(corr.shape[0]**0.5)
        except KeyError:
            corrs.loc[t] = np.nan, np.nan

    corrs['t'] = trajs.t.mean(level='t_stamp')

    return corrs

def autocorrel(trajs, col, window):
    data = trajs[col].unstack()
    times = data.index.values

    corrs = pd.DataFrame(np.zeros(times.size*2).reshape((times.shape[0], 2))*np.nan,
                         index=pd.Index(times, name='t_stamp'),
                         columns=('corr_{}'.format(col), 'corr_{}_stm'.format(col)))
    for t in times[window//2:-window//2]:
        try:
            sub_data = data.loc[t-window//2:t+window//2].dropna(axis=1).T
            if not sub_data.shape[0] > 1:
                corrs.loc[t] = np.nan, np.nan
                continue
            corr = np.corrcoef(sub_data)
            corr = np.ma.masked_equal(np.triu(corr, k=1), 0).ravel()
            corrs.loc[t] = corr.mean(), corr.std()/(corr.shape[0]**0.5)
        except KeyError:
            corrs.loc[t] = np.nan, np.nan

    corrs['t'] = trajs.t.mean(level='t_stamp')

    return corrs