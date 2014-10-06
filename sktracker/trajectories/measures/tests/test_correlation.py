# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import pandas as pd

import numpy as np

from sktracker.trajectories.measures import correlation

from sktracker.trajectories import Trajectories
from sktracker.data import trajectories_generator


from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_almost_equal


def test_correlation():
    trajs = Trajectories(
        trajectories_generator.straight_trajectories_generator(3, 20, noise=0, shuffle=False))
    corrs = correlation.crosscorel(trajs, 'x', 5)
    assert_array_almost_equal(corrs.corr_x.dropna().values, np.ones(15))
    assert_array_almost_equal(corrs.corr_x_stm.dropna().values, np.zeros(15))
