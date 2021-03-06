
# -*- coding: utf-8 -*-


from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import pandas as pd

from nose.tools import assert_raises

from sktracker import data

from sktracker.tracker.solver import AbstractSolver
from sktracker.tracker.cost_function import AbstractCostFunction
from sktracker.tracker.cost_function.diagonal import DiagonalCostFunction
from sktracker.tracker.cost_function.brownian import BrownianLinkCostFunction


def test_solver_check_cost_function_type():

    trajs = pd.DataFrame([])
    solver = AbstractSolver(trajs)
    cost_function = BrownianLinkCostFunction(parameters={'max_speed': 1})

    solver.check_cost_function_type(cost_function,
                                    AbstractCostFunction)

    assert True


def test_solver_check_cost_function_type_failure():

    trajs = pd.DataFrame([])
    solver = AbstractSolver(trajs)
    cost_function = BrownianLinkCostFunction(parameters={'max_speed': 1})

    assert_raises(TypeError,
                  solver.check_cost_function_type,
                  cost_function,
                  DiagonalCostFunction)


def test_solver_check_trajs_df_structure():

    trajs = data.brownian_trajs_df()
    solver = AbstractSolver(trajs)

    solver.trajs.check_trajs_df_structure(index=['t_stamp', 'label'])
    solver.trajs.check_trajs_df_structure(columns=['x', 'y', 't'])

    assert True


def test_solver_check_trajs_df_structure_failure():

    trajs = data.brownian_trajs_df()
    solver = AbstractSolver(trajs)

    assert_raises(ValueError, solver.trajs.check_trajs_df_structure, ['t_wrong_stamp', 'label'])
