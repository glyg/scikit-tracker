
# -*- coding: utf-8 -*-


from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


from nose.tools import with_setup

from sktracker.utils import in_ipython

def test_in_ipython():
    """Tests are supposed to be run outside an IPython process.
    """

    assert not in_ipython()
