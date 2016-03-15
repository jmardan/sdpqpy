# -*- coding: utf-8 -*-
"""sdpqpy
=====

Provides a high level interface to formulate and approximately solve
quantum ground state problems with the use of semi-definite programming
techniques by leveraging ncpo2sdpa.

@author: Christian Gogolin, Peter Wittek
"""
from .sdpqpy import BoseHubbardModel, FermiHubbardModel, LongRangeQuadraticFermiModel
from .ed import EDFermiHubbardModel


__all__ = ['BoseHubbardModel',
           'FermiHubbardModel',
           'LongRangeQuadraticFermiModel',
           'EDFermiHubbardModel'
       ]
