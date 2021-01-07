#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
QSDsan: Quantitative Sustainable Design for sanitation and resource recovery systems
Copyright (C) 2020, Quantitative Sustainable Design Group

This module is developed by:
    Yalin Li <zoe.yalin.li@gmail.com>
<<<<<<< HEAD
=======
    Joy Cheung
>>>>>>> 642b6690d694ad53b832005dff17b0833e6a9705

This module is under the UIUC open-source license. Please refer to 
https://github.com/QSD-Group/QSDsan/blob/master/LICENSE.txt
for license details.
'''

import biosteam as bst
CEPCI = bst.CE # Chemical Engineering Plant Cost Index
CEPCI_by_year = bst.units.design_tools.CEPCI_by_year
del bst
currency = 'USD'

from .utils import descriptors
from ._component import *
from ._components import *
from ._waste_stream import *
from ._impact_indicator import *
from ._impact_item import *
from ._construction import *
from ._transportation import *
from ._sanunit import *
from ._simple_tea import *
from ._lca import *
from ._cod import *

from . import (
    _units_of_measure, # if not included here, then need to add to setup.py
    _cod,
    _component,
    _components,
    _waste_stream,
    _impact_indicator,
    _impact_item,
    _construction,
    _transportation,
    _sanunit,
    _simple_tea,
    _lca,
    utils,
    sanunits,
    systems,
    )

utils.secondary_importing()

__all__ = (
    *_cod.__all__,
    *_component.__all__,
    *_components.__all__,
    *_waste_stream.__all__,
    *_impact_indicator.__all__,
    *_impact_item.__all__,
    *_construction.__all__,
    *_transportation.__all__,
    *_sanunit.__all__,
    *_simple_tea.__all__,
    *_lca.__all__,
           )





