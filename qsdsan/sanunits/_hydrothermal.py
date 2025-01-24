#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
QSDsan: Quantitative Sustainable Design for sanitation and resource recovery systems

This module is developed by:

    Jianan Feng <jiananf2@illinois.edu>

    Yalin Li <mailto.yalin.li@gmail.com>
    
    Andrew Koehler <koehler@mines.edu>

This module is under the University of Illinois/NCSA Open Source License.
Please refer to https://github.com/QSD-Group/QSDsan/blob/main/LICENSE.txt
for license details.
'''

from biosteam.units.decorators import cost
from biosteam.units.design_tools import CEPCI_by_year
from qsdsan import SanUnit, Stream
from qsdsan.utils import auom
from . import Reactor, HXutility, Pump
from scipy.integrate import odeint
import numpy as np
import pandas as pd
import math

__all__ = (
    'CatalyticHydrothermalGasification',
    'HydrothermalLiquefaction'
    )

_lb_to_kg = auom('lb').conversion_factor('kg')
_m3_to_gal = auom('m3').conversion_factor('gallon')
_in_to_m = auom('inch').conversion_factor('m')

# %%

# =============================================================================
# CHG
# =============================================================================

@cost(basis='Treatment capacity', ID='Hydrocyclone', units='lb/h',
      cost=5000000, S=968859,
      CE=CEPCI_by_year[2009], n=0.65, BM=2.1)
class CatalyticHydrothermalGasification(Reactor):
    '''
    CHG serves to reduce the COD content in the aqueous phase and produce fuel
    gas under elevated temperature (350°C) and pressure. The outlet will be
    cooled down and separated by a flash unit.
    
    Parameters
    ----------
    ins : Iterable(stream)
        chg_in, catalyst_in.
    outs : Iterable(stream)
        chg_out, catalyst_out.
    pump_pressure: float
        CHG influent pressure, [Pa].
    heat_temp: float
        CHG influent temperature, [K].
    cool_temp: float
        CHG effluent temperature, [K].
    WHSV: float
        Weight Hourly Space velocity, [kg feed/hr/kg catalyst].
    catalyst_lifetime: float
        CHG catalyst lifetime, [hr].
    gas_composition: dict
        CHG gas composition.
    gas_C_2_total_C: dict
        CHG gas carbon content to feed carbon content.
    CAPEX_factor: float
        Factor used to adjust CAPEX.
        
    References
    ----------
    [1] Jones, S. B.; Zhu, Y.; Anderson, D. B.; Hallen, R. T.; Elliott, D. C.; 
        Schmidt, A. J.; Albrecht, K. O.; Hart, T. R.; Butcher, M. G.; Drennan, C.; 
        Snowden-Swan, L. J.; Davis, R.; Kinchin, C. 
        Process Design and Economics for the Conversion of Algal Biomass to
        Hydrocarbons: Whole Algae Hydrothermal Liquefaction and Upgrading;
        PNNL--23227, 1126336; 2014; https://doi.org/10.2172/1126336.
    [2] Davis, R. E.; Grundl, N. J.; Tao, L.; Biddy, M. J.; Tan, E. C.;
        Beckham, G. T.; Humbird, D.; Thompson, D. N.; Roni, M. S. Process
        Design and Economics for the Conversion of Lignocellulosic Biomass
        to Hydrocarbon Fuels and Coproducts: 2018 Biochemical Design Case
        Update; Biochemical Deconstruction and Conversion of Biomass to Fuels
        and Products via Integrated Biorefinery Pathways; NREL/TP--5100-71949,
        1483234; 2018; p NREL/TP--5100-71949, 1483234.
        https://doi.org/10.2172/1483234.
    [3] Elliott, D. C.; Neuenschwander, G. G.; Hart, T. R.; Rotness, L. J.;
        Zacher, A. H.; Santosa, D. M.; Valkenburg, C.; Jones, S. B.;
        Rahardjo, S. A. T. Catalytic Hydrothermal Gasification of Lignin-Rich
        Biorefinery Residues and Algae Final Report. 87.
    '''
    _N_ins = 2
    _N_outs = 2
    
    _F_BM_default = {**Reactor._F_BM_default,
                      'Heat exchanger': 3.17,
                      'Sulfur guard': 2.0}
    _units= {'Treatment capacity': 'lb/h', # hydrocyclone
              'Hydrocyclone weight': 'lb'}
    
    auxiliary_unit_names=('pump','heat_ex_heating','heat_ex_cooling')
    
    def __init__(self, ID='', ins=None, outs=(), thermo=None,
                  init_with='Stream',
                  pump_pressure=3089.7*6894.76,
                  heat_temp=350+273.15,
                  cool_temp=60+273.15,
                  WHSV=3.562,
                  catalyst_lifetime=7920, # 1 year [1]
                  gas_composition={'CH4':0.527,
                                  'CO2':0.432,
                                  'C2H6':0.011,
                                  'C3H8':0.030,
                                  'H2':0.0001}, # [1]
                  gas_C_2_total_C=0.5981, # [1]
                  P=None, tau=20/60, void_fraction=0.5, # [2, 3]
                  length_to_diameter=2, diameter=None,
                  N=6, V=None, auxiliary=False,
                  mixing_intensity=None, kW_per_m3=0,
                  wall_thickness_factor=1,
                  vessel_material='Stainless steel 316',
                  vessel_type='Vertical',
                  CAPEX_factor=1):
        
        SanUnit.__init__(self, ID, ins, outs, thermo, init_with)
        
        self.pump_pressure = pump_pressure
        self.heat_temp = heat_temp
        self.cool_temp = cool_temp
        self.WHSV = WHSV
        self.catalyst_lifetime = catalyst_lifetime
        self.gas_composition = gas_composition
        self.gas_C_2_total_C = gas_C_2_total_C
        pump_in = Stream(f'{ID}_pump_in')
        pump_out = Stream(f'{ID}_pump_out')
        self.pump = Pump(ID=f'.{ID}_pump', ins=pump_in, outs=pump_out, P=pump_pressure)
        hx_ht_in = Stream(f'{ID}_hx_ht_in')
        hx_ht_out = Stream(f'{ID}_hx_ht_out')
        self.heat_ex_heating = HXutility(ID=f'.{ID}_hx_ht', ins=hx_ht_in, outs=hx_ht_out, T=heat_temp, rigorous=True)
        hx_cl_in = Stream(f'{ID}_hx_cl_in')
        hx_cl_out = Stream(f'{ID}_hx_cl_out')
        self.heat_ex_cooling = HXutility(ID=f'.{ID}_hx_cl', ins=hx_cl_in, outs=hx_cl_out, T=cool_temp, rigorous=True)
        self.P = P
        self.tau = tau
        self.V_wf = void_fraction
        # no headspace, gases produced will be vented, so V_wf = void fraction [2, 3]
        self.length_to_diameter = length_to_diameter
        self.diameter = diameter
        self.N = N
        self.V = V
        self.auxiliary = auxiliary
        self.mixing_intensity = mixing_intensity
        self.kW_per_m3 = kW_per_m3
        self.wall_thickness_factor = wall_thickness_factor
        self.vessel_material = vessel_material
        self.vessel_type = vessel_type
        self.CAPEX_factor = CAPEX_factor
        
    def _run(self):
        
        chg_in, catalyst_in = self.ins
        chg_out, catalyst_out = self.outs
        
        catalyst_in.imass['CHG_catalyst'] = chg_in.F_mass/self.WHSV/self.catalyst_lifetime
        catalyst_in.phase = 's'
        catalyst_out.copy_like(catalyst_in)
        # catalysts amount is quite low compared to the main stream, therefore do not consider
        # heating/cooling of catalysts
        
        # chg_out.phase='g'
        
        cmps = self.components
        gas_C_ratio = 0
        for name, ratio in self.gas_composition.items():
            gas_C_ratio += ratio*cmps[name].i_C
            
        gas_mass = chg_in.imass['C']*self.gas_C_2_total_C/gas_C_ratio
        
        for name,ratio in self.gas_composition.items():
            chg_out.imass[name] = gas_mass*ratio
                
        chg_out.imass['H2O'] = chg_in.F_mass - gas_mass
        # all C, N, and P are accounted in H2O here, but will be calculated as properties.
        
        chg_out.T = self.cool_temp
        chg_out.P = self.pump_pressure

        # chg_out.vle(T=chg_out.T, P=chg_out.P)

    @property
    def CHGout_C(self):
        # not include carbon in gas phase
        return self.ins[0].imass['C']*(1 - self.gas_C_2_total_C)
    
    @property
    def CHGout_N(self):
        return self.ins[0].imass['N']
    
    @property
    def CHGout_P(self):
        return self.ins[0].imass['P']
        
    def _design(self):
        Design = self.design_results
        Design['Treatment capacity'] = self.ins[0].F_mass/_lb_to_kg
        
        pump = self.pump
        pump.ins[0].copy_like(self.ins[0])
        pump.simulate()
        
        hx_ht = self.heat_ex_heating
        hx_ht_ins0, hx_ht_outs0 = hx_ht.ins[0], hx_ht.outs[0]
        hx_ht_ins0.copy_like(self.ins[0])
        hx_ht_outs0.copy_like(hx_ht_ins0)
        hx_ht_ins0.T = self.ins[0].T
        hx_ht_outs0.T = hx_ht.T
        hx_ht_ins0.P = hx_ht_outs0.P = pump.P
        
        hx_ht_ins0.vle(T=hx_ht_ins0.T, P=hx_ht_ins0.P)
        hx_ht_outs0.vle(T=hx_ht_outs0.T, P=hx_ht_outs0.P)
        
        hx_ht.simulate_as_auxiliary_exchanger(ins=hx_ht.ins, outs=hx_ht.outs)
            
        hx_cl = self.heat_ex_cooling
        hx_cl_ins0, hx_cl_outs0 = hx_cl.ins[0], hx_cl.outs[0]
        hx_cl_ins0.copy_like(self.outs[0])
        hx_cl_outs0.copy_like(hx_cl_ins0)
        hx_cl_ins0.T = hx_ht.T
        hx_cl_outs0.T = hx_cl.T
        hx_cl_ins0.P = hx_cl_outs0.P = self.outs[0].P

        hx_cl_ins0.vle(T=hx_cl_ins0.T, P=hx_cl_ins0.P)
        hx_cl_outs0.vle(T=hx_cl_outs0.T, P=hx_cl_outs0.P)        
        
        hx_cl.simulate_as_auxiliary_exchanger(ins=hx_cl.ins, outs=hx_cl.outs)

        self.P = self.pump_pressure
        Reactor._design(self)
        Design['Hydrocyclone weight'] = 0.3*Design['Weight']*Design['Number of reactors'] # assume stainless steel
        # based on [1], page 54, the purchase price of hydrocyclone to the purchase price of CHG
        # reactor is around 0.3, therefore, assume the weight of hydrocyclone is 0.3*single CHG weight*number of CHG reactors
        self.construction[0].quantity += Design['Hydrocyclone weight']*_lb_to_kg
    
    def _cost(self):
        Reactor._cost(self)
        purchase_costs = self.baseline_purchase_costs
        current_cost = 0 # cost w/o sulfur guard
        for item in purchase_costs.keys():
            current_cost += purchase_costs[item]
        purchase_costs['Sulfur guard'] = current_cost*0.05
        self._decorated_cost()
        
        purchase_costs = self.baseline_purchase_costs
        for item in purchase_costs.keys():
            purchase_costs[item] *= self.CAPEX_factor
        
# %%

# =============================================================================
# KOdrum
# =============================================================================

class KnockOutDrum(Reactor):
    '''
    Knockout drum is an auxiliary unit for :class:`HydrothermalLiquefaction`.
    
    References
    ----------
    [1] Knorr, D.; Lukas, J.; Schoen, P. Production of Advanced Biofuels via
        Liquefaction - Hydrothermal Liquefaction Reactor Design: April 5, 2013;
        NREL/SR-5100-60462, 1111191; 2013; p NREL/SR-5100-60462, 1111191.
        https://doi.org/10.2172/1111191.
        
    See Also
    --------
    :class:`qsdsan.sanunits.HydrothermalLiquefaction`
    '''
    _N_ins = 3
    _N_outs = 2
    _ins_size_is_fixed = False
    _outs_size_is_fixed = False
    
    def __init__(self, ID='', ins=None, outs=(), thermo=None,
                 init_with='Stream',
                 P=3049.7*6894.76, tau=0, V_wf=0,
                 length_to_diameter=2, diameter=None,
                 N=4, V=None,
                 auxiliary=True,
                 mixing_intensity=None, kW_per_m3=0,
                 wall_thickness_factor=1,
                 vessel_material='Stainless steel 316',
                 vessel_type='Vertical',
                 drum_steel_cost_factor=1.5):
        # drum_steel_cost_factor: so the cost matches [1]
        # when do comparison, if fully consider scaling factor (2000 tons/day to 100 tons/day),
        # drum_steel_cost_factor should be around 3
        # but that is too high, we use 1.5 instead.
        
        SanUnit.__init__(self, ID, ins, outs, thermo, init_with)
        self.P = P
        self.tau = tau
        self.V_wf = V_wf
        self.length_to_diameter = length_to_diameter
        self.diameter = diameter
        self.N = N
        self.V = V
        self.auxiliary = auxiliary
        self.mixing_intensity = mixing_intensity
        self.kW_per_m3 = kW_per_m3
        self.wall_thickness_factor = wall_thickness_factor
        self.vessel_material = vessel_material
        self.vessel_type = vessel_type
        self.drum_steel_cost_factor = drum_steel_cost_factor
    
    def _run(self):
        pass
    
    def _cost(self):
        Reactor._cost(self)
        
        purchase_costs = self.baseline_purchase_costs
        purchase_costs['Vertical pressure vessel'] *= self.drum_steel_cost_factor

# =============================================================================
# HTL_MCA
# =============================================================================

# separator
@cost(basis='Treatment capacity', ID='Solids filter oil/water separator', units='lb/h',
      cost=3945523, S=1219765,
      CE=CEPCI_by_year[2011], n=0.68, BM=1.9)
class HydrothermalLiquefaction(Reactor):
    '''
    HTL converts dewatered sludge to biocrude, aqueous, off-gas, and hydrochar
    under elevated temperature (350°C) and pressure. The products percentage
    (wt%) can be evaluated using revised MCA model (Li et al., 2017,
    Leow et al., 2018) with known sludge composition (protein%, lipid%,
    and carbohydrate%, all afdw%).
                                                      
    Notice that for HTL we just calculate each phases' total mass (except gas)
    and calculate C, N, and P amount in each phase as properties. We don't
    specify components for oil/char since we want to use MCA model to calculate
    C and N amount and it is not necessary to calculate every possible
    components since they will be treated in HT/AcidEx anyway. We also don't
    specify components for aqueous since we want to calculate aqueous C, N, and
    P based on mass balance closure. But later for CHG, HT, and HC, we specify
    each components (except aqueous phase) for the application of flash,
    distillation column, and CHP units.
    
    Parameters
    ----------
    ins : Iterable(stream)
        dewatered_sludge.
    outs : Iterable(stream)
        hydrochar, HTLaqueous, biocrude, offgas.
    HTL_model: str
        Can only be 'MCA' or 'kinetics'.
    feedstock: str
        Can only be 'sludge' or 'biosolid'.
    rxn_moisture: float
        The moisture content of the HTL reactor, ranging from 0 to 1.
    NaOH_M: int or float
        The concentration of NaOH, [M].
    HCl_neut: bool
        If True, HCl is added to neutralized NaOH,
        If False, HCl is not added.
    rxn_temp: int or float
        HTL reaction temperature, [°C].
    rxn_time: int or floar
        HTL reaction time, [min].
    lipid_2_biocrude: float
        Lipid to biocrude factor.
    protein_2_biocrude: float
        Protein to biocrude factor.
    carbo_2_biocrude: float
        Carbohydrate to biocrude factor.
    protein_2_gas: float
        Protein to gas factor.
    carbo_2_gas: float
        Carbohydrate to gas factor.
    biocrude_C_slope: float
        Biocrude carbon content slope.
    biocrude_C_intercept: float
        Biocrude carbon content intercept.
    biocrude_N_slope: float
        Biocrude nitrogen content slope.
    biocrude_H_slope: float
        Biocrude hydrogen content slope.
    biocrude_H_intercept: float
        Biocrude hydrogen content intercept.
    HTLaqueous_C_slope: float
        HTLaqueous carbon content slope.
    TOC_TC: float   
        HTL TOC/TC.
    hydrochar_C_slope: float
        Hydrochar carbon content slope.
    biocrude_moisture_content: float
        Biocrude moisture content.
    hydrochar_P_recovery_ratio: float
        Hydrochar phosphorus to total phosphorus ratio.
    gas_composition: dict
        HTL offgas compositions.
    hydrochar_pre: float
        Hydrochar pressure, [Pa].
    HTLaqueous_pre: float
        HTL aqueous phase pressure, [Pa].
    biocrude_pre: float
        Biocrude pressure, [Pa].
    offgas_pre: float
        Offgas pressure, [Pa].
    eff_T: float
        HTL effluent temperature, [K].
    CAPEX_factor: float
        Factor used to adjust CAPEX.
    HTL_steel_cost_factor: float
        Factor used to adjust the cost of stainless steel.
    mositure_adjustment_exist_in_the_system: bool
        If a moisture adjustment unit exists, set to true.

    References
    ----------
    [1] Leow, S.; Witter, J. R.; Vardon, D. R.; Sharma, B. K.;
        Guest, J. S.; Strathmann, T. J. Prediction of Microalgae Hydrothermal
        Liquefaction Products from Feedstock Biochemical Composition.
        Green Chem. 2015, 17 (6), 3584–3599. https://doi.org/10.1039/C5GC00574D.
    [2] Li, Y.; Leow, S.; Fedders, A. C.; Sharma, B. K.; Guest, J. S.;
        Strathmann, T. J. Quantitative Multiphase Model for Hydrothermal
        Liquefaction of Algal Biomass. Green Chem. 2017, 19 (4), 1163–1174.
        https://doi.org/10.1039/C6GC03294J.
    [3] Li, Y.; Tarpeh, W. A.; Nelson, K. L.; Strathmann, T. J.
        Quantitative Evaluation of an Integrated System for Valorization of
        Wastewater Algae as Bio-Oil, Fuel Gas, and Fertilizer Products.
        Environ. Sci. Technol. 2018, 52 (21), 12717–12727.
        https://doi.org/10.1021/acs.est.8b04035.
    [4] Jones, S. B.; Zhu, Y.; Anderson, D. B.; Hallen, R. T.; Elliott, D. C.; 
        Schmidt, A. J.; Albrecht, K. O.; Hart, T. R.; Butcher, M. G.; Drennan, C.; 
        Snowden-Swan, L. J.; Davis, R.; Kinchin, C. 
        Process Design and Economics for the Conversion of Algal Biomass to
        Hydrocarbons: Whole Algae Hydrothermal Liquefaction and Upgrading;
        PNNL--23227, 1126336; 2014; https://doi.org/10.2172/1126336.
    [5] Matayeva, A.; Rasmussen, S. R.; Biller, P. Distribution of Nutrients and
        Phosphorus Recovery in Hydrothermal Liquefaction of Waste Streams.
        BiomassBioenergy 2022, 156, 106323.
        https://doi.org/10.1016/j.biombioe.2021.106323.
    [6] Knorr, D.; Lukas, J.; Schoen, P. Production of Advanced Biofuels
        via Liquefaction - Hydrothermal Liquefaction Reactor Design:
        April 5, 2013; NREL/SR-5100-60462, 1111191; 2013; p NREL/SR-5100-60462,
        1111191. https://doi.org/10.2172/1111191.
    '''
    _N_ins = 4
    _N_outs = 4
    _units= {'Treatment capacity': 'lb/h',
             'Solid filter and separator weight': 'lb'}
    
    auxiliary_unit_names=('heat_exchanger','kodrum')

    _F_BM_default = {**Reactor._F_BM_default,
                     'Heat exchanger': 3.17}

    def __init__(self, ID='', ins=None, outs=(), thermo=None,
                 init_with='Stream',
                 HTL_model='MCA',
                 feedstock='sludge',
                 rxn_moisture=0.8,
                 NaOH_M=1,
                 HCl_neut=False,
                 rxn_temp=350,
                 rxn_time=60,
                 lipid_2_biocrude=0.846, # [1]
                 protein_2_biocrude=0.445, # [1]
                 carbo_2_biocrude=0.205, # [1]
                 protein_2_gas=0.074, # [1]
                 carbo_2_gas=0.418, # [1]
                 biocrude_C_slope=-8.37, # [2]
                 biocrude_C_intercept=68.55, # [2]
                 biocrude_N_slope=0.133, # [2]
                 biocrude_H_slope=-2.61, # [2]
                 biocrude_H_intercept=8.20, # [2]
                 HTLaqueous_C_slope=478, # [2]
                 TOC_TC=0.764, # [3]
                 hydrochar_C_slope=1.75, # [2]
                 biocrude_moisture_content=0.063, # [4]
                 hydrochar_P_recovery_ratio=0.86, # [5]
                 gas_composition={'CH4':0.050, 'C2H6':0.032,
                                  'CO2':0.918}, # [4]
                 hydrochar_pre=3029.7*6894.76, # [4]
                 HTLaqueous_pre=30*6894.76, # [4]
                 biocrude_pre=30*6894.76, # [4]
                 offgas_pre=30*6894.76, # [4]
                 eff_T=60+273.15, # [4]
                 P=None, V_wf=0.45, #tau is reaction time
                 length_to_diameter=None,
                 diameter=6.875*_in_to_m,
                 N=4,
                 V=None,
                 auxiliary=False,
                 mixing_intensity=None,
                 kW_per_m3=0,
                 wall_thickness_factor=1,
                 vessel_material='Stainless steel 316',
                 vessel_type='Horizontal',
                 CAPEX_factor=1,
                 HTL_steel_cost_factor=2.7, # so the cost matches [6]
                 mositure_adjustment_exist_in_the_system=False):
        
        SanUnit.__init__(self, ID, ins, outs, thermo, init_with)
        self.HTL_model = HTL_model
        self.NaOH_M = NaOH_M
        self.HCl_neut = HCl_neut
        self.feedstock = feedstock
        self.rxn_moisture = rxn_moisture
        self.rxn_temp = rxn_temp
        self.rxn_time = rxn_time
        self.lipid_2_biocrude = lipid_2_biocrude
        self.protein_2_biocrude = protein_2_biocrude
        self.carbo_2_biocrude = carbo_2_biocrude
        self.protein_2_gas = protein_2_gas
        self.carbo_2_gas = carbo_2_gas
        self.biocrude_C_slope = biocrude_C_slope
        self.biocrude_C_intercept = biocrude_C_intercept
        self.biocrude_N_slope = biocrude_N_slope
        self.biocrude_H_slope = biocrude_H_slope
        self.biocrude_H_intercept = biocrude_H_intercept
        self.HTLaqueous_C_slope = HTLaqueous_C_slope
        self.TOC_TC = TOC_TC
        self.hydrochar_C_slope = hydrochar_C_slope
        self.biocrude_moisture_content = biocrude_moisture_content
        self.hydrochar_P_recovery_ratio = hydrochar_P_recovery_ratio
        self.gas_composition = gas_composition
        self.hydrochar_pre = hydrochar_pre
        self.HTLaqueous_pre = HTLaqueous_pre
        self.biocrude_pre = biocrude_pre
        self.offgas_pre = offgas_pre
        hx_in = Stream(f'{ID}_hx_in')
        hx_out = Stream(f'{ID}_hx_out')
        self.heat_exchanger = HXutility(ID=f'.{ID}_hx', ins=hx_in, outs=hx_out, T=eff_T, rigorous=True)
        self.kodrum = KnockOutDrum(ID=f'.{ID}_KOdrum')
        self.P = P
        self.tau = self.rxn_time/60
        self.V_wf = V_wf
        self.length_to_diameter = length_to_diameter
        self.diameter = diameter
        self.N = N
        self.V = V
        self.auxiliary = auxiliary
        self.mixing_intensity = mixing_intensity
        self.kW_per_m3 = kW_per_m3
        self.wall_thickness_factor = wall_thickness_factor
        self.vessel_material = vessel_material
        self.vessel_type = vessel_type
        self.CAPEX_factor = CAPEX_factor
        self.HTL_steel_cost_factor = HTL_steel_cost_factor
        self.mositure_adjustment_exist_in_the_system = mositure_adjustment_exist_in_the_system

    def _run(self):
        
        if self.HTL_model not in ['MCA','kinetics','MCA_adj']:
            raise ValueError("invalid feedstock, select from 'MCA', 'kinetics', or 'MCA_adj'")
        
        if self.feedstock not in ['sludge','biosolid']:
            raise ValueError("invalid feedstock, select from 'sludge' and 'biosolid'")
        
        dewatered_sludge, NaOH_in, PFAS_in, HCl_in = self.ins
        hydrochar, HTLaqueous, biocrude, offgas = self.outs
        
        if self.mositure_adjustment_exist_in_the_system == True:
            self.WWTP = self.ins[0]._source.ins[0]._source.ins[0].\
                             _source.ins[0]._source
        else:
            self.WWTP = self.ins[0]._source.ins[0]._source.ins[0]._source
        
        dewatered_sludge_afdw = dewatered_sludge.imass['Sludge_lipid'] +\
                                dewatered_sludge.imass['Sludge_protein'] +\
                                dewatered_sludge.imass['Sludge_carbo'] +\
                                dewatered_sludge.imass['Sludge_lignin']  
        self.dewatered_sludge_afdw = dewatered_sludge_afdw
        
        # just use afdw in revised MCA model, other places use dw
        PFAS_in.imass['C8HF17O3S']=403*10**-9*dewatered_sludge_afdw # PFOS
        PFAS_in.imass['C8HF15O2']=34*10**-9*dewatered_sludge_afdw # PFOA
        PFAS_in.imass['C6HF13O3S']=5.9*10**-9*dewatered_sludge_afdw # PFHxS
        PFAS_in.imass['C6HF11O2']=6.2*10**-9*dewatered_sludge_afdw # PFHxA
        
        # Molarity * volume in L = moles
        # dewatered_sludge.imass['H2O'] is water mass in kg, converted to L (1 kg water = 1 L water)
        # divide by mass of dewatered_sludge_afdw
        
        destruction_potential = (self.NaOH_M*dewatered_sludge.ivol['H2O']*1000)/(self.dewatered_sludge_afdw+dewatered_sludge.imass['Sludge_ash'])
        # destruction_potential is mol of NaOH/kg of wastewater solid
        
        
        ### PFAS destruction approach one: PFAS destruction found at 350C for 1 hr, scaling factors applied for time and temperature       
        # # impact of time on PFAS destruction
        # if self.rxn_time < 25: # no noticable destruction before 25 minutes
        #     time_dest_PFHxS = 0
        #     time_dest_PFOS = 0
        # elif self.rxn_time >= 25 and self.rxn_time < 35: # between 25 and 35 minutes, destruction does not change
        #     time_dest_PFHxS = 0.547
        #     time_dest_PFOS = 0.572
        # elif self.rxn_time >= 35 and self.rxn_time < 60: # destruction is linear from 35-60 minutes
        #     time_dest_PFHxS = self.rxn_time * 1.647
        #     time_dest_PFOS =  self.rxn_time * 1.664
        # elif self.rxn_time >= 60:
        #     time_dest_PFHxS = 1
        #     time_dest_PFOS = 1    
        
        # # impact of temperature on PFAS destruction
        # # based on normalizing conditions to one hour
        # temp_dest_PFHxS = self.rxn_temp*0.016-4.694
        # temp_dest_PFOS = self.rxn_temp*0.016-4.187
        
        # # TODO: in experimental data, subtract ash weight from biosolids for all samples, make model based on ash free dry weight, NOT dry weight (current value)
        
        # self.PFOS_dest = (4.8351*destruction_potential*time_dest_PFOS*temp_dest_PFOS)/100 # values from experiment, divide by 100 to convert percent to decimals
        # self.PFOA_dest = 1 # values from experiment - all PFCAs destroyed w/ or w/o alkali
        # self.PFHxS_dest = (4.6453*destruction_potential*time_dest_PFHxS*temp_dest_PFHxS)/100 # values from experiment, divide by 100 to convert percent to decimals
        # self.PFHxA_dest = 1 # values from experiment - all PFCAs destroyed w/ or w/o alkali

        ###PFAS destruction approach two: PFAS destruction based on empirical relationship (modeled after 2nd order kinetic experiment)   
        #defines 'k' values from emperical data for PFOS and PFHxS; based on second order reaction, with WWRS concentration included
        #From Koehler et al, 2025
        k_emp350PFOS = 2.07*10**(-3) #kg WWRS/(mol NaOH*min)
        k_emp350PFHxS = 2.13*10**(-3) #kg WWRS/(mol NaOH*min)
        
        #defines activation energies divided by R
        #From Koehler et al, 2025
        Ea_sub_R_PFOS = 17118 #K
        Ea_sub_R_PFHxS = 15797 #K

        #Provides tmeperature adjusted k values from experimental conditions (310-365C)
        #derived from linear Arrhenius equation
        k_emp_rxntemp_PFOS = math.exp(-Ea_sub_R_PFOS*(1/(self.rxn_temp+273.15)-1/(350+273.15))+math.log(k_emp350PFOS))
        k_emp_rxntemp_PFHxS = math.exp(-Ea_sub_R_PFHxS*(1/(self.rxn_temp+273.15)-1/(350+273.15))+math.log(k_emp350PFHxS))

        self.PFOS_dest = math.exp(-k_emp_rxntemp_PFOS*destruction_potential*self.rxn_time)      
        self.PFOA_dest = 1 # values from experiment - all PFCAs destroyed w/ or w/o alkali
        self.PFHxS_dest = math.exp(-k_emp_rxntemp_PFHxS*destruction_potential*self.rxn_time)
        self.PFHxA_dest = 1 # values from experiment - all PFCAs destroyed w/ or w/o alkali

        self.afdw_lipid_ratio = self.WWTP.afdw_lipid
        self.afdw_protein_ratio = self.WWTP.afdw_protein
        self.afdw_carbo_ratio = self.WWTP.afdw_carbo
        self.afdw_lignin_ratio = self.WWTP.afdw_lignin
        
        NaOH_in.imass['NaOH'] = dewatered_sludge.ivol['H2O']*1000*self.NaOH_M*0.04
        if self.HCl_neut == True:
            HCl_in.imass['HCl'] = NaOH_in.imass['NaOH']*36.46/39.9997 # molar mass of HCl and NaOH
        else:
            HCl_in.imass['HCl'] = 0
        
        # pH calculations
        if destruction_potential <= 30:
            self.aq_pH = 0.1747*destruction_potential+9.85
        else:
            self.aq_pH = 14
        
        if self.HTL_model == 'kinetics':
            #allow reaction time to run past 1 hr
            #assumes rxn products are unchanged after 1 hr
            if self.rxn_time > 60:
                kin_time = 60
            else:
                kin_time = self.rxn_time
            
            #allow reactions to run at temperatures other than 250, 300, 350
            #assumes 250-275 are equal, 275-325 are the same, 325-365 are the same
            if 250 <= self.rxn_temp <275:
                kin_temp = 250
            elif 275 <= self.rxn_temp <325:
                kin_temp = 300
            elif 325 <= self.rxn_temp <375:
                kin_temp = 350
            
            # TODO: Andrew/Jeremy to decide: should here be multiplying by 60?
            # k values are per second - adjusted to per minute value by dividing by 60 in def kinetics_odes function
            sludge_kinetics = {250: [58.73, 6, 58.8, 43.55, 30, 33.3, 24, 42.51, 0.18, 0.3, 3.11, 1.18, 0.3, 0.18, 1.8, 0.18, 4.8],
                               300: [59.89, 7.67, 59.4, 53.99, 37.45, 42, 26,54, 0.24, 0.6, 7.24, 2.96, 2.56, 0.24, 21.7, 0.24, 59.4],
                               350: [60, 21, 60, 60, 60, 45, 60, 57, 0.3, 0.96, 16.55, 3, 3.98, 0.3, 30, 0.3, 60]}
            
            biosolid_kinetics = {250: [1.99, 6, 8.03, 0.3, 30, 11.02, 8.03, 0.3, 0.3, 0.3, 0.32, 0.3, 0.3, 0.3, 1.8, 0.3, 2.99],
                                 300: [2.1, 60, 58.9, 0.3, 60, 11.4, 58.9, 0.3, 4, 0.3, 0.33, 0.3, 6.99, 11.65, 1.86, 4.59, 6.14],
                                 350: [60, 60, 59.4, 60, 60, 60, 59.4, 60, 21.99, 2.81, 15.7, 2.81, 27.56, 38.06, 12.28, 27.39, 36.42]}
            
            sludge_df = pd.DataFrame(sludge_kinetics)
            biosolid_df = pd.DataFrame(biosolid_kinetics)
      
            if self.feedstock == 'sludge':
                kinetics_df = sludge_df
            elif self.feedstock == 'biosolid':
                kinetics_df = biosolid_df
                
            def kinetics_odes(x, t):
                # TODO: Andrew/Jeremy to decide: should here be multiplying by 60?
                k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17 = kinetics_df[kin_temp]/60 # 60 converts from seconds to minutes
                
                # assign each ODE to a vector element
                x1 = x[0]
                x2 = x[1]
                x3 = x[2]
                x4 = x[3]
                x5 = x[4]
                x6 = x[5]
                x7 = x[6]
                x8 = x[7]
                
                # define each ODE
                dx1dt= -(k1+k2)*x1
                dx2dt= -(k3+k4)*x2
                dx3dt= -(k5+k6)*x3
                dx4dt= -(k7+k8)*x4
                dx5dt= -(k10+k13)*x5+k3*x2+k5*x3+k7*x4+k9*x6+k14*x7
                dx6dt= -(k9+k12+k15)*x6+k1*x1+k4*x2+k6*x3+k8*x4+k10*x5+k11*x7+k17*x8
                dx7dt= -(k11+k14+k16)*x7+k2*x1+k12*x6+k13*x5
                dx8dt= -k17*x8+k15*x6+k16*x7
                
                return [dx1dt, dx2dt, dx3dt, dx4dt, dx5dt, dx6dt, dx7dt, dx8dt]
#TODO: evaluate x0 values, should be initialized with AFDW of lipid, carbohydrate, protein, lignin

            # initial conditions
            # lipid_init%, carbo_init%, protein_init%, lignin_init%
            x0 = [self.afdw_lipid_ratio, self.afdw_carbo_ratio, self.afdw_protein_ratio, self.afdw_lignin_ratio, 0, 0, 0, 0]
            
            # declare a time vector (time window)
            t = np.linspace(0, 60, 61)
            x = odeint(kinetics_odes, x0, t)  
            
            self.lipid_perc = x[:, 0] # lipid
            self.carbo_perc = x[:, 1] # carbohydrate
            self.protein_perc = x[:, 2] # protein
            self.lignin_perc = x[:, 3] # lignin
            self.hydrochar_perc = x[:, 4] # hydrochar
            self.aqueous_perc = x[:, 5] # aqueous
            self.biocrude_perc = x[:, 6] # crude
            self.gas_perc = x[:, 7] # gas       
            
            hydrochar.imass['Hydrochar'] = self.hydrochar_perc[kin_time]*self.dewatered_sludge_afdw
            
            # 2% of PFOS and 2% of PFHxS go to hydrochar post HTL, Yu et al. 2020
            hydrochar.imass['C8HF17O3S'] = PFAS_in.imass['C8HF17O3S']*(1-self.PFOS_dest)*0.02
            hydrochar.imass['C6HF13O3S'] = PFAS_in.imass['C6HF13O3S']*(1-self.PFHxS_dest)*0.02
            
            # HTLaqueous is just the TDS in the aqueous phase
            HTLaqueous.imass['HTLaqueous'] = self.aqueous_perc[kin_time]*self.dewatered_sludge_afdw
            
            gas_mass = self.gas_perc[kin_time]*self.dewatered_sludge_afdw
            for name, ratio in self.gas_composition.items():
                offgas.imass[name] = gas_mass*ratio
            
            biocrude.imass['Biocrude'] = self.biocrude_perc[kin_time]*self.dewatered_sludge_afdw     
           
            biocrude.imass['H2O'] = biocrude.imass['Biocrude']/(1 -\
                                    self.biocrude_moisture_content) -\
                                    biocrude.imass['Biocrude']
            
            # 98% of PFOS and 98% PFHxS go to biocrude post HTL, Yu et al. 2020
            biocrude.imass['C8HF17O3S'] = PFAS_in.imass['C8HF17O3S']*(1-self.PFOS_dest)*0.98
            biocrude.imass['C6HF13O3S'] = PFAS_in.imass['C6HF13O3S']*(1-self.PFHxS_dest)*0.98
        
        elif self.HTL_model == 'MCA':
            # the following calculations are based on revised MCA model
            # 0.377, 0.481, and 0.154 don't have uncertainties because they are calculated values
            hydrochar.imass['Hydrochar'] = 0.377*self.afdw_carbo_ratio*dewatered_sludge_afdw
            
            # 2% of PFOS and 2% of PFHxS go to hydrochar post HTL, Yu et al. 2020
            hydrochar.imass['C8HF17O3S'] = PFAS_in.imass['C8HF17O3S']*(1-self.PFOS_dest)*0.02
            hydrochar.imass['C6HF13O3S'] = PFAS_in.imass['C6HF13O3S']*(1-self.PFHxS_dest)*0.02
            
            # HTLaqueous is just the TDS in the aqueous phase
            HTLaqueous.imass['HTLaqueous'] = (0.481*self.afdw_protein_ratio +\
                                              0.154*self.afdw_lipid_ratio)*\
                                              dewatered_sludge_afdw
             
            gas_mass = (self.protein_2_gas*self.afdw_protein_ratio + self.carbo_2_gas*self.afdw_carbo_ratio)*\
                           dewatered_sludge_afdw    
            for name, ratio in self.gas_composition.items():
                offgas.imass[name] = gas_mass*ratio
                
            biocrude.imass['Biocrude'] = (self.protein_2_biocrude*self.afdw_protein_ratio +\
                                          self.lipid_2_biocrude*self.afdw_lipid_ratio +\
                                          self.carbo_2_biocrude*self.afdw_carbo_ratio)*\
                                          dewatered_sludge_afdw
            biocrude.imass['H2O'] = biocrude.imass['Biocrude']/(1 -\
                                    self.biocrude_moisture_content) -\
                                    biocrude.imass['Biocrude']
            
            # 98% of PFOS and 98% PFHxS go to biocrude post HTL, Yu et al. 2020
            biocrude.imass['C8HF17O3S'] = PFAS_in.imass['C8HF17O3S']*(1-self.PFOS_dest)*0.98
            biocrude.imass['C6HF13O3S'] = PFAS_in.imass['C6HF13O3S']*(1-self.PFHxS_dest)*0.98
 
            
        elif self.HTL_model == 'MCA_adj':
            def kinetics_adjustment(temperature = self.rxn_temp, time = self.rxn_time, afdw_lipid = self.afdw_lipid_ratio, afdw_carbo = self.afdw_carbo_ratio, afdw_protein = self.afdw_protein_ratio, afdw_lignin = self.afdw_lignin_ratio):
                
                #allow reaction time to run past 1 hr
                #assumes rxn products are unchanged after 1 hr
                if time > 60:
                    time = 60
                
                #allow reactions to run at temperatures other than 250, 300, 350
                #assumes 250-275 are equal, 275-325 are the same, 325-365 are the same
                if 250 <= temperature <275:
                    temperature = 250
                elif 275 <= temperature <325:
                    temperature = 300
                elif 325 <= temperature <375:
                    temperature = 350
                                       
                sludge_kinetics = {250: [58.73, 6, 58.8, 43.55, 30, 33.3, 24, 42.51, 0.18, 0.3, 3.11, 1.18, 0.3, 0.18, 1.8, 0.18, 4.8],
                                   300: [59.89, 7.67, 59.4, 53.99, 37.45, 42, 26,54, 0.24, 0.6, 7.24, 2.96, 2.56, 0.24, 21.7, 0.24, 59.4],
                                   350: [60, 21, 60, 60, 60, 45, 60, 57, 0.3, 0.96, 16.55, 3, 3.98, 0.3, 30, 0.3, 60]}
                
                biosolid_kinetics = {250: [1.99, 6, 8.03, 0.3, 30, 11.02, 8.03, 0.3, 0.3, 0.3, 0.32, 0.3, 0.3, 0.3, 1.8, 0.3, 2.99],
                                     300: [2.1, 60, 58.9, 0.3, 60, 11.4, 58.9, 0.3, 4, 0.3, 0.33, 0.3, 6.99, 11.65, 1.86, 4.59, 6.14],
                                     350: [60, 60, 59.4, 60, 60, 60, 59.4, 60, 21.99, 2.81, 15.7, 2.81, 27.56, 38.06, 12.28, 27.39, 36.42]}
                
                sludge_df = pd.DataFrame(sludge_kinetics)
                biosolid_df = pd.DataFrame(biosolid_kinetics)
          
                if self.feedstock == 'sludge':
                    kinetics_df = sludge_df
                elif self.feedstock == 'biosolid':
                    kinetics_df = biosolid_df
                    
                def kinetics_odes(x, t):
                    # TODO: Andrew/Jeremy to decide: should here be multiplying by 60?
                    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17 = kinetics_df[temperature]/60 # 60 converts from seconds to minutes
                    
                    # assign each ODE to a vector element
                    x1 = x[0]
                    x2 = x[1]
                    x3 = x[2]
                    x4 = x[3]
                    x5 = x[4]
                    x6 = x[5]
                    x7 = x[6]
                    x8 = x[7]
                    
                    # define each ODE
                    dx1dt= -(k1+k2)*x1
                    dx2dt= -(k3+k4)*x2
                    dx3dt= -(k5+k6)*x3
                    dx4dt= -(k7+k8)*x4
                    dx5dt= -(k10+k13)*x5+k3*x2+k5*x3+k7*x4+k9*x6+k14*x7
                    dx6dt= -(k9+k12+k15)*x6+k1*x1+k4*x2+k6*x3+k8*x4+k10*x5+k11*x7+k17*x8
                    dx7dt= -(k11+k14+k16)*x7+k2*x1+k12*x6+k13*x5
                    dx8dt= -k17*x8+k15*x6+k16*x7
                    
                    return [dx1dt, dx2dt, dx3dt, dx4dt, dx5dt, dx6dt, dx7dt, dx8dt]
    #TODO: evaluate x0 values, should be initialized with AFDW of lipid, carbohydrate, protein, lignin
    
                # initial conditions
                # lipid_init%, carbo_init%, protein_init%, lignin_init%
                x0 = [afdw_lipid, afdw_carbo, afdw_protein, afdw_lignin, 0, 0, 0, 0]
                
                # declare a time vector (time window)
                t = np.linspace(0, 60, 61)
                x = odeint(kinetics_odes, x0, t)  
                
                perc_lipid = x[:, 0] # lipid
                perc_carbo = x[:, 1] # carbohydrate
                perc_protein = x[:, 2] # protein
                perc_lignin = x[:, 3] # lignin
                perc_hydrochar = x[:, 4] # hydrochar
                perc_aqueous = x[:, 5] # aqueous
                perc_biocrude = x[:, 6] # crude
                perc_gas = x[:, 7] # gas       
                
                hydrochar_adj = perc_hydrochar[time]*self.dewatered_sludge_afdw
                aqueous_adj = perc_aqueous[time]*self.dewatered_sludge_afdw                
                gas_adj = perc_gas[time]*self.dewatered_sludge_afdw                
                biocrude_adj = perc_biocrude[time]*self.dewatered_sludge_afdw  
                
                kinetic_output = [hydrochar_adj, aqueous_adj, gas_adj, biocrude_adj]
                return kinetic_output
            #allows adjustment of MCA model by finding kinetic output at MCA conditions (300C and 1 hr), then making weighted calculations

            kinetic_baseline = kinetics_adjustment(temperature = 300, time = 60, afdw_lipid = self.afdw_lipid_ratio, afdw_carbo = self.afdw_carbo_ratio, afdw_protein = self.afdw_protein_ratio, afdw_lignin = self.afdw_lignin_ratio)
            kinetic_conditions = kinetics_adjustment(temperature = self.rxn_temp, time = self.rxn_time, afdw_lipid = self.afdw_lipid_ratio, afdw_carbo = self.afdw_carbo_ratio, afdw_protein = self.afdw_protein_ratio, afdw_lignin = self.afdw_lignin_ratio)
            kinetic_weight = np.array(kinetic_conditions) / np.array(kinetic_baseline)

            #next, apply to MCA model
            # the following calculations are based on revised MCA model
            # 0.377, 0.481, and 0.154 don't have uncertainties because they are calculated values
            hydrochar.imass['Hydrochar'] = 0.377*self.afdw_carbo_ratio*dewatered_sludge_afdw*kinetic_weight[0]
            
            # 2% of PFOS and 2% of PFHxS go to hydrochar post HTL, Yu et al. 2020
            hydrochar.imass['C8HF17O3S'] = PFAS_in.imass['C8HF17O3S']*(1-self.PFOS_dest)*0.02
            hydrochar.imass['C6HF13O3S'] = PFAS_in.imass['C6HF13O3S']*(1-self.PFHxS_dest)*0.02
            
            # HTLaqueous is just the TDS in the aqueous phase
            HTLaqueous.imass['HTLaqueous'] = (0.481*self.afdw_protein_ratio +\
                                              0.154*self.afdw_lipid_ratio)*\
                                              dewatered_sludge_afdw*kinetic_weight[1]
             
            gas_mass = (self.protein_2_gas*self.afdw_protein_ratio + self.carbo_2_gas*self.afdw_carbo_ratio)*\
                           dewatered_sludge_afdw*kinetic_weight[2]    
            for name, ratio in self.gas_composition.items():
                offgas.imass[name] = gas_mass*ratio
                
            biocrude.imass['Biocrude'] = (self.protein_2_biocrude*self.afdw_protein_ratio +\
                                          self.lipid_2_biocrude*self.afdw_lipid_ratio +\
                                          self.carbo_2_biocrude*self.afdw_carbo_ratio)*\
                                          dewatered_sludge_afdw*kinetic_weight[3]  
            biocrude.imass['H2O'] = biocrude.imass['Biocrude']/(1 -\
                                    self.biocrude_moisture_content) -\
                                    biocrude.imass['Biocrude']
            
            # 98% of PFOS and 98% PFHxS go to biocrude post HTL, Yu et al. 2020
            biocrude.imass['C8HF17O3S'] = PFAS_in.imass['C8HF17O3S']*(1-self.PFOS_dest)*0.98
            biocrude.imass['C6HF13O3S'] = PFAS_in.imass['C6HF13O3S']*(1-self.PFHxS_dest)*0.98
               
 ##########           
          
        # assume ash (all soluble based on Jones) and all other chemicals go to water
        HTLaqueous.imass['H2O'] = dewatered_sludge.F_mass + NaOH_in.F_mass + PFAS_in.F_mass +\
                                  HCl_in.F_mass - hydrochar.F_mass - biocrude.F_mass -\
                                  gas_mass - HTLaqueous.imass['HTLaqueous']
        
        hydrochar.phase = 's'
        offgas.phase = 'g'
        HTLaqueous.phase = biocrude.phase = 'l'
        
        hydrochar.P = self.hydrochar_pre
        HTLaqueous.P = self.HTLaqueous_pre
        biocrude.P = self.biocrude_pre
        offgas.P = self.offgas_pre
        
        for stream in self.outs: stream.T = self.heat_exchanger.T
    
    @property
    def aqueous_pH(self):
        return self.aq_pH   
    
    @property
    def reaction_temp(self):
        return self.rxn_temp
   
    @property
    def reaction_time(self):
        return self.rxn_time
    
    @property
    def NaOH_molarity(self):
        return self.NaOH_M
    
    @property
    def model_type(self):
        return self.HTL_model
    
    # yields (for biocrude, aqueous, hydrochar, and gas) are based on afdw
    @property
    def biocrude_yield(self):
        if self.HTL_model == 'MCA' or 'MCA_adj':
            return self.protein_2_biocrude*self.afdw_protein_ratio +\
                   self.lipid_2_biocrude*self.afdw_lipid_ratio +\
                   self.carbo_2_biocrude*self.afdw_carbo_ratio
        else:
            return self.biocrude_perc[self.rxn_time]
    
    @property
    def aqueous_yield(self):
        if self.HTL_model == 'MCA' or 'MCA_adj':
            return 0.481*self.afdw_protein_ratio + 0.154*self.afdw_lipid_ratio
        else:
            return self.aqueous_perc[self.rxn_time]
    
    @property
    def hydrochar_yield(self):
        if self.HTL_model == 'MCA' or 'MCA_adj':
            return 0.377*self.afdw_carbo_ratio
        else:
            return self.hydrochar_perc[self.rxn_time]
    
    @property
    def gas_yield(self):
        if self.HTL_model == 'MCA' or 'MCA_adj':
            return self.protein_2_gas*self.afdw_protein_ratio + self.carbo_2_gas*self.afdw_carbo_ratio
        else:
            return self.gas_perc[self.rxn_time]
    
    @property
    def biocrude_C_ratio(self):
        return (self.WWTP.AOSc*self.biocrude_C_slope + self.biocrude_C_intercept)/100 # [2]
    
    @property
    def biocrude_H_ratio(self):
        return (self.WWTP.AOSc*self.biocrude_H_slope + self.biocrude_H_intercept)/100 # [2]
    
    @property
    def biocrude_N_ratio(self):
        return self.biocrude_N_slope*self.WWTP.dw_protein # [2]
    
    @property
    def biocrude_C(self):
        return min(self.outs[2].F_mass*self.biocrude_C_ratio, self.WWTP.C)
    
    @property
    def HTLaqueous_C(self):
        return min(self.outs[1].F_vol*1000*self.HTLaqueous_C_slope*\
                   self.WWTP.dw_protein*100/1000000/self.TOC_TC,
                   self.WWTP.C - self.biocrude_C)
    
    @property
    def biocrude_H(self):
        return self.outs[2].F_mass*self.biocrude_H_ratio
    
    @property
    def biocrude_N(self):
        return min(self.outs[2].F_mass*self.biocrude_N_ratio, self.WWTP.N)
    
    @property
    def biocrude_HHV(self):
        return 30.74 - 8.52*self.WWTP.AOSc + 0.024*self.WWTP.dw_protein # [2]

    @property
    def energy_recovery(self):
        return self.biocrude_HHV*self.outs[2].imass['Biocrude']/\
               (self.WWTP.outs[0].F_mass -\
                self.WWTP.outs[0].imass['H2O'])/self.WWTP.HHV # [2]
    
    @property
    def offgas_C(self):
        carbon = sum(self.outs[3].imass[self.gas_composition]*
                     [cmp.i_C for cmp in self.components[self.gas_composition]])
        return min(carbon, self.WWTP.C - self.biocrude_C - self.HTLaqueous_C)
    
    @property
    def hydrochar_C_ratio(self):
        return min(self.hydrochar_C_slope*self.WWTP.dw_carbo, 0.65) # [2]
    
    @property
    def hydrochar_C(self):
        return min(self.outs[0].F_mass*self.hydrochar_C_ratio, self.WWTP.C -\
                   self.biocrude_C - self.HTLaqueous_C - self.offgas_C)
    
    @property
    def hydrochar_P(self):
        return min(self.WWTP.P*self.hydrochar_P_recovery_ratio, self.outs[0].F_mass)
    
    @property
    def HTLaqueous_N(self):
        return self.WWTP.N - self.biocrude_N
    
    @property
    def HTLaqueous_P(self):
        return self.WWTP.P*(1 - self.hydrochar_P_recovery_ratio)
    
    def _design(self):
        
        Design = self.design_results
        Design['Treatment capacity'] = self.ins[0].F_mass/_lb_to_kg
        
        hx = self.heat_exchanger
        hx_ins0, hx_outs0 = hx.ins[0], hx.outs[0]
        hx_ins0.mix_from((self.outs[1], self.outs[2], self.outs[3]))
        hx_outs0.copy_like(hx_ins0)
        hx_ins0.T = self.rxn_temp+273.15
        hx_outs0.T = hx.T
        hx_ins0.P = hx_outs0.P = self.outs[0].P # cooling before depressurized, heating after pressurized
        # in other words, both heating and cooling are performed under relatively high pressure
        # hx_ins0.vle(T=hx_ins0.T, P=hx_ins0.P)
        # hx_outs0.vle(T=hx_outs0.T, P=hx_outs0.P)
        hx.simulate_as_auxiliary_exchanger(ins=hx.ins, outs=hx.outs)

        self.P = self.ins[0].P
        Reactor._design(self)
        Design['Solid filter and separator weight'] = 0.2*Design['Weight']*Design['Number of reactors'] # assume stainless steel
        # based on [6], case D design table, the purchase price of solid filter and separator to
        # the purchase price of HTL reactor is around 0.2, therefore, assume the weight of solid filter
        # and separator is 0.2*single HTL weight*number of HTL reactors
        self.construction[0].quantity += Design['Solid filter and separator weight']*_lb_to_kg
        
        self.kodrum.V = self.F_mass_out/_lb_to_kg/1225236*4230/_m3_to_gal
        # in [6], when knockout drum influent is 1225236 lb/hr, single knockout
        # drum volume is 4230 gal
        
        self.kodrum.simulate()
        
    def _cost(self):
        Reactor._cost(self)
        self._decorated_cost()
        
        purchase_costs = self.baseline_purchase_costs
        for item in purchase_costs.keys():
            purchase_costs[item] *= self.CAPEX_factor
            
        purchase_costs['Horizontal pressure vessel'] *= self.HTL_steel_cost_factor
        
        for aux_unit in self.auxiliary_units:
            purchase_costs = aux_unit.baseline_purchase_costs
            installed_costs = aux_unit.installed_costs
            for item in purchase_costs.keys():
                purchase_costs[item] *= self.CAPEX_factor
                installed_costs[item] *= self.CAPEX_factor