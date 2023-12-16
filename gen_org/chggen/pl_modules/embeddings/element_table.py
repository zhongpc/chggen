from __future__ import annotations
from typing import List, Tuple, Dict, Union, Optional, Any, Callable, Sequence

import pandas as pd
import numpy as np
import torch
from torch.nn.functional import one_hot
import os


module_dir = os.path.dirname(os.path.abspath(__file__))


class PeriodicTable:
    """Class for periodic table of elements.
    Periodic Table information is from https://github.com/qiwsir/DataSet/blob/master/elemanets/elements.csv.
    """
    
    def __init__(
            self, path: str = 'elements.csv',
        ) -> None:
        """ Initialize periodic table from csv file.
        
        Args:
            Path (str): path to csv file with periodic table data.
        """
        self.table = pd.read_csv(os.path.join(module_dir, './elements.csv'))
        self.atomic_number = self.table['atomic number'].values
        self.symbol = self.table['symbol'].values
        self.atomic_mass = self.table['atomic mass'].values
        self.electronic_configuration = self.table['electronic configuration'].values
        self.electronegativity = self.table['electronegativity'].values
        self.atomic_radius = self.table['atomic radius'].values
        self.ion_radius = self.table['ion radius'].values
        self.van_der_waals_radius = self.table['van der Waals radius'].values
        self.IE_1 = self.table['IE-1'].values
        self.EA = self.table['EA'].values
        self.standard_state = self.table['standard state'].values
        self.bonding_type = self.table['bonding type'].values
        self.melting_point = self.table['melting point'].values
        self.boiling_point = self.table['boiling point'].values
        self.density = self.table['density'].values
        self.metal = self.table['metal'].values
        self.year_discovered = self.table['year discovered'].values
        self.group = self.table['group'].values
        self.period = self.table['period'].values
        
        # ------------------------------------------------------------
        # put La and Ac series in group 3 and period 6 and 7 separately
        La_series_ids = (self.atomic_number>=57) * (self.atomic_number<=70) 
        Ac_series_ids = (self.atomic_number>=89) * (self.atomic_number<=102)

        self.period[La_series_ids] = 6
        self.period[Ac_series_ids] = 7
        self.group[La_series_ids] = 3
        self.group[Ac_series_ids] = 3
        self.group = self.group.astype(np.int64)
        # ------------------------------------------------------------
        
        self.atomic_number = torch.tensor(self.atomic_number)
        self.group = torch.tensor(self.group)
        self.period = torch.tensor(self.period)
        
    
    def get_property_from_symbols(
        self, symbols: Sequence, prop_name: str,
    ) -> torch.Tensor:
        """ Get property from symbols in sequence.
        
        Args:
            symbols (Sequence): sequence of symbols.
            prop_name (str): name of property.
        
        Returns:
            out (Tensor): tensor of properties obtained.
        """
        assert prop_name in self.table.columns, f'Property {prop_name} not in table.'
        self.symbols = self.symbols.to(symbols.device)
        prop = eval('self.'+'_'.join(prop_name.split(' '))).to(symbols.device)
        return torch.cat([prop[self.symbol == symbol] for symbol in symbols])
    
    def get_property_from_atomic_numbers(
        self, atomic_numbers: Sequence, prop_name: str,
    ) -> torch.Tensor:
        """ Get property from atomic numbers in sequence.
        
        Args:
            atomic_numbers (Sequence): sequence of atomic numbers.
            prop_name (str): name of property.
        
        Returns:
            out (Tensor): tensor of properties obtained.
        """
        assert prop_name in self.table.columns, f'Property {prop_name} not in table.'
        self.atomic_number = self.atomic_number.to(atomic_numbers.device)
        prop = eval('self.'+'_'.join(prop_name.split(' '))).to(atomic_numbers.device)
        return torch.cat([prop[self.atomic_number == atomic_number] for atomic_number in atomic_numbers])
    
    def __repr__(self) -> str:
        return f"Periodic Table contains properties: {', '.join(self.table.columns)}"
    
if __name__ == '__main__':
    periodic_table = PeriodicTable()
    print(periodic_table)
    
    Z = torch.tensor([1,2,1,5,5,2,104])
    Z_period = periodic_table.get_property_from_atomic_numbers(Z, 'period')
    Z_group = periodic_table.get_property_from_atomic_numbers(Z, 'group')
    print('Period:', Z_period)
    print('Group:', Z_group)
    
    one_hot_emb_period = one_hot(torch.tensor(Z_period-1), num_classes=7)
    one_hot_emb_group = one_hot(torch.tensor(Z_group-1), num_classes=18)
    print('One hot embedding of Period:\n', one_hot_emb_period)
    print('One hot embedding of Group:\n', one_hot_emb_group)