import sys
sys.path.insert(0, '/Users/radu/git/spmn/SPFlow/src')
import numpy as np
import pandas as pd
import logging

from spn.algorithms.SPMN import SPMN
from spn.io.Graphics import plot_spn
from spn.structure.StatisticalTypes import MetaType
from spn.algorithms.SPMNDataUtil import align_data, cooper_tranformation
from spn.algorithms.EM import EM_optimization
from spn.structure.Base import get_nodes_by_type, Sum, Product, Max, Leaf
from spn.structure.leaves.spmnLeaves.SPMNLeaf import Utility
from spn.structure.leaves.histogram.Histograms import Histogram
from spn.algorithms.MEU import meu

logging.basicConfig(level=logging.INFO)

# load the data
file_name = '/Users/radu/git/ai-c-3356/ai-c-autodo/data/spmn/Export_textiles.tsv'
data_frame = pd.read_csv(file_name, sep='\t')

# set the params
feature_names = ['Economical_State', 'Export_Decision', 'U']
partial_order = [['Economical_State'], ['Export_Decision'], ['U']]
decision_nodes = ['Export_Decision']
utility_nodes = ['U']
meta_types = [MetaType.DISCRETE, MetaType.DISCRETE, MetaType.UTILITY]
util_to_bin = False
data_frame,_ = align_data(data_frame, partial_order)

#data = data[feature_names] # re-arrange the columns to match the feature_names
#logging.info(feature_names)
#print(data_frame['Economical_State'].unique())
#print(data_frame['Export_Decision'].unique())

# replace the values of categorical features by their indeces
for i,col in enumerate(list(data_frame.columns)) :
    vals = sorted(data_frame[col].unique())
    if meta_types[i] == MetaType.DISCRETE :
        to_replace = {k: int(v) for v, k in enumerate(vals)}
        data_frame.replace(to_replace, inplace=True)

#train_data = cooper_tranformation(data_frame.to_numpy(), 2)
train_data = data_frame.to_numpy()
print(train_data)

# create the SPMN
spmn = SPMN(
    partial_order=partial_order,
    decision_nodes=decision_nodes,
    utility_node=utility_nodes,
    feature_names=feature_names,
    meta_types=meta_types,
    util_to_bin=util_to_bin
)

# learn the structure
spmn.learn_spmn(train_data)

EM_optimization(spn=spmn.spmn_structure, data=train_data)

plot_spn(spmn.spmn_structure, 'test.pdf')

# debug the structure
all_nodes = get_nodes_by_type(spmn.spmn_structure)
logging.info(f'There are {len(all_nodes)} nodes in the SPMN')
for node in all_nodes :
    if isinstance(node, Sum) :
        print(f'{node.id}: Sum node with weights {node.weights}')
    elif isinstance(node, Max) :
        print(f'{node.id}: Max node with decisions {node.dec_values}')
    elif isinstance(node, Product) :
        print(f'{node.id}: Product node with {len(node.children)} children')
    elif isinstance(node, Utility) :
        print(f'{node.id}: Utility node with scope {node.scope}')
        print(f'  breaks: {node.breaks}')
        print(f'  bin_repr_points: {node.bin_repr_points}')
        print(f'  count: {node.count}')
    elif isinstance(node, Histogram) :
        print(f'{node.id}: Variable node with scope {node.scope}')
        print(f'  breaks: {node.breaks}')
        print(f'  bin_repr_points: {node.bin_repr_points}')
        print(f'  count: {node.count}')

input_data = np.array([np.nan, np.nan, np.nan]).reshape(1,3)
res = meu(spmn.spmn_structure, input_data)
print(f'MEU is {res}')