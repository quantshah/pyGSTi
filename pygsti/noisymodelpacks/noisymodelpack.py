"""
Base of the object-oriented model for noisy modelpacks
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np

from ..construction import modelconstruction as _mc
from ..objects import Basis as _Basis
from ..objects import operation as _op

def generate_pauli_stochastic_noise_model(pspec, noisy_gates=[]):
    # Generate the parameterization
    dummy_errors = {}
    for gate in noisy_gates:
        if isinstance(gate, tuple):
            gatename = gate[0]
        else:
            gatename = gate
        
        basis_size = pspec.root_gate_unitaries[gatename].shape[0]**2
        dummy_errors[gate] = _np.zeros(basis_size - 1)

    ideal_model = _mc.create_crosstalk_free_model(pspec.number_of_qubits, pspec.root_gate_names,
                                                  stochastic_error_probs=dummy_errors)
    return ideal_model

def _sample_stochastic_noise_op(snop, max_strength, max_weight=1,
                       symmetric_strengths=False,
                       random_total_strength=True,
                       random_relative_strengths=True,
                       rand_state=None):
    channel_labels = snop.basis.labels[1:]
    
    indices_map = {}
    count = 0
    for i, label in enumerate(channel_labels):
        if sum([c != 'I' for c in label]) <= max_weight:
            if symmetric_strengths:
                if label[::-1] not in indices_map.keys(): # Symmetric not present
                    indices_map[label] = count
                    count += 1
                else: # Symmetric already present
                    indices_map[label] = indices_map[label[::-1]]
            else: # No symmetric
                indices_map[label] = count
                count += 1
    
    total_strength = max_strength
    if random_total_strength:
        total_strength *= rand_state.rand()
    
    if random_relative_strengths:
        relative_strengths = rand_state.rand(len(set(indices_map.values())))
    else:
        relative_strengths = _np.ones(len(set(indices_map.values())))
    
    relative_strengths /= sum(relative_strengths)
    total_strengths = relative_strengths * total_strength
    
    error_probs = _np.zeros(len(channel_labels))
    for i, label in enumerate(channel_labels):
        if label not in indices_map.keys():
            continue
        
        # How many terms are using this strength when assuming symmetric
        factor = sum([v == indices_map[label] for v in indices_map.values()])
        
        error_probs[i] = total_strengths[indices_map[label]] / factor
    
    return error_probs

def sample_noise_model(ideal_model, max_total_strengths={}, max_weight=1,
                       symmetric_strengths=False,
                       random_total_strength=True,
                       random_relative_strengths=True,
                       rand_state=None):
    noisy_model = ideal_model.copy()

    if rand_state is None:
        rand_state = _np.random.RandomState()

    for label, op in noisy_model.operation_blks['gates'].items():
        if isinstance(op, _op.ComposedOp): # TODO: Would be easier if ComposedOp.num_params counted factorop num_params
            for subop in op.factorops:
                if subop.num_params == 0: # No sampling needed, skip
                    continue
                elif isinstance(subop, _op.StochasticNoiseOp):
                    max_strength = max_total_strengths.get(label.name, 0.0)
                    # Check if we have a qubit-specific override
                    max_strength = max_total_strengths.get(label, max_strength)

                    sampled_params = _sample_stochastic_noise_op(subop,
                        max_strength=max_strength,
                        max_weight=max_weight,
                        symmetric_strengths=symmetric_strengths,
                        random_total_strength=random_total_strength,
                        random_relative_strengths=random_relative_strengths,
                        rand_state=rand_state)
                    subop.from_vector(subop._rates_to_params(sampled_params))
                else:
                    raise NotImplementedError("Can't sample this type of op")
        elif op.num_params == 0: # No sampling needed (I think)
            pass
        else:
            raise NotImplementedError("Can't sample this type of op")

    return noisy_model

def generate_pauli_stochastic_noise_model_old(pspec, max_total_strengths={}, max_weight=1,
                                          symmetric_strengths=False,
                                          random_total_strength=True,
                                          random_relative_strengths=True,
                                          rand_state=None):
    if rand_state is None:
        rand_state = _np.random.RandomState()
    
    # Generate uniform random noise up to some max strength
    stochastic_error_probs = {}
    for gate, max_strength in max_total_strengths.items():
        if isinstance(gate, tuple):
            gatename = gate[0]
        else:
            gatename = gate
            
        basis_size = pspec.root_gate_unitaries[gatename].shape[0]**2
        
        pauli_basis = _Basis.cast('pp', basis_size)
        channel_labels = pauli_basis.labels[1:]
        
        indices_map = {}
        count = 0
        for i, label in enumerate(channel_labels):
            if sum([c != 'I' for c in label]) <= max_weight:
                if symmetric_strengths:
                    if label[::-1] not in indices_map.keys(): # Symmetric not present
                        indices_map[label] = count
                        count += 1
                    else: # Symmetric already present
                        indices_map[label] = indices_map[label[::-1]]
                else: # No symmetric
                    indices_map[label] = count
                    count += 1
        
        total_strength = max_strength
        if random_total_strength:
            total_strength *= rand_state.rand()
        
        if random_relative_strengths:
            relative_strengths = rand_state.rand(len(set(indices_map.values())))
        else:
            relative_strengths = _np.ones(len(set(indices_map.values())))
        
        relative_strengths /= sum(relative_strengths)
        total_strengths = relative_strengths * total_strength
        
        error_probs = _np.zeros(len(channel_labels))
        for i, label in enumerate(channel_labels):
            if label not in indices_map.keys():
                continue
            
            # How many terms are using this strength when assuming symmetric
            factor = sum([v == indices_map[label] for v in indices_map.values()])
            
            error_probs[i] = total_strengths[indices_map[label]] / factor
        
        stochastic_error_probs[gate] = error_probs
    
    return _mc.create_crosstalk_free_model(pspec.number_of_qubits, pspec.root_gate_names,
                                           stochastic_error_probs=stochastic_error_probs)
