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

#01234567890123456789012345678901234567890123456789012345678901234567890123456789

def create_pauli_stochastic_noise_model(pspec, noisy_gates = []):

    """
    Create a base Pauli stochastic noise model with the desired parametrization
    and zero error probabilities, using create_crosstalk_free_model.

    Parameters
    ----------
    pspec : ProcessorSpec object
        Specification for a qubit device.
    
    noisy_gates : list
        List of gates to add Pauli stochastic noise to downstream. List
        items may also include specific qubits as tuples, e.g.,
        (Gi, 9): 0.5.

    Returns
    -------
    Model object
        A create_crosstalk_free_model with the dimension of
        stochastic_error_probs determined by create_pauli_stochastic_noise_model
        and set to zero.
    """

    zero_error_probs = {}
    # Obtain gate name
    for gate in noisy_gates:
        if isinstance(gate, tuple):
            gatename = gate[0]
        else:
            gatename = gate

        # Generate zero_error_probs vector based on the gate dimension.
        basis_size = pspec.root_gate_unitaries[gatename].shape[0]**2
        zero_error_probs[gate] = _np.zeros(basis_size - 1)

    ideal_model = _mc.create_crosstalk_free_model(pspec.number_of_qubits,
                                                  pspec.root_gate_names,
                                                  stochastic_error_probs = zero_error_probs)
    return ideal_model

def _sample_stochastic_noise_op(snop, max_strength, max_weight = 1,
                                symmetric_strengths = False,
                                random_total_strength = True,
                                random_relative_strengths = True,
                                rand_state = None):

    """
    Create a NumPy array/vector of uniform error probabilities for a
    Pauli stochastic noise model. Elements of this array correspond
    to strengths for a set of Pauli stochastic noise operators for a
    given gate.

    Parameters
    ----------
    snop : StochasticNoiseOp
        TODO
    
    max_strength : float
        A float corresponding to the total noise maximum for the
        corresponding noise channel.
    
    max_weight : int, optional
        Maximum weight of the noise model operators. As such, weight 2
        also includes weight 1 operations, for example.
    
    symmetric_strengths : bool, optional
        If False, the strengths of each noise model operator for a gate
        are all independent. If True, and max_weight = 2, the strengths
        of noise model operations are symmetric, e.g., if max_weight = 1,
        then IX and XI both have noise strength s_{ix}, etc.; if
        max_weight = 2, then XY and YX both have strength s_{xy}, etc.
    
    random_total_strength : bool, optional
        If False, the total strength of each noise channel for a gate
        is the noise maximum in max_total_strength. If True, the total
        strength of each noise channel for a gate is randomly selected
        and bounded by the noise maximum in max_total_strength.
    
    random_relative_strengths : bool, optional
        If False, the relative strengths of each noise model operator for
        a gate are all equal and their sum is the noise maximum in
        max_total_strength. If True, the relative strengths of each noise 
        model operator for a gate are randomly selected and their sum
        is bounded by the noise maximum in max_total_strength.
    
    rand_state : RandomState, optional
        Container for a pseudo-random number generator, e.g., the slow
        Mersenne Twister, i.e., rand_state = np.random.RandomState(seed).

    Returns
    -------
    error_probs
        A NumPy array/vector of uniform error probabilities for a Pauli
        stochastic noise model. Elements of this array correspond to
        strengths for a set of Pauli stochastic noise operators for a 
        given gate.
    """

    # Obtain all channel labels except for the identity operator.
    channel_labels = snop.basis.labels[1:]
    
    indices_map = {}; count = 0
    for label in channel_labels:
        if sum([c != 'I' for c in label]) <= max_weight:
            if symmetric_strengths:
                if label[::-1] not in indices_map.keys():
                    # Symmetric counterpart is NOT present
                    indices_map[label] = count
                    count += 1
                else: # Symmetric counterpart is present
                    indices_map[label] = indices_map[label[::-1]]
            else: # Unsymmetric
                indices_map[label] = count
                count += 1

    # Generate the random total noise strength for the gate and the 
    # corresponding relative strengths for all channel operators.
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

        factor = 1.0
        if symmetric_strengths:
            # Calculate the number of terms that use this strength 
            # when using symmetric strengths.
            factor = sum([v == indices_map[label] for v in indices_map.values()])
        
        error_probs[i] = total_strengths[indices_map[label]] / factor
    
    return error_probs

def sample_noise_model(ideal_model, max_total_strengths = {}, max_weight = 1,
                       symmetric_strengths = False,
                       random_total_strength = True,
                       random_relative_strengths = True,
                       rand_state = None):

    """
    Create a NumPy array/vector of uniform error probabilities for a
    Pauli stochastic noise model. Elements of this array correspond
    to strengths for a set of Pauli stochastic noise operators for a
    given gate.

    Parameters
    ----------
    ideal_model : Model object
        A create_crosstalk_free_model with the dimension of
        stochastic_error_probs determined by create_pauli_stochastic_noise_model
        and set to zero.
    
    max_strength : float
        A float corresponding to the total noise maximum for the
        corresponding noise channel.
    
    max_weight : dict
        Dictionary of gates and corresponding total noise maximum for
        the corresponding channel operators. Key values may also 
        include specific qubits as tuples, e.g., (Gi, 9): 0.5.
    
    symmetric_strengths : bool, optional
        If False, the strengths of each noise model operator for a gate
        are all independent. If True, and max_weight = 2, the strengths
        of noise model operations are symmetric, e.g., if max_weight = 1,
        then IX and XI both have noise strength s_{ix}, etc.; if
        max_weight = 2, then XY and YX both have strength s_{xy}, etc.
    
    random_total_strength : bool, optional
        If False, the total strength of each noise channel for a gate
        is the noise maximum in max_total_strength. If True, the total
        strength of each noise channel for a gate is randomly selected
        and bounded by the noise maximum in max_total_strength.
    
    random_relative_strengths : bool, optional
        If False, the relative strengths of each noise model operator for
        a gate are all equal and their sum is the noise maximum in
        max_total_strength. If True, the relative strengths of each noise 
        model operator for a gate are randomly selected and their sum
        is bounded by the noise maximum in max_total_strength.
    
    rand_state : RandomState, optional
        Container for a pseudo-random number generator, e.g., the slow
        Mersenne Twister, i.e., rand_state = np.random.RandomState(seed).

    Returns
    -------
    Model object
        A create_crosstalk_free_model with stochastic_error_probs
        determined by sample_noise_model and its inputs.
    """

    noisy_model = ideal_model.copy()

    if rand_state is None:
        rand_state = _np.random.RandomState()

    for label, op in noisy_model.operation_blks['gates'].items():
        if isinstance(op, _op.ComposedOp):
            # TODO: This would be easier if ComposedOp.num_params counted factorop num_params
            for subop in op.factorops:
                if subop.num_params == 0: # No sampling is needed, so skip
                    continue
                elif isinstance(subop, _op.StochasticNoiseOp):
                    max_strength = max_total_strengths.get(label.name, 0.0)
                    # Check if we have a qubit-specific override
                    max_strength = max_total_strengths.get(label, max_strength)

                    sampled_params = _sample_stochastic_noise_op(subop,
                        max_strength = max_strength,
                        max_weight = max_weight,
                        symmetric_strengths = symmetric_strengths,
                        random_total_strength = random_total_strength,
                        random_relative_strengths = random_relative_strengths,
                        rand_state = rand_state)
                    subop.from_vector(subop._rates_to_params(sampled_params))
                else:
                    raise NotImplementedError("This type of operation cannot be sampled!")
        elif op.num_params == 0: # No sampling needed (I think)
            pass
        else:
            raise NotImplementedError("This type of operation cannot be sampled!")

    return noisy_model

def create_pauli_stochastic_noise_model_old(pspec, max_total_strengths = {},
                                            max_weight = 1,
                                            symmetric_strengths = False,
                                            random_total_strength = True,
                                            random_relative_strengths = True,
                                            rand_state = None):
    """
    Create a Pauli stochastic noise model with uniform random noise,
    using create_crosstalk_free_model.

    Parameters
    ----------
    pspec : ProcessorSpec object
        Specification for a qubit device.
    
    max_total_strength : dict
        Dictionary of gates and corresponding total noise maximum for
        the corresponding channel operators. Key values may also 
        include specific qubits as tuples, e.g., (Gi, 9): 0.5.
    
    max_weight : int, optional
        Maximum weight of the noise model operators. As such, weight 2
        also includes weight 1 operations, for example.
    
    symmetric_strengths : bool, optional
        If False, the strengths of each noise model operator for a gate
        are all independent. If True, and max_weight = 2, the strengths
        of noise model operations are symmetric, e.g., if max_weight = 1,
        then IX and XI both have noise strength s_{ix}, etc.; if
        max_weight = 2, then XY and YX both have strength s_{xy}, etc.
    
    random_total_strength : bool, optional
        If False, the total strength of each noise channel for a gate
        is the noise maximum in max_total_strength. If True, the total
        strength of each noise channel for a gate is randomly selected
        and bounded by the noise maximum in max_total_strength.
    
    random_relative_strengths : bool, optional
        If False, the relative strengths of each noise model operator for
        a gate are all equal and their sum is the noise maximum in
        max_total_strength. If True, the relative strengths of each noise 
        model operator for a gate are randomly selected and their sum
        is bounded by the noise maximum in max_total_strength.
    
    rand_state : RandomState, optional
        Container for a pseudo-random number generator, e.g., the slow
        Mersenne Twister, i.e., rand_state = np.random.RandomState(seed).

    Returns
    -------
    Model
        A create_crosstalk_free_model with stochastic_error_probs determined
        by create_pauli_stochastic_noise_model and its inputs.
    """

    if rand_state is None:
        rand_state = _np.random.RandomState()
    
    # Generate uniform random noise up to a specified maximum strength.
    stochastic_error_probs = {}
    for gate, max_strength in max_total_strengths.items():
        if isinstance(gate, tuple):
            gatename = gate[0]
        else:
            gatename = gate

        # Obtain all channel labels except for the identity operator.
        basis_size = pspec.root_gate_unitaries[gatename].shape[0]**2
        pauli_basis = _Basis.cast('pp', basis_size)
        channel_labels = pauli_basis.labels[1:]
        
        # Create a map of indices for channel operators with non-zero
        # strengths and weights <= max_weight.
        indices_map = {}; count = 0
        for label in channel_labels:
            if sum([c != 'I' for c in label]) <= max_weight:
                if symmetric_strengths:
                    if label[::-1] not in indices_map.keys():
                        # Symmetric counterpart is NOT present
                        indices_map[label] = count
                        count += 1
                    else: # Symmetric counterpart is present
                        indices_map[label] = indices_map[label[::-1]]
                else: # Unsymmetric
                    indices_map[label] = count
                    count += 1
            #print(count, label)
        #print(indices_map.items())

        # Generate the random total noise strength for the gate and the 
        # corresponding relative strengths for all channel operators.
        total_strength = max_strength
        if random_total_strength:
            total_strength *= rand_state.rand()
        
        if random_relative_strengths:
            relative_strengths = rand_state.rand(max(indices_map.values())+1)
        else:
            relative_strengths = _np.ones(max(indices_map.values())+1)
        
        relative_strengths /= sum(relative_strengths)
        total_strengths = relative_strengths * total_strength
        
        error_probs = _np.zeros(len(channel_labels))
        for i, label in enumerate(channel_labels):
            if label not in indices_map.keys():
                continue

            factor = 1.0
            if symmetric_strengths:
                # Calculate the number of terms that use this strength 
                # when using symmetric strengths.
                factor = sum([v == indices_map[label] for v in indices_map.values()])
            
            error_probs[i] = total_strengths[indices_map[label]] / factor
        
        stochastic_error_probs[gate] = error_probs
    
    return _mc.create_crosstalk_free_model(pspec.number_of_qubits, pspec.root_gate_names,
                                           stochastic_error_probs = stochastic_error_probs)
