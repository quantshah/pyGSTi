"""
Defines OrderedDict-derived classes used to store specific pyGSTi objects
"""
import copy as _copy
import numbers as _numbers
import sys as _sys

# ***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
# ***************************************************************************************************
import numpy as _np


class StateSpace(object):
    """
    Base class for defining a state space (Hilbert or Hilbert-Schmidt space).

    This class mainly just sets the API for a "state space" in pyGSTi, accessed
    as the direct sum of one or more tensor products of Hilbert spaces.

    TODO: docstrings for this module
    """

    @classmethod
    def cast(cls, obj):
        """
        Casts `obj` into a :class:`StateSpace` object if possible.

        If `obj` is already of this type, it is simply returned without modification.

        Parameters
        ----------
        obj : StateSpace or int or list
            Either an already-built state space object or an integer specifying the number of qubits,
            or a list of labels as would be provided to the first argument of :method:`ExplicitStateSpace.__init__`.

        Returns
        -------
        StateSpace
        """
        if isinstance(obj, StateSpace):
            return obj
        if isinstance(obj, int) or all([isinstance(x, int) or (isinstance(x, str) and x.startswith('Q')) for x in obj]):
            return QubitSpace(obj)
        return ExplicitStateSpace(obj)

    def is_entirely_qubits(self):
        try:
            self.num_qubits
            return True
        except Exception:
            return False

    @property
    def udim(self):
        raise NotImplementedError("Derived classes should implement this!")

    @property
    def dim(self):
        raise NotImplementedError("Derived classes should implement this!")

    @property
    def num_qubits(self):  # may raise ValueError if the state space doesn't consist entirely of qubits
        raise NotImplementedError("Derived classes should implement this!")

    @property
    def num_tensor_product_blocks(self):
        """
        Get the number of tensor-product blocks which are direct-summed to get the final state space.

        Returns
        -------
        int
        """
        raise NotImplementedError("Derived classes should implement this!")

    @property
    def tensor_product_blocks_labels(self):
        """
        Get the labels for all the tensor-product blocks.

        Returns
        -------
        tuple of tuples
        """
        raise NotImplementedError("Derived classes should implement this!")

    @property
    def tensor_product_blocks_dimensions(self):
        """
        Get the superoperator dimensions for all the tensor-product blocks.

        Returns
        -------
        tuple of tuples
        """
        raise NotImplementedError("Derived classes should implement this!")

    @property
    def tensor_product_blocks_udimensions(self):
        """
        Get the unitary-operator dimensions for all the tensor-product blocks.

        Returns
        -------
        tuple of tuples
        """
        raise NotImplementedError("Derived classes should implement this!")

    @property
    def tensor_product_blocks_types(self):
        """
        Get the type (quantum vs classical) of all the tensor-product blocks.

        Returns
        -------
        tuple of tuples
        """
        raise NotImplementedError("Derived classes should implement this!")

    def label_dimension(self, label):
        raise NotImplementedError("Derived classes should implement this!")

    def label_udimension(self, label):
        raise NotImplementedError("Derived classes should implement this!")

    def label_tensor_product_block_index(self, label):
        raise NotImplementedError("Derived classes should implement this!")

    def label_type(self, label):
        raise NotImplementedError("Derived classes should implement this!")

    def tensor_product_block_labels(self, i_tpb):
        """
        Get the labels for the `iTBP`-th tensor-product block.

        Parameters
        ----------
        i_tpb : int
            Tensor-product block index.

        Returns
        -------
        tuple
        """
        return self.tensor_product_blocks_labels[i_tpb]

    def tensor_product_block_dimensions(self, i_tpb):
        """
        Get the superoperator dimensions for the factors in the `iTBP`-th tensor-product block.

        Parameters
        ----------
        i_tpb : int
            Tensor-product block index.

        Returns
        -------
        tuple
        """
        return self.tensor_product_blocks_dimensions[i_tpb]

    def copy(self):
        """
        Return a copy of this StateSpace.

        Returns
        -------
        StateSpace
        """
        return _copy.deepcopy(self)

    def is_compatible_with(self, other_state_space):
        try:
            if self.num_qubits == other_state_space.num_qubits:
                return True
        except Exception:
            if self.udim == other_state_space.udim:
                return True
        return False

    def __repr__(self):
        return self.__class__.__name__ + "[" + str(self) + "]"

    def __hash__(self):
        return hash((self.tensor_product_blocks_labels,
                     self.tensor_product_blocks_dimensions,
                     self.tensor_product_blocks_types))

    def __eq__(self, other_statespace):
        if isinstance(other_statespace, StateSpace):
            return (self.tensor_product_blocks_labels == other_statespace.tensor_product_blocks_labels
                    and self.tensor_product_blocks_dimensions == other_statespace.tensor_product_blocks_dimensions
                    and self.tensor_product_blocks_types == other_statespace.tensor_product_blocks_types)
        else:
            return False  # this state space is not equal to anything that isn't another state space


class QubitSpace(StateSpace):
    """
    A state space consisting of N qubits.
    """

    def __init__(self, nqubits_or_labels):
        if isinstance(nqubits_or_labels, int):
            self.qubit_labels = tuple(range(nqubits_or_labels))
        else:
            self.qubit_labels = tuple(nqubits_or_labels)

    @property
    def udim(self):
        return 2**self.num_qubits

    @property
    def dim(self):
        return 4**self.num_qubits

    @property
    def num_qubits(self):  # may raise ValueError if the state space doesn't consist entirely of qubits
        return len(self.qubit_labels)

    @property
    def num_tensor_product_blocks(self):
        """
        Get the number of tensor-product blocks which are direct-summed to get the final state space.

        Returns
        -------
        int
        """
        return 1

    @property
    def tensor_product_blocks_labels(self):
        """
        Get the labels for all the tensor-product blocks.

        Returns
        -------
        tuple of tuples
        """
        return (self.qubit_labels,)

    @property
    def tensor_product_blocks_dimensions(self):
        """
        Get the superoperator dimensions for all the tensor-product blocks.

        Returns
        -------
        tuple of tuples
        """
        return ((4,) * self.num_qubits,)

    @property
    def tensor_product_blocks_udimensions(self):
        """
        Get the unitary operator dimensions for all the tensor-product blocks.

        Returns
        -------
        tuple of tuples
        """
        return ((2,) * self.num_qubits,)

    @property
    def tensor_product_blocks_types(self):
        """
        Get the type (quantum vs classical) of all the tensor-product blocks.

        Returns
        -------
        tuple of tuples
        """
        return (('Q',) * self.num_qubits,)

    def label_dimension(self, label):
        if label in self.qubit_labels:
            return 4
        else:
            raise KeyError("Invalid qubit label: %s" % label)

    def label_udimension(self, label):
        if label in self.qubit_labels:
            return 2
        else:
            raise KeyError("Invalid qubit label: %s" % label)

    def label_tensor_product_block_index(self, label):
        if label in self.qubit_labels:
            return 0
        else:
            raise KeyError("Invalid qubit label: %s" % label)

    def label_type(self, label):
        if label in self.qubit_labels:
            return 'Q'
        else:
            raise KeyError("Invalid qubit label: %s" % label)

    def __str__(self):
        if len(self.qubit_labels) <= 10:
            return 'QubitSpace(' + str(self.qubit_labels) + ")"
        else:
            return 'QubitSpace(' + str(len(self.qubit_labels)) + ")"


class ExplicitStateSpace(StateSpace):
    """
    A customizable definition of a state space.

    An ExplicitStateSpace object describes, using string/int labels, how an entire
    Hilbert state space is decomposed into the direct sum of terms which
    themselves are tensor products of smaller (typically qubit-sized) Hilbert
    spaces.

    Parameters
    ----------
    label_list : str or int or iterable
        Most generally, this can be a list of tuples, where each tuple
        contains the state-space labels (which can be strings or integers)
        for a single "tensor product block" formed by taking the tensor
        product of the spaces asociated with the labels.  The full state
        space is the direct sum of all the tensor product blocks.
        E.g. `[('Q0','Q1'), ('Q2',)]`.

        If just an iterable of labels is given, e.g. `('Q0','Q1')`, it is
        assumed to specify the first and only tensor product block.

        If a single state space label is given, e.g. `'Q2'`, then it is
        assumed to completely specify the first and only tensor product
        block.

    udims : int or iterable, optional
        The dimension of each state space label as an integer, tuple of
        integers, or list or tuples of integers to match the structure
        of `label_list` (i.e., if `label_list=('Q0','Q1')` then `dims` should
        be a tuple of 2 integers).  Values specify unitary evolution state-space
        dimensions, i.e. 2 for a qubit, 3 for a qutrit, etc.  If None, then the
        dimensions are inferred, if possible, from the following naming rules:

        - if the label starts with 'L', udim=1 (a single Level)
        - if the label starts with 'Q' OR is an int, udim=2 (a Qubit)
        - if the label starts with 'T', udim=3 (a quTrit)

    types : str or iterable, optional
        A list of label types, either `'Q'` or `'C'` for "quantum" and
        "classical" respectively, indicating the type of state-space
        associated with each label.  Like `dims`, `types` must match
        the structure of `label_list`.  A quantum state space of dimension
        `d` is a `d`-by-`d` density matrix, whereas a classical state space
        of dimension d is a vector of `d` probabilities.  If `None`, then
        all labels are assumed to be quantum.
    """

    def __init__(self, label_list, udims=None, types=None):

        #Allow initialization via another CustomStateSpace object
        #if isinstance(label_list, CustomStateSpace):
        #    assert(dims is None and types is None), "Clobbering non-None 'dims' and/or 'types' arguments"
        #    dims = [tuple((label_list.labeldims[lbl] for lbl in tpbLbls))
        #            for tpbLbls in label_list.labels]
        #    types = [tuple((label_list.labeltypes[lbl] for lbl in tpbLbls))
        #             for tpbLbls in label_list.labels]
        #    label_list = label_list.labels

        #Step1: convert label_list (and dims, if given) to a list of
        # elements describing each "tensor product block" - each of
        # which is a tuple of string labels.

        def is_label(x):
            """ Return whether x is a valid space-label """
            return isinstance(x, str) or isinstance(x, _numbers.Integral)

        if is_label(label_list):
            label_list = [(label_list,)]
            if udims is not None: udims = [(udims,)]
            if types is not None: types = [(types,)]
        else:
            #label_list must be iterable if it's not a string
            label_list = list(label_list)

        if len(label_list) > 0 and is_label(label_list[0]):
            # assume we've just been give the labels for a single tensor-prod-block
            label_list = [label_list]
            if udims is not None: udims = [udims]
            if types is not None: types = [types]

        self.labels = tuple([tuple(tpbLabels) for tpbLabels in label_list])

        #Type check - labels must be strings or ints
        for tpbLabels in self.labels:  # loop over tensor-prod-blocks
            for lbl in tpbLabels:
                if not is_label(lbl):
                    raise ValueError("'%s' is an invalid state-space label (must be a string or integer)" % lbl)

        # Get the type of each labeled space
        self.labeltypes = {}
        if types is None:  # use defaults
            for tpbLabels in self.labels:  # loop over tensor-prod-blocks
                for lbl in tpbLabels:
                    self.labeltypes[lbl] = 'C' if (isinstance(lbl, str) and lbl.startswith('C')) else 'Q'  # default
        else:
            for tpbLabels, tpbTypes in zip(self.labels, types):
                for lbl, typ in zip(tpbLabels, tpbTypes):
                    self.labeltypes[lbl] = typ

        # Get the dimension of each labeled space
        self.label_udims = {}
        self.label_dims = {}
        if udims is None:
            for tpbLabels in self.labels:  # loop over tensor-prod-blocks
                for lbl in tpbLabels:
                    if isinstance(lbl, _numbers.Integral): d = 2  # ints = qubits
                    elif lbl.startswith('T'): d = 3  # qutrit
                    elif lbl.startswith('Q'): d = 2  # qubits
                    elif lbl.startswith('L'): d = 1  # single level
                    elif lbl.startswith('C'): d = 2  # classical bits
                    else: raise ValueError("Cannot determine state-space dimension from '%s'" % lbl)
                    self.label_udims[lbl] = d
                    self.label_dims[lbl] = d**2 if (isinstance(lbl, _numbers.Integral) or lbl[0] in ('Q', 'T')) else d
        else:
            for tpbLabels, tpbDims in zip(self.labels, udims):
                for lbl, udim in zip(tpbLabels, tpbDims):
                    self.label_udims[lbl] = udim
                    self.label_dims[lbl] = udim**2

        # Store the starting index (within the density matrix / state vec) of
        # each tensor-product-block (TPB), and which labels belong to which TPB
        self.tpb_index = {}

        self.tpb_dims = []
        self.tpb_udims = []
        for iTPB, tpbLabels in enumerate(self.labels):
            float_prod = _np.product(_np.array([self.label_dims[lbl] for lbl in tpbLabels], 'd'))
            if float_prod >= float(_sys.maxsize):  # too many qubits to hold dimension in an integer
                self.tpb_dims.append(_np.inf)
            else:
                self.tpb_dims.append(int(_np.product([self.label_dims[lbl] for lbl in tpbLabels])))

            float_prod = _np.product(_np.array([self.label_udims[lbl] for lbl in tpbLabels], 'd'))
            if float_prod >= float(_sys.maxsize):  # too many qubits to hold dimension in an integer
                self.tpb_udims.append(_np.inf)
            else:
                self.tpb_udims.append(int(_np.product([self.label_udims[lbl] for lbl in tpbLabels])))

            self.tpb_index.update({lbl: iTPB for lbl in tpbLabels})

        self._dim = sum(self.tpb_dims)
        self._udim = sum(self.tpb_udims)

        if len(self.labels) == 1 and all([v == 2 for v in self.label_udims.values()]):
            self._nqubits = len(self.labels[0])  # there's a well-defined number of qubits
        else:
            self._nqubits = None

    #REMOVE - this shouldn't be needed anymore
    #def reduce_dims_densitymx_to_state_inplace(self):
    #    """
    #    Reduce all state space dimensions appropriately for moving from a density-matrix to state-vector rep
    #
    #    Returns
    #    -------
    #    None
    #    """
    #    for lbl in self.label_dims:
    #        if self.labeltypes[lbl] == 'Q':
    #            self.label_dims[lbl] = int(_np.sqrt(self.label_dims[lbl]))
    #
    #    #update tensor-product-block dims and overall dim too:
    #    self.tpb_dims = []
    #    for iTPB, tpbLabels in enumerate(self.labels):
    #        self.tpb_dims.append(int(_np.product([self.label_dims[lbl] for lbl in tpbLabels])))
    #        self.tpb_index.update({lbl: iTPB for lbl in tpbLabels})
    #    self.dim = sum(self.tpb_dims)

    @property
    def udim(self):
        return self._udim

    @property
    def dim(self):
        return self._dim

    @property
    def num_qubits(self):  # may raise ValueError if the state space doesn't consist entirely of qubits
        if self._nqubits is None:
            raise ValueError("This state space is not a tensor product of qubit factors spaces!")
        return self._nqubits

    @property
    def num_tensor_product_blocks(self):
        """
        Get the number of tensor-product blocks which are direct-summed to get the final state space.

        Returns
        -------
        int
        """
        return len(self.labels)

    @property
    def tensor_product_blocks_labels(self):
        """
        Get the labels for all the tensor-product blocks.

        Returns
        -------
        tuple of tuples
        """
        return self.labels

    @property
    def tensor_product_blocks_dimensions(self):
        """
        Get the superoperator dimensions for all the tensor-product blocks.

        Returns
        -------
        tuple of tuples
        """
        return tuple([tuple([self.label_dims[lbl] for lbl in tpb_labels]) for tpb_labels in self.labels])

    @property
    def tensor_product_blocks_udimensions(self):
        """
        Get the unitary operator dimensions for all the tensor-product blocks.

        Returns
        -------
        tuple of tuples
        """
        return tuple([tuple([self.label_udims[lbl] for lbl in tpb_labels]) for tpb_labels in self.labels])

    @property
    def tensor_product_blocks_types(self):
        """
        Get the type (quantum vs classical) of all the tensor-product blocks.

        Returns
        -------
        tuple of tuples
        """
        return tuple([tuple([self.labeltypes[lbl] for lbl in tpb_labels]) for tpb_labels in self.labels])

    def label_dimension(self, label):
        return self.label_dims[label]

    def label_udimension(self, label):
        return self.label_udims[label]

    def label_tensor_product_block_index(self, label):
        return self.tpb_index[label]

    def label_type(self, label):
        return self.labeltypes[label]

    #REMOVE
    #def product_dim(self, labels):  # only in modelconstruction
    #    """
    #    Computes the product of the state-space dimensions associated with each label in `labels`.
    #
    #    Parameters
    #    ----------
    #    labels : list
    #        A list of state space labels (strings or integers).
    #
    #    Returns
    #    -------
    #    int
    #    """
    #    return int(_np.product([self.label_dims[l] for l in labels]))

    def __str__(self):
        if len(self.labels) == 0: return "ZeroDimSpace"
        return ' + '.join(
            ['*'.join(["%s(%d%s)" % (lbl, self.label_dims[lbl], 'c' if (self.labeltypes[lbl] == 'C') else '')
                       for lbl in tpb]) for tpb in self.labels])


def default_space_for_dim(dim):
    nqubits = int(round(_np.log2(dim) / 2))
    if 4**nqubits == dim:
        return QubitSpace(nqubits)
    else:
        udim = int(round(_np.sqrt(dim)))
        assert(udim**2 == dim), "`dim` must be a perfect square: %d is not" % dim
        return ExplicitStateSpace(('all',), udims=(udim,), types=('quantum',))


def default_space_for_udim(udim):
    nqubits = int(round(_np.log2(udim)))
    if 2**nqubits == udim:
        return QubitSpace(nqubits)
    else:
        return ExplicitStateSpace(('all',), udims=(udim,), types=('quantum',))


def default_space_for_num_qubits(num_qubits):
    return QubitSpace(num_qubits)
