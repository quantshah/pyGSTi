"""
State representation classes for the `densitymx_slow` evolution type.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import functools as _functools

import numpy as _np

from .. import basereps as _basereps
from pygsti.baseobjs.statespace import StateSpace as _StateSpace
from ...tools import basistools as _bt
from ...tools import optools as _ot

try:
    from ...tools import fastcalc as _fastcalc
except ImportError:
    _fastcalc = None


class StateRep(_basereps.StateRep):
    def __init__(self, data, state_space):
        #vec = _np.asarray(vec, dtype='d')
        assert(data.dtype == _np.dtype('d'))
        self.data = _np.require(data.copy(), requirements=['OWNDATA', 'C_CONTIGUOUS'])
        self.state_space = _StateSpace.cast(state_space)
        assert(len(self.data) == self.state_space.dim)

    def __reduce__(self):
        return (StateRep, (self.data, self.state_space), (self.data.flags.writeable,))

    def __setstate__(self, state):
        writeable, = state
        self.data.flags.writeable = writeable

    def copy_from(self, other):
        self.data[:] = other.data

    def actionable_staterep(self):
        # return a state rep that can be acted on by op reps or mapped to
        # a probability/amplitude by POVM effect reps.
        return self  # for most classes, the rep itself is actionable

    def to_dense(self, on_space):
        if on_space not in ('minimal', 'HilbertSchmidt'):
            raise ValueError("'densitymx' evotype cannot produce Hilbert-space ops!")
        return self.data

    @property
    def dim(self):
        return self.state_space.dim

    def __str__(self):
        return str(self.data)


class StateRepDense(StateRep):
    @property
    def base(self):
        return self.data

    def base_has_changed(self):
        pass

    def __reduce__(self):
        return (StateRepDense, (self.base, self.state_space), (self.base.flags.writeable,))


class StateRepDensePure(StateRep):
    def __init__(self, purevec, basis, state_space):
        assert(purevec.dtype == _np.dtype(complex))
        self.base = _np.require(purevec.copy(), requirements=['OWNDATA', 'C_CONTIGUOUS'])
        self.basis = basis
        dmVec_std = _ot.state_to_dmvec(self.base)
        super(StateRepDensePure, self).__init__(_bt.change_basis(dmVec_std, 'std', self.basis), state_space)

    def base_has_changed(self):
        dmVec_std = _ot.state_to_dmvec(self.base)
        self.data[:] = _bt.change_basis(dmVec_std, 'std', self.basis)

    def __reduce__(self):
        return (StateRepDensePure, (self.base, self.basis, self.state_space), (self.base.flags.writeable,))


class StateRepComputational(StateRep):
    def __init__(self, zvals, basis, state_space):

        #Convert zvals to dense vec:
        assert(basis.name == 'pp'), "Only Pauli-product-basis computational states are supported so far"
        self.basis = basis
        factor_dim = 4
        v0 = 1.0 / _np.sqrt(2) * _np.array((1, 0, 0, 1), 'd')  # '0' qubit state as Pauli dmvec
        v1 = 1.0 / _np.sqrt(2) * _np.array((1, 0, 0, -1), 'd')  # '1' qubit state as Pauli dmvec
        v = (v0, v1)

        if _fastcalc is None:  # do it the slow way using numpy
            vec = _functools.reduce(_np.kron, [v[i] for i in zvals])
        else:
            typ = 'd'
            fast_kron_array = _np.ascontiguousarray(
                _np.empty((len(zvals), factor_dim), 'd'))
            fast_kron_factordims = _np.ascontiguousarray(_np.array([factor_dim] * len(zvals), _np.int64))
            for i, zi in enumerate(zvals):
                fast_kron_array[i, :] = v[zi]
            vec = _np.ascontiguousarray(_np.empty(factor_dim**len(zvals), typ))
            _fastcalc.fast_kron(vec, fast_kron_array, fast_kron_factordims)

        self.zvals = zvals
        super(StateRepComputational, self).__init__(vec, state_space)

    def __reduce__(self):
        return (StateRepComputational, (self.zvals, self.basis, self.state_space), (self.data.flags.writeable,))


class StateRepComposed(StateRep):
    def __init__(self, state_rep, op_rep, state_space):
        self.state_rep = state_rep
        self.op_rep = op_rep
        super(StateRepComposed, self).__init__(state_rep.to_dense('HilbertSchmidt'), state_space)
        self.reps_have_changed()

    def reps_have_changed(self):
        rep = self.op_rep.acton(self.state_rep)
        self.data[:] = rep.data[:]

    def __reduce__(self):
        return (StateRepComposed, (self.state_rep, self.op_rep, self.state_space), (self.data.flags.writeable,))


class StateRepTensorProduct(StateRep):
    def __init__(self, factor_state_reps, state_space):
        self.factor_reps = factor_state_reps
        dim = _np.product([fct.dim for fct in self.factor_reps])
        super(StateRepTensorProduct, self).__init__(_np.zeros(dim, 'd'), state_space)
        self.reps_have_changed()

    def reps_have_changed(self):
        if len(self.factor_reps) == 0:
            vec = _np.empty(0, 'd')
        else:
            vec = self.factor_reps[0].to_dense('HilbertSchmidt')
            for i in range(1, len(self.factor_reps)):
                vec = _np.kron(vec, self.factor_reps[i].to_dense('HilbertSchmidt'))
        self.data[:] = vec

    def __reduce__(self):
        return (StateRepTensorProduct, (self.factor_reps, self.state_space), (self.data.flags.writeable,))

    #REMOVE - or do something with this for a to_dense method?
    #def _fill_fast_kron(self):
    #    """ Fills in self._fast_kron_array based on current self.factors """
    #    for i, factor_dim in enumerate(self._fast_kron_factordims):
    #        self._fast_kron_array[i][0:factor_dim] = self.factors[i].to_dense()

#        if self._evotype in ("statevec", "densitymx"):
#            if self._prep_or_effect == "prep":
#                self._rep.base[:] = self.to_dense()
#            else:
#                self._fill_fast_kron()  # updates effect reps
#        elif self._evotype == "stabilizer":
#            if self._prep_or_effect == "prep":
#                #we need to update self._rep, which is a SBStateRep object.  For now, we
#                # kinda punt and just create a new rep and copy its data over to the existing
#                # rep instead of having some kind of update method within SBStateRep...
#                # (TODO FUTURE - at least a .copy_from method?)
#                sframe_factors = [f.to_dense() for f in self.factors]  # StabilizerFrame for each factor
#                new_rep = _stabilizer.sframe_kronecker(sframe_factors).to_rep()
#                self._rep.smatrix[:, :] = new_rep.smatrix[:, :]
#                self._rep.pvectors[:, :] = new_rep.pvectors[:, :]
#                self._rep.amps[:, :] = new_rep.amps[:, :]
#            else:
#                pass  # I think the below (e.g. 'outcomes') is not altered by any parameters
#                #factor_povms = self.factors
#                #factorVecs = [factor_povms[i][self.effectLbls[i]] for i in range(1, len(factor_povms))]
#                #outcomes = _np.array(list(_itertools.chain(*[f.outcomes for f in factorVecs])), _np.int64)
#                #rep = replib.SBEffectRep(outcomes)
#
#    def to_dense(self):  # from tensorprodstate
#        if self._evotype in ("statevec", "densitymx"):
#            if len(self.factors) == 0: return _np.empty(0, complex if self._evotype == "statevec" else 'd')
#            #NOTE: moved a fast version of to_dense to replib - could use that if need a fast to_dense call...
#
#            ret = self.factors[0].to_dense()  # factors are just other SPAMVecs
#            for i in range(1, len(self.factors)):
#                ret = _np.kron(ret, self.factors[i].to_dense())
#            return ret
#        elif self._evotype == "stabilizer":
#
#            # => self.factors should all be StabilizerSPAMVec objs
#            #Return stabilizer-rep tuple, just like StabilizerSPAMVec
#            sframe_factors = [f.to_dense() for f in self.factors]
#            return _stabilizer.sframe_kronecker(sframe_factors)
#        else:  # self._evotype in ("svterm","cterm")
#            raise NotImplementedError("to_dense() not implemented for %s evolution type" % self._evotype)
#
#
#
#    def init_from_dense_vec(self, vec):
#
#
#        pass
#
#    def init_from_dense_purevec(self, purevec):
#
#        if not isinstance(pure_state_vec, _State):
#            pure_state_vec = StaticState(_State._to_vector(pure_state_vec), 'statevec')
#        self.pure_state_vec = pure_state_vec
#        self.basis = dm_basis  # only used for dense conversion
#
#        pure_evo = pure_state_vec._evotype
#        if pure_evo == "statevec":
#            if evotype not in ("densitymx", "svterm"):
#                raise ValueError(("`evotype` arg must be 'densitymx' or 'svterm'"
#                                  " when `pure_state_vec` evotype is 'statevec'"))
#        elif pure_evo == "stabilizer":
#            if evotype not in ("cterm",):
#                raise ValueError(("`evotype` arg must be 'densitymx' or 'svterm'"
#                                  " when `pure_state_vec` evotype is 'statevec'"))
#        else:
#            raise ValueError("`pure_state_vec` evotype must be 'statevec' or 'stabilizer' (not '%s')" % pure_evo)
#
#        dim = self.pure_state_vec.dim**2
#
#        pass
#
#    def init_from_zvalues(self, zvals):
#        pass
#
#    @property
#    def base(self):
#        pass  # numpy array of dense super-bra (real) vector
#
#
#    @property
#    def purebase(self):
#        pass  # numpy array of dense pure-state (complex) vector
#
#
#    def copy_from(self, other_state_rep):
#        pass
#
#
#
#        dtype = complex if evotype == "statevec" else 'd'
#        if evotype == "statevec":
#            rep = replib.SVStateRep(vec)
#        elif evotype == "densitymx":
#            rep = replib.DMStateRep(vec)
#        else:
#            raise ValueError("Invalid evotype for DenseSPAMVec: %s" % evotype)
#
#
#    #Computational:
#           if evotype == "statevec":
#                rep = replib.SVStateRep(self.to_dense())
#            elif evotype == "densitymx":
#                vec = _np.require(self.to_dense(), requirements=['OWNDATA', 'C_CONTIGUOUS'])
#                rep = replib.DMStateRep(vec)
#            elif evotype == "stabilizer":
#                sframe = _stabilizer.StabilizerFrame.from_zvals(len(self._zvals), self._zvals)
#                rep = sframe.to_rep()
#            else:
#                rep = dim  # no representations for term-based evotypes
#
#
#    def init_as_tensor_product_of_states(self, state_factors):
#
#        #Create representation
#        dim = _np.product([fct.dim for fct in factors])
#        if evotype == "statevec":
#            rep = replib.SVStateRep(_np.ascontiguousarray(_np.zeros(dim, complex)))
#        elif evotype == "densitymx":
#            vec = _np.require(_np.zeros(dim, 'd'), requirements=['OWNDATA', 'C_CONTIGUOUS'])
#            rep = replib.DMStateRep(vec)
#        elif evotype == "stabilizer":
#            #Rep is stabilizer-rep tuple, just like StabilizerSPAMVec
#            sframe_factors = [f.to_dense() for f in self.factors]  # StabilizerFrame for each factor
#            rep = _stabilizer.sframe_kronecker(sframe_factors).to_rep()
#        else:  # self._evotype in ("svterm","cterm")
#            rep = dim  # no reps for term-based evotypes
#
#
#
#
##UPDATE REP? (from tensorprod)
#                                if self._prep_or_effect == "prep":
#                #we need to update self._rep, which is a SBStateRep object.  For now, we
#                # kinda punt and just create a new rep and copy its data over to the existing
#                # rep instead of having some kind of update method within SBStateRep...
#                # (TODO FUTURE - at least a .copy_from method?)
#                sframe_factors = [f.to_dense() for f in self.factors]  # StabilizerFrame for each factor
#                new_rep = _stabilizer.sframe_kronecker(sframe_factors).to_rep()
#                self._rep.smatrix[:, :] = new_rep.smatrix[:, :]
#                self._rep.pvectors[:, :] = new_rep.pvectors[:, :]
#                self._rep.amps[:, :] = new_rep.amps[:, :]
#            else:
