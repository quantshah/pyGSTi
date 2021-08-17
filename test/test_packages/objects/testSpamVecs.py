import itertools
import pickle
import unittest

import numpy as np
from numpy.random import random, seed

import pygsti
from pygsti.models import modelconstruction
from pygsti.modelpacks.legacy import std1Q_XYI
from pygsti.modelmembers.states import State
from pygsti.modelmembers import states
from pygsti.modelmembers import povms
from ..testutils import BaseTestCase


class SPAMVecTestCase(BaseTestCase):
    def test_cptp_state(self):
        vec = states.CPTPState([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2) - 0.1], "pp")
        print(vec)
        print(vec.base.shape)


        v = vec.to_vector()
        vec.from_vector(v)
        print(v)
        print(vec)

        vec_std = pygsti.change_basis(vec, "pp", "std")
        print(vec_std)

        def analyze(spamvec):
            stdmx = pygsti.vec_to_stdmx(spamvec, "pp")
            evals = np.linalg.eigvals(stdmx)
            #print(evals)
            assert( np.all(evals > -1e-10) )
            assert( np.linalg.norm(np.imag(evals)) < 1e-8)
            return np.real(evals)

        print( analyze(vec) )

        states.check_deriv_wrt_params(vec)


        seed(1234)

        #Nice cases - when parameters are small
        nRandom = 1000
        for randvec in random((nRandom,4)):
            r = 2*(randvec-0.5)
            vec.from_vector(r)
            evs = analyze(vec)
            states.check_deriv_wrt_params(vec)
            #print(r, "->", evs)
        print("OK1")

        #Mean cases - when parameters are large
        nRandom = 1000
        for randvec in random((nRandom,4)):
            r = 10*(randvec-0.5)
            vec.from_vector(r)
            evs = analyze(vec)
            states.check_deriv_wrt_params(vec)
            #print(r, "->", evs)
        print("OK2")

        vec.depolarize(0.01)
        vec.depolarize((0.1,0.09,0.08))


    #TODO
    def test_complement_spamvec(self):
        model = pygsti.models.modelconstruction.create_explicit_model_from_expressions(
            [('Q0',)],['Gi','Gx','Gy'],
            [ "I(Q0)","X(pi/8,Q0)", "Y(pi/8,Q0)"])

        E0 = model.povms['Mdefault']['0']
        E1 = model.povms['Mdefault']['1']
        Ec = povms.ComplementPOVMEffect(
            modelconstruction._create_identity_vec([4], "pp"),
            [E0])
        print(Ec.gpindices)

        #Test TPPOVM which uses a complement evec
        model.povms['Mtest'] = povms.TPPOVM([('+', E0), ('-', E1)])
        E0 = model.povms['Mtest']['+']
        Ec = model.povms['Mtest']['-']

        v = model.to_vector()
        model.from_vector(v)

        #print(Ec.num_params) #not implemented for complement vecs - only for POVM
        identity = np.array([[np.sqrt(2)], [0], [0], [0]],'d')
        print("TEST1")
        print(E0)
        print(Ec)
        print(E0 + Ec)
        self.assertArraysAlmostEqual(E0+Ec, identity)

        #TODO: add back if/when we can set parts of a POVM directly...
        #print("TEST2")
        #model.effects['E0'] = [1/np.sqrt(2), 0, 0.4, 0.6]
        #print(model.effects['E0'])
        #print(model.effects['E1'])
        #print(model.effects['E0'] + model.effects['E1'])
        #self.assertArraysAlmostEqual(model.effects['E0'] + model.effects['E1'], identity)
        #
        #print("TEST3")
        #model.effects['E0'][0,0] = 1.0 #uses dirty processing
        #model._update_paramvec(model.effects['E0'])
        #print(model.effects['E0'])
        #print(model.effects['E1'])
        #print(model.effects['E0'] + model.effects['E1'])
        #self.assertArraysAlmostEqual(model.effects['E0'] + model.effects['E1'], identity)


    def test_povms(self):
        model = pygsti.models.modelconstruction.create_explicit_model_from_expressions(
            [('Q0',)],['Gi'], ["I(Q0)"])
        gateset2Q = pygsti.models.modelconstruction.create_explicit_model_from_expressions(
            [('Q0','Q1')],['Gi'], ["I(Q0)"])

        povm = model.povms['Mdefault'].copy()
        E0 = povm['0']
        E1 = povm['1']
        model.povms['Munconstrained'] = povm # so gpindices get setup

        with self.assertRaises(ValueError):
            povms.convert(povm, "foobar", model.basis)
        with self.assertRaises(ValueError):
            povms.UnconstrainedPOVM("NotAListOrDict")

        povm['0'] = E0 # assignment
        tp_povm = povms.convert(povm, "TP", model.basis)
        tp_povm['0'] = E0 # ok
        with self.assertRaises(KeyError):
            tp_povm['1'] = E0 # can't assign complement vector
        model.povms['Mtp'] = tp_povm # so gpindices get setup

        factorPOVMs = [povm, povm.copy()]
        tensor_povm = povms.TensorProductPOVM(factorPOVMs)
        gateset2Q.povms['Mtensor'] = tensor_povm # so gpindices get setup

        for i,p in enumerate([povm, tp_povm, tensor_povm]):
            print("Testing POVM of type ", type(p))
            Nels = p.num_elements
            cpy = p.copy()
            s = str(p)

            s = pickle.dumps(p)
            x = pickle.loads(s)

            T = pygsti.models.gaugegroup.FullGaugeGroupElement(
                np.array( [ [1,0,0,0],
                            [0,0,1,0],
                            [0,1,0,0],
                            [0,0,0,1]], 'd') )

            v = p.to_vector()
            p.from_vector(v)

            v = model.to_vector() if i < 2 else gateset2Q.to_vector()
            effects = p.simplify_effects(prefix="ABC")
            for Evec in effects.values():
                print("inds = ",Evec.gpindices, len(v))
                Evec.from_vector(v[Evec.gpindices]) # gpindices should be setup relative to Model's param vec


            try:
                p.transform_inplace(T)
            except ValueError:
                pass #OK - tensorprod doesn't allow transform for instance

            try:
                p.depolarize(0.01)
            except ValueError:
                pass #OK - tensorprod doesn't allow transform for instance

    def test_compbasis_povm(self):
        cv = states.ComputationalBasisState([0, 1], 'pp', 'densitymx')
        v = modelconstruction._basis_create_spam_vector("1", pygsti.baseobjs.Basis.cast("pp", 4 ** 2))
        self.assertTrue(np.linalg.norm(cv.to_dense()-v.flat) < 1e-6)

        cv = states.ComputationalBasisState([0, 0, 1], 'pp', 'densitymx')
        v = modelconstruction._basis_create_spam_vector("1", pygsti.baseobjs.Basis.cast("pp", 4 ** 3))
        self.assertTrue(np.linalg.norm(cv.to_dense()-v.flat) < 1e-6)

        cv = states.ComputationalBasisState([0, 0, 1], 'pp', 'densitymx')
        v = modelconstruction._basis_create_spam_vector("1", pygsti.baseobjs.Basis.cast("pp", 4 ** 3))
        self.assertTrue(np.linalg.norm(cv.to_dense()-v.flat) < 1e-6)

        cv = states.ComputationalBasisState([0, 0, 1], 'pp', 'densitymx')
        v = modelconstruction._basis_create_spam_vector("1", pygsti.baseobjs.Basis.cast("pp", 4 ** 3))
        self.assertTrue(np.linalg.norm(cv.to_dense()-v.flat) < 1e-6)

        #Only works with Python replib (only there is to_dense implemented)
        #cv = pygsti.obj.ComputationalSPAMVec([0,1,1],'densitymx')
        #v = modelconstruction._basis_create_spam_vector("3", pygsti.obj.Basis.cast("pp",4**3))
        #s = pygsti.obj.FullSPAMVec(v)
        #assert(np.linalg.norm(cv.to_rep("effect").todense(np.empty(cv.dim,'d'))-v.flat) < 1e-6)
        #
        #cv = pygsti.obj.ComputationalSPAMVec([0,1,0,1],'densitymx')
        #v = modelconstruction._basis_create_spam_vector("5", pygsti.obj.Basis.cast("pp",4**4))
        #assert(np.linalg.norm(cv.to_rep("effect").todense(np.empty(cv.dim,'d'))-v.flat) < 1e-6)

        nqubits = 3
        iterover = [(0,1)]*nqubits
        items = [ (''.join(map(str,outcomes)), povms.ComputationalBasisPOVMEffect(outcomes, 'pp', "densitymx"))
                  for outcomes in itertools.product(*iterover) ]
        povm = povms.UnconstrainedPOVM(items)
        self.assertEqual(povm.num_params,0)

        mdl = std1Q_XYI.target_model()
        mdl.preps['rho0'] = states.ComputationalBasisState([0], 'pp', 'densitymx')
        mdl.povms['Mdefault'] = povms.UnconstrainedPOVM({'0': povms.ComputationalBasisPOVMEffect([0], 'pp', 'densitymx'),
                                                         '1': povms.ComputationalBasisPOVMEffect([1], 'pp', 'densitymx')})

        ps0 = mdl.probabilities(())
        ps1 = mdl.probabilities(('Gx',))
        self.assertAlmostEqual(ps0['0'], 1.0)
        self.assertAlmostEqual(ps0['1'], 0.0)
        self.assertAlmostEqual(ps1['0'], 0.5)
        self.assertAlmostEqual(ps1['1'], 0.5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
