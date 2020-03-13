
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

import unittest
import os
import sys
import numpy as np
import pickle
import collections

import pygsti
from pygsti.modelpacks.legacy import std1Q_XY as std
import pygsti.io.json as json
import pygsti.io.msgpack as msgpack
from pygsti.objects.label import CircuitLabel

from ..testutils import BaseTestCase, compare_files, temp_files


class ObjDerivedFromStdType(list):
    def __init__(self,listInit):
        self.extra = "Hello"
        super(ObjDerivedFromStdType,self).__init__(listInit)


testObj = ObjDerivedFromStdType( (1,2,3) )
testObj.__class__.__module__ = "pygsti.objects" # make object look like a pygsti-native object so it gets special serialization treatment.
sys.modules['pygsti.objects'].ObjDerivedFromStdType = ObjDerivedFromStdType

class CodecsTestCase(BaseTestCase):

    def setUp(self):
        std.target_model()._check_paramvec()
        super(CodecsTestCase, self).setUp()
        self.model = std.target_model()

        self.germs = pygsti.construction.circuit_list( [('Gx',), ('Gy',) ] ) #abridged for speed
        self.fiducials = std.fiducials
        self.maxLens = [1,2]
        self.opLabels = list(self.model.operations.keys())

        self.lsgstStrings = pygsti.construction.make_lsgst_lists(
            self.opLabels, self.fiducials, self.fiducials, self.germs, self.maxLens )

        self.datagen_gateset = self.model.depolarize(op_noise=0.05, spam_noise=0.1)
        test = self.datagen_gateset.copy()
        self.ds = pygsti.construction.generate_fake_data(
            self.datagen_gateset, self.lsgstStrings[-1],
            n_samples=1000,sample_error='binomial', seed=100)

        #Make an model with instruments
        E = self.datagen_gateset.povms['Mdefault']['0']
        Erem = self.datagen_gateset.povms['Mdefault']['1']
        Gmz_plus = np.dot(E,E.T)
        Gmz_minus = np.dot(Erem,Erem.T)
        self.mdl_withInst = self.datagen_gateset.copy()
        self.mdl_withInst.instruments['Iz'] = pygsti.obj.Instrument({'plus': Gmz_plus, 'minus': Gmz_minus})
        self.mdl_withInst.instruments['Iztp'] = pygsti.obj.TPInstrument({'plus': Gmz_plus, 'minus': Gmz_minus})

        self.results = self.runSilent(pygsti.do_long_sequence_gst,
                                     self.ds, std.target_model(), self.fiducials, self.fiducials,
                                     self.germs, self.maxLens)

        #make a confidence region factory
        estLbl = "default"
        crfact = self.results.estimates[estLbl].add_confidence_region_factory('go0', 'final')
        crfact.compute_hessian(comm=None)
        crfact.project_hessian('std')

        #create a Workspace object
        self.ws = pygsti.report.create_standard_report(self.results, None,
                                                       title="GST Codec TEST Report",
                                                       confidence_level=95)
        std.target_model()._check_paramvec()

        #create miscellaneous other objects
        self.miscObjects = []
        self.miscObjects.append( pygsti.objects.labeldicts.OutcomeLabelDict(
            [( ('0',), 90 ), ( ('1',), 10)]) )


class TestCodecs(CodecsTestCase):

    def test_json(self):

        #basic types
        s = json.dumps(range(10))
        x = json.loads(s)
        s = json.dumps(4+3.0j)
        x = json.loads(s)
        s = json.dumps(np.array([1,2,3,4],'d'))
        x = json.loads(s)
        s = json.dumps( testObj )
        x = json.loads(s)

        #string list
        s = json.dumps(self.lsgstStrings)
        x = json.loads(s)
        self.assertEqual(x, self.lsgstStrings)

        # DataSet
        s = json.dumps(self.ds)
        x = json.loads(s)
        self.assertEqual(list(x.keys()), list(self.ds.keys()))
        self.assertEqual(x[('Gx',)].as_dict(), self.ds[('Gx',)].as_dict())

        # Model
        s = json.dumps(self.datagen_gateset)
        with open(temp_files + "/model.json",'w') as f:
            json.dump(self.datagen_gateset, f)
        with open(temp_files + "/model.json",'r') as f:
            x = json.load(f)
        s = json.dumps(self.mdl_withInst)
        x = json.loads(s)
        self.assertAlmostEqual(self.mdl_withInst.frobeniusdist(x),0)

        #print(s)
        x._check_paramvec(True)
        self.assertAlmostEqual(self.datagen_gateset.frobeniusdist(x),0)

        # Results (containing confidence region)
        std.target_model()._check_paramvec()
        print("target_model = ",id(std.target_model()))
        print("rho0 parent = ",id(std.target_model().preps['rho0'].parent))

        #import bpdb; bpdb.set_trace()
        with open(temp_files + "/results.json",'w') as f:
            json.dump(self.results, f)
        print("mdl_target2 = ",id(std.target_model()))
        print("rho0 parent2 = ",id(std.target_model().preps['rho0'].parent))
        std.target_model()._check_paramvec()
        with open(temp_files + "/results.json",'r') as f:
            x = json.load(f)
        self.assertEqual(list(x.estimates.keys()), list(self.results.estimates.keys()))
        self.assertEqual(list(x.estimates['default'].confidence_region_factories.keys()),
                         list(self.results.estimates['default'].confidence_region_factories.keys()))

        # Workspace
        s = json.dumps(self.ws)
        x = json.loads(s)
         #TODO: comparison (?)

        #Misc other objects
        for obj in self.miscObjects:
            s = json.dumps(obj)
            x = json.loads(s)



    def test_msgpack(self):

        #basic types
        s = msgpack.dumps(range(10))
        x = msgpack.loads(s)
        s = msgpack.dumps(4+3.0j)
        x = msgpack.loads(s)
        s = msgpack.dumps(np.array([1,2,3,4],'d'))
        x = msgpack.loads(s)
        s = msgpack.dumps( testObj )
        x = msgpack.loads(s)

        #string list
        s = msgpack.dumps(self.lsgstStrings)
        x = msgpack.loads(s)
        self.assertEqual(x, self.lsgstStrings)

        # DataSet
        s = msgpack.dumps(self.ds)
        x = msgpack.loads(s)
        self.assertEqual(list(x.keys()), list(self.ds.keys()))
        self.assertEqual(x[('Gx',)].as_dict(), self.ds[('Gx',)].as_dict())

        # Model
        s = msgpack.dumps(self.datagen_gateset)
        with open(temp_files + "/model.mpk",'wb') as f:
            msgpack.dump(self.datagen_gateset, f)
        with open(temp_files + "/model.mpk",'rb') as f:
            x = msgpack.load(f)
        self.assertAlmostEqual(self.datagen_gateset.frobeniusdist(x),0)
        s = msgpack.dumps(self.mdl_withInst)
        x = msgpack.loads(s)
        self.assertAlmostEqual(self.mdl_withInst.frobeniusdist(x),0)

        # Results (containing confidence region)
        with open(temp_files + "/results.mpk",'wb') as f:
            msgpack.dump(self.results, f)
        with open(temp_files + "/results.mpk",'rb') as f:
            x = msgpack.load(f)
        self.assertEqual(list(x.estimates.keys()), list(self.results.estimates.keys()))
        self.assertEqual(list(x.estimates['default'].confidence_region_factories.keys()),
                         list(self.results.estimates['default'].confidence_region_factories.keys()))

        # Workspace
        s = msgpack.dumps(self.ws)
        x = msgpack.loads(s)
         #TODO: comparison (?)

        #Misc other objects
        for obj in self.miscObjects:
            s = msgpack.dumps(obj)
            x = msgpack.loads(s)



    def test_pickle(self):

        #basic types
        s = pickle.dumps(range(10))
        x = pickle.loads(s)
        s = pickle.dumps(4+3.0j)
        x = pickle.loads(s)
        s = pickle.dumps(np.array([1,2,3,4],'d'))
        x = pickle.loads(s)
        s = pickle.dumps( testObj ) #b/c we've messed with its __module__ this won't work...
        x = pickle.loads(s)

        #string list
        s = pickle.dumps(self.lsgstStrings)
        x = pickle.loads(s)
        self.assertEqual(x, self.lsgstStrings)

        # DataSet
        s = pickle.dumps(self.ds)
        x = pickle.loads(s)
        self.assertEqual(list(x.keys()), list(self.ds.keys()))
        self.assertEqual(x[('Gx',)].as_dict(), self.ds[('Gx',)].as_dict())

        # Model
        s = pickle.dumps(self.datagen_gateset)
        with open(temp_files + "/model.pickle",'wb') as f:
            pickle.dump(self.datagen_gateset, f)
        with open(temp_files + "/model.pickle",'rb') as f:
            x = pickle.load(f)
        self.assertAlmostEqual(self.datagen_gateset.frobeniusdist(x),0)
        s = pickle.dumps(self.mdl_withInst)
        x = pickle.loads(s)
        self.assertAlmostEqual(self.mdl_withInst.frobeniusdist(x),0)

        # Results (containing confidence region)
        with open(temp_files + "/results.pickle",'wb') as f:
            pickle.dump(self.results, f)
        with open(temp_files + "/results.pickle",'rb') as f:
            x = pickle.load(f)
        self.assertEqual(list(x.estimates.keys()), list(self.results.estimates.keys()))
        self.assertEqual(list(x.estimates['default'].confidence_region_factories.keys()),
                         list(self.results.estimates['default'].confidence_region_factories.keys()))

        # Workspace
        pygsti.report.workspace.enable_plotly_pickling() # b/c workspace cache may contain plotly figures
        s = pickle.dumps(self.ws)
        x = pickle.loads(s)
        pygsti.report.workspace.disable_plotly_pickling()
         #TODO: comparison (?)

        #Misc other objects
        for obj in self.miscObjects:
            s = pickle.dumps(obj)
            x = pickle.loads(s)

    def test_std_decode(self):
        # test decode_std_base function since it isn't easily reached/covered:
        binary = False

        mock_json_obj = {'__tuple__': True}
        with self.assertRaises(AssertionError):
            pygsti.io.jsoncodec.decode_std_base(mock_json_obj,"",binary)

        mock_json_obj = {'__list__': ['a','b']}
        pygsti.io.jsoncodec.decode_std_base(mock_json_obj,[],binary)

        mock_json_obj = {'__set__': ['a','b']}
        pygsti.io.jsoncodec.decode_std_base(mock_json_obj,set(),binary)

        mock_json_obj = {'__ndict__': [('key1','val1'),('key2','val2')]}
        pygsti.io.jsoncodec.decode_std_base(mock_json_obj,{},binary)

        mock_json_obj = {'__odict__': [('key1','val1'),('key2','val2')]}
        pygsti.io.jsoncodec.decode_std_base(mock_json_obj,collections.OrderedDict(),binary)

        mock_json_obj = {'__uuid__': True}
        with self.assertRaises(AssertionError):
            pygsti.io.jsoncodec.decode_std_base(mock_json_obj,"",binary)

        mock_json_obj = {'__ndarray__': True}
        with self.assertRaises(AssertionError):
            pygsti.io.jsoncodec.decode_std_base(mock_json_obj,"",binary)

        mock_json_obj = {'__npgeneric__': True}
        with self.assertRaises(AssertionError):
            pygsti.io.jsoncodec.decode_std_base(mock_json_obj,"",binary)

        mock_json_obj = {'__complex__': True}
        with self.assertRaises(AssertionError):
            pygsti.io.jsoncodec.decode_std_base(mock_json_obj,"",binary)

        mock_json_obj = {'__counter__': True}
        with self.assertRaises(AssertionError):
            pygsti.io.jsoncodec.decode_std_base(mock_json_obj,"",binary)

        mock_json_obj = {'__slice__': True}
        with self.assertRaises(AssertionError):
            pygsti.io.jsoncodec.decode_std_base(mock_json_obj,"",binary)


    def test_helpers(self):
        pygsti.io.jsoncodec.tostr("Hi")
        pygsti.io.jsoncodec.tostr(b"Hi")
        pygsti.io.jsoncodec.tobin("Hi")
        pygsti.io.jsoncodec.tobin(b"Hi")

    def test_pickle_dataset_with_circuitlabels(self):
        #A later-added test checking whether Circuits containing CiruitLabels
        # are correctly pickled within a DataSet.  In particular correct
        # preservation of the circuit's .str property
        pygsti.obj.Circuit.default_expand_subcircuits = False # so exponentiation => CircuitLabels
        ds = pygsti.obj.DataSet(outcome_labels=('0','1'))
        c0 = pygsti.obj.Circuit(None,stringrep="[Gx:0Gy:1]")
        c = c0**2
        self.assertTrue(isinstance(c.tup[0], CircuitLabel))
        self.assertEqual(c.str, "([Gx:0Gy:1])^2")
        ds.add_count_dict(c, {'0': 50, '1': 50})
        s = pickle.dumps(ds)
        ds2 = pickle.loads(s)
        c2 = list(ds2.keys())[0]
        self.assertEqual(c2.str, "([Gx:0Gy:1])^2")
        pygsti.obj.Circuit.default_expand_subcircuits = True

    #Debugging, because there was some weird python3 vs 2 json incompatibility with string labels
    # - turned out to be that the unit test files needed to import unicode_literals from __future__
    #def test_labels(self):
    #    strLabel = pygsti.obj.Label("Gi")
    #    #strLabel = ("Gi",)
    #    from pygsti.modelpacks.legacy import std1Q_XYI as std
    #
    #    s = json.dumps(strLabel)
    #    print("s = ",str(s))
    #    x = msgpack.loads(s)
    #    print("x = ",x)
    #
    #    print("-----------------------------")
    #
    #    s = json.dumps(std.prepStrs[2])
    #    print("s = ",s)
    #    x = json.loads(s)
    #    print("x = ",x)
    #    assert(False),"STOP"



if __name__ == "__main__":
    unittest.main(verbosity=2)
