import os

from ..util import BaseCase

import pygsti
from pygsti.modelpacks import smq1Q_XYI, smq2Q_XXYYII

class ModelIOTester(BaseCase):
    def setUp(self):
        self.mdl_1q = smq1Q_XYI.target_model()
        self.mdl_2q = smq2Q_XXYYII.target_model()

    def test_1q_model(self):
        pygsti.io.write_model(self.mdl_1q, 'mdl_1q.txt')
        mdl = pygsti.io.load_model('mdl_1q.txt')

        for op_label, op in mdl.operations.items():
            self.assertArraysAlmostEqual(op.to_dense(), self.mdl_1q.operations[op_label].to_dense())
        
        os.remove('mdl_1q.txt')
    
    def test_2q_model(self):
        pygsti.io.write_model(self.mdl_2q, 'mdl_2q.txt')
        mdl = pygsti.io.load_model('mdl_2q.txt')

        for op_label, op in mdl.operations.items():
            self.assertArraysAlmostEqual(op.to_dense(), self.mdl_2q.operations[op_label].to_dense())

        os.remove('mdl_2q.txt')