import unittest
import numpy as np
from mndynamics.models.py.ch04 import HH_SOLUTION
from mndynamics.models.py.HH_Base import HH, HH_Reduced
from mndynamics.models.py.ch03 import HH_GATING_VARIABLES

class TestModules(unittest.TestCase):

    def test_gating_variables(self):

        obj = HH_GATING_VARIABLES({"i_ext": 10.0, "v0":-70.0})
        data = obj.simulate(tspan=np.arange(0, 50, 0.01))
        t = data["t"]
        v = data["v"]
        m = data["m"]
        h = data["h"]
        n = data["n"]
        mean = np.mean(v)
        std = np.std(v)
        
        self.assertEqual(np.abs(mean+58.86)<0.1, True)
        self.assertEqual(np.abs(std-26.37)<0.1, True)
    
    def test_HH(self):

        obj = HH({"i_ext": 10.0, "v0":-70.0})
        data = obj.simulate(tspan=np.arange(0, 50, 0.01))
        t = data["t"]
        v = data["v"]
        m = data["m"]
        h = data["h"]
        n = data["n"]
        mean = np.mean(v)
        std = np.std(v)
        
        self.assertEqual(np.abs(mean+58.86)<0.1, True)
        self.assertEqual(np.abs(std-26.37)<0.1, True)
    
    def test_HH_Reduced(self):

        obj = HH_Reduced({"i_ext": 10.0, "v0":-20.0})
        data = obj.simulate(tspan=np.arange(0, 50, 0.01))
        t = data["t"]
        v = data["v"]
        m = data["m"]
        h = data["h"]
        n = data["n"]
        mean = np.mean(v)
        std = np.std(v)

        self.assertEqual(np.abs(mean+63.107)<0.1, True)
        self.assertEqual(np.abs(std-27.35)<0.1, True)
    
    def test_HH_Solution(self):

        obj = HH_SOLUTION()

        obj = HH_SOLUTION({"i_ext": 10.0, "v0":-50.0})
        data = obj.simulate(tspan=np.arange(0, 50, 0.01))
        t = data["t"]
        v = data["v"]

        mean = np.mean(v)
        std = np.std(v)
        # print(mean, std)
        self.assertEqual(np.abs(mean+62.21)<0.1, True)
        self.assertEqual(np.abs(std-22.47)<0.1, True)



if __name__ == '__main__':
    unittest.main()
    # obj = TestModules()
    # obj.test_HH_Solution()