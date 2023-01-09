import unittest
import numpy as np
from mndynamics.models.py.WB_Base import WB


class TestModules(unittest.TestCase):

    def test_WB(self):

        obj = WB({"i_ext": 0.75, "v0":-63.0})
        data = obj.simulate(tspan=np.arange(0, 50, 0.01))
        t = data["t"]
        v = data["v"]
        h = data["h"]
        n = data["n"]
        mean = np.mean(v)
        std = np.std(v)

        self.assertEqual(np.abs(mean+58.02)<0.1, True)
        self.assertEqual(np.abs(std-11.90)<0.1, True)


if __name__ == '__main__':
    unittest.main()
    # obj = TestModules()
    # obj.test_WB()

