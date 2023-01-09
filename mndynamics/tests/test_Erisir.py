import unittest
import numpy as np
from mndynamics.models.py.Erisir_Base import Erisir


class TestModules(unittest.TestCase):

    def test_Erisir(self):

        obj = Erisir({"i_ext": 7.0, "v0":-70.0})
        data = obj.simulate(tspan=np.arange(0, 50, 0.01))
        t = data["t"]
        v = data["v"]
        h = data["h"]
        n = data["n"]
        mean = np.mean(v)
        std = np.std(v)

        self.assertEqual(np.abs(mean+56.60)<0.1, True)
        self.assertEqual(np.abs(std-20.47)<0.1, True)


if __name__ == '__main__':
    unittest.main()
    # obj = TestModules()
    # obj.test_Erisir()

