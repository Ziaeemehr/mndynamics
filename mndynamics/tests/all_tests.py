import os
import glob
import unittest
import mndynamics
from os.path import join
from unittest import TestLoader, TextTestRunner, TestSuite



def get_module_path():
    '''
    Returns the location of the tests folder
    '''
    tests_folder = "tests"
    location = mndynamics.__file__
    location = location.replace('__init__.py', '')
    location = join(location, tests_folder)

    return location


def tests():
    """
    Find all test_*.py files in the tests folder and run them

    """

    path = get_module_path()
    cwd = os.getcwd()
    os.chdir(path)
    test_file_strings = glob.glob('test_*.py')
    module_strings = [test_file[0:len(test_file)-3] for test_file in test_file_strings]
    suites = [unittest.defaultTestLoader.loadTestsFromName(name)
              for name in module_strings]
    test_suite = TestSuite(suites)
    test_runner = TextTestRunner().run(test_suite)
    os.chdir(cwd)


if __name__ == '__main__':
    tests()
