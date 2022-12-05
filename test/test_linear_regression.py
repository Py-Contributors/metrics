import unittest
import numpy as np
import sys
sys.path.append('.')

from src.main import r2_score


class LinearRegressionTest(unittest.TestCase):
    def test_r2_score_negative(self):
        a = np.array([10, 20, 30])
        b = np.array([30, 10, 20])
        self.assertEqual(r2_score(a, b), -2.0)
    
    def test_r2_score_with_one(self):
        a = np.array([10, 20, 30])
        b = np.array([10, 20, 30])
        self.assertEqual(r2_score(a, b), 1.0)
    
    def test_r2_score_with_list(self):
        a = [10, 20, 30]
        b = [30, 10, 20]
        self.assertEqual(r2_score(a, b), -2.0)
        
    def test_r2_score_with_different_shape(self):
        a = np.array([10, 20, 30])
        b = np.array([30, 10])
        self.assertRaises(ValueError, r2_score, a, b)
        
    
        
        
if __name__ == '__main__':
    unittest.main()