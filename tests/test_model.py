import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "sklearn")

#-
import unittest
import os
import sys
import numpy as np

# Integrate system paths to ensure import clarity across any workspace
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from plug_in_ver.DNN_main import Mass_Loss_Model

#-
class TestMassLossInference(unittest.TestCase):
    
    def setUp(self):
        # Default baseline configuration for generic structural component verification
        self.mss = 0.85
        self.d0 = 0.08
        self.euv = 955.0
        self.rpl = 2.4
        self.mpl = 8.88
        
        self.model = Mass_Loss_Model(mss = self.mss, d0 = self.d0, euv0 = self.euv, rpl = self.rpl, mpl = self.mpl)

    def test_temperature_interpolation(self):
        """Validates that SMAXIS_inv resolves expected equilibrium bounds."""
        teq = self.model.SMAXIS_inv()
        self.assertIsInstance(teq, (int, float, np.float64))
        self.assertTrue(300 <= teq <= 3000, f"Unexpected Teq found: {teq} K")

    def test_dimensionless_parameters(self):
        """Verifies geometric scaling logic remains non-negative."""
        k = self.model.cap_k()
        lambda_k, calculated_k = self.model.Lambda_k_cal()
        
        self.assertGreater(k, 0)
        self.assertGreater(lambda_k, 0)
        self.assertEqual(k, calculated_k)

    def test_pipeline_inference(self):
        """Runs forward passes across all 8 validation planets and prints structural comparisons."""
        # Verification registry matching your paper benchmarks
        validation_planets = {
            'HD209458b':  {'mss': 1.148, 'd0': 0.047,   'euv': 1086.,  'rpl': 15.6,  'mpl': 232.,  'expected': '2.43e+10'},
            'GJ436b':     {'mss': 0.452, 'd0': 0.02887, 'euv': 1760.,  'rpl': 4.17,  'mpl': 22.1,  'expected': '4.52e+09'},
            'HD189733b':  {'mss': 0.8,   'd0': 0.03,    'euv': 24778., 'rpl': 12.7,  'mpl': 359.,  'expected': '7.66e+10'},
            'GJ3470b':    {'mss': 0.539, 'd0': 0.03557, 'euv': 1868.,  'rpl': 4.57,  'mpl': 13.9,  'expected': '1.72e+10'},
            'HD149026b':  {'mss': 1.3,   'd0': 0.04288, 'euv': 6886.,  'rpl': 8.04,  'mpl': 121.,  'expected': '2.55e+10'},
            'HAT-P-11b':  {'mss': 0.81,  'd0': 0.053,   'euv': 3236.,  'rpl': 4.8,   'mpl': 29.,   'expected': '1.01e+10'},
            '55-Cnc-e':   {'mss': 0.905, 'd0': 0.01544, 'euv': 570.,   'rpl': 1.875, 'mpl': 7.99,  'expected': '1.45e+10'},
            'HD97658b':   {'mss': 0.85,  'd0': 0.08,    'euv': 955.,   'rpl': 2.4,   'mpl': 8.88,  'expected': '2.21e+09'}
        }

        print("\n" + "="*68)
        print("NEURAL NETWORK  MASS-LOSS INFERENCE LOGS        ")
        print("="*68 + "\n")

        for name, params in validation_planets.items():
            # Setup dynamic verification instance
            eval_model = Mass_Loss_Model(mss = params['mss'], d0 = params['d0'], euv0 = params['euv'], rpl = params['rpl'], mpl = params['mpl'])
            
            # Compute evaluation parameters
            prediction = eval_model.calculate_mass_loss_rate()
            
            # Strip out scalar values cleanly regardless of matrix dimensionality
            pred_scalar = prediction[0] if hasattr(prediction, '__len__') else prediction

            print(f"Target System: {name}")
            print(f"  -> NN Predicted Mass-loss rate : {pred_scalar:.4e} g/s")
            print(f"  -> Expected Mass-loss rate : ~ {params['expected']} g/s")
            print("-" * 68)

        print("\n" + "="*68)

if __name__ == '__main__':
    unittest.main()

