# Grid-Based Exoplanet Atmospheric Mass-Loss Predictions via Neural Networks

Official repository for the *Astronomy & Astrophysics* (2024) paper, featuring a neural network implementation for predicting exoplanet atmospheric escape rates.

## Overview & Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com
cd Mass-loss-rate
pip install -r requirements.txt
```

## Usage Styles
* **Plug-in Version (`plug_in_ver/`)**: Import `Mass_Loss_Model` directly into your own modeling scripts to calculate escape rates based on planetary parameters.
* **Command Line Version (`command-line-ver/`)**: Run batch predictions from terminal using `test.py` with custom data and model weight files.

## Testing
* **Unit Testing**: Run `python tests/test_model.py` from the root directory to verify your local setup.

---

## Citation

```bibtex
@article{reza2025grid,
  title={Grid-based exoplanet atmospheric mass-loss predictions via neural networks},
  author={Reza, Amit and Kubyshkina, Daria and Fossati, Luca and Helling, Christiane},
  journal={Astronomy \& Astrophysics},
  volume={694},
  pages={A88},
  year={2025},
  publisher={EDP Sciences}
}



