import warnings

warnings.filterwarnings("ignore", category = UserWarning, module = "sklearn")
#-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import joblib
import numpy as np
from tensorflow.keras.models import load_model

#-

class Mass_Loss_Model:
    def __init__(self, mss, d0, euv0, rpl, mpl):
        self.Mss = mss  # Stellar mass [Msun]
        self.d0 = d0  # Orbital separation [AU]
        self.Rpl = rpl  # Planetary radius [Rearth]
        self.Mpl = mpl  # Planetary mass [Mearth]
        self.EUV = euv0  # EUV flux [erg/s/cm^2]
        self.Teq = self.SMAXIS_inv()  # Equilibrium temperature [K]
        self.lambda_K, _ = self.Lambda_k_cal()

    def SMAXIS_inv(self):
        """
        Calculate the corresponding temperature in the grid [K].
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        teq_file_path = os.path.join(base_dir, 'teq-sma.txt')
        mstd = np.loadtxt(teq_file_path)
        tg = np.linspace(3000, 300, 28)
        d1 = np.zeros(len(tg))
        mstd0 = mstd[:, 0]
        mstd2 = mstd[:, 2]

        for i in range(len(tg)):
            n1 = np.where(mstd[:, 1] == tg[i])[0]    
            d1[i] = np.interp(self.Mss, mstd0[n1], mstd2[n1])

        teq = np.interp(self.d0, d1, tg)
        return teq

    @staticmethod
    def Lambda(rpl, mpl, t0):
        """
        Calculate the parameter Lambda.
        """
        Re = 6.378e8   # Earth radius [cm]
        Me = 5.9722e27 # Earth mass [g]
        kb = 1.3807e-16 # Boltzmann constant [erg/K]
        mh = 1.6726e-24 # Hydrogen atom mass [g]
        G0 = 6.6726e-8  # Gravitational constant [cm^3/g/s^2]
        lamb = G0 * mpl * Me / (rpl * Re * (kb * t0 / mh))
        return lamb

    def cap_k(self):
        """
        Calculate the parameter k.
        """
        AU = 1.4960e13  # Astronomical unit [cm]
        Re = 6.378e8    # Earth radius [cm]
        Me = 5.9722e27  # Earth mass [g]
        Msun = 1.98855e33 # Stellar mass [g]
        d_sp = (AU/Re) * (self.d0/self.Rpl)
        alpha = (Me/Msun) * (self.Mpl/self.Mss)
        R_0 = d_sp * (alpha/3)**(1/3)
        num = ((R_0 - 1)**2) * (2 * R_0 + 1)
        denom = 2 * R_0**3
        return num / denom

    def Lambda_k_cal(self):
        """
        Calculate the parameter Lambda_k.
        """
        k = self.cap_k()
        lambda_k = self.Lambda(self.Rpl, self.Mpl, self.Teq) * k
        return lambda_k, k

    # Persistent Class-Level Object Storage Caches
    _cached_model = None
    _cached_scaler = None

    def DNN_pred(self):
        """
        Use a deep neural network model to predict the mass loss rate.
        """
        input_data = np.zeros((1, 6))
        input_data[:, 0] = self.Mss
        input_data[:, 1] = np.log10(self.Teq)
        input_data[:, 2] = self.d0
        input_data[:, 3] = np.log10(self.EUV)
        input_data[:, 4] = self.Rpl
        input_data[:, 5] = self.lambda_K
        
        # Build filepaths dynamically relative to script location
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, '3l_model_lambda_k.h5')
        scaler_path = os.path.join(base_dir, 'scaler_y_3l_model_lambda_k.save')
        
        if Mass_Loss_Model._cached_model is None:
           Mass_Loss_Model._cached_model = load_model(model_path, compile = False)
        if Mass_Loss_Model._cached_scaler is None:
            Mass_Loss_Model._cached_scaler = joblib.load(scaler_path)
        
        output_pred = Mass_Loss_Model._cached_model.predict(input_data, verbose = 0)
        output_trans1 = Mass_Loss_Model._cached_scaler.inverse_transform(output_pred.reshape(-1, 1))
        output_trans2 = 10**output_trans1

        return output_trans2.flatten()

    def calculate_mass_loss_rate(self):
        """
        Calculate the mass-loss rate and print it up to three decimal places.
        """
        mass_loss_rate = self.DNN_pred()
        
        return mass_loss_rate
        
#-

'''
planet_name = 'HD209458b'
Mss = 1.148 #stellar mass [Msun]
d0 = 0.047 #orbital separation [AU]
T_i = 1448. #Teq [K]
EUV0 = 1086. #Feuv erg/s/cm^2
Rpl0 = 15.6 #planetary radius [Rearth]
Mpl = 232. #planetary mass [Mearth]

model =  Mass_Loss_Model(mss = Mss, d0 = d0, euv0 = EUV0, rpl = Rpl0, mpl = Mpl)
mass_loss_rate = model.calculate_mass_loss_rate()

print('Planet Name:HD209458b')
print(mass_loss_rate) 
'''
#-
'''
planet_name = 'GJ436b'
Mss = 0.452 #stellar mass [Msun]
d0 = 0.02887 #orbital separation [AU]
T_i = 686. #Teq [K]
EUV0 = 1760. #Feuv erg/s/cm^2
Rpl0 = 4.17 #planetary radius [Rearth]
Mpl = 22.1 #planetary mass [Mearth]

model =  Mass_Loss_Model(mss = Mss, d0 = d0, euv0 = EUV0, rpl = Rpl0, mpl = Mpl)
mass_loss_rate = model.calculate_mass_loss_rate()
print('Planet Name:GJ436b')
print(mass_loss_rate) 
'''
#-
'''
planet_name = 'HD189733b'
Mss = 0.8 #stellar mass [Msun]
d0 = 0.03 #orbital separation [AU]
T_i = 1209. #Teq [K]
EUV0 = 24778. #Feuv erg/s/cm^2
Rpl0 = 12.7 #planetary radius [Rearth]
Mpl = 359. #planetary mass [Mearth]

model =  Mass_Loss_Model(mss = Mss, d0 = d0, euv0 = EUV0, rpl = Rpl0, mpl = Mpl)
mass_loss_rate = model.calculate_mass_loss_rate()
print('Planet Name:HD189733b')
print(mass_loss_rate) 
'''
#-
'''
planet_name = 'GJ3470b'
Mss = 0.539 #stellar mass [Msun]
d0 = 0.03557 #orbital separation [AU]
T_i = 593.5 #Teq [K]
EUV0 = 1868. #Feuv erg/s/cm^2
Rpl0 = 4.57 #planetary radius [Rearth]
Mpl = 13.9 #planetary mass [Mearth]

model = Mass_Loss_Model(mss = Mss, d0 = d0, euv0 = EUV0, rpl = Rpl0, mpl = Mpl)
mass_loss_rate = model.calculate_mass_loss_rate()
print('Planet Name:GJ3470b')
print(mass_loss_rate) 
'''
#-
'''
planet_name = 'HD149026b'
Mss = 1.3 #stellar mass [Msun]
d0 = 0.04288 #orbital separation [AU]
T_i = 1626. #Teq [K]
EUV0 = 6886. #Feuv erg/s/cm^2
Rpl0 = 8.04 #planetary radius [Rearth]
Mpl = 121. #planetary mass [Mearth]

model = Mass_Loss_Model(mss = Mss, d0 = d0, euv0 = EUV0, rpl = Rpl0, mpl = Mpl)
mass_loss_rate = model.calculate_mass_loss_rate()
print('Planet Name:HD149026b')
print(mass_loss_rate)
'''
#-
'''
planet_name = 'HAT-P-11b'
Mss = 0.81 #stellar mass [Msun]
d0 = 0.053 #orbital separation [AU]
T_i = 930. #Teq [K]
EUV0 = 3236. #Feuv erg/s/cm^2
Rpl0 = 4.8 #planetary radius [Rearth]
Mpl = 29. #planetary mass [Mearth]

model = Mass_Loss_Model(mss = Mss, d0 = d0, euv0 = EUV0, rpl = Rpl0, mpl = Mpl)
mass_loss_rate = model.calculate_mass_loss_rate()
print('Planet Name:HAT-P-11b')
print(mass_loss_rate)
'''
#-
'''
planet_name = '55-Cnc-e'
Mss = 0.905 #stellar mass [Msun]
d0 = 0.01544 #orbital separation [AU]
T_i = 1958. #Teq [K]
EUV0 = 570. #Feuv erg/s/cm^2
Rpl0 = 1.875 #planetary radius [Rearth]
Mpl = 7.99 #planetary mass [Mearth]

model = Mass_Loss_Model(mss = Mss, d0 = d0, euv0 = EUV0, rpl = Rpl0, mpl = Mpl)
mass_loss_rate = model.calculate_mass_loss_rate()
print('Planet Name:55-Cnc-e')
print(mass_loss_rate)
'''
#-
'''
planet_name = 'HD97658b'
Mss = 0.85 #stellar mass [Msun]
d0 = 0.08 #orbital separation [AU]
T_i = 757. #Teq [K]
EUV0 = 955. #Feuv erg/s/cm^2
Rpl0 = 2.4 #planetary radius [Rearth]
Mpl = 8.88 #planetary mass [Mearth]

model = Mass_Loss_Model(mss = Mss, d0 = d0, euv0 = EUV0, rpl = Rpl0, mpl = Mpl)
mass_loss_rate = model.calculate_mass_loss_rate()

formatted_mass_loss_rate = [f"{rate:.3f}" for rate in mass_loss_rate]

print('Planet Name:HD97658b')
print(mass_loss_rate) 
print('The mass-loss rate is:', formatted_mass_loss_rate)
'''

#expected outputs

#The escape rate of HD209458b is 2.43e+10 g/s
#The escape rate of GJ436b is 4.52e+09 g/s
#The escape rate of HD189733b is 7.66e+10 g/s
#The escape rate of GJ3470b is 1.72e+10 g/s
#The escape rate of HD149026b is 2.55e+10 g/s
#The escape rate of HAT-P-11b is 1.01e+10 g/s
#The escape rate of 55-Cnc-e is 1.45e+10 g/s
#The escape rate of HD97658b is 2.21e+09 g/s
#-

if __name__ == '__main__':
    print("Executing standalone pipeline diagnostics...\n")

    # Baseline planet for calculation verification
    planet_name = 'HD97658b'
    Mss = 0.85
    d0 = 0.08
    T_i = 757.
    EUV0 = 955.
    Rpl0 = 2.4
    Mpl = 8.88

    model = Mass_Loss_Model(mss = Mss, d0 = d0, euv0 = EUV0, rpl = Rpl0, mpl = Mpl)
    mass_loss_rate = model.calculate_mass_loss_rate()
    formatted_mass_loss_rate = [f"{rate:.3e}" for rate in mass_loss_rate]

    print(f'Planet Name: {planet_name}')
    print(f'The predicted mass-loss rate is: {formatted_mass_loss_rate} g/s')
    print('Expected mass-loss rate is: ~ 2.21e+09 g/s')

