import numpy as np
import scipy.stats as statistics
from os import path


class Stationary:
    # Modeling hazards as stochastic processes: winds are modeled as broadband stationary stochastic processes.

    def __init__(self, power_spectrum_object=None, power_spectrum_script=None, pdf_object=None, pdf_script=None,
                 ndof=None):

        self.power_spectrum_object = power_spectrum_object
        self.power_spectrum_script = power_spectrum_script
        self.pdf_object = pdf_object
        self.pdf_script = pdf_script
        self.ndof = ndof

        # Initial checks.
        
        # Number of degrees-of-freedom: ndof.
        if ndof is None:
            raise ValueError('NDOF cannot be None.')

        # Cehck if the power spectrum (PS) is provided as a script.
        if power_spectrum_script is not None:
            self.user_ps_check = path.exists(power_spectrum_script)
        else:
            self.user_ps_check = False

        if self.user_ps_check:
            try:
                self.module_ps = __import__(self.power_spectrum_script[:-3])
            except ImportError:
                raise ImportError('There is no module implementing a power spectrum.')

        # Check if the probability density function of the hazard is provided as a script.
        if pdf_script is not None:
            self.user_pdf_check = path.exists(pdf_script)
        else:
            self.user_pdf_check = False

        if self.user_pdf_check:
            try:
                self.module_pdf = __import__(self.pdf_script[:-3])
            except ImportError:
                raise ImportError('There is no module implementing the PDF.')

        if self.power_spectrum_object is 'white_noise':
            self.pdf_object = 'gaussian'

            
    def power_spectrum_excitation(self, freq=None, **kwargs):
        # implementing the power spectrum of the excitation (Hazard).

        if self.user_ps_check:
            if self.power_spectrum_script is None:
                raise TypeError('power_spectrum_script cannot be None')

            exec('from ' + self.power_spectrum_script[:-3] + ' import ' + self.power_spectrum_object)
            power_spectrum_fun = eval(self.power_spectrum_object)
        else:
            if self.power_spectrum_object is None:
                raise TypeError('power_spectrum_object cannot be None')

            power_spectrum_fun = eval("Stationary." + self.power_spectrum_object)

        power_spectrum, U = power_spectrum_fun(self, freq=freq, **kwargs)

        return power_spectrum, U

    def white_noise(self, freq=None, **kwargs):
        # If the PS of the excitation is modeled as a white noise.

        if 'S0' in kwargs.keys():
            S0 = kwargs['S0']
        else:
            raise ValueError('S0 cannot be None.')

        ndof = self.ndof
        power_spectrum = np.zeros((len(freq),ndof))

        for i in range(len(freq)):
            for j in range(ndof):
                power_spectrum[i,j] = S0[j]

        U = []
        return power_spectrum, U

    def windpsd(self, u10=None, freq=None, **kwargs):
        # For Wind PSD.
        
        if 'z' in kwargs.keys():
            z = kwargs['z']
        else:
            raise ValueError('z cannot be None.')

        # Parameters of the wind PSD.
        ak = 6.868
        bk = 1
        ck = 10.302
        dk = 5 / 3
        z0 = 0.025
        Cd = 1.45
        ro = 1.225
        kk = 0.4
        #u10 = 6.2371
        sigV = 7.04
        kV = 2.04
        L0 = 0.2

        uast = np.sqrt(0.006)*u10
        sd = (6 - 1.1 * np.arctan(np.log(z0) + 1.75)) * (uast ** 2)
  
        ndof = self.ndof
        power_spectrum = np.zeros((len(freq),ndof))
        freq = freq/(2*np.pi) # Frequency in Hz.

        U = []
        
        # Loop over each floor to define a height-dependent hazard (Wind).
        for i in range(ndof):
            ub = (uast / kk) * np.log(z[i] / z0)
            U.append(ub)
            # Lu = 300 * (z[i] / 300) ** (0.46 + 0.074 * np.log(z0))
            #Lu = L0 * (kk * z[i]) / (kk * z[i] + L0) # Ref: Wind_0
            Lu = 300*(z[i]/200)**(0.67+0.05*np.log(z0)) # Code petrini Thesis
            for j in range(len(freq)):
                f = (Lu / ub) * freq[j]
                power_spectrum[j,i] = (sd / freq[j]) * (ak * f / ((bk + ck * f ** 2) ** dk))

        return power_spectrum, np.array(U)

    def pdf_intensity_measure(self, im=None, **kwargs):
        if self.user_pdf_check:
            if self.pdf_script is None:
                raise TypeError('pdf_script cannot be None')

            exec('from ' + self.pdf_script[:-3] + ' import ' + self.pdf_object)
            pdf_fun = eval(self.pdf_object)
        else:
            if self.pdf_object is None:
                raise TypeError('pdf_object cannot be None')

            pdf_fun = eval("Stationary." + self.pdf_object)

        probability_density = pdf_fun(self, im=im, **kwargs)

        return probability_density

    @staticmethod
    def gaussian(im=None, **kwargs):
        # Static method for the Normal distribution modeling the probabilistic characteristics of the hazard.
        
        if 'mean' in kwargs.keys():
            mean = kwargs['mean']
        else:
            raise ValueError('mean cannot be None.')

        if 'std' in kwargs.keys():
            std = kwargs['std']
        else:
            raise ValueError('std cannot be None.')

        probability_density = statistics.norm.pdf(im, mean, std)

        return probability_density

    @staticmethod
    def weibull(im=None, **kwargs):
        # static method for the Weibull distribution used to model the probabilist characteristics of the hazard.

        if 'kv' in kwargs.keys():
            kv = kwargs['kv']
        else:
            raise ValueError('kv cannot be None.')

        if 'Av' in kwargs.keys():
            Av = kwargs['Av']
        else:
            raise ValueError('Av cannot be None.')

        probability_density = (kv/Av)*((im/Av)**(kv-1))*np.exp(-(im/Av)**kv)

        return probability_density

    @staticmethod
    def gumbel(im=None, **kwargs):
        # static method for the Gumbel distribution used to model the probabilist characteristics of the hazard.

        if 'kv' in kwargs.keys():
            kv = kwargs['kv']
        else:
            raise ValueError('kv cannot be None.')

        if 'Av' in kwargs.keys():
            Av = kwargs['Av']
        else:
            raise ValueError('Av cannot be None.')

        probability_density = (1/Av)*np.exp(-((im-kv)/Av)-np.exp(-((im-kv)/Av)))

        return probability_density

