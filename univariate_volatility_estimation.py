import numpy as np
from decimal import Decimal, getcontext
import math 
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from scipy.stats import truncnorm, invgamma

class Gibbs:
	def __init__(self, mu, phi, sigma_square, alpha_sigma = 2.5, beta_sigma = 0.025, alpha_phi = 0, beta_phi_square = 1, alpha_mu = 0, beta_mu_square = 100):
		self.mu = Decimal(str(mu))
		self.phi = Decimal(str(phi))
		self.sigma_square = Decimal(str(sigma_square))
		self.alpha_sigma = Decimal(str(alpha_sigma))
		self.beta_sigma = Decimal(str(beta_sigma))
		self.alpha_phi = Decimal(str(alpha_phi))
		self.beta_phi_square = Decimal(str(beta_phi_square))
		self.alpha_mu = Decimal(str(alpha_mu))
		self.beta_mu_square = Decimal(str(beta_mu_square))

	def get_data_from_csv(self, file_name):
		df = pd.read_csv(file_name, header=0, names=['Date', 'Close'])
		arr = df['Close'].values[::-1]
		arr = arr[-(len(arr)-1):]
		arr = [Decimal(str(np.log(float(i.replace(',','.'))))) for i in arr]
		self.data = np.diff(arr)
		self.num_of_data = Decimal(str(len(self.data)))
		self.data_date = pd.to_datetime(df['Date'].values[::-1][-int(self.num_of_data):], format='%Y-%m-%d')
		self.h = [Decimal('-1') for i in range(self.num_of_data)]

	def calc_absolute_volatility(self):
		self.absolute_volatility2 = [Decimal(str(np.log(float(np.var(self.data[:i]))))) for i in range(1,self.num_of_data+1)]
		self.absolute_volatility = [Decimal(str(np.log(np.power(float(self.data[i]), 2)))) for i in range(0,self.num_of_data)]

	def sample_from_normal(self, mean, variance, sample_size=None):
		return Decimal(str(np.random.normal(loc = mean, scale = np.sqrt(variance), size = sample_size)))

	def sample_from_truncated_normal(self, mean, variance, lower_bound, upper_bound, sample_size=None):
		X = truncnorm((lower_bound - mean) / np.sqrt(variance), (upper_bound - mean) / np.sqrt(variance), loc=mean, scale=np.sqrt(variance))
		return Decimal(str(X.rvs()))

	def sample_from_inverse_gamma(self, alpha, beta, sample_size=None):
		return Decimal(str(invgamma(a = float(alpha), scale = float(beta)).rvs()))

	def calc_h_hyperparameters(self, t):
		beta_h_square = self.sigma_square / (1 + Decimal(str(np.power(self.phi, 2))))
		h_t_minus1 = h_t_plus1 = Decimal('0')
		if (t - 1) < 0:
			h_t_minus1 = (self.h[t] - self.mu + (self.phi * self.mu) - (np.sqrt(self.sigma_square) * self.sample_from_normal(0,1))) / self.phi
		else:
			h_t_minus1 = self.h[t - 1]
		if (t+1) >= self.num_of_data:
			h_t_plus1 = self.mu + (self.phi * self.h[t]) - (self.phi * self.mu) + (np.sqrt(self.sigma_square) * self.sample_from_normal(0,1))
		else:
			h_t_plus1 = self.h[t + 1]
		temp = self.mu + ((self.phi * (h_t_minus1 + h_t_plus1 - (2 * self.mu))) / (1 + Decimal(str(np.power(self.phi, 2)))))
		alpha_h_hat = temp + ((beta_h_square / 2) * ((Decimal(str(np.power(self.data[t], 2))) * Decimal(str(np.exp(-temp)))) - 1))

		self.alpha_h = temp
		self.alpha_h_hat = alpha_h_hat
		self.beta_h_square = beta_h_square

	def calc_sigma_hyperparameters(self):
		alpha_sigma_hat = self.alpha_sigma + (self.num_of_data / 2)

		temp = sum([Decimal(str(np.power(self.h[i + 1] - self.mu - (self.phi * (self.h[i] - self.mu)), 2))) for i in range(self.num_of_data-1)])
		beta_sigma_hat = self.beta_sigma + (Decimal('0.5') * ((Decimal(str(np.power(self.h[0] - self.mu, 2))) * (1 - Decimal(str(np.power(self.phi, 2))))) + temp))

		self.alpha_sigma_hat = alpha_sigma_hat		
		self.beta_sigma_hat = beta_sigma_hat

	def calc_phi_hyperparameters(self):
		temp = sum([Decimal(str(np.power(self.h[i] - self.mu, 2))) for i in range(1, self.num_of_data-1)])
		beta_phi_square = Decimal(str(np.power((temp / self.sigma_square) + (1 / self.beta_phi_square), -1)))

		temp = sum([(self.h[i + 1] - self.mu) * (self.h[i] - self.mu) for i in range(self.num_of_data-1)])
		alpha_phi = beta_phi_square * ((temp / self.sigma_square) + (self.alpha_phi / self.beta_phi_square))

		self.alpha_phi_hat = alpha_phi
		self.beta_phi_square_hat = beta_phi_square

	def calc_mu_hyperparameters(self):
		temp = 1 - Decimal(str(np.power(self.phi, 2))) + ((self.num_of_data - 1) * Decimal(str(np.power(1 - self.phi, 2))))
		beta_mu_square = Decimal(str(np.power((temp / self.sigma_square) + (1 / self.beta_mu_square), -1)))

		temp = self.h[0] * (1 - Decimal(str(np.power(self.phi, 2)))) + ((1 - self.phi) * sum([self.h[i + 1] - (self.phi * self.h[i]) for i in range(self.num_of_data-1)]))
		alpha_mu = beta_mu_square * ((temp / self.sigma_square) + (self.alpha_mu / self.beta_mu_square))

		self.alpha_mu_hat = alpha_mu
		self.beta_mu_square_hat = beta_mu_square

	def calc_acceptance_ratio(self, t, h_temp):
		log_f_calculation = Decimal(str(np.exp(Decimal(str(-0.5 * np.log(2 * np.pi))) - (Decimal('0.5')* h_temp) - ((np.power(self.data[t], 2) / 2 ) * np.exp(-h_temp)))))
		log_g_calculation = Decimal(str(np.exp(Decimal(str(-0.5 * np.log(2 * np.pi))) - (Decimal('0.5') * h_temp) - ((np.power(self.data[t], 2) / 2 ) * np.exp(-self.alpha_h) * (1 + self.alpha_h - (h_temp * np.exp(-self.alpha_h)))))))
		return log_f_calculation / log_g_calculation

	def run_gibbs_sampler(self, num_of_sample):
		for i in range(num_of_sample):
			for t in range(self.num_of_data):
				count = 0
				self.calc_h_hyperparameters(t)
				u = np.random.random()
				h_temp = self.sample_from_normal(self.alpha_h_hat, self.beta_h_square)
				acceptance_ratio = self.calc_acceptance_ratio(t, h_temp)

				if u <= acceptance_ratio:
					count += 1
					self.h[t] = h_temp
			
			self.calc_sigma_hyperparameters()
			self.sigma_square = self.sample_from_inverse_gamma(self.alpha_sigma_hat, self.beta_sigma_hat)
			self.calc_phi_hyperparameters()
			phi_sample = self.sample_from_truncated_normal(mean = float(self.alpha_phi_hat), variance = float(self.beta_phi_square_hat), lower_bound = -1, upper_bound = 1)
			if not np.isinf(float(phi_sample)):
				self.phi = phi_sample
			else:
				print 'inf'

			self.calc_mu_hyperparameters()
			self.mu = self.sample_from_normal(self.alpha_mu_hat, self.beta_mu_square_hat)

	def plot_log_returns(self):
		plt.plot(self.data_date, self.data, 'r')
		plt.ylabel(r'$log(\frac{S_t}{S_{t-1}})$', fontsize = 18)
		plt.title('S&P 500 Index Log Returns')
		plt.grid(True)
		plt.show()

	def plot_h_estimate_with_absolute_values(self):
		self.calc_absolute_volatility()
		
		plt.subplot(211)
		plt.plot(self.data_date, np.exp([i / 2 for i in self.h]), 'b')
		plt.ylabel(r'$exp(h_t / 2)$')
		plt.title('Volatility Estimate')
		plt.grid(True)

		plt.subplot(212)
		plt.plot(self.data_date, [abs(i) for i in self.data], 'r')
		plt.ylabel('Return')
		plt.title('Absolute Log Return of S&P 500 Index')
		plt.grid(True)
		plt.show()

start_time = datetime.utcnow()
t = Gibbs(mu = -8, phi = 1, sigma_square = 0.01)
t.get_data_from_csv(file_name = 'SP500.csv')
t.plot_log_returns()
t.run_gibbs_sampler(num_of_sample = 4000)
end_time = datetime.utcnow()
print 'Duration: {}'.format(end_time - start_time)
t.plot_h_estimate_with_absolute_values()




