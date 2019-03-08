import abc
import numpy as np

class Activation(metaclass = abc.ABCMeta):
	def __init__(self):
		pass

	@abc.abstractmethod
	def _calculate(self, x):
		pass

	@abc.abstractmethod
	def _derivative(self, x):
		pass


class Sigmoid(Activation):
	def __init__(self):
		super(Activation, self).__init__()

	def _calculate(self, x):
		return 1/(1+np.exp(-x)) 

	def _derivative(self, x):
		x = self._calculate(x)
		return x*(1-x)

	def _derivative_with_calculate(self, x):
		return x*(1-x)


class Tanh(Activation):
	def __init__(self):
		super(Activation, self).__init__()

	def _calculate(self, x):
		return np.tanh(x) 

	def _derivative(self, x):
		x = self._calculate(x)
		return (1-x*x)

	def _derivative_with_calculate(self, x):
		return (1-x*x)
