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
	"""docstring for Task1"""
	def __init__(self):
		super(Activation, self).__init__()

	def _calculate(self, x):
		return 1/(1+np.exp(-x)) 

	def _derivative(self, x):
		x = self._calculate(x)
		return x*(1-x)

	def _derivative_with_calculate(self, x):
		return x*(1-x)



a = Task1(1)
a.run()

