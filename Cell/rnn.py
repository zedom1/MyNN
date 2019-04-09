import copy, numpy as np

class RNN(object):

	def __init__(self, arg):
		super(RNN, self).__init__()

		self.input_dim = arg["input_dim"]
		self.hidden_dim = arg["hidden_dim"]
		self.output_dim = arg["output_dim"]

		self.w_input_hidden = 2*np.random.random((self.input_dim, self.hidden_dim)) - 1
		self.w_hidden_output = 2*np.random.random((self.hidden_dim, self.output_dim)) - 1
		self.w_hidden_hidden = 2*np.random.random((self.hidden_dim, self.hidden_dim)) - 1

		self.activation = arg["activation"]()

	def train(self, X, y, learning_rate = 0.1):

		assert len(np.shape(X)) == 3, "Invalid input shape. The input must be 3d (batch, timestep, input_dim)"
		batch_size, steps, input_dim = np.shape(X)
		assert self.input_dim == input_dim, "Unmatch input_dim."

		deltas = list()
		overallError = 0
		predicts = list()
		hidden_layer = list()
		
		hidden_layer.append(np.zeros([batch_size, self.hidden_dim]))
		# Forwarding
		# moving along the positions in the binary encoding

		for position in range(steps):

			# generate input and output
			current_X = np.atleast_2d(X[:batch_size, position])
			current_y = np.atleast_2d(y[:batch_size, position])
			# hidden layer (input ~+ pre_hidden)
			new_hidden = self.activation._calculate((np.dot(current_X, self.w_input_hidden) + np.dot(hidden_layer[-1], self.w_hidden_hidden)))
			# output layer (new binary representation)
			predict = self.activation._calculate(np.dot(new_hidden, self.w_hidden_output))
			predicts.append(predict)
			# compute error
			error = current_y - predict
			delta = (error*self.activation._derivative_with_calculate(predict))
			deltas.append(delta)
			
			overallError += np.abs(np.mean(error))
			
			# store hidden layer so we can use it in the next timestep
			hidden_layer.append(new_hidden)

		# Backward update initialize
		accumulate_hidden_delta = np.zeros([batch_size, self.hidden_dim])
		# initialize update matrix
		w_ih_update = np.zeros_like(self.w_input_hidden)
		w_ho_update = np.zeros_like(self.w_hidden_output)
		w_hh_update = np.zeros_like(self.w_hidden_hidden)
		# Backward update
		for position in range(steps):

			current_X =  np.atleast_2d(X[:batch_size, steps - position - 1])
			now_hidden_layer = np.atleast_2d(hidden_layer[-position-1])
			pre_hidden_layer = np.atleast_2d(hidden_layer[-position-2])
			current_deltas = deltas[-position-1]
			
			delta_w_ho = now_hidden_layer.T.dot((np.atleast_2d(current_deltas)))
			w_ho_update += np.expand_dims(np.mean(delta_w_ho, axis=1), axis=1)/batch_size

			delta_c_h = current_deltas.dot(self.w_hidden_output.T)
			delta_c_h += np.atleast_2d(accumulate_hidden_delta).dot(self.w_hidden_hidden)
			delta_c_h *= self.activation._derivative_with_calculate(now_hidden_layer)
			
			accumulate_hidden_delta = delta_c_h
			w_ih_update += current_X.T.dot(delta_c_h)/batch_size
			w_hh_update += pre_hidden_layer.T.dot(delta_c_h)/batch_size
			

		# update using gradient descent
		self.w_input_hidden += w_ih_update * learning_rate
		self.w_hidden_hidden += w_hh_update * learning_rate
		self.w_hidden_output += w_ho_update * learning_rate 

		# clear update
		w_ih_update *= 0
		w_hh_update *= 0
		w_ho_update *= 0

		return predicts, hidden_layer[-1], overallError