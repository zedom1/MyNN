import copy, numpy as np
from activation import Sigmoid, Tanh

class LSTM(object):

	def __init__(self, arg):
		super(LSTM, self).__init__()

		self.input_dim 	= arg["input_dim"]
		self.hidden_dim = arg["hidden_dim"]
		self.output_dim = arg["output_dim"]

		self.w_forget 			= 2*np.random.random((self.hidden_dim, self.input_dim+self.hidden_dim)) - 1
		self.w_memory_weight 	= 2*np.random.random((self.hidden_dim, self.input_dim+self.hidden_dim)) - 1
		self.w_memory_content 	= 2*np.random.random((self.hidden_dim, self.input_dim+self.hidden_dim)) - 1
		self.w_output 			= 2*np.random.random((self.hidden_dim, self.input_dim+self.hidden_dim)) - 1
		self.w_predict 			= 2*np.random.random((self.output_dim, self.hidden_dim)) - 1

		self.activation_forget = arg["activation_forget"]() 				if "activation_forget" 			in arg else Sigmoid()
		self.activation_memory_weight = arg["activation_memory_weight"]() 	if "activation_memory_weight" 	in arg else Sigmoid()
		self.activation_memory_content = arg["activation_memory_content"]() if "activation_memory_content" 	in arg else Tanh()
		self.activation_output_weight = arg["activation_output_weight"]() 	if "activation_output_weight" 	in arg else Sigmoid()
		self.activation_output_content = arg["activation_output_content"]() if "activation_output_content" 	in arg else Tanh()

	def train(self, X, y, learning_rate = 0.1):

		assert len(np.shape(X)) == 3, "Invalid input shape. The input must be 3d (batch, timestep, input_dim)"
		batch_size, steps, input_dim = np.shape(X)
		assert self.input_dim == input_dim, "Unmatch input_dim."

		overallError = 0
		list_delta = list()
		list_predict = np.zeros([steps, batch_size, self.output_dim])

		list_hidden_state = list()
		list_cell_state = list()
		
		list_derivative_forget_weight = list()
		list_memory_weight = list()
		list_memory_content = list()
		list_output_weight = list()
		list_output_content = list()

		list_hidden_state.append(np.zeros([batch_size, self.hidden_dim]))
		list_cell_state.append(np.zeros([batch_size, self.hidden_dim]))
		

		# Forwarding
		for position in range(steps):

			# generate input and output
			current_X = np.atleast_2d(X[:batch_size, position])
			current_y = np.atleast_2d(y[:batch_size, position])
			concate_X = np.concatenate((list_hidden_state[-1], current_X), axis=1)
			
			# forget gate
			forget_weight = self.activation_forget._calculate(concate_X.dot(self.w_forget.T))
			forget_cell = list_cell_state[-1]*forget_weight
			
			list_derivative_forget_weight.append(self.activation_forget._derivative_with_calculate(forget_weight))
			
			# memory gate
			memory_weight = self.activation_memory_weight._calculate(concate_X.dot(self.w_memory_weight.T))
			memory_content = self.activation_memory_content._calculate(concate_X.dot(self.w_memory_content.T))
			memory_cell = memory_weight * memory_content
			
			list_memory_weight.append(memory_weight)
			list_memory_content.append(memory_content)

			# new cell state
			cell_state = forget_cell + memory_cell
			list_cell_state.append(cell_state)

			# output gate
			output_weight = self.activation_output_weight._calculate(concate_X.dot(self.w_output.T))
			output_content = self.activation_output_content._calculate(cell_state)
			output = output_weight * output_content
			
			list_hidden_state.append(output)
			list_output_weight.append(output_weight)
			list_output_content.append(output_content)

			# predict
			predict = output.dot(self.w_predict.T)
			list_predict[position] = predict
			
			# compute error
			error = current_y - predict
			delta = error*2
			list_delta.append(delta)
			overallError += np.mean(np.abs(error))
			
		# Backward update initialize
		accumulate_hidden_delta = np.zeros([batch_size, self.hidden_dim])
		# initialize update matrix

		w_update_forget = np.zeros_like(self.w_forget)
		w_update_memory_weight = np.zeros_like(self.w_memory_weight)
		w_update_memory_content = np.zeros_like(self.w_memory_content)
		w_update_output = np.zeros_like(self.w_output)
		w_update_predict = np.zeros_like(self.w_predict)

		# Backward update
		for position in range(steps):

			# get data in this position
			current_X 				= np.atleast_2d(X[:batch_size, steps - position - 1])
			current_hidden_state 	= np.atleast_2d(list_hidden_state[-position-1])
			pre_hidden_state 		= np.atleast_2d(list_hidden_state[-position-2])
			current_cell_state 		= np.atleast_2d(list_cell_state[-position-1])
			pre_cell_state 			= np.atleast_2d(list_cell_state[-position-2])

			current_delta 	= np.atleast_2d(list_delta[-position-1])
			output_weight 	= np.atleast_2d(list_output_weight[-position-1])
			output_content 	= np.atleast_2d(list_output_content[-position-1])
			memory_weight 	= np.atleast_2d(list_memory_weight[-position-1])
			memory_content 	= np.atleast_2d(list_memory_content[-position-1])
			derivative_forget_weight = np.atleast_2d(list_derivative_forget_weight[-position-1])
			
			# preprocess data
			concate_X = np.concatenate((pre_hidden_state, current_X), axis=1)
			derivative_memory_weight = self.activation_memory_weight._derivative_with_calculate(memory_weight)
			derivative_memory_content = self.activation_memory_content._derivative_with_calculate(memory_content)
			derivative_output_weight = self.activation_output_weight._derivative_with_calculate(output_weight)
			derivative_output_content = self.activation_output_content._derivative_with_calculate(output_content)

			# calculate predict weight update
			current_hidden_delta = current_delta.dot(self.w_predict)
			w_update_predict += (current_hidden_state.T.dot(current_delta)).T

			# calculate new accumulate_hidden_delta
			# \partial{L (t.. T)}{h_t} = \partial{L (t+1.. T)}{h_t+1} * \partial{h_t+1}{h_t} + \partial{L_t}{h_t} 
			derivative_out_w_h = derivative_output_weight.dot(self.w_output[:, :self.hidden_dim])
			
			derivative_out_c_c = derivative_output_content
			derivative_for_c_h = pre_cell_state * (derivative_forget_weight.dot(self.w_forget[:, :self.hidden_dim]))
			derivative_men_c_h = (derivative_memory_weight.dot(self.w_memory_weight[:, :self.hidden_dim])) * (derivative_memory_content.dot(self.w_memory_content[:, :self.hidden_dim]))
			derivative_out_c_h = derivative_out_c_c * derivative_for_c_h * derivative_men_c_h

			derivative_h_h = derivative_out_w_h * derivative_out_c_h

			accumulate_hidden_delta = accumulate_hidden_delta * derivative_h_h + current_hidden_delta

			# calculate \paritial{L}{Cell_state_t}
			derivative_cell_state = accumulate_hidden_delta * output_weight * derivative_output_content

			# calculate the rest update weights:
			w_update_output += (accumulate_hidden_delta * output_content * derivative_output_weight).T.dot(concate_X)
			w_update_memory_content += (derivative_cell_state * memory_weight * derivative_memory_weight).T.dot(concate_X)
			w_update_memory_weight += (derivative_cell_state * memory_content * derivative_memory_content).T.dot(concate_X)
			w_update_forget += (derivative_cell_state * pre_cell_state * derivative_forget_weight).T.dot(concate_X)

			#exit()
			

		# update using gradient descent
		self.w_forget += learning_rate * w_update_forget /batch_size
		self.w_memory_weight += w_update_memory_weight * learning_rate /batch_size
		self.w_memory_content += w_update_memory_content * learning_rate /batch_size
		self.w_output += w_update_output * learning_rate /batch_size
		self.w_predict += w_update_predict * learning_rate /batch_size

		# clear update
		w_update_forget *= 0
		w_update_memory_weight *= 0
		w_update_memory_content *= 0
		w_update_output *= 0
		w_update_predict *= 0

		return list_predict, list_hidden_state[-1], overallError