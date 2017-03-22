import numpy as np
import itertools

def markov_seq(n_state, n_order, trans_dict, seq_length, batch_size):
	"""
	Arg:
	n_state: number of single-states (e.g. if n=5, state is from 0 to 4).
	n_order: order of the Markov chain.
	trans_dict: a dictionary where key is an 1D list representing the total state of shape (n_order,), 
	and value is an 1D np.array representing transition probability distribution of shape (n_state,).
	sequence length.
	batch_size: the size of batch.
	
	Return:
	markov_seq: (batch_size, seq_length)-shaped np.array representing random sequences generated by the Markov chain. 
	entropy_rate: entropy rate of the Markov chain.(We demand the entropy of each row of the transition matrix to be the same, so that the entropy rate doesn't depend on the asympototic ditribution of states.)
	"""	
	init_state = np.random.randint(n_state, size=(batch_size, n_order)) #(batch_size, n_order) 2d array.
	current_state = init_state #This is not a single-state, but the total state of size n_order for each batch.
	markov_seq = np.empty((batch_size, seq_length))
	markov_seq[:,:n_order] = init_state
	for i in range(seq_length-n_order):
		for j in range(batch_size):
			next_single_state = np.random.choice(n_state, p=trans_dict[tuple(current_state[j])]) # This is an integer between 0 and n_state-1, representing the next single state for each batch.
			current_state[j] = np.append(current_state[j,1:], next_single_state)
			markov_seq[j,i+n_order] = next_single_state

	trans_dist = trans_dict[trans_dict.keys()[0]] # This is a 1d array representing transition probability distribution from a random total state. 

	return markov_seq.astype(int) #astype(int) is to make markov_seq integer-type.

def generate_trans_dict(n_state, n_order, trans_prob):
	"""
	Arg:
	n_state: the number of single states
	n_order: the order of the Markov chain
	trans_prob: a transition probability from a total state (for e.g., if n_order=2, the total state is a size-2 tuple consisting of single states.)

	Return:
	trans_dict: transition dictionary. Since we want the entropy of the transition probability distribution is constant across all possible starting
	states, the transition probability distribution for each starting state is a random permutation of the trans_prob.
	"""
	trans_dict = {}
	state_list = list(itertools.product(range(n_state), repeat=n_order))
	for state in state_list:
		trans_dict[state] = np.random.permutation(trans_prob)
	entropy_rate = -np.sum(trans_prob*np.log(trans_prob))	
	return trans_dict, entropy_rate



