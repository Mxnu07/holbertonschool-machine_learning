#!/usr/bin/env python3
"""This modlue containes the function backward that performs the backward
algorithm for a hidden markov model"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """This function performs the backward algorithm for a hidden markov model
    Args:
        Observation is a numpy.ndarray of shape (T,) that contains the index of
                   the observation
            T is the number of observations
        Emission is a numpy.ndarray of shape (N, M) containing the emission
                 probability of a specific observation given a hidden state
            Emission[i, j] is the probability of observing j given the hidden
                          state i
            N is the number of hidden states
            M is the number of all possible observations
        Transition is a 2D numpy.ndarray of shape (N, N) containing the
                   transition probabilities
            Transition[i, j] is the probability of transitioning from the
             hidden
                           state i to j
        Initial is a numpy.ndarray of shape (N, 1) containing the
            probability of
                starting in a particular hidden state
    Returns:
        P, B, or None, None on failure
        P is the likelihood of the observations given the model
        B is a numpy.ndarray of shape (N, T) containing the backward path
            probabilities
            B[i, j] is the probability of generating the future observations
            from hidden state i at time j
    """
    # Spte 1: Validate the inputs
    # Observation check
    if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
        return None, None

    # Emmision check
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    # Chek the sum of the emission values
    if np.all(np.isclose(np.sum(Emission, axis=1), 1)) is False:
        return None, None

    # Initial check
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None
    # Check the shape of Initial to be (N, 1)
    if Initial.shape[1] != 1:
        return None, None
    # Check the sum of the Initial values
    if np.all(np.isclose(np.sum(Initial), 1)) is False:
        return None, None

    # Transition check
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    # Check the values of transition N match Initial N
    if Transition.shape[0] != Initial.shape[0]:
        return None, None
    if Transition.shape[1] != Initial.shape[0]:
        return None, None
    # Check the sum of the transition values
    if np.all(np.isclose(np.sum(Transition, axis=1), 1)) is False:
        return None, None

    # Step 2: Initialize the variables

    # extract the number of hidden states from the shape of Initial
    hidden_states = Initial.shape[0]
    # extract the number of observations from the shape of Observation
    observations = Observation.shape[0]
    # Prints to see the values for understanding the code
    # print(f"Hidden states: {hidden_states}")
    # print(f"Observations: {observations}")

    # Step 3: Initialize the backward probabilities
    B = np.zeros((hidden_states, observations))
    # print(f"B: {B}")
    # Initialize the last column of B
    B[:, observations - 1] = 1
    # print(f"B: {B}")

    # Step 4: Perform the backward algorithm
    # range(start, stop, step)
    for t in range(observations - 2, -1, -1):
        for s in range(hidden_states):
            # set the value of B[s, t] to the sum of the
            # probability of transitioning from state s to state t
            B[s, t] = np.sum(B[:, t + 1] * Transition[s, :] *
                             Emission[:, Observation[t + 1]])

    # Step 5: Termination
    # Calculate the likelihood of the observations
    # sum the product of the initial state and the emission
    # Observation[0] is the first observation
    # B[:, 0] is the first column of B
    # Initial.T is the transpose of the initial state
    # Emission[:, Observation[0]] is the emission probability
    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])

    return P, B
