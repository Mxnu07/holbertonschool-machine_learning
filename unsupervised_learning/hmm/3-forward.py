#!/usr/bin/env python3
"""This modlue containes the function forward that performs the forward
algorithm for a hidden markov model"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """Thias function calculates the forward algorithm for a hidden markov
    model
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
        P, F, or None, None on failure
        P is the likelihood of the observations given the model
        F is a numpy.ndarray of shape (N, T) containing the forward path
            probabilities
            F[i, j] is the probability of being in hidden state i at time j
            given the previous observations
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

    # Create the alpha variable to store the forward probabilities
    F = np.zeros((hidden_states, observations))
    # print to see the results of the initialization
    # print(f"Initial F\n:{F}")

    # Extract the observation index from the Observation
    # print the observation 1D array to understand the code
    # print(f"Observation: {Observation}")
    index = Observation[0]

    # Extract the Emission probabilities for the first observation
    # emission contains the probabilities of a specific observation
    # given a hidden state
    # print(f"Emission: {Emission}")
    E = Emission[:, index]

    # Calculate the forward probabilities for the first observation
    # Transpose the initial
    F[:, 0] = Initial.T * E
    # print the results to understand the code
    # print(f"Initial F\n:{F}")

    # Step 3: Iterate over the observations to calculate the forward
    # probabilities
    for i in range(1, observations):
        for j in range(hidden_states):
            # Calculate the transition probabilities
            # print(f"Transition: {Transition}")
            # print(f"F: {F}")
            # print(f"j: {j}")
            # print(f"i: {i}")
            # print(f"Observation: {Observation[i]}")
            # print(f"Emission: {Emission[:, Observation[i]]}")
            # print(f"Hidden states: {hidden_states}")
            # print(f"Observations: {observations}")
            # print(f"Initial: {Initial}")
            # print(f"Initial: {Initial.T}")
            # print(f"Initial: {Initial.T * Emission[:, Observation[i]]}")
            # print(f"Initial: {Initial.T * Emission[
            #   :, Observation[i]] * Transition[j]}")
            # print(f"Initial: {Initial.T * Emission[
            #   :, Observation[i]] * Transition[j]}")
            # print(f"Initial: {np.sum(Initial.T * Emission[
            #   :, Observation[i]] * Transition[j])}")
            F[j, i] = np.sum(F[:, i-1] * Transition[
                :, j] * Emission[j, Observation[i]])

    # Step 4: Calculate the likelihood of the observations given the model
    # [:, observations-1] to get the last observation
    # axis=0 to sum the values of the last observation
    P = np.sum(F[:, observations-1], axis=0)

    return P, F
