#!/usr/bin/env python3
"""This modlue containes the function viterbi that calculates the most likely
sequence of hidden states for a hidden markov model"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """This fucntion calculates the most likely sequence of hidden states for a
    hidden markov model
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
        path, P, or None, None on failure
        path is the a list of length T containing the most likely sequence of
             hidden states
        P is the probability of obtaining the path sequence
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

    # Initialize the viterbi path
    viterbi = np.zeros((hidden_states, observations))
    # print for visualization of the viterbi path
    # print(f"Viterbi: {viterbi}")

    # this creates a matrix that will store the path
    # set the first column of the viterbi matrix to the
    # initial probabilities
    viterbi[:, 0] = Initial.T * Emission[:, Observation[0]]
    # print for visualization of the viterbi path
    # print(f"Viterbi: {viterbi}")

    # initialize the backpointer
    backpointer = np.zeros((hidden_states, observations))

    # Step 3: iterate over the observations
    for t in range(1, observations):
        for s in range(hidden_states):
            # Calculate the viterbi path for the current state
            # and observation
            # print(f"Viterbi: {viterbi[:, t]}")
            # print(f"Transition: {Transition[:, s]}")
            # print(f"Emission: {Emission[s, Observation[t]]}")
            viterbi[s, t] = np.max(viterbi[:, t - 1] * Transition[:, s] *
                                   Emission[s, Observation[t]])
            # print(f"Viterbi: {viterbi}")
            # print(f"Backpointer: {backpointer}")
            backpointer[s, t] = np.argmax(
                viterbi[:, t - 1] * Transition[:, s] *
                Emission[s, Observation[t]])
            # print(f"Backpointer: {backpointer}")

    # Step 4: Termination
    # Calculate the path probablity
    # P is the probability of obtaining the path sequence
    P = np.max(viterbi[:, observations - 1])
    # print(f"P: {P}")

    # infer the last state of the most likely path
    Last_state = np.argmax(viterbi[:, observations - 1])
    # print(f"Last state: {Last_state}")

    # Add the last state to the path
    path = [Last_state]

    # Step 5: Backtrack
    for t in range(observations - 1, 0, -1):
        # INSERT the state into the path
        path.insert(0, int(backpointer[Last_state, t]))
        # Update the last state
        Last_state = int(backpointer[Last_state, t])

    return path, P
