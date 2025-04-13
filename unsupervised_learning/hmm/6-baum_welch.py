#!/usr/bin/env python3
"""This mdlue contains the function baum_welch that performs the Baum-Welch
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


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    This function performs the Baum-Welch algorithm for a hidden markov model
    Args:
        Observations is a numpy.ndarray of shape (T,) that contains
            the index of
                    the observation
            T is the number of observations
        Transition is a 2D numpy.ndarray of shape (N, N) containing the
                    initialized transition probabilities
            Transition[i, j] is the probability of transitioning from the
             hidden
                            state i to j
            N is the number of hidden states
        Emission is a 2D numpy.ndarray of shape (N, M) containing the
                 initialized emission probabilities
            Emission[i, j] is the probability of observing j given the hidden
                          state i
            M is the number of all possible observations
        Initial is a numpy.ndarray of shape (N, 1) containing the
             initialized probability of starting in a particular hidden state
        iterations is the number of times expectation-maximization should be
                    performed
    Returns:
        the converged Transition, Emission, or None, None on failure
        Transition is the converged Transition
        Emission is the converged Emission
    """
    # Spte 1: Validate the inputs
    # Observation check
    if not isinstance(Observations, np.ndarray) or Observations.ndim != 1:
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

    # Iterations check
    if not isinstance(iterations, int) or iterations < 0:
        return None, None

    # Step 2: Initialize the variables
    hidden_states = Initial.shape[0]
    # print(f"hidden_states: {hidden_states}")

    observations = Observations.shape[0]
    # print(f"observations: {observations}")

    # number of outoput states
    output_states = Emission.shape[1]
    # print(f"output_states: {output_states}")

    # make copies of the Transition and Emission matrices
    # for erarly stopping
    transition_prev = Transition.copy()
    emission_prev = Emission.copy()

    # Step 3: Perform the Baum-Welch algorithm
    for iteration in range(iterations):
        # print(f"iteration: {iteration}")
        # Step 3.1: Perform the forward algorithm
        _, F = forward(Observations, Emission, Transition, Initial)
        # print(f"F: {F}")

        # Step 3.2: Perform the backward algorithm
        _, B = backward(Observations, Emission, Transition, Initial)
        # print(f"B: {B}")

        # Step 3.3: Compute the numerator and denominator
        # for the transition matrix
        NUM = np.zeros((hidden_states, hidden_states, observations - 1))
        for t in range(observations - 1):
            for i in range(hidden_states):
                for j in range(hidden_states):
                    Fit = F[i, t]
                    aij = Transition[i, j]
                    bjt1 = Emission[j, Observations[t + 1]]
                    Bjt1 = B[j, t + 1]
                    NUM[i, j, t] = Fit * aij * bjt1 * Bjt1

        # Compute the denominator for normalization
        DEN = np.sum(NUM, axis=(0, 1))
        X = NUM / DEN

        # Compute gamma, the aggregate helper variable
        G = np.zeros((hidden_states, observations))
        # print(f"G: {G}")
        NUM = np.zeros((hidden_states, observations))
        # print(f"NUM: {NUM}")
        for t in range(observations):
            for i in range(hidden_states):
                # Fit is the forward probability at time t for hidden state i
                Fit = F[i, t]
                # Bit is the backward probability at time t for hidden state i
                Bit = B[i, t]
                # NUM[i, t] is the numerator of the gamma expression
                NUM[i, t] = Fit * Bit
                # print(f"NUM: {NUM}")
        # DEN is the denominator of the gamma expression
        DEN = np.sum(NUM, axis=0)
        # G is the gamma expression
        G = NUM / DEN
        # print(f"G: {G}")

        # Update the Transition matrix
        Transition = np.sum(
            X, axis=2) / np.sum(
                G[:, :observations - 1], axis=1)[..., np.newaxis]

        # Update the Emission matrix
        DEN = np.sum(G, axis=1)
        NUM = np.zeros((hidden_states, output_states))
        for k in range(output_states):
            NUM[:, k] = np.sum(G[:, Observations == k], axis=1)
        Emission = NUM / DEN[..., np.newaxis]

        # Early stopping; exit condition on Transition and Emission matrices
        if np.all(
            np.isclose(
                Transition, transition_prev)) or np.all(
                    np.isclose(Emission, emission_prev)):
            return Transition, Emission

        # Make deep copies of Transition and Emission matrices for early stop
        transition_prev = np.copy(Transition)
        emission_prev = np.copy(Emission)

    return Transition, Emission
