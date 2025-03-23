#!/usr/bin/env python3
"""This modlue contains the poisson distribution class
"""


class Poisson:
    """This class represents a poisson distribution
    """
    def __init__(self, data=None, lambtha=1.):
        """Initialized the Poisson object with the given lambtha value
        Args:
            data: list of the data to be used to estimate the distribution
            lambtha: expected number of occurences in a given time frame
        """
        # Check if data is given
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # calculate the lambtha value
            self.lambtha = float(sum(data) / len(data))
        else:
            # If data is not given
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of successes.

        The Probability Mass Function (PMF) for the
        Poisson distribution calculates
        the probability of observing exactly k successes
        (events, arrivals, etc.)
        given the average rate (lambda) of successes in
        the same time period under
        consideration.

        Args:
            k: The number of successes for which to calculate
            the PMF. This is
            the specific value of the random variable for which
            the probability
            is calculated.

        Returns:
            The PMF value for observing exactly k successes,
            given the Poisson
            distribution's lambda (average rate of success).

        Note:
            - The PMF is defined only for non-negative integers.
            If a non-integer
            is provided, it is converted to an integer.
            - If k is negative, the function returns 0, as the
            PMF is not defined
            for negative numbers of successes.
        """
        # Define the base of the natural logarithm (e) to
        # approximate calculations.
        e = 2.7182818285

        # Ensure k is an integer, as the PMF is defined only
        # for integer values.
        if not isinstance(k, int):
            k = int(k)

        # The PMF is 0 for negative values of k, as negative
        # successes are not possible.
        if k < 0:
            return 0

        # Calculate k factorial (k!) as it's required for the
        # PMF formula.
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        # Calculate and return the PMF using the formula
        # P(k; lambda) = (e^-lambda) * (lambda^k) / k!
        # This formula represents the probability of
        # observing exactly k successes.
        return ((e ** -self.lambtha) * (self.lambtha ** k)) / factorial

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of successes.

        The Cumulative Distribution Function (CDF) for the Poisson distribution
        calculates the probability that the Poisson random variable takes on a
        value less than or equal to k.

        Args:
            k: The upper limit of successes for which to calculate the CDF.

        Returns:
            The CDF value for observing up to k successes, given the Poisson
            distribution's lambda (average rate of success).

        Note:
            - The CDF is defined for non-negative integers. If a non-integer
            is provided, it is converted to an integer.
            - If k is negative, the function returns 0, as the CDF is not
            defined
            for negative numbers of successes.
    """
        # Define the base of the natural logarithm (e) to
        # approximate calculations.
        e = 2.7182818285

        # Ensure k is an integer, as the CDF
        # is defined only for integer values.
        if not isinstance(k, int):
            k = int(k)
        # The CDF is 0 for negative values of k,
        if k < 0:
            return 0

        # Initialize the cumulative distribution
        # function (CDF) value to 0.
        # This variable will accumulate the probabilities
        # from 0 to k, inclusive.
        cdf_value = 0

        # Iterate over each value from 0 to k (inclusive) to
        # calculate the sum of probabilities.
        # This loop calculates the probability for each i
        # (from 0 to k) and adds it to the CDF value.
        for i in range(k + 1):
            # Initialize the factorial value for the current i.
            # Factorial is used in the denominator of the Poisson PMF formula.
            factorial = 1
            # Calculate the factorial of i (i!).
            # This inner loop multiplies 1 by all integers up to i to get i!.
            for j in range(1, i + 1):
                factorial *= j

            # Calculate the probability of exactly i successes
            # using the Poisson PMF formula
            # and add it to the cumulative probability (cdf_value).
            # The formula is: (e^-lambda) * (lambda^i) / i!
            # - e^-lambda scales the probability based on the average
            # rate of success (lambda),
            # - lambda^i adjusts for the probability of i successes,
            # - dividing by i! accounts for the order in which successes
            # occur not mattering.
            cdf_value += (e ** -self.lambtha) * (self.lambtha ** i) / factorial

        # Return the cumulative probability of observing up to k successes.
        # This value represents the CDF at k for the Poisson distribution.
        return cdf_value
