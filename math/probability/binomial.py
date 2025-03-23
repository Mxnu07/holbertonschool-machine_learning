#!/usr/bin/env python3
"""This modluels contains the binomial distribution class"""


class Binomial:
    """This class represents a binomial distribution
    """

    def __init__(self, data=None, n=1, p=0.5):
        """Binomial class constructor
        Args:
            data (List): List of the data to be used to estimate
            the distribution
            n (int): number of Bernoulli trials
            p (float): probability of a "success"
        """
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # Caclculate mean
            mean = float(sum(data) / len(data))
            # Calaculate stddev
            stddev = float((sum(
                [(data[i] - mean) ** 2 for i in range(len(data))]
            ) / len(data)) ** 0.5)
            # calculate the lambtha value
            self.p = 1 - ((stddev ** 2) / mean)
            self.n = int(round(mean / self.p))
            self.p = float(mean / self.n)
        else:
            # If data is not given
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = round(n)
            self.p = float(p)

    def pmf(self, k):
        """
        function that calculates the probability mass function
        for k successes
        """
        if not isinstance(k, int):
            k = int(k)
        if k is None or k < 0 or k > self.n:
            return 0
        return (self.fact(self.n) / (self.fact(k) * self.fact(self.n - k))) * (
            (self.p ** k) * (1 - self.p) ** (self.n - k))

    def fact(self, k):
        """function that returns the factorial of k"""
        if k in [0, 1]:
            return 1
        return k * self.fact(k - 1)

    def cdf(self, k):
        """
        function that calculates the cumulative distribution function
        for k successes
        """
        if not isinstance(k, int):
            k = int(k)
        if k is None or k < 0 or k > self.n:
            return 0
        return sum([(self.fact(self.n) /
                    (self.fact(i) * self.fact(self.n - i)))
                    * ((self.p ** i) * (1 - self.p) ** (self.n - i))
                    for i in range(0, k + 1)])
