# -*- coding: utf-8 -*-
import math
import copy
from .trie import Trie


class SCW(object):
    """Soft Confidence Weighted (SCW)
    """

    def __init__(self, phi, c):
        """
        Args:
            <int> phi
            <float> c
        """
        self.phi = phi
        self.c = c

        square_phi = phi * phi
        self.psi = 1. + square_phi / 2.
        self.zeta = 1. + square_phi

        self.mu = {}
        self.sigma = {}

    def calc_margin(self, x, y):
        """
        Args:
            <dict> x
            <int> y
        Return:
            <float> margin
        """
        return y * Trie.inner_prod(self.mu, x)

    def calc_variance(self, x):
        """
        Args:
            <dict> x
        Return:
            <float> variance
        """
        ret = Trie.mult(copy.deepcopy(x), self.sigma)
        return Trie.inner_prod(ret, x)

    def calc_alpha(self, margin, variance):
        """
        Args:
            <float> margin
            <float> variance
        Return:
            <float> alpha
        """
        term1 = margin * self.phi * 0.5
        alpha_denom = variance * self.zeta
        alpha = (-1 * margin * self.psi + self.phi *
                 math.sqrt(term1**2 + alpha_denom)) / alpha_denom
        if alpha < 0:
            return 0.
        return alpha if alpha < self.c else self.c

    def calc_beta(self, margin, variance, alpha):
        """
        Args:
            <float> margin
            <float> variance
            <float> alpha
        Return:
            <float> beta
        """
        beta_numer = alpha * self.phi
        term1 = beta_numer * variance
        beta_denom = ((-1 * term1 + math.sqrt(term1**2 + 4 * variance))
                      / 2. + term1)
        return beta_numer / beta_denom

    def update_mu_sigma(self, x, y, alpha, beta):
        """
        Args:
            <dict> x
            <float> y
            <float> alpha
            <float> beta
        """
        x_sigma = Trie.mult(x.copy(), self.sigma)
        x_sigma2 = copy.deepcopy(x_sigma)
        x_sigma2 = Trie.mult(copy.deepcopy(x_sigma), x_sigma)

        self.mu = Trie.add_coef(self.mu, x_sigma, alpha * y)
        self.sigma = Trie.add_coef(self.sigma, x_sigma2, -1 * beta, 1.)

    def update(self, x, y):
        """
        Args:
            <dict> x
            <float> y
        """
        margin = self.calc_margin(x, y)
        variance = self.calc_variance(x)
        alpha = self.calc_alpha(margin, variance)
        beta = self.calc_beta(margin, variance, alpha)
        self.update_mu_sigma(x, y, alpha, beta)

    def prune(self, _lambda, sigma_th):
        """Feature selection by L1 regularization (FOBOS)
        Args:
            <float> _lambda
            <float> sigma_th
        """
        new_mu = {}
        new_sigma = {}
        old_sigma = self.sigma

        def fobos(key, mu_val, *args):
            vals = args[0]
            sigma_val = Trie.find(old_sigma, key)
            if mu_val < -_lambda:
                vals['new_mu'] = Trie.insert(vals['new_mu'], key, mu_val + _lambda)
                vals['new_sigma'] = Trie.insert(vals['new_sigma'], key, sigma_val)
            elif mu_val > _lambda:
                vals['new_mu'] = Trie.insert(vals['new_mu'], key, mu_val - _lambda)
                vals['new_sigma'] = Trie.insert(vals['new_sigma'], key, sigma_val)
            else:
                if (sigma_val < sigma_th):
                    vals['new_mu'] = Trie.insert(vals['new_mu'], key, 0)
                    vals['new_sigma'] = Trie.insert(vals['new_sigma'], key, sigma_val)

        vals = {'new_mu': new_mu, 'new_sigma': new_sigma}
        Trie.each(self.mu, fobos, [], vals)

        self.mu = vals['new_mu']
        self.sigma = vals['new_sigma']
