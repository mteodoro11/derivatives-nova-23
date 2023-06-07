import numpy as np
from scipy.stats import norm

class BlackScholes:

    def __init__(self, S, X, r, Tp, vol):

        self.S = S
        self.X = X
        self.r = r
        self.Tp = Tp
        self.vol = vol


    def calculate_d1(self):

        d1 = (np.log(self.S / self.X) + (self.r + (self.vol ** 2) / 2) * self.Tp) / (self.vol * np.sqrt(self.Tp))

        return d1


    def calculate_d2(self):

        d1 = self.calculate_d1()
        d2 = d1 - self.vol * np.sqrt(self.Tp)

        return d2


    def calculate_call(self):

        d1 = self.calculate_d1()
        d2 = self.calculate_d2()
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        call = self.S * N_d1 - self.X * np.exp(-self.r * self.Tp) * N_d2

        return call


    def calculate_put(self):

        d1 = self.calculate_d1()
        d2 = self.calculate_d2()
        N_minus_d1 = norm.cdf(-d1)
        N_minus_d2 = norm.cdf(-d2)
        put = self.X * np.exp(-self.r * self.Tp) * N_minus_d2 - self.S * N_minus_d1

        return put


    def print_values(self):

        d1 = self.calculate_d1()
        d2 = self.calculate_d2()
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        N_minus_d1 = norm.cdf(-d1)
        N_minus_d2 = norm.cdf(-d2)
        call_value = self.calculate_call()
        put_value = self.calculate_put()

        print("S: ", self.S)
        print("K: ", self.X)
        print("r: ", self.r)
        print("T: ", self.Tp)
        print("sigma: ", self.vol)
        print("d1: ", d1)
        print("d2: ", d2)
        print("N(d1): ", N_d1)
        print("N(d2): ", N_d2)
        print("N(-d1): ", N_minus_d1)
        print("N(-d2): ", N_minus_d2)
        print("Call value: ", call_value)
        print("Put value: ", put_value)


class BinomialTree:

    def __init__(self, S_, T_, sigma_, N):

        self.S_ = S_
        self.T_ = T_
        self.sigma_ = sigma_
        self.N = N
        self.tree = self._build_tree()


    def _build_tree(self):

        u = np.exp(self.sigma_ * np.sqrt(self.T_ / self.N))
        d = np.exp(-self.sigma_ * np.sqrt(self.T_ / self.N))

        tree = [[self.S_]]

        for step in range(1, self.N+1):
            St = [0] * (step+1)
            St[0] = self.S_ * (d**step)

            for j in range(1, step+1):
                St[j] = tree[step-1][j-1] * u

            tree.append(St)

        return tree


    def _calculate_option_price(self, K_, r_, option_type):

        u = np.exp(self.sigma_ * np.sqrt(self.T_ / self.N))
        d = np.exp(-self.sigma_ * np.sqrt(self.T_ / self.N))
        R = 1 + r_
        qu = (u - R) / (u - d)
        qd = 1 - qu

        ctree = self.tree
        St_prices = self.tree[-1]

        if option_type == "call":
            for state in range(len(St_prices)):
                ctree[-1][state] = max(St_prices[state] - K_, 0)
                ctree[-1][state] = ctree[-1][state] / R
        elif option_type == "put":
            for state in range(len(St_prices)):
                ctree[-1][state] = max(K_ - St_prices[state], 0)
                ctree[-1][state] = ctree[-1][state] / R

        for step in range(2, len(ctree) + 1):
            for state in range(len(ctree[-step])):
                ctree[-step][state] = (
                    qd * ctree[-(step - 1)][state] +
                    qu * ctree[-(step - 1)][(state + 1)]
                )
                ctree[-step][state] = ctree[-step][state] / R

        return ctree


    def calculate_call_option_price(self, K_, r_):
        return self._calculate_option_price(K_, r_, option_type="call")


    def calculate_put_option_price(self, K_, r_):
        return self._calculate_option_price(K_, r_, option_type="put")