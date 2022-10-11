import itertools

import numpy as np
import pandas as pd


class BestBet:
    def __init__(self, C, Proba_bookmaker=None):
        self.C = C
        s = 1 / C
        self.S = 0
        P_imp = np.exp(s) / sum(np.exp(s))  ## proba induite
        if Proba_bookmaker is None:
            self.P = P_imp
        else:
            self.P = Proba_bookmaker
        self.P_imp = P_imp

        if len(C) == 3:
            phi = {'A': [], 'N': [], 'B': [], 'Payoff': [], 'Payoff_imp': [],
                   'Variance': [], 'Variance_imp': []}
        elif len(C) == 2:
            phi = {'A': [], 'B': [], 'Payoff': [], 'Payoff_imp': [],
                   'Variance': [], 'Variance_imp': []}
        else:
            raise ValueError("incorrect size of odd list it should be 2 or 3")

        self.phi = pd.DataFrame.from_dict(phi)
        self.n_issue = len(C)

    def check_risk_free(self):
        r = np.array(list(map(np.array, list(itertools.combinations(self.C,
                                                                    self.n_issue - 1)))))  ## assume that the game has 3 outcomes Victory, Defeat, Draw
        if self.n_issue == 3:
            coef_arb = sum(np.array([i[0] * i[1] for i in r])) / np.prod(self.C)
        else:
            coef_arb = sum(np.array([i[0] for i in r])) / np.prod(self.C)

        if coef_arb <= 1:
            return ('Arbitrage existe')
        else:
            return ("Absence d'Opportunuité d'Arbitrage")
    def kelly_criterion(self):
        if any((self.C-1)==0):
            return []

        S = (self.P * self.C -1)*1/(self.C-1)
        S[S<0] = 0
        if len(self.C)==3:
            res = pd.DataFrame(S.reshape([1,len(self.C)]),columns = ['A','N','B'])
        else:
            res = pd.DataFrame(S.reshape([1,len(self.C)]),columns = ['A','B'])
        return  res

    def strategy(self, S, step=0.05):
        self.S = S
        if S >= 15:
                step = round(S/300,2)
        if self.n_issue == 3:
            phi = {'A': [], 'N': [], 'B': [], 'Payoff': [], 'Payoff_imp': [],
                   'Variance': [], 'Variance_imp': [], 'GL_min': [], 'GL_sum': []}
            self.phi = pd.DataFrame.from_dict(phi)

            A = B = N = np.arange(start=0, stop=S + step, step=step)
            for a in A:
                for n in N:
                    for b in B:
                        if a + n + b == S:
                            phi_alpha = np.array([a, n, b])

                            payoff = (phi_alpha * self.P).dot(self.C)
                            payoff_imp = (phi_alpha * self.P_imp).dot(self.C)  ## formule du payofff
                            variance = ((phi_alpha ** 2) * (self.C ** 2)).dot(self.P * (1 - self.P))
                            variance_imp = ((phi_alpha ** 2) * (self.C ** 2)).dot(self.P_imp * (1 - self.P_imp))
                            GL_min = np.min(self.C * np.array([a, n, b]) - self.S)
                            GL_sum = np.sum(self.C * np.array([a, n, b]) - self.S)
                            # self.C * best[['A', 'N', 'B']].to_numpy() - self.S
                            self.phi = pd.concat(
                                [self.phi, pd.DataFrame([{'A': a, 'N': n, 'B': b, 'Payoff': round(payoff, 3),
                                                          'Payoff_imp': round(payoff_imp, 3),
                                                          'Variance': round(variance, 4), 'Variance_imp':
                                                              round(variance_imp, 4), 'GL_min': round(GL_min, 3),
                                                          'GL_sum': round(GL_sum, 3)}])])
                            self.phi.reset_index()
        else:
            phi = {'A': [], 'B': [], 'Payoff': [], 'Payoff_imp': [],
                   'Variance': [], 'Variance_imp': [], 'GL_min': [], 'GL_sum': []}
            self.phi = pd.DataFrame.from_dict(phi)
            A = B = np.arange(start=0, stop=S + step, step=step)
            for a in A:
                for b in B:
                    if a + b == S:
                        phi_alpha = np.array([a, b])

                        payoff = (phi_alpha * self.P).dot(self.C)
                        payoff_imp = (phi_alpha * self.P_imp).dot(self.C)  ## formule du payofff
                        variance = ((phi_alpha ** 2) * (self.C ** 2)).dot(self.P * (1 - self.P))
                        variance_imp = ((phi_alpha ** 2) * (self.C ** 2)).dot(self.P_imp * (1 - self.P_imp))
                        GL_min = np.min(self.C * np.array([a, b]) - self.S)
                        GL_sum = np.sum(self.C * np.array([a, b]) - self.S)
                        self.phi = pd.concat([self.phi, pd.DataFrame([{'A': a, 'B': b, 'Payoff': round(payoff, 3),
                                                                       'Payoff_imp': round(payoff_imp, 3),
                                                                       'Variance': round(variance, 4), 'Variance_imp':
                                                                           round(variance_imp, 4),
                                                                       'GL_min': round(GL_min, 3),
                                                                       'GL_sum': round(GL_sum, 3)}])])
                        self.phi.reset_index()

    def best_bet(self, type='variance'):
        if self.phi.shape[0] == 0:
            return 'Please run strategy with the amount to bet '
        else:
            if self.n_issue == 3:
                if type == 'variance':
                    best = self.phi.query('Variance == Variance.min()')
                    return {'bet': best[['A', 'N', 'B']],
                            'Gain/Loss': self.C * best[['A', 'N', 'B']].to_numpy() - self.S}
                elif type == 'GL_min':
                    best = self.phi.query('GL_min == GL_min.max()')
                    return {'bet': best[['A', 'N', 'B']],
                            'Gain/Loss': self.C * best[['A', 'N', 'B']].to_numpy() - self.S}

                elif type == 'GL_sum':
                    best = self.phi.query('GL_sum == GL_sum.max()')
                    return {'bet': best[['A', 'N', 'B']],
                            'Gain/Loss': self.C * best[['A', 'N', 'B']].to_numpy() - self.S}

                else:
                    best = self.phi.query('Variance_imp == Variance_imp.min()')
                    return {'bet': best[['A', 'N', 'B']],
                            'Gain/Loss': self.C * best[['A', 'N', 'B']].to_numpy() - self.S}
            else:
                if type == 'variance':
                    best = self.phi.query('Variance == Variance.min()')
                    return {'bet': best[['A', 'B']],
                            'Gain/Loss': self.C * best[['A', 'B']].to_numpy() - self.S}
                elif type == 'GL_min':
                    best = self.phi.query('GL_min == GL_min.max()')
                    return {'bet': best[['A', 'B']],
                            'Gain/Loss': self.C * best[['A', 'B']].to_numpy() - self.S}

                elif type == 'GL_sum':
                    best = self.phi.query('GL_sum == GL_sum.max()')
                    return {'bet': best[['A', 'B']],
                            'Gain/Loss': self.C * best[['A', 'B']].to_numpy() - self.S}

                else:
                    best = self.phi.query('Variance_imp == Variance_imp.min()')
                    return {'bet': best[['A', 'B']],
                            'Gain/Loss': self.C * best[['A', 'B']].to_numpy() - self.S}

    def plot_simul(self):
        import plotly.express as px
        if self.n_issue == 3:
            fig = px.scatter_3d(self.phi, x='A', y='B', z='N',
                                color='Payoff_imp', size='Variance_imp', size_max=28,
                                hover_data=['GL_min', 'GL_sum'])
        else:
            fig = px.scatter_3d(self.phi, x='A', y='B', z='Variance_imp',
                                color='Payoff_imp', size_max=28, hover_data=['GL_min', 'GL_sum'])
        fig.show()

    def graph_VarianceReturn(self, xlab="Variance", ylab="Payoff"):
        import plotly.express as px
        if self.n_issue == 3:
            fig = px.scatter(self.phi, x=xlab, y=ylab, hover_data=['A', 'B', 'GL_min', 'GL_sum'], color='GL_min')
        else:
            fig = px.scatter(self.phi, x=xlab, y=ylab, hover_data=['A', 'B', 'GL_min', 'GL_sum'], color='GL_min')
        return fig
