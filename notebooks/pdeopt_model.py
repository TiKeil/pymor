import numpy as np

from pymor.models.basic import StationaryModel
from pymor.operators.constructions import VectorOperator

class LinearPdeoptStationaryModel(StationaryModel):

    def __init__(self, operator, rhs, output_functional=None, products=None,
                 error_estimator=None, visualizer=None, name='LinearPdeoptModel'):
        super().__init__(operator, rhs, output_functional, products, error_estimator, visualizer, name)
        self.__auto_init(locals())

    @property
    def dual(self):
        if not hasattr(self, '_dual'):
            assert self.output_functional is not None
            assert self.output_functional.linear
            assert 1 # TODO: assert that the operator is symmetric
            self._dual = self.with_(rhs=self.output_functional.H)
        return self._dual

    def solve_d_mu(self, parameter, index, mu, U=None):
        if U is None:
            U = self.solve(mu)
        residual_dmu_lhs = VectorOperator(self.operator.d_mu(parameter, index).apply(U, mu=mu))
        residual_dmu_rhs = self.rhs.d_mu(parameter, index)
        rhs_operator = residual_dmu_rhs-residual_dmu_lhs
        return self.operator.apply_inverse(rhs_operator.as_range_array(mu), mu=mu)

    def output_d_mu(self, mu, U=None, P=None, adjoint_approach=True):
        if U is None:
            U = self.solve(mu)
        gradient = []
        if adjoint_approach:
            if P is None:
                P = self.dual.solve(mu)
        for (parameter, size) in self.parameters.items():
            for index in range(size):
                output_partial_dmu = self.output_functional.d_mu(parameter, index).apply(U, mu=mu).to_numpy()[0,0]
                if adjoint_approach:
                    residual_dmu_lhs = self.operator.d_mu(parameter, index).apply2(U, P, mu=mu)
                    residual_dmu_rhs = self.rhs.d_mu(parameter, index).apply_adjoint(P, mu=mu).to_numpy()[0,0]
                    gradient.append((output_partial_dmu + residual_dmu_rhs - residual_dmu_lhs)[0,0])
                else:
                    U_d_mu = self.solve_d_mu(parameter, index, mu, U=U)
                    gradient.append(output_partial_dmu + \
                            self.output_functional.apply(U_d_mu, mu).to_numpy()[0,0])
        return np.array(gradient)
