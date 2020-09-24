import numpy as np

from pymor.models.basic import StationaryModel
from pymor.operators.constructions import VectorOperator

class LinearPdeoptStationaryModel(StationaryModel):
    
    def __init__(self, operator, rhs, output_functional=None, products=None, 
                 error_estimator=None, visualizer=None, name='LinearPdeoptModel'):
        super().__init__(operator, rhs, output_functional, products, error_estimator, visualizer, name)
        self.__auto_init(locals())

    @property
    def dual_model(self):
        if not hasattr(self, '_dual_model'):
            assert self.output_functional is not None
            assert self.output_functional.linear
            assert 1 # TODO: how to assert that the operator is symmetric
            self._dual_model = self.with_(rhs=self.output_functional.H)
        return self._dual_model
    
    def solution_sensitivity(self, parameter, index, mu, U=None): 
        if U is None:
            U = self.solve(mu)
        residual_dmu_lhs = VectorOperator(self.operator.d_mu(parameter, index).apply(U, mu=mu))
        residual_dmu_rhs = self.rhs.d_mu(parameter, index)
        rhs_operator = residual_dmu_rhs-residual_dmu_lhs
        return self.operator.apply_inverse(rhs_operator.as_range_array(mu), mu=mu)

    def solve_dual(self, mu):
        return self.dual_model.solve(mu)

    def output_functional_gradient(self, mu, U=None, P=None, adjoint_approach=True):
        if U is None:
            U = self.solve(mu)
        gradient = []
        if adjoint_approach:
            if P is None:
                P = self.solve_dual(mu)
        for (parameter, size) in self.parameters.items(): 
            for index in range(size):
                output_partial_dmu = self.output_functional.d_mu(parameter, index).apply(U, mu=mu).to_numpy()[0,0]
                if adjoint_approach:
                    residual_dmu_lhs = self.operator.d_mu(parameter, index).apply2(U, P, mu=mu)             
                    residual_dmu_rhs = self.rhs.d_mu(parameter, index).apply_adjoint(P, mu=mu).to_numpy()[0,0]
                    gradient.append((output_partial_dmu + residual_dmu_rhs - residual_dmu_lhs)[0,0])
                else:
                    primal_sensitivity = self.primal_sensitivity(parameter, index, mu, U=U)
                    gradient.append(output_partial_dmu + \
                            self.output_functional.apply(primal_sensitivity, mu).to_numpy()[0,0])
        return np.array(gradient)
