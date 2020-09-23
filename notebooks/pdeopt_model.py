import numpy as np

from pymor.models.basic import StationaryModel
from pymor.operators.constructions import VectorOperator

class LinearPdeoptStationaryModel(StationaryModel):
    
    def __init__(self, primal_model, name='LinearPdeoptModel'):
        super().__init__(primal_model.operator, primal_model.rhs, primal_model.output_functional, 
                primal_model.products, primal_model.error_estimator, primal_model.visualizer, name)
        self.__auto_init(locals())
        assert primal_model.output_functional.linear
        self.dual_model = primal_model.with_(rhs=self.output_functional.H)

    def primal_sensitivity(self, parameter, index, mu, U=None):
        if U is None:
            U = self.solve(mu)
        residual_dmu_lhs = VectorOperator(self.primal_model.operator.d_mu(parameter, index).apply(U, mu=mu))
        residual_dmu_rhs = self.primal_model.rhs.d_mu(parameter, index)
        rhs_operator = residual_dmu_rhs-residual_dmu_lhs
        return self.primal_model.operator.apply_inverse(rhs_operator.as_range_array(mu), mu=mu)

    def solve_dual(self, mu):
        return self.dual_model.solve(mu)

    def output_functional_gradient(self, mu, U=None, P=None, adjoint_approach=True):
        if U is None:
            U = self.solve(mu)
        gradient = []
        if adjoint_approach:
            if P is None:
                P = self.solve_dual(mu)
        for (parameter, size) in self.primal_model.parameters.items():             
            for index in range(size):
                output_partial_dmu = self.output_functional.d_mu(parameter, index).apply(U, mu=mu).to_numpy()[0][0]
                if adjoint_approach:
                    residual_dmu_lhs = self.primal_model.operator.d_mu(parameter, index).apply2(U, P, mu=mu)             
                    residual_dmu_rhs = self.primal_model.rhs.d_mu(parameter, index).apply_adjoint(P, mu=mu).to_numpy()[0][0]
                    gradient.append((output_partial_dmu + residual_dmu_rhs - residual_dmu_lhs)[0][0])
                else:
                    primal_sensitivity = self.primal_sensitivity(parameter, index, mu, U=U)
                    gradient.append(output_partial_dmu + self.output_functional.apply(primal_sensitivity, mu).to_numpy()[0][0])
        return np.array(gradient)


