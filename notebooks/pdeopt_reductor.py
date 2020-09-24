from pymor.models.basic import StationaryModel
from pymor.reductors.coercive import CoerciveRBReductor

from pdeopt_model import LinearPdeoptStationaryModel

class LinearPdeoptStationaryCoerciveRBReductor(CoerciveRBReductor):
    
    def __init__(self, fom, RB=None, product=None, coercivity_estimator=None,
                 check_orthonormality=None, check_tol=None):
        super().__init__(fom, RB, product, coercivity_estimator,
                 check_orthonormality, check_tol)
        assert isinstance(fom, LinearPdeoptStationaryModel)
    
    def build_rom(self, projected_operators, error_estimator):
        return LinearPdeoptStationaryModel(error_estimator=error_estimator, **projected_operators)
