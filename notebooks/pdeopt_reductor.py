from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.core.exceptions import ExtensionError
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
        rom = StationaryModel(error_estimator=error_estimator, **projected_operators)
        return LinearPdeoptStationaryModel(rom)

    def extend_primal_and_dual_basis(self, U, P, basis='RB', method='gram_schmidt', copy_U=True):
        basis_length = len(self.bases[basis])

        extend_primal_and_dual_basis(U, P, self.bases[basis], self.products.get(basis), method=method, copy_U=copy_U)

        self._check_orthonormality(basis, basis_length)

    def extend_basis(self, mu):
        U = self.fom.solve(mu)
        P = self.fom.solve_dual(mu)
        self.extend_primal_and_dual_basis(U, P)

def extend_primal_and_dual_basis(U, P, basis, product=None, method='gram_schmidt', pod_modes=1, copy_U=True):
    assert method in ('trivial', 'gram_schmidt', 'pod')

    basis_length = len(basis)

    if method == 'trivial':
        return NotImplemented
    elif method == 'gram_schmidt':
        basis.append(U, remove_from_other=(not copy_U))
        basis.append(P, remove_from_other=(not copy_U))
        gram_schmidt(basis, offset=basis_length, product=product, copy=False, check=False)
        print('My basis size is {}'.format(len(basis)))
    elif method == 'pod':
        return NotImplemented

    if len(basis) <= basis_length:
        raise ExtensionError
