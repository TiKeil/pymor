import numpy as np

from pymor.algorithms.greedy import WeakGreedySurrogate, weak_greedy, _rb_surrogate_evaluate
from pymor.parallel.dummy import dummy_pool
from pymor.parallel.interface import RemoteObject
from pdeopt_reductor import LinearPdeoptStationaryCoerciveRBReductor
from pdeopt_model import LinearPdeoptStationaryModel

def pdeopt_greedy(fom, reductor, training_set, use_error_estimator=True, error_norm=None,
              atol=None, rtol=None, max_extensions=None, extension_params=None, pool=None):

    surrogate = PdeoptSurrogate(fom, reductor, use_error_estimator, error_norm, extension_params, pool or dummy_pool)

    result = weak_greedy(surrogate, training_set, atol=atol, rtol=rtol, max_extensions=max_extensions, pool=pool)
    result['rom'] = surrogate.rom

    return result

class PdeoptSurrogate(WeakGreedySurrogate):
    """Surrogate for the :func:`weak_greedy` error used in :func:`rb_greedy`.
    Not intended to be used directly.
    """

    def __init__(self, fom, reductor, use_error_estimator, error_norm, extension_params, pool):
        assert isinstance(fom, LinearPdeoptStationaryModel)
        assert isinstance(reductor, LinearPdeoptStationaryCoerciveRBReductor)
        self.__auto_init(locals())
        if use_error_estimator:
            self.remote_fom, self.remote_error_norm, self.remote_reductor = None, None, None
        else:
            self.remote_fom, self.remote_error_norm, self.remote_reductor = \
                pool.push(fom), pool.push(error_norm), pool.push(reductor)
        self.rom = None

    def evaluate(self, mus, return_all_values=False):
        if self.rom is None:
            with self.logger.block('Reducing ...'):
                self.rom = self.reductor.reduce()

        if not isinstance(mus, RemoteObject):
            mus = self.pool.scatter_list(mus)

        result = self.pool.apply(_rb_surrogate_evaluate,
                                 rom=self.rom,
                                 fom=self.remote_fom,
                                 reductor=self.remote_reductor,
                                 mus=mus,
                                 error_norm=self.remote_error_norm,
                                 return_all_values=return_all_values)
        if return_all_values:
            return np.hstack(result)
        else:
            errs, max_err_mus = list(zip(*result))
            max_err_ind = np.argmax(errs)
            return errs[max_err_ind], max_err_mus[max_err_ind]

    def extend(self, mu):
        with self.logger.block(f'Computing primal solution snapshot for mu = {mu} ...'):
            U = self.fom.solve(mu)
        with self.logger.block(f'Computing dual solution snapshot for mu = {mu} ...'):
            P = self.fom.solve_dual(mu)
        with self.logger.block('Extending basis with solution snapshot ...'):
            extension_params = self.extension_params
            if len(U) > 1 and extension_params is None:
                extension_params = {'method': 'pod'}
            self.reductor.extend_primal_and_dual_basis(U, P, copy_U=False, **(extension_params or {}))
            if not self.use_error_estimator:
                self.remote_reductor = self.pool.push(self.reductor)
        with self.logger.block('Reducing ...'):
            self.rom = self.reductor.reduce()
