# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.base import abstractmethod
from pymor.core.cache import CacheableObject
from pymor.operators.constructions import induced_norm
from pymor.parameters.base import ParametricObject, Mu
from pymor.tools.frozendict import FrozenDict
from pymor.tools.deprecated import Deprecated


class Model(CacheableObject, ParametricObject):
    """Interface for model objects.

    A model object defines a discrete problem
    via its `class` and the |Operators| it contains.
    Furthermore, models can be
    :meth:`solved <Model.solve>` for given
    |parameter values| resulting in a solution |VectorArray|.

    Attributes
    ----------
    solution_space
        |VectorSpace| of the solution |VectorArrays| returned by :meth:`solve`.
    output_space
        |VectorSpace| of the model output |VectorArrays| returned by
        :meth:`output` (typically `NumpyVectorSpace(k)` where `k` is a small).
    linear
        `True` if the model describes a linear problem.
    products
        Dict of inner product operators associated with the model.
    """

    solution_space = None
    output_space = None
    linear = False
    products = FrozenDict()

    def __init__(self, products=None, error_estimator=None, visualizer=None,
                 name=None, **kwargs):
        products = FrozenDict(products or {})
        if products:
            for k, v in products.items():
                setattr(self, f'{k}_product', v)
                setattr(self, f'{k}_norm', induced_norm(v))

        self.__auto_init(locals())

    @abstractmethod
    def _solve(self, mu=None, return_output=False, **kwargs):
        """Perform the actual solving."""
        pass

    def solve(self, mu=None, return_output=False, **kwargs):
        """Solve the discrete problem for the |parameter values| `mu`.

        The result will be :mod:`cached <pymor.core.cache>`
        in case caching has been activated for the given model.

        Parameters
        ----------
        mu
            |Parameter values| for which to solve.
        return_output
            If `True`, the model output for the given |parameter values| `mu` is
            returned as a |VectorArray| from :attr:`output_space`.

        Returns
        -------
        The solution |VectorArray|. When `return_output` is `True`,
        the output |VectorArray| is returned as second value.
        """
        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        assert self.parameters.assert_compatible(mu)
        return self.cached_method_call(self._solve, mu=mu, return_output=return_output, **kwargs)

    def solution_sensitivity(self, parameter, index, mu, U=None):
        """Solve for the derivative of the solution w.r.t. a parameter index

        Parameters
        ----------
        parameter
            parameter for which to compute the sensitivity
        index
            parameter index for which to compute the sensitivity
        mu
            |Parameter value| for which to solve
        U
            |VectorArray| containing the solutions for the |Parameter values| mu

        Return
        ------
        The sensitivity of the solution as a |VectorArray|.
        """
        return NotImplemented

    @property
    def dual_model(self):
        """Instantiate the dual model which is used to solve for a dual solution of the |Model|."""
        return NotImplemented

    def solve_dual(self, mu):
        """Solve the dual problem for the |parameter values| `mu`.

        Parameters
        ----------
        mu
            |Parameter value| for which to solve.

        Returns
        -------
        The dual solution |VectorArray|.
        """
        return self.dual_model.solve(mu)

    def output(self, mu=None, **kwargs):
        """Return the model output for given |parameter values| `mu`.

        Parameters
        ----------
        mu
            |Parameter values| for which to compute the output.

        Returns
        -------
        The computed model output as a |VectorArray| from `output_space`.
        """
        return self.solve(mu=mu, return_output=True, **kwargs)[1]

    def output_gradient(self, mu, U=None, P=None, adjoint_approach=True):
        """compute the gradient w.r.t. the parameter of the output functional

        Parameters
        ----------
        mu
            |Parameters value| for which to compute the gradient
        U
            |VectorArray| containing the solutions for the |Parameter values| mu
        P
            |VectorArray| containing the dual solutions for the |Parameter values| mu
        adjoint_approach
            decided whether to use the adjoint approach for computing the gradient
            True: use dual solution
            False: use sensitivities of solutions
        """
        return NotImplemented

    def estimate_error(self, U, mu=None):
        """Estimate the model error for a given solution.

        The model error could be the error w.r.t. the analytical
        solution of the given problem or the model reduction error w.r.t.
        a corresponding high-dimensional |Model|.

        Parameters
        ----------
        U
            The solution obtained by :meth:`~solve`.
        mu
            |Parameter values| for which `U` has been obtained.

        Returns
        -------
        The estimated error.
        """
        if getattr(self, 'error_estimator') is not None:
            return self.error_estimator.estimate_error(U, mu=mu, m=self)
        else:
            raise NotImplementedError('Model has no error estimator.')

    @Deprecated('estimate_error')
    def estimate(self, U, mu=None):
        return self.estimate_error(U, mu)

    def visualize(self, U, **kwargs):
        """Visualize a solution |VectorArray| U.

        Parameters
        ----------
        U
            The |VectorArray| from
            :attr:`~pymor.models.interface.Model.solution_space`
            that shall be visualized.
        kwargs
            See docstring of `self.visualizer.visualize`.
        """
        if getattr(self, 'visualizer') is not None:
            return self.visualizer.visualize(U, self, **kwargs)
        else:
            raise NotImplementedError('Model has no visualizer.')
