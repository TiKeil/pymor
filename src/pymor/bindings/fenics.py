# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config


if config.HAVE_FENICS:
    import dolfin as df
    import ufl
    import numpy as np

    from pymor.core.defaults import defaults
    from pymor.core.interfaces import BasicInterface
    from pymor.operators.basic import OperatorBase
    from pymor.operators.numpy import NumpyMatrixOperator
    from pymor.vectorarrays.interfaces import _create_random_values
    from pymor.vectorarrays.list import CopyOnWriteVector, ListVectorSpace
    from pymor.vectorarrays.numpy import NumpyVectorSpace

    class FenicsVector(CopyOnWriteVector):
        """Wraps a FEniCS vector to make it usable with ListVectorArray."""

        def __init__(self, impl):
            self.impl = impl

        @classmethod
        def from_instance(cls, instance):
            return cls(instance.impl)

        def _copy_data(self):
            self.impl = self.impl.copy()

        def to_numpy(self, ensure_copy=False):
            if ensure_copy:
                return self.impl.copy().get_local()
            return self.impl.get_local()

        def _scal(self, alpha):
            self.impl *= alpha

        def _axpy(self, alpha, x):
            if x is self:
                self.scal(1. + alpha)
            else:
                self.impl.axpy(alpha, x.impl)

        def dot(self, other):
            return self.impl.inner(other.impl)

        def l1_norm(self):
            return self.impl.norm('l1')

        def l2_norm(self):
            return self.impl.norm('l2')

        def l2_norm2(self):
            return self.impl.norm('l2') ** 2

        def sup_norm(self):
            return self.impl.norm('linf')

        def dofs(self, dof_indices):
            dof_indices = np.array(dof_indices, dtype=np.intc)
            if len(dof_indices) == 0:
                return np.array([], dtype=np.intc)
            assert 0 <= np.min(dof_indices)
            assert np.max(dof_indices) < self.impl.size()
            dofs = self.impl.gather(dof_indices)
            # in the mpi distributed case, gather returns the values
            # at the *global* dof_indices on each rank
            return dofs

        def amax(self):
            A = np.abs(self.impl.get_local())
            # there seems to be no way in the interface to compute amax without making a copy.
            max_ind_on_rank = np.argmax(A)
            max_val_on_rank = A[max_ind_on_rank]
            from pymor.tools import mpi
            if not mpi.parallel:
                return max_ind_on_rank, max_val_on_rank
            else:
                max_global_ind_on_rank = max_ind_on_rank + self.impl.local_range()[0]
                comm = self.impl.mpi_comm()
                comm_size = comm.Get_size()

                max_inds = np.empty(comm_size, dtype='i')
                comm.Allgather(np.array(max_global_ind_on_rank, dtype='i'), max_inds)

                max_vals = np.empty(comm_size, dtype=np.float64)
                comm.Allgather(np.array(max_val_on_rank), max_vals)

                i = np.argmax(max_inds)
                return max_inds[i], max_vals[i]

        def __add__(self, other):
            return FenicsVector(self.impl + other.impl)

        def __iadd__(self, other):
            self._copy_data_if_needed()
            self.impl += other.impl
            return self

        __radd__ = __add__

        def __sub__(self, other):
            return FenicsVector(self.impl - other.impl)

        def __isub__(self, other):
            self._copy_data_if_needed()
            self.impl -= other.impl
            return self

        def __mul__(self, other):
            return FenicsVector(self.impl * other)

        def __neg__(self):
            return FenicsVector(-self.impl)

    class FenicsVectorSpace(ListVectorSpace):

        def __init__(self, V, id_='STATE'):
            self.V = V
            self.id = id_

        @property
        def dim(self):
            return df.Function(self.V).vector().size()

        def __eq__(self, other):
            return type(other) is FenicsVectorSpace and self.V == other.V and self.id == other.id

        # since we implement __eq__, we also need to implement __hash__
        def __hash__(self):
            return id(self.V) + hash(self.id)

        def zero_vector(self):
            impl = df.Function(self.V).vector()
            return FenicsVector(impl)

        def full_vector(self, value):
            impl = df.Function(self.V).vector()
            impl += value
            return FenicsVector(impl)

        def random_vector(self, distribution, random_state, **kwargs):
            impl = df.Function(self.V).vector()
            values = _create_random_values(impl.local_size(), distribution, random_state, **kwargs)
            impl[:] = values
            return FenicsVector(impl)

        def make_vector(self, obj):
            return FenicsVector(obj)

    class FenicsMatrixOperator(OperatorBase):
        """Wraps a FEniCS matrix as an |Operator|."""

        linear = True

        def __init__(self, matrix, source_space, range_space, solver_options=None, name=None):
            assert matrix.rank() == 2
            self.source = FenicsVectorSpace(source_space)
            self.range = FenicsVectorSpace(range_space)
            self.matrix = matrix
            self.solver_options = solver_options
            self.name = name

        def apply(self, U, mu=None):
            assert U in self.source
            R = self.range.zeros(len(U))
            for u, r in zip(U._list, R._list):
                self.matrix.mult(u.impl, r.impl)
            return R

        def apply_adjoint(self, V, mu=None):
            assert V in self.range
            U = self.source.zeros(len(V))
            for v, u in zip(V._list, U._list):
                self.matrix.transpmult(v.impl, u.impl)  # there are no complex numbers in FEniCS
            return U

        def apply_inverse(self, V, mu=None, least_squares=False):
            assert V in self.range
            if least_squares:
                raise NotImplementedError
            R = self.source.zeros(len(V))
            options = self.solver_options.get('inverse') if self.solver_options else None
            for r, v in zip(R._list, V._list):
                _apply_inverse(self.matrix, r.impl, v.impl, options)
            return R

        def _assemble_lincomb(self, operators, coefficients, identity_shift=0., solver_options=None, name=None):
            if not all(isinstance(op, FenicsMatrixOperator) for op in operators):
                return None
            if identity_shift != 0:
                return None
            assert not solver_options

            if coefficients[0] == 1:
                matrix = operators[0].matrix.copy()
            else:
                matrix = operators[0].matrix * coefficients[0]
            for op, c in zip(operators[1:], coefficients[1:]):
                matrix.axpy(c, op.matrix, False)
                # in general, we cannot assume the same nonzero pattern for # all matrices. how to improve this?

            return FenicsMatrixOperator(matrix, self.source.V, self.range.V, name=name)

    class FenicsOperator(OperatorBase):
        """Wraps a FEniCS form as an |Operator|."""

        linear = False

        @defaults('restriction_method')
        def __init__(self, form, source_space, range_space, source_function, dirichlet_bc=None,
                     parameter_setter=None, parameter_type=None, solver_options=None,
                     restriction_method='submesh', name=None):
            assert restriction_method in ('assemble_local', 'submesh')
            assert len(form.arguments()) == 1
            self.form = form
            self.source = source_space
            self.range = range_space
            self.source_function = source_function
            self.dirichlet_bc = dirichlet_bc
            self.parameter_setter = parameter_setter
            self.build_parameter_type(parameter_type)
            self.solver_options = solver_options
            self.restriction_method = restriction_method
            self.name = name

        def _set_mu(self, mu=None):
            mu = self.parse_parameter(mu)
            if self.parameter_setter:
                self.parameter_setter(mu)

        def apply(self, U, mu=None):
            assert U in self.source
            self._set_mu(mu)
            R = []
            source_vec = self.source_function.vector()
            for u in U._list:
                source_vec[:] = u.impl
                r = df.assemble(self.form)
                if self.dirichlet_bc:
                    self.dirichlet_bc.apply(r, source_vec)
                R.append(r)
            return self.range.make_array(R)

        def jacobian(self, U, mu=None):
            assert U in self.source and len(U) == 1
            self._set_mu(mu)
            source_vec = self.source_function.vector()
            source_vec[:] = U._list[0].impl
            matrix = df.assemble(df.derivative(self.form, self.source_function))
            if self.dirichlet_bc:
                self.dirichlet_bc.apply(matrix)
            return FenicsMatrixOperator(matrix, self.source.V, self.range.V)

        def restricted(self, dofs):
            assert self.source.V.mesh().id() == self.range.V.mesh().id()

            # first determine affected cells
            self.logger.info('Computing affected cells ...')
            mesh = self.source.V.mesh()
            range_dofmap = self.range.V.dofmap()
            affected_cell_indices = set()
            for c in df.cells(mesh):
                cell_index = c.index()
                local_dofs = range_dofmap.cell_dofs(cell_index)
                for ld in local_dofs:
                    if ld in dofs:
                        affected_cell_indices.add(cell_index)
                        continue
            affected_cell_indices = list(sorted(affected_cell_indices))
            affected_cells = [df.Cell(mesh, ci) for ci in affected_cell_indices]

            # increase stencil if needed
            # TODO

            # determine source dofs
            self.logger.info('Computing source DOFs ...')
            source_dofmap = self.source.V.dofmap()
            source_dofs = set()
            for cell_index in affected_cell_indices:
                local_dofs = source_dofmap.cell_dofs(cell_index)
                source_dofs.update(local_dofs)
            source_dofs = np.array(sorted(source_dofs), dtype=np.intc)

            if self.restriction_method == 'assemble_local':
                # range local-to-restricted dof mapping
                to_restricted = np.zeros(self.range.dim, dtype=np.int32)
                to_restricted[:] = len(dofs)
                to_restricted[dofs] = np.arange(len(dofs))
                range_local_restricted = np.array([to_restricted[range_dofmap.cell_dofs(ci)]
                                                   for ci in affected_cell_indices])

                # source local-to-restricted dof mapping
                to_restricted = np.zeros(self.source.dim, dtype=np.int32)
                to_restricted[:] = len(source_dofs)
                to_restricted[source_dofs] = np.arange(len(source_dofs))
                source_local_restricted = np.array([to_restricted[source_dofmap.cell_dofs(ci)]
                                                   for ci in affected_cell_indices])

                # compute dirichlet DOFs
                if self.dirichlet_bc:
                    self.logger.warn('Dirichlet DOF handling will only work for constant, non-paramentric '
                                     'Dirichlet boundary conditions')
                    v1 = self.source.zeros()._list[0].impl
                    v1[:] = 42
                    v2 = self.source.zeros()._list[0].impl
                    v2[:] = 0
                    self.dirichlet_bc.apply(v1)
                    self.dirichlet_bc.apply(v2)
                    dir_dofs = [i for i in range(self.source.dim) if (v1[i] != 42) or (v2[i] != 0)]
                    dir_dofs_r, dir_vals_r = zip(*((i, v1[dof]) for i, dof in enumerate(dofs) if dof in dir_dofs))
                    dir_dofs_r = np.array(dir_dofs_r, dtype=np.int32)
                    dir_vals_r = np.array(dir_vals_r)
                    dir_dofs_r_source = to_restricted[dofs[dir_dofs_r]]
                else:
                    dir_dofs_r = None
                    dir_vals_r = None
                    dir_dofs_r_source = None

                return (
                    RestrictedFenicsOperatorAssembleLocal(self, np.array(dofs), source_dofs.copy(), affected_cells,
                                                          source_local_restricted, range_local_restricted,
                                                          dir_dofs_r, dir_vals_r, dir_dofs_r_source),
                    source_dofs
                )

            elif self.restriction_method == 'submesh':
                # generate restricted spaces
                self.logger.info('Building submesh ...')
                subdomain = df.MeshFunction('size_t', mesh, mesh.geometry().dim())
                for ci in affected_cell_indices:
                    subdomain.set_value(ci, 1)
                submesh = df.SubMesh(mesh, subdomain, 1)

                # build restricted form
                self.logger.info('Building UFL form on submesh ...')
                V_r_source = df.FunctionSpace(submesh, self.source.V.ufl_element())
                V_r_range = df.FunctionSpace(submesh, self.range.V.ufl_element())
                assert V_r_source.dim() == len(source_dofs)

                if self.source.V != self.range.V:
                    assert all(arg.ufl_function_space() != self.source.V for arg in self.form.arguments())
                args = tuple((df.function.argument.Argument(V_r_range, arg.number(), arg.part())
                              if arg.ufl_function_space() == self.range.V else arg)
                             for arg in self.form.arguments())
                if any(isinstance(coeff, df.Function) and coeff != self.source_function for coeff in
                       self.form.coefficients()):
                    raise NotImplementedError
                source_function_r = df.Function(V_r_source)
                form_r = ufl.replace_integral_domains(
                    self.form(*args, coefficients={self.source_function: source_function_r}),
                    submesh.ufl_domain()
                )
                if self.dirichlet_bc:
                    bc = self.dirichlet_bc
                    if not bc.user_subdomain():
                        raise NotImplementedError
                    bc_r = df.DirichletBC(V_r_source, bc.value(), bc.user_subdomain(), bc.method())
                else:
                    bc_r = None

                # source dof mapping
                self.logger.info('Computing source DOF mapping ...')
                u = df.Function(self.source.V)
                u_vec = u.vector()
                restricted_source_dofs = []
                for source_dof in source_dofs:
                    u_vec.zero()
                    u_vec[source_dof] = 1
                    u_r = df.interpolate(u, V_r_source)
                    u_r = u_r.vector().get_local()
                    if not np.all(np.logical_or(np.abs(u_r) < 1e-10, np.abs(u_r - 1.) < 1e-10)):
                        raise NotImplementedError
                    r_dof = np.where(np.abs(u_r - 1.) < 1e-10)[0]
                    if not len(r_dof) == 1:
                        raise NotImplementedError
                    restricted_source_dofs.append(r_dof[0])
                restricted_source_dofs = np.array(restricted_source_dofs, dtype=np.int32)
                assert len(set(restricted_source_dofs)) == len(source_dofs)

                # source dof mapping
                self.logger.info('Computing range DOF mapping ...')
                u = df.Function(self.range.V)
                u_vec = u.vector()
                restricted_range_dofs = []
                for range_dof in dofs:
                    u_vec.zero()
                    u_vec[range_dof] = 1
                    u_r = df.interpolate(u, V_r_range)
                    u_r = u_r.vector().get_local()
                    if not np.all(np.logical_or(np.abs(u_r) < 1e-10, np.abs(u_r - 1.) < 1e-10)):
                        raise NotImplementedError
                    r_dof = np.where(np.abs(u_r - 1.) < 1e-10)[0]
                    if not len(r_dof) == 1:
                        raise NotImplementedError
                    restricted_range_dofs.append(r_dof[0])
                restricted_range_dofs = np.array(restricted_range_dofs, dtype=np.int32)

                op_r = FenicsOperator(form_r, FenicsVectorSpace(V_r_source), FenicsVectorSpace(V_r_range),
                                      source_function_r, dirichlet_bc=bc_r, parameter_setter=self.parameter_setter,
                                      parameter_type=self.parameter_type)

                return (RestrictedFenicsOperatorSubMesh(op_r, restricted_range_dofs),
                        source_dofs[np.argsort(restricted_source_dofs)])
            else:
                assert False

    class RestrictedFenicsOperatorAssembleLocal(OperatorBase):

        linear = False

        def __init__(self, operator, range_dofs, source_dofs, cells, source_local_restricted, range_local_restricted,
                     dirichlet_dofs, dirichlet_values, dirichlet_source_dofs):
            self.source = NumpyVectorSpace(len(source_dofs))
            self.range = NumpyVectorSpace(len(range_dofs))
            self.operator = operator
            self.range_dofs = range_dofs
            self.source_dofs = source_dofs
            self.cells = cells
            self.source_local_restricted = source_local_restricted
            self.range_local_restricted = range_local_restricted
            self.dirichlet_dofs = dirichlet_dofs
            self.dirichlet_values = dirichlet_values
            self.dirichlet_source_dofs = dirichlet_source_dofs
            self.build_parameter_type(operator)

        def apply(self, U, mu=None):
            assert U in self.source
            operator = self.operator
            source_vec = operator.source_function.vector()
            operator._set_mu(mu)
            R = np.zeros((len(U), self.range.dim + 1))
            for u, r in zip(U.data, R):
                source_vec[self.source_dofs] = u
                for cell, local_restricted in zip(self.cells, self.range_local_restricted):
                    local_evaluations = df.assemble_local(operator.form, cell)
                    r[local_restricted] += local_evaluations
                r[self.dirichlet_dofs] = u[self.dirichlet_source_dofs] - self.dirichlet_values
            return self.range.make_array(R[:, :-1])

        def jacobian(self, U, mu=None):
            assert U in self.source and len(U) == 1
            operator = self.operator
            source_vec = operator.source_function.vector()
            operator._set_mu(mu)
            J = np.zeros((self.range.dim + 1, self.source.dim + 1))
            source_vec[self.source_dofs] = U.data[0]
            for cell, range_local_restricted, source_local_restricted in zip(self.cells,
                                                                             self.range_local_restricted,
                                                                             self.source_local_restricted):
                local_matrix = df.assemble_local(df.derivative(operator.form, operator.source_function), cell)
                J[np.meshgrid(range_local_restricted, source_local_restricted, indexing='ij')] += local_matrix
            J[self.dirichlet_dofs, :] = 0.
            J[np.meshgrid(self.dirichlet_dofs, self.dirichlet_source_dofs, indexing='ij')] = 1.
            return NumpyMatrixOperator(J[:-1, :-1])

        def restricted(self, dofs):
            raise NotImplementedError

    class RestrictedFenicsOperatorSubMesh(OperatorBase):

        linear = False

        def __init__(self, op, restricted_range_dofs):
            self.source = NumpyVectorSpace(op.source.dim)
            self.range = NumpyVectorSpace(len(restricted_range_dofs))
            self.op = op
            self.restricted_range_dofs = restricted_range_dofs
            self.build_parameter_type(op)

        def apply(self, U, mu=None):
            assert U in self.source
            UU = self.op.source.zeros(len(U))
            for uu, u in zip(UU._list, U.data):
                uu.impl[:] = u
            VV = self.op.apply(UU, mu=mu)
            V = self.range.zeros(len(VV))
            for v, vv in zip(V.data, VV._list):
                v[:] = vv.impl[self.restricted_range_dofs]
            return V

        def jacobian(self, U, mu=None):
            assert U in self.source and len(U) == 1
            UU = self.op.source.zeros()
            UU._list[0].impl[:] = U.data[0]
            JJ = self.op.jacobian(UU, mu=mu)
            return NumpyMatrixOperator(JJ.matrix.array()[self.restricted_range_dofs, :])

    @defaults('solver', 'preconditioner')
    def _solver_options(solver='bicgstab', preconditioner='amg'):
        return {'solver': solver, 'preconditioner': preconditioner}

    def _apply_inverse(matrix, r, v, options=None):
        options = options or _solver_options()
        solver = options.get('solver')
        preconditioner = options.get('preconditioner')
        # preconditioner argument may only be specified for iterative solvers:
        options = (solver, preconditioner) if preconditioner else (solver,)
        df.solve(matrix, r, v, *options)

    class FenicsVisualizer(BasicInterface):
        """Visualize a FEniCS grid function.

        Parameters
        ----------
        space
            The `FenicsVectorSpace` for which we want to visualize DOF vectors.
        mesh_refinements
            Number of uniform mesh refinements to perform for vtk visualization
            (of functions from higher-order FE spaces).
        """

        def __init__(self, space, mesh_refinements=0):
            self.space = space
            self.mesh_refinements = mesh_refinements

        def visualize(self, U, m, title='', legend=None, filename=None, block=True,
                      separate_colorbars=True):
            """Visualize the provided data.

            Parameters
            ----------
            U
                |VectorArray| of the data to visualize (length must be 1). Alternatively,
                a tuple of |VectorArrays| which will be visualized in separate windows.
                If `filename` is specified, only one |VectorArray| may be provided which,
                however, is allowed to contain multipled vectors that will be interpreted
                as a time series.
            m
                Filled in by :meth:`pymor.models.ModelBase.visualize` (ignored).
            title
                Title of the plot.
            legend
                Description of the data that is plotted. If `U` is a tuple of |VectorArrays|,
                `legend` has to be a tuple of the same length.
            filename
                If specified, write the data to that file. `filename` needs to have an extension
                supported by FEniCS (e.g. `.pvd`).
            separate_colorbars
                If `True`, use separate colorbars for each subplot.
            block
                If `True`, block execution until the plot window is closed.
            """
            if filename:
                assert not isinstance(U, tuple)
                assert U in self.space
                f = df.File(filename)
                coarse_function = df.Function(self.space.V)
                if self.mesh_refinements:
                    mesh = self.space.V.mesh()
                    for _ in range(self.mesh_refinements):
                        mesh = df.refine(mesh)
                    V_fine = df.FunctionSpace(mesh, self.space.V.ufl_element())
                    function = df.Function(V_fine)
                else:
                    function = coarse_function
                if legend:
                    function.rename(legend, legend)
                for u in U._list:
                    coarse_function.vector()[:] = u.impl
                    if self.mesh_refinements:
                        function.vector()[:] = df.interpolate(coarse_function, V_fine).vector()
                    f << function
            else:
                from matplotlib import pyplot as plt

                assert U in self.space and len(U) == 1 \
                    or (isinstance(U, tuple) and all(u in self.space for u in U) and all(len(u) == 1 for u in U))
                if not isinstance(U, tuple):
                    U = (U,)
                if isinstance(legend, str):
                    legend = (legend,)
                assert legend is None or len(legend) == len(U)

                if not separate_colorbars:
                    vmin = np.inf
                    vmax = -np.inf
                    for u in U:
                        vec = u._list[0].impl
                        vmin = min(vmin, vec.min())
                        vmax = max(vmax, vec.max())

                for i, u in enumerate(U):
                    function = df.Function(self.space.V)
                    function.vector()[:] = u._list[0].impl
                    if legend:
                        tit = title + ' -- ' if title else ''
                        tit += legend[i]
                    else:
                        tit = title
                    if separate_colorbars:
                        plt.figure()
                        df.plot(function, title=tit)
                    else:
                        plt.figure()
                        df.plot(function, title=tit,
                                range_min=vmin, range_max=vmax)
                plt.show(block=block)
