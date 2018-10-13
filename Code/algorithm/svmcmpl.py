# Copyright 2010-2018 M. S. Andersen & L. Vandenberghe
#
# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this software. If not, see <http://www.gnu.org/licenses/>.

from cvxopt import base, blas, lapack, solvers
from cvxopt.base import matrix, spmatrix, exp, mul, spdiag, div, sparse
import cvxopt, chompack

from os import times
def cputime(T0 = (0.0, 0.0)):
    """
    Returns tuple (utime, stime) with CPU time spent since start.

    CPU time since T0 = (utime0, stime0) is returned if
    the optional argument T0 is supplied.
    """
    T = times()
    return (T[0]-T0[0], T[1]-T0[1])

# relative threshold for support vector selection
Tsv = 1e-5

weights = 'equal' # 'equal' or 'proportional'
verbose = True

def kernel_matrix(X, kernel, sigma = 1.0, theta = 1.0, degree = 1, V = None, width = None):
    """
    Computes the kernel matrix or a partial kernel matrix.

    Input arguments.

        X is an N x n matrix.

        kernel is a string with values 'linear', 'rfb', 'poly', or 'tanh'.
        'linear': k(u,v) = u'*v/sigma.
        'rbf':    k(u,v) = exp(-||u - v||^2 / (2*sigma)).
        'poly':   k(u,v) = (u'*v/sigma)**degree.
        'tanh':   k(u,v) = tanh(u'*v/sigma - theta).        kernel is a

        sigma and theta are positive numbers.

        degree is a positive integer.

        V is an N x N sparse matrix (default is None).

        width is a positive integer (default is None).

    Output.

        Q, an N x N matrix or sparse matrix.
        If V is a sparse matrix, a partial kernel matrix with the sparsity
        pattern V is returned.
        If width is specified and V = 'band', a partial kernel matrix
        with band sparsity is returned (width is the half-bandwidth).

        a, an N x 1 matrix with the products <xi,xi>/sigma.

    """
    N,n = X.size

    #### dense (full) kernel matrix
    if V is None:
        if verbose: print("building kernel matrix ..")

        # Qij = xi'*xj / sigma
        Q = matrix(0.0, (N,N))
        blas.syrk(X, Q, alpha = 1.0/sigma)
        a = Q[::N+1] # ai = ||xi||**2 / sigma

        if kernel == 'linear':
            pass

        elif kernel == 'rbf':
            # Qij := Qij - 0.5 * ( ai + aj )
            #      = -||xi - xj||^2 / (2*sigma)
            ones = matrix(1.0, (N,1))
            blas.syr2(a, ones, Q, alpha = -0.5)

            Q = exp(Q)

        elif kernel == 'tanh':
            Q = exp(Q - theta)
            Q = div(Q - Q**-1, Q + Q**-1)

        elif kernel == 'poly':
            Q = Q**degree

        else:
            raise ValueError('invalid kernel type')

    #### general sparse partial kernel matrix
    elif type(V) is cvxopt.base.spmatrix:

        if verbose: print("building projected kernel matrix ...")
        Q = +V
        base.syrk(X,Q,partial=True,alpha=1.0/sigma)

        # ai = ||xi||**2 / sigma
        a = matrix(Q[::N+1],(N,1))

        if kernel == 'linear':
            pass

        elif kernel == 'rbf':

            ones = matrix(1.0, (N,1))

            # Qij := Qij - 0.5 * ( ai + aj )
            #      = -||xi - xj||^2 / (2*sigma)
            p = chompack.maxcardsearch(V)
            symb = chompack.symbolic(Q,p)
            Qc = chompack.cspmatrix(symb) + Q
            chompack.syr2(Qc,a,ones,alpha = -0.5)
            Q = Qc.spmatrix(reordered = False)
            Q.V = exp(Q.V)

        elif kernel == 'tanh':

            v = +Q.V
            v = exp(v - theta)
            v = div(v - v**-1, v + v**-1)
            Q.V = v

        elif kernel == 'poly':

            Q.V = Q.V**degree

        else:
            raise ValueError('invalid kernel type')


    #### banded partial kernel matrix
    elif V == 'band' and width is not None:

        # Lower triangular part of band matrix with bandwidth 2*w+1.
        if verbose: print("building projected kernel matrix ...")
        I = [ i for k in range(N) for i in range(k, min(width+k+1, N)) ]
        J = [ k for k in range(N) for i in range(min(width+1,N-k)) ]
        V = matrix(0.0, (len(I), 1))
        oy = 0
        for k in range(N):   # V[:,k] = Xtrain[k:k+w, :] * Xtrain[k,:].T
            m = min(width+1, N-k)
            blas.gemv(X, X, V, m = m, ldA = N, incx = N, offsetA = k,
                offsetx = k, offsety = oy)
            oy += m
        blas.scal(1.0/sigma,V)

        # ai = ||xi||**2 / sigma
        a = matrix(V[[i for i in range(len(I)) if I[i] == J[i]]],(N,1))

        if kernel == 'linear':

            Q = spmatrix(V, I, J, (N,N))

        elif kernel == 'rbf':

            Q = spmatrix(V, I, J, (N,N))

            ones = matrix(1.0, (N,1))

            # Qij := Qij - 0.5 * ( ai + aj )
            #      = -||xi - xj||^2 / (2*sigma)
            symb = chompack.symbolic(Q)
            Qc = chompack.cspmatrix(symb) + Q
            chompack.syr2(Qc,a,ones,alpha = -0.5)
            Q = Qc.spmatrix(reordered = False)
            Q.V = exp(Q.V)

        elif kernel == 'tanh':

            V = exp(V - theta)
            V = div(V - V**-1, V + V**-1)
            Q = spmatrix(V, I, J, (N,N))

        elif kernel == 'poly':

            Q = spmatrix(V**degree, I, J, (N,N))

        else:
            raise ValueError('invalid kernel type')
    else:
        raise TypeError('invalid type V')

    return Q,a


def softmargin(X, d, gamma, kernel = 'linear', sigma = 1.0, degree = 1, theta = 1.0, Q = None):
    """
    Solves the 'soft-margin' SVM problem

        maximize    -(1/2)*z'*Q*z + d'*z
        subject to  0 <= diag(d)*z <= gamma*ones
                    sum(z) = 0

    (with variables z), and its dual problem

        minimize    (1/2)*y'*Q^{-1}*y + gamma*sum(v)
        subject to  diag(d)*(y + b*ones) + v >= 1
                    v >= 0

    (with variables y, v, b).

    Q is given by Q_ij = K(xi, xj) where K is a kernel function and xi is
    the ith row of X (xi' = X[i,:]).  If Q is singular, we replace Q^{-1}
    in the dual with its pseudo-inverse and add a constraint y in Range(Q).
    We can also use make a change of variables y = Q*u to obtain

        minimize    (1/2)*u'*Q*u + gamma*sum(v)
        subject to  diag(d)*(Q*u + b*ones) + v >= 1
                    v >= 0

    For the linear kernel (Q = X*X'), a change of variables w = X'*u allows
    us to write this in the more common form

        minimize    (1/2)*w'*w + gamma*sum(v)
        subject to  diag(d)*(X*w + b*ones) + v >= 1
                    v >= 0.


    Input arguments.

        X is an N x n matrix.

        d is an N-vector with elements -1 or 1; d[i] is the label of
        row X[i,:].

        gamma is a positive parameter.

        kernel is a string with values 'linear', 'rfb', 'poly', or 'tanh'.
        'linear': k(u,v) = u'*v/sigma.
        'rbf':    k(u,v) = exp(-||u - v||^2 / (2*sigma)).
        'poly':   k(u,v) = (u'*v/sigma)**degree.
        'tanh':   k(u,v) = tanh(u'*v/sigma - theta).

        sigma and theta are positive numbers.

        degree is a positive integer.


    Output.

        Returns a dictionary with the keys:

        'classifier'
           a Python function object that takes an M x n matrix with
           test vectors as rows and returns a vector with labels

        'z'
           a sparse m-vector

        'cputime'
           a tuple (Ttot, Tqp, Tker) where Ttot is the total
           CPU time, Tqp is the CPU time spent solving the QP, and
           Tker is the CPU time spent computing the kernel matrix

        'iterations'
           the number of interior-point iteations

        'misclassified'
           a tuple (L1, L2) where L1 is a list of indices of
           misclassified training vectors from class 1, and L2 is a
           list of indices of misclassified training vectors from
           class 2
    """

    Tstart = cputime()

    if verbose: solvers.options['show_progress'] = True
    else: solvers.options['show_progress'] = False
    N, n = X.size

    if Q is None:
        Q,a = kernel_matrix(X, kernel, sigma = sigma, degree = degree, theta = theta)
    else:
        if not (Q.size[0] == N and Q.size[1] == N):
            raise ValueError("invalid kernel matrix dimensions")
        elif not type(Q) is cvxopt.base.matrix:
            raise ValueError("invalid kernel matrix type")
        elif verbose:
            print("using precomputed kernel matrix ..")
        if kernel == 'rbf':
            Ad = spmatrix(0.0,range(N),range(N))
            base.syrk(X,V,partial = True)
            a = Ad.V
            del Ad

    Tkernel = cputime(Tstart)

    # build qp
    Tqp = cputime()
    q = matrix(-d, tc = 'd')
    # Inequality constraints  0 <= diag(d)*z <= gamma*ones
    G = spmatrix([], [], [], size = (2*N, N))
    G[::2*N+1], G[N::2*N+1] = d, -d
    h = matrix(0.0, (2*N,1))

    if weights is 'proportional':
        dlist = list(d)
        C1 = 0.5*N*gamma/dlist.count(1)
        C2 = 0.5*N*gamma/dlist.count(-1)
        gvec = matrix([C1 if w == 1 else C2 for w in dlist],(N,1))
        del dlist
        h[:N] = gvec
    elif weights is 'equal':
        h[:N] = gamma
    else:
        raise ValueError("invalid weight type")

    # solve qp
    sol = solvers.qp(Q, q, G, h, matrix(1.0, (1,N)), matrix(0.0))
    Tqp = cputime(Tqp)
    if verbose: print("utime = %f, stime = %f." % Tqp)

    # extract solution
    z, b, v = sol['x'], sol['y'][0], sol['z'][:N]

    # extract nonzero support vectors
    nrmz = max(abs(z))
    sv = [ k for k in range(N) if abs(z[k]) > Tsv * nrmz ]
    N, X, z = len(sv), X[sv,:], z[sv]
    zs = spmatrix(z,sv,[0 for i in range(N)],(len(d),1))
    if verbose: print("%d support vectors." %N)

    # find misclassified training vectors
    err1 = [i for i in range(Q.size[0]) if (v[i] > 1 and d[i] == 1)]
    err2 = [i for i in range(Q.size[0]) if (v[i] > 1 and d[i] == -1)]
    if verbose:
        e1,n1 = len(err1),list(d).count(1)
        e2,n2 = len(err2),list(d).count(-1)
        print("class 1: %i/%i = %.1f%% misclassified." % (e1,n1,100.*e1/n1))
        print("class 2: %i/%i = %.1f%% misclassified." % (e2,n2,100.*e2/n2))
        del e1,e2,n1,n2

    # create classifier function object
    if kernel == 'linear':

        # w = X'*z / sigma
        w = matrix(0.0, (n,1))
        blas.gemv(X, z, w, trans = 'T', alpha = 1.0/sigma)

        def classifier(Y, soft = False):
            M = Y.size[0]
            x = matrix(b, (M,1))
            blas.gemv(Y, w, x, beta = 1.0)
            if soft: return x
            else:    return matrix([ 2*(xk > 0.0) - 1 for xk in x ])

    elif kernel == 'rbf':

        def classifier(Y, soft = False):
            M = Y.size[0]

            # K = Y*X' / sigma
            K = matrix(0.0, (M, N))
            blas.gemm(Y, X, K, transB = 'T', alpha = 1.0/sigma)

            # c[i] = ||Yi||^2 / sigma
            ones = matrix(1.0, (max([M,N,n]),1))
            c = Y**2 * ones[:n]
            blas.scal(1.0/sigma, c)

            # Kij := Kij - 0.5 * (ci + aj)
            #      = || yi - xj ||^2 / (2*sigma)
            blas.ger(c, ones, K, alpha = -0.5)
            blas.ger(ones, a[sv], K, alpha = -0.5)
            x = exp(K) * z + b
            if soft: return x
            else:    return matrix([ 2*(xk > 0.0) - 1 for xk in x ])

    elif kernel == 'tanh':

        def classifier(Y, soft = False):
            M = Y.size[0]

            # K = Y*X' / sigma - theta
            K = matrix(theta, (M, N))
            blas.gemm(Y, X, K, transB = 'T', alpha = 1.0/sigma, beta = -1.0)

            K = exp(K)
            x = div(K - K**-1, K + K**-1) * z + b
            if soft: return x
            else:    return matrix([ 2*(xk > 0.0) - 1 for xk in x ])

    elif kernel == 'poly':

        def classifier(Y, soft = False):
            M = Y.size[0]

            # K = Y*X' / sigma
            K = matrix(0.0, (M, N))
            blas.gemm(Y, X, K, transB = 'T', alpha = 1.0/sigma)

            x = K**degree * z + b
            if soft: return x
            else:    return matrix([ 2*(xk > 0.0) - 1 for xk in x ])

    Ttotal = cputime(Tstart)

    return {'classifier': classifier,
            'cputime': (sum(Ttotal),sum(Tqp),sum(Tkernel)),
            'iterations':sol['iterations'],
            'z':zs,
            'misclassified':(err1,err2)}

def softmargin_completion(Q, d, gamma):
    """
    Solves the QP

        minimize    (1/2)*y'*Qc^{-1}*y + gamma*sum(v)
        subject to  diag(d)*(y + b*ones) + v >= 1
                    v >= 0

    (with variables y, b, v) and its dual, the 'soft-margin' SVM problem,

        maximize    -(1/2)*z'*Qc*z + d'*z
        subject to  0 <= diag(d)*z <= gamma*ones
                    sum(z) = 0

    (with variables z).

    Qc is the max determinant completion of Q.


    Input arguments.

        Q is a sparse N x N sparse matrix with chordal sparsity pattern
            and a positive definite completion

        d is an N-vector of labels -1 or 1.

        gamma is a positive parameter.

        F is the chompack pattern corresponding to Q.  If F is None, the
            pattern is computed.


    Output.

        z, y, b, v, optval, L, iters

    """

    if verbose: solvers.options['show_progress'] = True
    else: solvers.options['show_progress'] = False

    N  = Q.size[0]
    p = chompack.maxcardsearch(Q)
    symb = chompack.symbolic(Q,p)
    Qc = chompack.cspmatrix(symb) + Q

    # Qinv is the inverse of the max. determinant p.d. completion of Q
    Lc = Qc.copy()
    chompack.completion(Lc)
    Qinv = Lc.copy()
    chompack.llt(Qinv)
    Qinv = Qinv.spmatrix(reordered=False)
    Qinv = chompack.symmetrize(Qinv)

    def P(u, v, alpha = 1.0, beta = 0.0):
        """
            v := alpha * [ Qc^-1, 0, 0;  0, 0, 0;  0, 0, 0 ] * u + beta * v
        """

        v *= beta
        base.symv(Qinv, u, v, alpha = alpha, beta = 1.0)


    def G(u, v, alpha = 1.0, beta = 0.0, trans = 'N'):
        """
        If trans is 'N':

            v := alpha * [ -diag(d),  -d,  -I;  0,  0,  -I ] * u + beta * v.

        If trans is 'T':

            v := alpha * [ -diag(d), 0;  -d', 0;  -I, -I ] * u + beta * v.
        """

        v *= beta

        if trans is 'N':
            v[:N] -= alpha * ( base.mul(d, u[:N] + u[N]) + u[-N:] )
            v[-N:] -= alpha * u[-N:]

        else:
            v[:N] -= alpha * base.mul(d, u[:N])
            v[N] -= alpha * blas.dot(d, u, n = N)
            v[-N:] -= alpha * (u[:N] + u[N:])


    K = spmatrix(0.0, Qinv.I, Qinv.J)
    dy1, dy2 = matrix(0.0, (N,1)), matrix(0.0, (N,1))

    def Fkkt(W):
        """
        Custom KKT solver for

            [  Qinv  0   0  -D    0  ] [ ux_y ]   [ bx_y ]
            [  0     0   0  -d'   0  ] [ ux_b ]   [ bx_b ]
            [  0     0   0  -I   -I  ] [ ux_v ] = [ bx_v ]
            [ -D    -d  -I  -D1   0  ] [ uz_z ]   [ bz_z ]
            [  0     0  -I   0   -D2 ] [ uz_w ]   [ bz_w ]

        with D1 = diag(d1), D2 = diag(d2), d1 = W['d'][:N]**2,
        d2 = W['d'][N:])**2.
        """

        d1, d2 = W['d'][:N]**2, W['d'][N:]**2
        d3, d4 = (d1 + d2)**-1, (d1**-1 + d2**-1)**-1

        # Factor the chordal matrix K = Qinv + (D_1+D_2)^-1.
        K.V = Qinv.V
        K[::N+1] = K[::N+1] + d3
        L = chompack.cspmatrix(symb) + K
        chompack.cholesky(L)

        # Solve (Qinv + (D1+D2)^-1) * dy2 = (D1 + D2)^{-1} * 1
        blas.copy(d3, dy2)
        chompack.trsm(L, dy2, trans = 'N')
        chompack.trsm(L, dy2, trans = 'T')

        def g(x, y, z):

            # Solve
            #
            #     [ K    d3    ] [ ux_y ]
            #     [            ] [      ] =
            #     [ d3'  1'*d3 ] [ ux_b ]
            #
            #         [ bx_y ]   [ D  ]
            #         [      ] - [    ] * D3 * (D2 * bx_v + bx_z - bx_w).
            #         [ bx_b ]   [ d' ]

            x[:N] -= mul(d, mul(d3, mul(d2, x[-N:]) + z[:N] - z[-N:]))
            x[N] -= blas.dot(d, mul(d3, mul(d2, x[-N:]) + z[:N] - z[-N:]))

            # Solve dy1 := K^-1 * x[:N]
            blas.copy(x, dy1, n = N)
            chompack.trsm(L, dy1, trans = 'N')
            chompack.trsm(L, dy1, trans = 'T')

            # Find ux_y = dy1 - ux_b * dy2 s.t
            #
            #     d3' * ( dy1 - ux_b * dy2 + ux_b ) = x[N]
            #
            # i.e.  x[N] := ( x[N] - d3'* dy1 ) / ( d3'* ( 1 - dy2 ) ).

            x[N] = ( x[N] - blas.dot(d3, dy1) ) / \
                ( blas.asum(d3) - blas.dot(d3, dy2) )
            x[:N] = dy1 - x[N] * dy2


            # ux_v = D4 * ( bx_v -  D1^-1 (bz_z + D * (ux_y + ux_b))
            #     - D2^-1 * bz_w )

            x[-N:] = mul(d4, x[-N:] - div(z[:N] + mul(d, x[:N] + x[N]), d1)
                - div(z[N:],d2))

            # uz_z = - D1^-1 * ( bx_z - D * ( ux_y + ux_b ) - ux_v )
            # uz_w = - D2^-1 * ( bx_w - uz_w )
            z[:N] +=  base.mul(d, x[:N] + x[N]) + x[-N:]
            z[-N:] += x[-N:]
            blas.scal(-1.0, z)

            # Return W['di'] * uz
            blas.tbmv(W['di'], z, n = 2*N, k = 0, ldA = 1)

        return g

    q = matrix(0.0, (2*N+1,1))

    if weights is 'proportional':
        dlist = list(d)
        C1 = 0.5*N*gamma/dlist.count(1)
        C2 = 0.5*N*gamma/dlist.count(-1)
        gvec = matrix([C1 if w == 1 else C2 for w in dlist],(N,1))
        del dlist
        q[-N:] = gvec
    elif weights is 'equal':
        q[-N:] = gamma

    h = matrix(0.0, (2*N,1))
    h[:N] = -1.0
    sol = solvers.coneqp(P, q, G, h, kktsolver = Fkkt)
    u = matrix(0.0, (N,1))
    y, b, v =  sol['x'][:N], sol['x'][N], sol['x'][N+1:]
    z = mul(d, sol['z'][:N])
    base.symv(Qinv, y, u)
    optval = 0.5 * blas.dot(y, u) + gamma * sum(v)
    return y, b, v, z, optval, Lc, sol['iterations']


def softmargin_appr(X, d, gamma, width, kernel = 'linear', sigma = 1.0, degree = 1, theta = 1.0, Q = None):
    """
    Solves the approximate 'soft-margin' SVM problem

        maximize    -(1/2)*z'*Qc*z + d'*z
        subject to  0 <= diag(d)*z <= gamma*ones
                    sum(z) = 0

    (with variables z), and its dual problem

        minimize    (1/2)*y'*Qc^{-1}*y + gamma*sum(v)
        subject to  diag(d)*(y + b*ones) + v >= 1
                    v >= 0

    (with variables y, v, b).

    Qc is the maximum determinant completion of the projection of Q
    on a band with bandwidth 2*w+1.  Q_ij = K(xi, xj) where K is a kernel
    function and xi is the ith row of X (xi' = X[i,:]).

    Input arguments.

        X is an N x n matrix.

        d is an N-vector with elements -1 or 1; d[i] is the label of
        row X[i,:].

        gamma is a positive parameter.

        kernel is a string with values 'linear', 'rfb', 'poly', or 'tanh'.
        'linear': k(u,v) = u'*v/sigma.
        'rbf':    k(u,v) = exp(-||u - v||^2 / (2*sigma)).
        'poly':   k(u,v) = (u'*v/sigma)**degree.
        'tanh':   k(u,v) = tanh(u'*v/sigma - theta).

        sigma and theta are positive numbers.

        degree is a positive integer.

        width is a positive integer.

    Output.

        Returns a dictionary with the keys:

        'classifier'
           a Python function object that takes an M x n matrix with
           test vectors as rows and returns a vector with labels

        'completion classifier'
          a Python function object that takes an M x n matrix with
          test vectors as rows and returns a vector with labels

        'z'
           a sparse m-vector

        'cputime'
           a tuple (Ttot, Tqp, Tker) where Ttot is the total
           CPU time, Tqp is the CPU time spent solving the QP, and
           Tker is the CPU time spent computing the kernel matrix

        'iterations'
           the number of interior-point iteations

        'misclassified'
           a tuple (L1, L2) where L1 is a list of indices of
           misclassified training vectors from class 1, and L2 is a
           list of indices of misclassified training vectors from
           class 2
    """

    Tstart = cputime()

    if verbose: solvers.options['show_progress'] = True
    else: solvers.options['show_progress'] = False
    N, n = X.size

    if Q is None:
        Q,a = kernel_matrix(X, kernel, sigma = sigma, degree = degree, theta = theta, V = 'band', width = width)
    else:
        if not (Q.size[0] == N and Q.size[1] == N):
            raise ValueError("invalid kernel matrix dimensions")
        elif not type(Q) is cvxopt.base.spmatrix:
            raise ValueError("invalid kernel matrix type")
        elif verbose:
            print("using precomputed kernel matrix ..")
        if kernel == 'rbf':
            Ad = spmatrix(0.0,range(N),range(N))
            base.syrk(X,V,partial = True)
            a = Ad.V
            del Ad

    Tkernel = cputime(Tstart)

    # solve qp
    Tqp = cputime()
    y, b, v, z, optval, Lc, iters  = softmargin_completion(Q, matrix(d, tc='d'), gamma)
    Tqp = cputime(Tqp)
    if verbose: print("utime = %f, stime = %f." % Tqp)

    # extract nonzero support vectors
    nrmz = max(abs(z))
    sv = [ k for k in range(N) if abs(z[k]) > Tsv * nrmz ]
    zs = spmatrix(z[sv],sv,[0 for i in range(len(sv))],(len(d),1))
    if verbose: print("%d support vectors." % len(sv))
    Xr, zr, Nr = X[sv,:], z[sv], len(sv)

    # find misclassified training vectors
    err1 = [i for i in range(Q.size[0]) if (v[i] > 1 and d[i] == 1)]
    err2 = [i for i in range(Q.size[0]) if (v[i] > 1 and d[i] == -1)]
    if verbose:
        e1,n1 = len(err1),list(d).count(1)
        e2,n2 = len(err2),list(d).count(-1)
        print("class 1: %i/%i = %.1f%% misclassified." % (e1,n1,100.*e1/n1))
        print("class 2: %i/%i = %.1f%% misclassified." % (e2,n2,100.*e2/n2))
        del e1,e2,n1,n2

    # create classifier function object

    # CLASSIFIER 1 (standard kernel classifier)
    if kernel == 'linear':
        # w = X'*z / sigma
        w = matrix(0.0, (n,1))
        blas.gemv(Xr, zr, w, trans = 'T', alpha = 1.0/sigma)

        def classifier(Y, soft = False):
            M = Y.size[0]
            x = matrix(b, (M,1))
            blas.gemv(Y, w, x, beta = 1.0)
            if soft: return x
            else:    return matrix([ 2*(xk > 0.0) - 1 for xk in x ])

    elif kernel == 'rbf':

        def classifier(Y, soft = False):
            M = Y.size[0]
            # K = Y*X' / sigma
            K = matrix(0.0, (M, Nr))
            blas.gemm(Y, Xr, K, transB = 'T', alpha = 1.0/sigma)

            # c[i] = ||Yi||^2 / sigma
            ones = matrix(1.0, (max([M,Nr,n]),1))
            c = Y**2 * ones[:n]
            blas.scal(1.0/sigma, c)

            # Kij := Kij - 0.5 * (ci + aj)
            #      = || yi - xj ||^2 / (2*sigma)
            blas.ger(c, ones, K, alpha = -0.5)
            blas.ger(ones, a[sv], K, alpha = -0.5)
            x = exp(K) * zr + b
            if soft: return x
            else:    return matrix([ 2*(xk > 0.0) - 1 for xk in x ])

    elif kernel == 'tanh':

        def classifier(Y, soft = False):
            M = Y.size[0]
            # K = Y*X' / sigma - theta
            K = matrix(theta, (M, Nr))
            blas.gemm(Y, Xr, K, transB = 'T', alpha = 1.0/sigma, beta = -1.0)

            K = exp(K)
            x = div(K - K**-1, K + K**-1) * zr + b
            if soft: return x
            else:    return matrix([ 2*(xk > 0.0) - 1 for xk in x ])

    elif kernel == 'poly':

        def classifier(Y, soft = False):
            M = Y.size[0]
            # K = Y*X' / sigma
            K = matrix(0.0, (M, Nr))
            blas.gemm(Y, Xr, K, transB = 'T', alpha = 1.0/sigma)

            x = K**degree * zr + b
            if soft: return x
            else:    return matrix([ 2*(xk > 0.0) - 1 for xk in x ])

    else:
        pass

    # CLASSIFIER 2 (completion kernel classifier)
    # TODO: generalize to arbitrary sparsity pattern
    L11 = matrix(Q[:width,:width])
    lapack.potrf(L11)

    if kernel == 'linear':

        def classifier2(Y, soft = False):
            M = Y.size[0]
            W = matrix(0., (width,M))
            blas.gemm(X, Y, W, transB = 'T', alpha = 1.0/sigma, m = width)
            lapack.potrs(L11,W)
            W = matrix([W, matrix(0.,(N-width,M))])
            chompack.trsm(Lc,W,trans='N')
            chompack.trsm(Lc,W,trans='T')

            x = matrix(b,(M,1))
            blas.gemv(W,z,x,trans='T', beta = 1.0)
            if soft: return x
            else:    return matrix([ 2*(xk > 0.0) - 1 for xk in x ])

    elif kernel == 'poly':

        def classifier2(Y, soft = False):
            if Y is None: return zs

            M = Y.size[0]
            W = matrix(0., (width,M))
            blas.gemm(X, Y, W, transB = 'T', alpha = 1.0/sigma, m = width)
            W = W**degree
            lapack.potrs(L11,W)
            W = matrix([W, matrix(0.,(N-width,M))])
            chompack.trsm(Lc,W,trans='N')
            chompack.trsm(Lc,W,trans='T')

            x = matrix(b,(M,1))
            blas.gemv(W,z,x,trans='T', beta = 1.0)
            if soft: return x
            else:    return matrix([ 2*(xk > 0.0) - 1 for xk in x ])

    elif kernel == 'rbf':

        def classifier2(Y, soft = False):

            M = Y.size[0]

            # K = Y*X' / sigma
            K = matrix(0.0, (width, M))
            blas.gemm(X, Y, K, transB = 'T', alpha = 1.0/sigma, m = width)

            # c[i] = ||Yi||^2 / sigma
            ones = matrix(1.0, (max(width,n,M),1))
            c = Y**2 * ones[:n]
            blas.scal(1.0/sigma, c)

            # Kij := Kij - 0.5 * (ci + aj)
            #      = || yi - xj ||^2 / (2*sigma)
            blas.ger(ones[:width], c, K, alpha = -0.5)
            blas.ger(a[:width], ones[:M], K, alpha = -0.5)
            # Kij = exp(Kij)
            K = exp(K)

            # complete K
            lapack.potrs(L11,K)
            K = matrix([K, matrix(0.,(N-width,M))],(N,M))
            chompack.trsm(Lc,K,trans='N')
            chompack.trsm(Lc,K,trans='T')

            x = matrix(b,(M,1))
            blas.gemv(K,z,x,trans='T', beta = 1.0)

            if soft: return x
            else:    return matrix([ 2*(xk > 0.0) - 1 for xk in x ])

    elif kernel == 'tanh':

        def classifier2(Y, soft = False):

            M = Y.size[0]

            # K = Y*X' / sigma
            K = matrix(theta, (width, M))
            blas.gemm(X, Y, K, transB = 'T',
                      alpha = 1.0/sigma, beta = -1.0, m = width)

            K = exp(K)
            K = div(K - K**-1, K + K**-1)

            # complete K
            lapack.potrs(L11,K)
            K = matrix([K, matrix(0.,(N-width,M))],(N,M))
            chompack.trsm(Lc,K,trans='N')
            chompack.trsm(Lc,K,trans='T')

            x = matrix(b,(M,1))
            blas.gemv(K,z,x,trans='T', beta = 1.0)

            if soft: return x
            else:    return matrix([ 2*(xk > 0.0) - 1 for xk in x ])

    Ttotal = cputime(Tstart)

    return {'classifier': classifier,
            'completion classifier': classifier2,
            'cputime': (sum(Ttotal),sum(Tqp),sum(Tkernel)),
            'iterations':iters,
            'z':zs,
            'misclassified':(err1,err2)}
