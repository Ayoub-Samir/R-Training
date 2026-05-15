# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 23:05:11 2024

@author: musta
"""

from typing import Optional, Tuple
import math
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor, matmul
from scipy.special import legendre

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor
from torch.nn import Parameter

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.linalg import expm
import scipy.io as sio

from scipy.linalg import eig, eigh
from scipy.sparse.linalg import eigs, eigsh


from scipy.special import gamma, factorial

#! /usr/bin/env python
#
def imtqlx ( n, d, e, z ):

#*****************************************************************************80
#
## IMTQLX diagonalizes a symmetric tridiagonal matrix.
#
#  Discussion:
#
#    This routine is a slightly modified version of the EISPACK routine to
#    perform the implicit QL algorithm on a symmetric tridiagonal matrix.
#
#    The authors thank the authors of EISPACK for permission to use this
#    routine.
#
#    It has been modified to produce the product Q' * Z, where Z is an input
#    vector and Q is the orthogonal matrix diagonalizing the input matrix.
#    The changes consist (essentially) of applying the orthogonal 
#    transformations directly to Z as they are generated.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    15 June 2015
#
#  Author:
#
#    John Burkardt.
#
#  Reference:
#
#    Sylvan Elhay, Jaroslav Kautsky,
#    Algorithm 655: IQPACK, FORTRAN Subroutines for the Weights of
#    Interpolatory Quadrature,
#    ACM Transactions on Mathematical Software,
#    Volume 13, Number 4, December 1987, pages 399-415.
#
#    Roger Martin, James Wilkinson,
#    The Implicit QL Algorithm,
#    Numerische Mathematik,
#    Volume 12, Number 5, December 1968, pages 377-383.
#
#  Parameters:
#
#    Input, integer N, the order of the matrix.
#
#    Input, real D(N), the diagonal entries of the matrix.
#
#    Input, real E(N), the subdiagonal entries of the
#    matrix, in entries E(1) through E(N-1). 
#
#    Input, real Z(N), a vector to be operated on.
#
#    Output, real LAM(N), the diagonal entries of the diagonalized matrix.
#
#    Output, real QTZ(N), the value of Q' * Z, where Q is the matrix that 
#    diagonalizes the input symmetric tridiagonal matrix.
#
  import numpy as np
  #from r8_epsilon import r8_epsilon
  
  from sys import exit

  lam = np.zeros ( n )
  for i in range ( 0, n ):
    lam[i] = d[i]

  qtz = np.zeros ( n )
  for i in range ( 0, n ):
    qtz[i] = z[i]

  if ( n == 1 ):
    return lam, qtz

  itn = 30

  prec = 2.220446049250313E-016

  e[n-1] = 0.0

  for l in range ( 1, n + 1 ):

    j = 0

    while ( True ):

      for m in range ( l, n + 1 ):

        if ( m == n ):
          break

        if ( abs ( e[m-1] ) <= prec * ( abs ( lam[m-1] ) + abs ( lam[m] ) ) ):
          break

      p = lam[l-1]

      if ( m == l ):
        break

      if ( itn <= j ):
        print ( '' )
        print ( 'IMTQLX - Fatal error!' )
        print ( '  Iteration limit exceeded.' )
        exit ( 'IMTQLX - Fatal error!' )

      j = j + 1
      g = ( lam[l] - p ) / ( 2.0 * e[l-1] )
      r = np.sqrt ( g * g + 1.0 )

      if ( g < 0.0 ):
        t = g - r
      else:
        t = g + r

      g = lam[m-1] - p + e[l-1] / ( g + t )
 
      s = 1.0
      c = 1.0
      p = 0.0
      mml = m - l

      for ii in range ( 1, mml + 1 ):

        i = m - ii
        f = s * e[i-1]
        b = c * e[i-1]

        if ( abs ( g ) <= abs ( f ) ):
          c = g / f
          r = np.sqrt ( c * c + 1.0 )
          e[i] = f * r
          s = 1.0 / r
          c = c * s
        else:
          s = f / g
          r = np.sqrt ( s * s + 1.0 )
          e[i] = g * r
          c = 1.0 / r
          s = s * c

        g = lam[i] - p
        r = ( lam[i-1] - g ) * s + 2.0 * c * b
        p = s * r
        lam[i] = g + p
        g = c * r - b
        f = qtz[i]
        qtz[i]   = s * qtz[i-1] + c * f
        qtz[i-1] = c * qtz[i-1] - s * f

      lam[l-1] = lam[l-1] - p
      e[l-1] = g
      e[m-1] = 0.0

  for ii in range ( 2, n + 1 ):

     i = ii - 1
     k = i
     p = lam[i-1]

     for j in range ( ii, n + 1 ):

       if ( lam[j-1] < p ):
         k = j
         p = lam[j-1]

     if ( k != i ):

       lam[k-1] = lam[i-1]
       lam[i-1] = p

       p        = qtz[i-1]
       qtz[i-1] = qtz[k-1]
       qtz[k-1] = p

  return lam, qtz


#! /usr/bin/env python
#
def p_polynomial_zeros ( nt ):

#*****************************************************************************80
#
## P_POLYNOMIAL_ZEROS: zeros of Legendre function P(n,x).
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    16 March 2016
#
#  Author:
#
#    John Burkardt
#
#  Parameters:
#
#    Input, integer NT, the order of the rule.
#
#    Output, real T(NT), the zeros.
#


  a = np.zeros ( nt )

  b = np.zeros ( nt )

  for i in range ( 0, nt ):
    ip1 = i + 1
    b[i] = ip1 / np.sqrt ( 4 * ip1 * ip1 - 1 )

  c = np.zeros ( nt )
  c[0] = np.sqrt ( 2.0 )

  t, w = imtqlx ( nt, a, b, c )

  return  t+1# for [0, 2] interval


def j_polynomial_zeros ( nt, alpha, beta):

#*****************************************************************************80
#
## P_POLYNOMIAL_ZEROS: zeros of Legendre function P(n,x).
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    19 October 2023
#
#  Author:
#
#    Dr. Mustafa Coşkun
#
#  Parameters:
#
#    Input, integer NT, the order of the rule, upper and lower are bounds.
#
#    Output, real T(NT), the zeros.
#

  ab = alpha + beta
  abi = 2.0 + ab
  # define the zero-th moment
  zemu = (np.power(2.0,(ab + 1.0))*gamma(alpha + 1.0)*gamma(beta + 1.0))/gamma(abi)
  
  
  x = np.zeros(nt)
  bj = np.zeros(nt)
  
  
  x[0] = (beta -alpha)/abi
  bj[0] = np.sqrt(4.0*(1.0 + alpha)*(1.0 + beta)/((abi + 1.0)*abi*abi))
  a2b2 = beta*beta-alpha*alpha

  for i in range ( 2, nt +1 ):
    abi = 2.0*i + ab
    x[i-1] = a2b2 / ((abi -2.0)*abi)
    abi = np.power(abi,2)
    bj[i-1] = np.sqrt((4.0*i*(i+alpha)*(i + beta)* (i + ab))/((abi -1.0)*abi))

 #bjs = np.sqrt(bj)
  c = np.zeros ( nt )
  c[0] = np.sqrt ( zemu )

  t, w = imtqlx ( nt, x, bj, c )

  return  t + 1 # for [0, 2] interval

def g_fullRWR(x):
    return (1)/(1 - x) 
    #return x/(1-x) - x**2 


def g_0(x):
    return (0.1)/(1 - x) 
def g_1(x):
    return (1)/(1 - x)
def g_2(x):
    return ((x)/(1 - x))
def g_3(x):
    return (x**2)/(1 - x)   
def g_4(x):
    return (1)/(1 + 25*x**2)


def g_par(x):
    return 1/(1 + x)

def g_appRWR(x,Ksteps):
    sum = 0
    for k in range(Ksteps):
        sum = sum + x**k
    return 0.1*sum

def g_heat(x,Ksteps):
    t = 5
    sum = 0
    for k in range(Ksteps):
        sum = sum + (t**k)/np.math.factorial(k)
    return np.math.exp(-sum)
def g_band_rejection(x):
    return (1-np.exp(-10*(x-1)**2))

def g_band_pass(x):
    return np.exp(-10*(x-1)**2)

def g_low_pass(x):
    return np.exp(-10*x**2)
def g_high_pass(x):
    return 1-np.exp(-10*x**2)

def g_hetero(x):
    # Tri-band heterophily filter on Laplacian spectrum lambda in [0, 2].
    w0, w1, w2 = 0.2, 0.5, 0.3
    g0, g1, g2 = 6.0, 10.0, 6.0
    return (
        w0 * np.exp(-g0 * x**2)
        + w1 * np.exp(-g1 * (x - 1.0) ** 2)
        + w2 * np.exp(-g2 * (x - 2.0) ** 2)
    )

def g_hetero_adj(x):
    # Same filter on normalized adjacency spectrum omega in [-1, 1].
    # Mapping: lambda = 1 - omega, so g^(A)(omega) = g^(L)(1 - omega).
    return g_hetero(1.0 - x)

def g_comb(x):
    return np.abs(np.sin(np.pi*x))
    
def filter_jackson(c):
    N = len(c)
    n = np.arange(N)
    tau = np.pi/(N+1)
    g = ((N-n+1)*np.cos(tau*n) + np.sin(tau*n)/np.tan(tau))/(N+1)
    c = np.multiply(g,c)
    return c

# def filter_jackson(c):
# 	"""
# 	Apply the Jackson filter to a sequence of Chebyshev	moments. The moments 
# 	should be arranged column by column.

# 	Args:
# 		c: Unfiltered Chebyshev moments

# 	Output:
# 		cf: Jackson filtered Chebyshev moments
# 	"""

# 	N = len(c)
# 	n = np.arange(N)
# 	tau = np.pi/(N+1)
# 	g = ((N-n+1)*np.cos(tau*n)+np.sin(tau*n)/np.tan(tau))/(N+1)
# 	g.shape = (N,1)
# 	c = g*c 
#     #print(c)
    
# 	return c

def g_Ours(x):
    sum = 1*1 + 1*x + 4*x**2 + 5*x**3
    return sum
def runge(x):
  """In some places the x range is expanded and the formula give as 1/(1+x^2)
  """
  return 1 / (1 +  x ** 2)

def polyfitA(x,y,n):

    m = x.size
    Q = np.ones((m, 1), dtype=object)
    H = np.zeros((n+1, n), dtype=object)
    k = 0
    j = 0
    for k in range(n):
        q = np.multiply(x,Q[:,k])
        #print(q)
        for j in range(k):
            H[j,k] = np.dot(Q[:,j].T,(q/m))
            q = q - np.dot(H[j,k],(Q[:,j]))
        H[k+1,k] = np.linalg.norm(q)/np.sqrt(m)
        Q = np.column_stack((Q, q/H[k+1,k]))
    #print(Q)
    #print(Q.shape)
    d = np.linalg.solve(Q.astype(np.float64), y.astype(np.float64))
    return d, H

def polyvalA(d,H,s):
    inputtype = H.dtype.type
    M = len(s)
    W = np.ones((M,1), dtype=inputtype)
    n = H.shape[1]
    #print("Complete H", H)
    k = 0
    j = 0
    for k in range(n):
        w = np.multiply(s,W[:,k])
        for j in range(k):
            #print( "H[j,k]",H[j,k])
            w = w -np.dot(H[j,k],( W[:,j]))
        W = np.column_stack((W, w/H[k+1,k]))
    y = W @ d
    return y, W


def t_polynomial_zeros(x0, x1, n):
  return (x1 - x0) * (np.cos((2*np.arange(1, n + 1) - 1)/(2*n)*np.pi) + 1) / 2  + x0

def cheby(i,x):
    if i==0:
        return 1
    elif i==1:
        return x
    else:
        T0=1
        T1=x
        for ii in range(2,i+1):
            T2=2*x*T1-T0
            T0,T1=T1,T2
        return T2

def s_polynomial_zeros(n):
    temp = Parameter(torch.Tensor(n+1))
    temp.data.fill_(1.0)
    coe_tmp=F.relu(temp)
    coe=coe_tmp.clone()
    for i in range(n):
        coe[i]=coe_tmp[0]*cheby(i,math.cos((n+0.5)*math.pi/(n+1)))
        for j in range(1,n+1):
            x_j=math.cos((n-j+0.5)*math.pi/(n+1))
            coe[i]=coe[i]+coe_tmp[j]*cheby(i,x_j)
        coe[i]=2*coe[i]/(n+1)
    return coe

def compare_fitA(f, x, Vander, x0, x1):
  y = f(x)
  n = x.size-1

  if(Vander):
      coefficients = Vandermonde(x, y)
  else:
      coefficients, H = polyfitA(x, y, n)
  #K = coefficients.shape[0]
  # for k in range(K-1, -1, -1):
  #     print(coefficients[k], k)
  return coefficients

def m_polynomial_zeros (x0, x1, n):
    return  np.linspace(x0, x1,n)
def compare_fit_panelA(f, polyname, Vandermonde, degree, x0, x1,zoom=False):
   # Male equedistance
   #x = np.linspace(x0, x1,10)
   if (polyname == 'Monomial'):
       x = m_polynomial_zeros(x0, x1, degree)
   elif (polyname == 'Chebyshev'):    
       x = t_polynomial_zeros(x0, x1, degree)
   elif (polyname == 'Legendre'):
       x = p_polynomial_zeros(degree)
   elif (polyname == 'Jacobi'):    
       x = j_polynomial_zeros(degree,0,1)
   else:
    print ('Give proper polynomial to interpolate\n')
    print ('Calling Monimal as default\n')
    x = m_polynomial_zeros(x0, x1, degree)
    
   return compare_fitA(f, x, Vandermonde, x0,x1)
def Vandermonde(x, y):
  """Return a polynomial fit of order n+1 to n points"""
  #z = np.polyfit(x, y, x.size + 1)
 
  V = np.vander(x) # Vandermonde matrix
  coeffs = np.linalg.solve(V, y) # f_nodes must be a column vector
  return coeffs

class ARNOLDI(MessagePassing):
    r"""The approximate personalized propagation of neural predictions layer
    from the `"Predict then Propagate: Graph Neural Networks meet Personalized
    PageRank" <https://arxiv.org/abs/1810.05997>`_ paper

    .. math::
        \mathbf{X}^{(0)} &= \mathbf{X}

        \mathbf{X}^{(k)} &= (1 - \alpha) \mathbf{\hat{D}}^{-1/2}
        \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \mathbf{X}^{(k-1)} + \alpha
        \mathbf{X}^{(0)}

        \mathbf{X}^{\prime} &= \mathbf{X}^{(K)},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Args:
        K (int): Number of iterations :math:`K`.
        alpha (float): Teleport probability :math:`\alpha`.
        dropout (float, optional): Dropout probability of edges during
            training. (default: :obj:`0`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F)`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F)`
    """
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, K: int, alpha: float, lower: float, upper: float,  homophily: bool, nameFunc: str, namePoly: str, Vandermonde: str, dropout: float = 0.,
                 cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.K = K
        self.alpha = alpha
        self.homophily = homophily
        self.Vandermonde = Vandermonde
        self.lower = lower
        self.upper = upper
        self.dropout = dropout
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        if(nameFunc == 'g_0'):
            self.coeffs = compare_fit_panelA(g_0, namePoly, Vandermonde, self.K, self.lower, self.upper)
        elif(nameFunc == 'g_1'):
            self.coeffs = compare_fit_panelA(g_1, namePoly, Vandermonde, self.K, self.lower, self.upper) 
        elif(nameFunc == 'g_2'):
            self.coeffs = compare_fit_panelA(g_2,namePoly, Vandermonde, self.K,self.lower, self.upper)
        elif(nameFunc == 'g_3'):
            self.coeffs = compare_fit_panelA(g_3,namePoly, Vandermonde, self.K, self.lower, self.upper)
        elif(nameFunc == 'g_4'):
            self.coeffs = compare_fit_panelA(g_4,namePoly, Vandermonde, self.K,self.lower, self.upper)
        elif(nameFunc == 'g_band_rejection'):
            self.coeffs = compare_fit_panelA(g_band_rejection,namePoly, Vandermonde, self.K,self.lower, self.upper)
        elif(nameFunc == 'g_band_pass'):
            self.coeffs = compare_fit_panelA(g_band_pass,namePoly, Vandermonde, self.K,self.lower, self.upper)
        elif(nameFunc == 'g_low_pass'):
            self.coeffs = compare_fit_panelA(g_low_pass,namePoly, Vandermonde, self.K,self.lower, self.upper)
        elif(nameFunc == 'g_high_pass'):
            self.coeffs = compare_fit_panelA(g_high_pass,namePoly, Vandermonde, self.K,self.lower, self.upper)
        elif(nameFunc == 'g_hetero'):
            self.coeffs = compare_fit_panelA(g_hetero,namePoly, Vandermonde, self.K,self.lower, self.upper)
        elif(nameFunc == 'g_hetero_adj'):
            self.coeffs = compare_fit_panelA(g_hetero_adj,namePoly, Vandermonde, self.K,self.lower, self.upper)
        elif(nameFunc == 'g_comb'):
            self.coeffs = compare_fit_panelA(g_comb,namePoly, Vandermonde, self.K,self.lower, self.upper)
        else:
            self.coeffs = compare_fit_panelA(g_fullRWR,namePoly, Vandermonde,self.K,self.lower, self.upper)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        h = x
        # Here this code could be more efficient 
        myb = self.coeffs[self.K-1]*x
        for k in range(self.K-2,-1,-1):
            if self.dropout > 0 and self.training:
                if isinstance(edge_index, Tensor):
                    assert edge_weight is not None
                    edge_weight = F.dropout(edge_weight, p=self.dropout)
                else:
                    value = edge_index.storage.value()
                    assert value is not None
                    value = F.dropout(value, p=self.dropout)
                    edge_index = edge_index.set_value(value, layout='coo')

            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                               size=None)
            #x = x * (1 - self.alpha)
            if (self.homophily):
                x = x + self.coeffs[k]*myb
            else:
                x = self.coeffs[k]*x + myb

        return x

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(K={self.K}, alpha={self.alpha})'
