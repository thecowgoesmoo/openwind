#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2019-2023, INRIA
#
# This file is part of Openwind.
#
# Openwind is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Openwind is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Openwind.  If not, see <https://www.gnu.org/licenses/>.
#
# For more informations about authors, see the CONTRIBUTORS file

"""
Optimisation of coefficients a_i, b_i.

Perform coefficient optimization for the "diffusive representation"
model of viscothermal losses, as described in our paper: ...
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.special import jve
from scipy.optimize import minimize


# Coefficient a_0 is fixed and will always be 8
# (to ensure the low-frequency behavior of GN to be correct)
a_0 = 8.0

def phi(alpha):
    return 2 * jve(1, alpha) / (alpha * jve(0, alpha))

def exact_G(zeta):
    """The function G we are trying to approximate."""
    phi_z = phi(np.sqrt(-1j*zeta))
    return 1j*zeta * phi_z / (1 - phi_z)

def GN(zeta, a_i, b_i):
    """Evaluate an approximation of function G, using coefs a_i and b_i."""
    j_zeta = 1j * zeta[..., np.newaxis]
    return a_0 + np.sum(a_i * j_zeta / (b_i*j_zeta + 1), axis=-1)

def d_GN_d_ai(zeta, a_i, b_i):
    """Gradient of GN wrt coefficients a_i.
    Adds a new axis with size N at the last position.
    """
    zeta = np.array(zeta)[..., np.newaxis]
    return 1/(b_i + 1/(1j*zeta))

def d_GN_d_bi(zeta, a_i, b_i):
    """Gradient of GN wrt coefficients b_i."""
    zeta = np.array(zeta)[..., np.newaxis]
    return -a_i / (b_i + 1/(1j*zeta))**2


zeta_min, zeta_max = 8, 2e6


def optimize_ai_bi(N=4):
    M = 100  # Number of values of zeta
    zeta_k = np.geomspace(zeta_min, zeta_max, M)
    G_zeta = exact_G(zeta_k)   # Evaluate only once

    # Find reasonable initial values for a_i, b_i
    zeta_tilde_i = np.geomspace(zeta_min, zeta_max, N)
    # Take each term to have the correct real part when b_i*zeta is 1
    b_i_init = 1 / zeta_tilde_i
    a_i_init = zeta_tilde_i ** -0.5

    x_init = np.log(np.concatenate((a_i_init, b_i_init)))

    def cost(log_ai_bi):
        """
        The cost function.

        Parameters
        ----------
        log_ai_bi : array(float)
            The 2N parameters are
            [log(a_1), ..., log(a_N), log(b_1), ..., log(b_N)].

        Returns
        -------
        Value of the cost function:
            E = 1/2 * sum(abs(GN(zeta_k)/G(zeta_k) - 1)**2).
        """
        a_i = np.exp(log_ai_bi[:N])
        b_i = np.exp(log_ai_bi[N:])
        GN_zeta = GN(zeta_k, a_i, b_i)
        residue = GN_zeta / G_zeta - 1
        E = 0.5 * np.sum(np.abs(residue)**2)
        return E

    def grad_cost(log_ai_bi):
        """Gradient of the cost wrt log_ai_bi."""
        a_i = np.exp(log_ai_bi[:N])
        b_i = np.exp(log_ai_bi[N:])
        GN_zeta = GN(zeta_k, a_i, b_i)
        dGN_ai = d_GN_d_ai(zeta_k, a_i, b_i)
        dGN_bi = d_GN_d_bi(zeta_k, a_i, b_i)
        residue = GN_zeta / G_zeta - 1
        dE_ai = np.real(residue.conj().dot(dGN_ai/G_zeta[:, None]))
        dE_bi = np.real(residue.conj().dot(dGN_bi/G_zeta[:, None]))
        return np.concatenate((dE_ai*a_i, dE_bi*b_i))

    result = minimize(cost, x_init, jac=grad_cost,
                      method='BFGS', options={'gtol': 1e-8})
    a_i, b_i = np.exp(result.x[:N]), np.exp(result.x[N:])
    return result, a_i, b_i


def _plot_G():
    zeta = np.geomspace(zeta_min/10, zeta_max*10, 300)
    G_zeta = exact_G(zeta)
    plt.loglog(zeta, np.real(G_zeta), '-k', label="Re(G)")
    plt.loglog(zeta, np.imag(G_zeta), '--k', label="Im(G)")
    plt.fill_betweenx([0, 1e4], zeta_min, zeta_max, color=(0,0,0,0.1))
    plt.grid(True)
    plt.xlabel(r"Dimensionless frequency $\zeta$")
    plt.ylabel(r"$G(\zeta)$")
    plt.legend()

def plot_GN(a_i, b_i, **kwargs):
    N = len(a_i)
    fig = plt.figure("G_N")

    zeta = np.geomspace(zeta_min/10, zeta_max*10, 300)
    GN_zeta = GN(zeta, a_i, b_i)

    if not fig.get_axes():
        # If the plot is empty, first draw function G
        _plot_G()

    line, = plt.loglog(zeta, np.real(GN_zeta), '-', label=f"Re(G_{N})", **kwargs)
    plt.loglog(zeta, np.imag(GN_zeta), '--', color=line.get_color(), label=f"Im(G_{N})", **kwargs)

    plt.legend()


def plot_difference(a_i, b_i, **kwargs):
    N = len(a_i)
    fig = plt.figure("difference")
    if not fig.get_axes():
        plt.fill_betweenx([0, 1], zeta_min, zeta_max, color=(0,0,0,0.1))
        plt.grid(True)
        plt.xlabel(r"Dimensionless frequency $\zeta$")
        plt.ylabel(r"Relative error $| G_N(\zeta)/G(\zeta) - 1 |$")

    zeta = np.geomspace(zeta_min/10, zeta_max*10, 300)
    GN_zeta = GN(zeta, a_i, b_i)
    G_zeta = exact_G(zeta)

    rel_err = np.abs((GN_zeta - G_zeta)/G_zeta)
    plt.loglog(zeta, rel_err, label=f"N={N}", **kwargs)

    #plt.legend(loc='upper left')
    plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

def latex_table(a_i, b_i):
    N = len(a_i)
    beginning = r"""\begin{tabular}{c|c|c}
$i$ & $a_i$ & $b_i$\\ \hline
0 & 8 &  \\"""
    lines = [beginning]
    for i in range(N):
        lines.append(f"{i+1} & \\num{{{a_i[i]:.5e}}} & \\num{{{b_i[i]:.5e}}} \\\\")
    ending = r"""\end{tabular}
\vskip 1mm
Coefficients for $N=%d$

""" % N
    lines.append(ending)
    return "\n".join(lines)

def python_code(a_i, b_i):
    import textwrap
    """Export coefs a_i, b_i to a string of Python code."""
    N = len(a_i)
    a_repr = "[" + ", ".join([f"{x:.6e}" for x in a_i]) + "]"
    b_repr = "[" + ", ".join([f"{x:.6e}" for x in b_i]) + "]"
    code =  f"DIFF_REPR_COEFS_{N} = (8, \n{a_repr}, \n{b_repr})"
    wrapped_code = textwrap.fill(code, subsequent_indent=' '*8)
    return wrapped_code




if __name__ == "__main__":

    plt.figure("Function G", figsize=(4,2.7))
    _plot_G()
    plt.tight_layout()
    plt.savefig('Figure1.pdf')

    plt.figure("difference", figsize=(5,3.3))
    for N in [2,4,8,16]:
        result, a_i, b_i = optimize_ai_bi(N)
        #print(latex_table(a_i, b_i))
        print(python_code(a_i, b_i))
        plot_difference(a_i, b_i, linestyle='--', linewidth=np.log(N))
    plt.tight_layout()
    plt.savefig('Figure2.pdf')




# Output should be:
"""
DIFF_REPR_COEFS_2 = (8,  [1.023152e-01, 6.452520e-03],  [1.031475e-03,
        4.096967e-06])
DIFF_REPR_COEFS_4 = (8,  [2.101566e-01, 4.075433e-02, 8.148254e-03,
        1.961590e-03],  [1.046286e-02, 4.020925e-04, 1.622093e-05,
        5.688605e-07])
DIFF_REPR_COEFS_8 = (8,  [1.864109e-01, 8.063381e-02, 3.520992e-02,
        1.533509e-02, 6.695831e-03, 2.932507e-03, 1.328246e-03,
        9.403662e-04],  [3.168418e-02, 5.883906e-03, 1.112007e-03,
        2.116664e-04, 4.045028e-05, 7.735957e-06, 1.444918e-06,
        1.483825e-07])
DIFF_REPR_COEFS_16 = (8,  [6.878425e-05, 6.303345e-02, 1.526123e-01,
        4.035140e-02, 2.692309e-02, 1.762996e-02, 1.147201e-02,
        7.455354e-03, 4.844708e-03, 3.148891e-03, 2.047603e-03,
        1.333695e-03, 8.757532e-04, 5.984964e-04, 4.806952e-04,
        5.137644e-04],  [3.105538e+01, 1.342843e-02, 3.782608e-02,
        5.747350e-03, 2.450279e-03, 1.036183e-03, 4.375007e-04,
        1.847515e-04, 7.804746e-05, 3.297976e-05, 1.393604e-05,
        5.883296e-06, 2.469819e-06, 1.006158e-06, 3.482622e-07,
        4.253793e-08])
"""
