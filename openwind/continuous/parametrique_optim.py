# -*- coding: utf-8 -*-
r"""
Created on Wed Dec 14 13:12:06 2022

Optimisation des Ri, Li pour la rugosité paramétrique.

On part de:

Zv_t(s) = zeta_0 + zeta_1 s / sqrt(s + w1)
                 + zeta_2 s / sqrt(s + w2)
                 + zeta_3 s

qui peut être représentée comme

Zv_t(s) = zeta_0 + zeta_1 \int_w1^\infty s/(s+xi) mu(xi-w1) dxi
                 + zeta_2 \int_w2^\infty s/(s+xi) mu(xi-w2) dxi
                 + zeta_3 s

et on veut arriver à

Zv^N(s) = R_0 + \sum_{i=1}^N Ri Li s / (Ri + Li s) + m_u s

On peut directement poser

    R_0 = zeta_0
    m_u = zeta_3

et pour les Ri, Li, on effectue une optimisation. On pose

    Zl(s) =   zeta_1 \int_w1^\infty s/(s+xi) mu(xi-w1) dxi
            + zeta_2 \int_w2^\infty s/(s+xi) mu(xi-w2) dxi

la partie correspondant aux pertes intégrales, que l'on souhaite approximer
par

    Zl^N(s) = \sum_{i=1}^N Ri Li s / (Ri + Li s).

Afin de garantir la positivité des paramètres Ri, Li, on pose

    Ri = exp(ri)
    Li = exp(li)

et on choisit comme fonction de coût:

    J(ri, li) = \sum_{k=1}^M |Zl^n(iwk) / Zl(iwk) - 1|^2


On peut utiliser la même fonction pour obtenir les coefficients Gi, Ci
correspondant à Yt_tilde.



@author: Alexis
"""

import warnings

import numpy as np
from numpy import sqrt, exp, log
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#%% On définit la fonction Zlosses_N qui approxime Zlosses avec N pôles

def Zlosses_N(s, Ri, Li):
    # Ri, Li must have shape (N,1)
    return np.sum(Ri * Li * s / (Ri + Li * s), axis=0)

#%% Définition de la méthode d'optimisation

def optimize_RiLi_roughness(omega_1, omega_2,         # Parametric cutoff freqs
                            zeta_0, zeta_1, zeta_2, zeta_3,  # Coefs of Zv_tilde
                            omega_min, omega_max,     # Frequency range
                            N=8,                      # Number of poles
                            M=100,                    # Number of freqs for optim
                            display=False             # Plot the curves?
                            ):

    #% On définit les fonctions Zv_tilde et Zlosses (modèle irrationnel)

    def Zv_tilde(s):
        return zeta_0 + zeta_1 * s / sqrt(s + omega_1) \
                      + zeta_2 * s / sqrt(s + omega_2) \
                      + zeta_3 * s

    def Zlosses(s):
        return zeta_1 * s / sqrt(s + omega_1) \
             + zeta_2 * s / sqrt(s + omega_2)

    #% On définit la fonction de coût

    omega_kk = np.geomspace(omega_min, omega_max, M)

    def param_to_physical(rili):
        # rili is the vector of parameters (ri, li) of size (N,2)
        # print(rili.shape)
        assert rili.shape == (2*N,)
        ri = rili[:N,None]
        li = rili[N:,None]
        Ri = exp(ri)
        Li = exp(li)
        assert Ri.shape == (N,1)
        assert Li.shape == (N,1)
        return Ri, Li

    def physical_to_param(Ri, Li):
        assert Ri.shape == (N,1)
        assert Li.shape == (N,1)
        rili = log(np.ravel([Ri.T, Li.T]))
        assert rili.shape == (2*N,)
        return rili

    def cost_function(rili):
        Ri, Li = param_to_physical(rili)
        Zl = Zlosses(1j*omega_kk)
        Zl_N = Zlosses_N(1j*omega_kk, Ri, Li)
        assert Zl.shape == (M,)
        assert Zl_N.shape == (M,)

        return np.sum(np.abs(Zl_N / Zl - 1)**2)

    #% Condition initiale: on prend une approximation grossière de notre intégrale

    # On répartit des pôles géométriquement sur ]omega_start, omega_max].
    omega_start = max(omega_1, omega_min)
    xi_i = np.geomspace(omega_max, omega_start, N, endpoint=False)
    dxi = abs(np.gradient(xi_i))
    # Le coefficient vaut zeta_1 / sqrt(xi-w1) si on est sur [omega_1, omega_2]
    # et on y rajoute zeta_2 / sqrt(xi-w2) si on est au delà de omega_2
    coef_i  = zeta_1 / sqrt(xi_i - omega_1) * dxi
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value encountered in sqrt", RuntimeWarning)
        coef_i += np.where(xi_i > omega_2, zeta_2 / sqrt(xi_i - omega_2) * dxi, 0)
    # A partir de là on calcule les coefficients Ri, Li de l'approximation initiale
    Ri0 = coef_i[:, np.newaxis]
    Li0 = (coef_i / xi_i)[:, np.newaxis]


    if display:
        plt.figure(1, figsize=(4,2.6))
        # On affiche la fonction Zlosses sur la gamme [omega_min, omega_max]
        # et son approximation initiale
        omegas = np.geomspace(omega_min, omega_max, 300)
        ss = 1j*omegas
        if not plt.gca().lines: # show only when plot is empty
            plt.loglog(omegas, Zlosses(ss).real, '-k', label=r"$Z_{loss}^\otimes(i\omega)$", linewidth=0.8)
            plt.loglog(omegas, Zlosses(ss).imag, '--k', linewidth=0.8)
        # plt.loglog(omegas, Zlosses_N(ss, Ri0, Li0).real, '-C0', label="initial approx")
        # plt.loglog(omegas, Zlosses_N(ss, Ri0, Li0).imag, '--C0')

    #% Optimisation des paramètres Ri, Li

    rili0 = physical_to_param(Ri0, Li0)
    res = minimize(cost_function, rili0, method='BFGS')

    Ri_opti = exp(res.x[:N,None])
    Li_opti = exp(res.x[N:,None])

    #%% Affichage de la fonction Zlosses_N optimisée

    if display:
        plt.figure(1)
        # line, = plt.loglog(omegas, Zlosses_N(ss, Ri_opti, Li_opti).real, '-',
        #                    # label=f"$Z_{{loss}}^{N}(i\\omega)$",
        #                    label=f"N={N}",
        #                    )
        # plt.loglog(omegas, Zlosses_N(ss, Ri_opti, Li_opti).imag, '--',
        #            color=line.get_color())

        Zl = Zlosses(1j*omega_kk)
        Zl_N = Zlosses_N(1j*omega_kk, Ri_opti, Li_opti)


        plt.grid(True, 'both')
        plt.legend()

        plt.figure(2, figsize=(4,2.6))
        plt.loglog(omega_kk, np.abs(Zl_N / Zl - 1), '-', linewidth=0.8, label=f"Rel. err. N={N}",
                   # color=line.get_color()
                   )
        plt.text(omega_kk[0], np.abs(Zl_N / Zl - 1)[0], f"N={N}")

        plt.grid(True, 'both')
        # plt.legend()

    return Ri_opti, Li_opti


#%% Testons la fonction!


if __name__ == "__main__":
    # Valeurs arbitraires juste pour tester!
    # zeta_0 = 0
    # zeta_1 = 1
    # zeta_2 = 2
    # zeta_3 = 0
    # omega_1 = 1
    # omega_2 = 30

    # omega_min = 0.1
    # omega_max = 1000

    zeta_0, zeta_1, zeta_2, zeta_3 = (3.7566873863533847, 1.370526794038224, 0.0, 1.0000000000000002)
    nu_0, nu_1, nu_2, nu_3 = (0, 0.6413241014369092, 0.0, 1.0)
    omega_1, omega_2 = (0.469585923294173, 6762.037295436091)
    omega_min, omega_max = (125.66370614359172, 125663.70614359173)

    for N in [3,4,8,
               12,20
              ]:
        Ri, Li = optimize_RiLi_roughness(omega_1, omega_2, zeta_0, zeta_1, zeta_2, zeta_3,
                                         omega_min, omega_max,
                                         N=N, display=True)
        print("N =", N)
        print("Ri =", Ri)
        print("Li =", Li)
        print("*"*20)


    plt.figure(1)
    plt.xlabel("$\\omega$")
    plt.tight_layout()
    plt.savefig("parametrique-optim-test-Zv.png", dpi=300)

    plt.figure(2)
    # plt.legend(ncol=3, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left')
    plt.xlabel("$\\omega$")
    plt.tight_layout()
    plt.savefig("parametrique-optim-test-err.png", dpi=300)
    # TODO récupérer les coefficients zeta_0, zeta_1, zeta_2, zeta_3,
    # omega_1, omega_2, issus de la géométrie 'tournesol'
