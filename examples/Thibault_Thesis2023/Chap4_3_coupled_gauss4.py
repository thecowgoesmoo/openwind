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
This script is part of the numerical examples accompanying Alexis THIBAULT's
Ph.D. thesis.

Gauss4 with a "mortar"-type coupling between two 1D subdomains.

This script generates Figures 4.3, 4.4a, 4.4b, and 4.4c of the thesis.
"""

import warnings

import numpy as np
from numpy import zeros, sqrt, arange
from numpy.linalg import inv
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from scipy.sparse import bmat, eye, coo_matrix, diags
from scipy.sparse.linalg import factorized

from openwind import InstrumentGeometry, InstrumentPhysics, Player
from openwind.continuous import Physics
from openwind.discretization import DiscretizedPipe
from openwind.tracker import SimulationTracker

# %% Construction des matrices d'éléments finis de nos tuyaux

rayon = 4e-2
longueur1 = 25e-2
longueur2 = 30e-2
tuyau = [[0, rayon], [longueur1, rayon],
         [longueur1+longueur2, rayon]]
instr_geom = InstrumentGeometry(tuyau)
player = Player("ZERO_FLOW")
physics = Physics(25)
celerity = physics.c(0)
instr_physics = InstrumentPhysics(
    instr_geom, temperature=25, player=player, losses=False, radiation_category='closed')
pipe1, _ = instr_physics.netlist.get_pipe_and_ends('bore0')
pipe2, _ = instr_physics.netlist.get_pipe_and_ends('bore1')


# %% Condition initiale: une fonction bump

def bump(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning) # Ignore overflows in exp
        return np.where(abs(x) < 1,
                        np.exp(-1/(1-x**2)),
                        0)


bump_c = 10e-2  # Center of the bump
bump_w = 7e-2  # Width of the bump

# %% Define the numerical scheme


def run_scheme_gauss4_coupled(l_ele=0.03, order=6, duration=0.001, n_steps=200,
                              plot_final_result=False):
    dt = duration / n_steps
    print("dt =", dt)

    # ------ Calcul des matrices d'éléments finis ------

    print("Computing FEM matrices...")

    dpipe1 = DiscretizedPipe(pipe1, l_ele=l_ele, order=order)
    dpipe2 = DiscretizedPipe(pipe2, l_ele=l_ele, order=order)
    Zc = physics.rho(0) * celerity / (np.pi * rayon**2)
    xP_1 = dpipe1.mesh.get_xH1()*longueur1
    xU_1 = dpipe1.mesh.get_xL2()*longueur1
    xP_2 = longueur1 + dpipe2.mesh.get_xH1()*longueur2
    xU_2 = longueur1 + dpipe2.mesh.get_xL2()*longueur2
    nU_1 = dpipe1.nL2
    nP_1 = dpipe1.nH1
    nU_2 = dpipe2.nL2
    nP_2 = dpipe2.nH1
    mU_1, mP_1 = dpipe1.get_mass_matrices()
    mU_2, mP_2 = dpipe2.get_mass_matrices()
    MU_1 = diags([mU_1], [0])
    MP_1 = diags([mP_1], [0])
    MU_2 = diags([mU_2], [0])
    MP_2 = diags([mP_2], [0])
    Bh_1 = dpipe1.get_Bh()
    Bh_2 = dpipe2.get_Bh()

    n1 = nU_1 + nP_1
    n2 = nU_2 + nP_2

    # Condition initiale

    P_1_init = bump((xP_1 - bump_c) / bump_w)
    U_1_init = bump((xU_1 - bump_c) / bump_w) / Zc
    P_2_init = zeros(nP_2)
    U_2_init = zeros(nU_2)

    def exact_sol(t):
        """Return U_1, P_1, U_2, P_2 at time t (assuming no reflection)"""
        P_1_t = bump((xP_1 - celerity*t - bump_c) / bump_w)
        U_1_t = bump((xU_1 - celerity*t - bump_c) / bump_w) / Zc
        P_2_t = bump((xP_2 - celerity*t - bump_c) / bump_w)
        U_2_t = bump((xU_2 - celerity*t - bump_c) / bump_w) / Zc
        return U_1_t, P_1_t, U_2_t, P_2_t

    # --- On met notre système sous la forme souhaitée ---
    """
    X_1' = A_1 X_1 + E_1 L
    X_2' = A_2 X_2 + E_2 L
    E_1.T X_1 + E_2.T X_2 = 0

    avec une distorsion du temps: un pas de temps de durée simulée dt
    correspond à un pas de temps unitaire pour X_1 et X_2.
    Pour faire cela on multiplie A_1 et A_2 par dt.
    """

    print("Transforming matrices...")

    mU_1_mh = (mU_1)**(-1/2)
    mP_1_mh = (mP_1)**(-1/2)
    mU_2_mh = (mU_2)**(-1/2)
    mP_2_mh = (mP_2)**(-1/2)
    MU_1_mh = diags(mU_1_mh)
    MP_1_mh = diags(mP_1_mh)
    MU_2_mh = diags(mU_2_mh)
    MP_2_mh = diags(mP_2_mh)

    # /!\ ON MULTIPLIE PAR DT ICI /!\
    A_1_block = dt * MU_1_mh @ Bh_1 @ MP_1_mh
    A_2_block = dt * MU_2_mh @ Bh_2 @ MP_2_mh

    A_1 = bmat([[coo_matrix((nU_1, nU_1)),                 A_1_block],
                [-A_1_block.T,  coo_matrix((nP_1, nP_1))]])

    A_2 = bmat([[coo_matrix((nU_2, nU_2)),                 A_2_block],
                [-A_2_block.T,  coo_matrix((nP_2, nP_2))]])

    E_1_tilde = zeros(nP_1)
    E_1_tilde[-1] = 1       # Select last dof
    E_2_tilde = zeros(nP_2)
    E_2_tilde[0] = -1       # Select first dof

    E_1 = np.concatenate([zeros(nU_1), mP_1_mh * E_1_tilde])
    E_2 = np.concatenate([zeros(nU_2), mP_2_mh * E_2_tilde])

    assert A_1.shape == (n1, n1)
    assert A_2.shape == (n2, n2)
    assert E_1.shape == (n1,)
    assert E_2.shape == (n2,)

    # Opérateurs de reconstruction

    getP_1 = bmat([[coo_matrix((nP_1, nU_1)), MP_1_mh]])
    getP_2 = bmat([[coo_matrix((nP_2, nU_2)), MP_2_mh]])

    assert getP_1.shape == (nP_1, n1)
    assert getP_2.shape == (nP_2, n2)

    # --- Conditions initiales ---

    X_1_init = np.concatenate([U_1_init / mU_1_mh, P_1_init / mP_1_mh])
    X_2_init = np.concatenate([U_2_init / mU_2_mh, P_2_init / mP_2_mh])

    # Plot initial conditions

    if l_ele == 0.03 and order == 6:
        plt.figure(1, figsize=(5, 3))
        plt.plot(xP_1, getP_1 @ X_1_init, "x-", label="$P_1^0$")
        plt.plot(xP_2, getP_2 @ X_2_init, "x-", label="$P_2^0$")
        plt.xlabel("x")
        plt.ylabel("Pressure")
        plt.legend()
        plt.tight_layout()
        plt.savefig("Chap4_init_cond_lele0.03_order6.png", dpi=300)

    # --- On construit les matrices du numérateur et du dénominateur ---

    print("Creating matrices N and D...")

    # Pour des gros systèmes, faire des produits matrice-vecteur successifs
    N_1 = eye(n1) + 1/2 * A_1 + 1/12 * A_1 @ A_1
    N_2 = eye(n2) + 1/2 * A_2 + 1/12 * A_2 @ A_2

    # Pour de gros systèmes, il serait préférable d'utiliser la technique de Mamadou NDiaye
    # (factorisation du polynôme et inversion pour une seule des deux racines conjuguées)
    D_1 = eye(n1) - 1/2 * A_1 + 1/12 * A_1 @ A_1
    D_2 = eye(n2) - 1/2 * A_2 + 1/12 * A_2 @ A_2

    print("Inverting D_1 and D_2...")

    invD_1 = factorized(D_1)
    invD_2 = factorized(D_2)

    # --- On calcule les matrices G et H servant à calculer les multiplicateurs de
    # Lagrange Lambda_0, Lambda_1 à chaque pas de temps ---

    print("Constructing boundary matrices...")

    I_1 = eye(n1)
    I_2 = eye(n2)

    # S'il y avait plusieurs dof de frontière ces coefficients seraient
    # des blocs matriciels
    G_00 = (E_1.T @ invD_1((I_1 / 4 - A_1 / 12) @ E_1)
            + E_2.T @ invD_2((I_2 / 4 - A_2 / 12) @ E_2))
    G_shared = E_1.T @ invD_1(E_1) + E_2.T @ invD_2(E_2)
    G_01 = G_shared * (1/4 - sqrt(3)/6)
    G_10 = G_shared * (1/4 + sqrt(3)/6)
    G_11 = G_00

    # Comme ici Lambda est scalaire, G est une matrice 2x2
    G = np.block([[G_00, G_01],
                  [G_10, G_11]])
    invG = inv(G)

    # Matrices du membre de droite de la solution de couplage
    def H_01(X_1): return - E_1.T @ invD_1((I_1 - sqrt(3)/6 * A_1) @ X_1)
    def H_02(X_2): return - E_2.T @ invD_2((I_2 - sqrt(3)/6 * A_2) @ X_2)
    # H_11 = - E_1.T @ invD_1 @ (I_1 + sqrt(3)/6 * A_1)
    # H_12 = - E_2.T @ invD_2 @ (I_2 + sqrt(3)/6 * A_2)
    def H_11(X_1): return - E_1.T @ invD_1((I_1 + sqrt(3)/6 * A_1) @ X_1)
    def H_12(X_2): return - E_2.T @ invD_2((I_2 + sqrt(3)/6 * A_2) @ X_2)
    # H_1 = np.block([[H_01],[H_11]])
    # H_2 = np.block([[H_02],[H_12]])

    # assert H_1.shape == (2,n1)
    # assert H_2.shape == (2,n2)

    # Visualisation: quels degrés de liberté influent sur les multiplicateurs?
    if False:
        plt.figure(2)
        plt.title("Influence de chaque d.d.l. sur le mult. de Lagrange $\Lambda_1$")
        plt.plot(arange(nU_1), H_11[:nU_1], "o", label="débit 1")
        plt.plot(nU_1+arange(nP_1), H_11[nU_1:], "o", label="pression 1")
        plt.plot(n1+arange(nU_2), H_12[:nU_2], "o", label="débit 2")
        plt.plot(n1+nU_2+arange(nP_2), H_12[nU_2:], "o", label="pression 2")
        plt.legend()
        plt.tight_layout()
        plt.savefig("13_couplage_influence_dof_on_Lambda1.png")

    # Préparation du vecteur A_1 E_1 qui sert dans la mise à jour
    A1E1 = A_1 @ E_1
    A2E2 = A_2 @ E_2

    # -------------------------- Boucle en temps ------------------------------

    print("Starting time loop...")

    X_1_n = X_1_init
    X_2_n = X_2_init
    # Liste des valeurs de la solution à chaque pas de temps
    XX1 = [X_1_init]
    XX2 = [X_2_init]
    EE = [X_1_init@X_1_init + X_2_init@X_2_init]
    nrj0 = EE[0]
    errs = [0]
    tracker = SimulationTracker(n_steps)
    for n in range(n_steps):
        # Calcul des multiplicateurs de Lagrange
        lambda_rhs0 = H_01(X_1_n) + H_02(X_2_n)
        lambda_rhs1 = H_11(X_1_n) + H_12(X_2_n)
        # print("lambda_rhs", lambda_rhs)
        # print(H_01 @ X_1_n + H_02 @ X_2_n)
        # print(H_11 @ X_1_n + H_12 @ X_2_n)
        Lambda0, Lambda1 = invG @ np.array([lambda_rhs0, lambda_rhs1])
        # Lambda0, Lambda1 = 0, 0 # No coupling
        Lplus = Lambda0 + Lambda1
        Lminus = Lambda0 - Lambda1

        # Mise à jour dans chaque sous-domaine
        X_1_np1 = invD_1(N_1 @ X_1_n + 1/2 * E_1 * Lplus +
                         sqrt(3)/12 * A1E1 * Lminus)
        X_2_np1 = invD_2(N_2 @ X_2_n + 1/2 * E_2 * Lplus +
                         sqrt(3)/12 * A2E2 * Lminus)

        # DEBUG Vérification de la condition de raccord au pas n+1
        assert abs(E_1.T @ X_1_np1 + E_2.T @
                   X_2_np1) < 1e-10, f"Echec de la condition de couplage (n={n})"

        # DEBUG Calcul de l'énergie
        E_np1 = X_1_np1 @ X_1_np1 + X_2_np1 @ X_2_np1
        assert abs(E_np1 - EE[-1]) / \
            EE[-1] < 1e-10, f"Echec du bilan d'énergie (n={n})"
        EE.append(E_np1)

        # DEBUG Vérification de la condition de raccord aux pas de collocation
        # X_1_c0 = invD_1 @ ((I_1 - sqrt(3)/6 * A_1) @ X_1_n + (1/4*I_1 - 1/12*A_1) @ E_1 * Lambda0 +    (1/4 - sqrt(3)/6) * E_1 * Lambda1)
        # X_2_c0 = invD_2 @ ((I_2 - sqrt(3)/6 * A_2) @ X_2_n + (1/4*I_2 - 1/12*A_2) @ E_2 * Lambda0 +    (1/4 - sqrt(3)/6) * E_2 * Lambda1)
        # X_1_c1 = invD_1 @ ((I_1 + sqrt(3)/6 * A_1) @ X_1_n +    (1/4 + sqrt(3)/6) * E_1 * Lambda0 + (1/4*I_1 - 1/12*A_1) @ E_1 * Lambda1)
        # X_2_c1 = invD_2 @ ((I_2 + sqrt(3)/6 * A_2) @ X_2_n +    (1/4 + sqrt(3)/6) * E_2 * Lambda0 + (1/4*I_2 - 1/12*A_2) @ E_2 * Lambda1)
        # err_c0 = E_1.T @ X_1_c0 + E_2.T @ X_2_c0
        # err_c1 = E_1.T @ X_1_c1 + E_2.T @ X_2_c1
        # err_c0 = G_00 * Lambda0 + G_01 * Lambda1 - (H_01 @ X_1_n + H_02 @  X_2_n)
        # err_c1 = G_10 * Lambda0 + G_11 * Lambda1 - (H_11 @ X_1_n + H_12 @ X_2_n)
        # print("err_c0 =", err_c0)
        # print("err_c1 =", err_c1)
        # print(f"err_c0 / err_c1 = {err_c0 / err_c1:.3e}")

        # CONVERGENCE Calcul de l'énergie de l'erreur avec la solution exacte
        U_1x, P_1x, U_2x, P_2x = exact_sol((n+1)*dt)
        X_1x = np.concatenate([U_1x / mU_1_mh, P_1x / mP_1_mh])
        X_2x = np.concatenate([U_2x / mU_2_mh, P_2x / mP_2_mh])
        err_1 = X_1x - X_1_np1
        err_2 = X_2x - X_2_np1
        nrj_err = err_1 @ err_1 + err_2 @ err_2
        errs.append(np.sqrt(nrj_err / nrj0))

        # On boucle la boucle
        X_1_n = X_1_np1
        X_2_n = X_2_np1
        XX1.append(X_1_n)
        XX2.append(X_2_n)
        tracker.update()

    # Plot final result

    if plot_final_result:
        plt.figure(11, figsize=(4, 2.6))
        plt.clf()
        plt.plot(xP_1, getP_1 @ X_1_n, "x-", label="$P_1^N$")
        plt.plot(xP_2, getP_2 @ X_2_n, "x-", label="$P_2^N$")
        _, P_1_exact, _, P_2_exact = exact_sol(duration)
        plt.plot(xP_1, P_1_exact, ":k", label="Exact")
        plt.plot(xP_2, P_2_exact, ":k")
        plt.xlabel("x")
        plt.ylabel("Pressure")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"Chap4_final_nsteps{n_steps}_order{order}.png", dpi=300)

    return locals()


# %% Show animation

def show_gauss4_animation(write_to_file=False, l_ele=0.01, order=6, n_steps=200):
    res = run_scheme_gauss4_coupled(l_ele=l_ele, order=order, n_steps=n_steps)

    xP_1 = res["xP_1"]
    xP_2 = res["xP_2"]
    xU_1 = res["xU_1"]
    xU_2 = res["xU_2"]
    getP_1 = res["getP_1"]
    getP_2 = res["getP_2"]
    P_1_init = res["P_1_init"]
    P_2_init = res["P_2_init"]
    dt = res["dt"]
    XX1 = res["XX1"]
    XX2 = res["XX2"]
    exact_sol = res["exact_sol"]  # a function

    fig = plt.figure(figsize=(4, 2.6))
    ax = fig.gca()
    line1, = plt.plot(xP_1, getP_1 @ XX1[0], "x-")
    line2, = plt.plot(xP_2, getP_2 @ XX2[0], "x-")
    def cat(x, y): return np.concatenate([x, y])
    line3, = plt.plot(cat(xP_1, xP_2), cat(P_1_init, P_2_init), ":k")
    plt.xlabel("x")
    plt.ylabel("Pressure")
    ax.set_title("$\\Delta t = 0$")
    plt.tight_layout()

    def next_frame(frame):
        line1.set_data(xP_1, getP_1 @ XX1[frame])
        line2.set_data(xP_2, getP_2 @ XX2[frame])
        t = frame * dt
        _, P_1_exact, _, P_2_exact = exact_sol(t)
        line3.set_data(cat(xP_1, xP_2), cat(P_1_exact, P_2_exact))
        ax.set_title(f"$\\Delta t = {int(dt*frame*1e6)}$ µs")
        return line1, line2

    anim = FuncAnimation(fig, next_frame, len(XX1), interval=60*100//n_steps)

    if write_to_file:  # Export the animation instead of showing it interactively?
        f = f"Chap4_animation_{n_steps}.gif"
        print(f"Exporting video to {f}...")
        # writergif = animation.PillowWriter(fps=17)
        # fps = int(np.round(17*n_steps/100))
        fps = 17*n_steps/100
        writergif = animation.ImageMagickFileWriter(fps=fps)
        anim.save(f, writer=writergif)




# %% Run the same computation with several discretizations

def run_and_plot_gauss4(nn_steps, show_animation=False, write_to_file=False):
    duration = 0.001
    P_1s = []
    P_2s = []
    xP_1s = []
    xP_2s = []
    dts = duration / nn_steps
    order = 6        # en espace
    # 1 élément = distance parcourue en order pas de temps
    l_eles = celerity * dts * order
    maxerrs = []
    for l_ele, n_steps in zip(l_eles, nn_steps):
        print("*"*50)
        print(f"Running coupled Gauss4, n_steps={n_steps}, l_ele={l_ele}")
        print("*"*50)
        res = run_scheme_gauss4_coupled(l_ele=l_ele, order=order, duration=duration, n_steps=n_steps,
                                        plot_final_result=True)
        errs = res["errs"]
        maxerrs.append(np.max(errs))

        P_1s.append(res["getP_1"] @ res["X_1_n"])
        P_2s.append(res["getP_2"] @ res["X_2_n"])
        xP_1s.append(res["xP_1"])
        xP_2s.append(res["xP_2"])

        if show_animation:
            show_gauss4_animation(write_to_file=write_to_file, l_ele=l_ele, order=order, n_steps=n_steps)



if __name__ == "__main__":
    nn_steps = np.array([20, 50, 100])
    run_and_plot_gauss4(nn_steps, show_animation=True, write_to_file=True)
