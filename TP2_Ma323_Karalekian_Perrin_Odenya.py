############# Ma323 -- TP2 : Equation d'advection ###############
###### David Karalekian - Emerick Perrin - Emmanuel Odenya ######

### Imports

import numpy as np
import matplotlib.pyplot as plt


### Fonctions utiles

def matriceM(N, a, b, c):
    """Construit une matrice carrée de taille N, tridiagonale.
    Le coefficient sur la diagonale est a.
    Le coefficient au dessus est b et celui au dessous est c."""
    A = a*np.eye(N)
    for i in range(N-1):
        A[i, i+1] = b
        A[i+1, i] = c
        A[0, N-1] = c # On rajoute ici les conditons de périodicité
        A[N-1, 0] = b
    return A


def U0(x):
    res = (np.sin(np.pi*x))**10
    return res


V = 2

Tmin = 0
Tmax = 2

H = np.array([0.02, 0.002, 0.002, 0.005])
N = np.array([int(1/H[0]), int(1/H[1]), int(1/H[2]), int(1/H[3])])
Tau = np.array([0.01, 0.005, 0.002, 0.0002])
C = np.array([V*Tau[0]/H[0], V*Tau[1]/H[1], V*Tau[2]/H[2], V*Tau[3]/H[3] ])





### SCHEMA EXPLICITE CENTRE

# Me = matriceM(N, -c/2, c/2)

Me0 = matriceM(N[0], 1, -C[0]/2, C[0]/2)
Me1 = matriceM(N[1], 1, -C[1]/2, C[1]/2)
Me2 = matriceM(N[2], 1, -C[2]/2, C[2]/2)
Me3 = matriceM(N[3], 1, -C[3]/2, C[3]/2)


def SolutionExplicite(h, tau, Me, N):
    """ Dans le schéma explicite U_n+1 = Me U_n """
    ntfinal = int(Tmax/tau)
    ntdemi = int(ntfinal/2)
    X = np.linspace(0,1,N)
    T = np.arange(ntfinal + 1)*tau
    U = np.zeros((ntfinal, N))
    U[0, : ] = U0(X)
    for i in range(ntfinal-1):
        U[i+1, : ] = Me@U[i , : ]
    rep = U[ntdemi, : ]
    return U, T, X, rep


# Courbes pour le schéma explicite

""" Cas h = 0.02 tau = 0.01 """

Ue1, Te1, Xe1, rep1 = SolutionExplicite(H[0], Tau[0], Me0, N[0])

plt.plot(Xe1, Ue1[0, : ], label = 't = 0')
# plt.plot(Xe1, rep1, label = 't = 1')
# plt.plot(Xe1, Ue1[-1, : ], label = 't = 2')
plt.grid()
plt.legend()
plt.title('Explicite centré h = 0.02 tau = 0.01')
plt.show()

plt.plot(Xe1, rep1, label = 't = 1')
plt.grid()
plt.legend()
plt.title('Explicite centré h = 0.02 tau = 0.01')
plt.show()

plt.plot(Xe1, Ue1[-1, : ], label = 't = 2')
plt.grid()
plt.legend()
plt.title('Explicite centré h = 0.02 tau = 0.01')
plt.show()

""" Cas h = 0.002 tau = 0.005 """

Ue2, Te2, Xe2, rep2 = SolutionExplicite(H[0], Tau[0], Me1, N[1])

plt.plot(Xe2, Ue2[0, : ], label = 't = 0')
plt.plot(Xe2, rep2, label = 't = 1')
plt.plot(Xe2, Ue2[-1, : ], label = 't = 2')
plt.grid()
plt.legend()
plt.title('Explicite centré h = 0.002 tau = 0.005')
plt.show()

""" Cas h = 0.002 tau = 0.002 """

Ue3, Te3, Xe3, rep3 = SolutionExplicite(H[0], Tau[0], Me2, N[2])

plt.plot(Xe3, Ue3[0, : ], label = 't = 0')
plt.plot(Xe3, rep3, label = 't = 1')
plt.plot(Xe3, Ue3[-1, : ], label = 't = 2')
plt.grid()
plt.legend()
plt.title('Explicite centré h = 0.002 tau = 0.002')
plt.show()

""" Cas h = 0.005 tau = 0.0002 """

Ue4, Te4, Xe4, rep4 = SolutionExplicite(H[0], Tau[0], Me3, N[3])

plt.plot(Xe4, Ue4[0, : ], label = 't = 0')
plt.plot(Xe4, rep4, label = 't = 1')
plt.plot(Xe4, Ue4[-1, : ], label = 't = 2')
plt.grid()
plt.legend()
plt.title('Explicite centré h = 0.005 tau = 0.0002')
plt.show()





### SCHEMA IMPLICITE CENTRE

# Mi = matriceM(N, 1, c/2, -c/2)

Mi0 = matriceM(N[0], 1, C[0]/2, -C[0]/2)
Mi1 = matriceM(N[1], 1, C[1]/2, -C[1]/2)
Mi2 = matriceM(N[2], 1, C[2]/2, -C[2]/2)
Mi3 = matriceM(N[3], 1, C[3]/2, -C[3]/2)

def SolutionImplicite(h, tau, Mi, N):
    """ Dans le schéma implicite Mi U_n+1 = U_n """
    ntfinal = int(Tmax/tau)
    ndemi = int(ntfinal/2)
    X = np.linspace(0,1,N)
    T = np.arange(ntfinal + 1)*tau
    U = np.zeros((ntfinal, N))
    U[0, : ] = U0(X)
    for i in range(ntfinal-1):
        U[i+1, : ] = np.linalg.solve(Mi, U[i , : ])
    answ = U[ndemi, : ]
    return U, T, X, answ


# Courbes pour le schéma implicite

""" Cas h = 0.02 tau = 0.01 """

Ui1, Ti1, Xi1, answ1 = SolutionImplicite(H[0], Tau[0], Mi0, N[0])

plt.plot(Xi1, Ui1[0, : ], ':', label = 't = 0')
plt.plot(Xi1, answ1, '--', label = 't = 1')
plt.plot(Xi1, Ui1[-1, : ], '-.', label = 't = 2')
plt.grid()
plt.legend()
plt.title('Implicite centré h = 0.02 tau = 0.01')
plt.show()

""" Cas h = 0.002 tau = 0.005 """

Ui2, Ti2, Xi2, answ2 = SolutionImplicite(H[1], Tau[1], Mi1, N[1])

plt.plot(Xi2, Ui2[0, : ], label = 't = 0')
plt.plot(Xi2, answ2, label = 't = 1')
plt.plot(Xi2, Ui2[-1, : ], label = 't = 2')
plt.grid()
plt.legend()
plt.title('Implicite centré h = 0.002 tau = 0.005')
plt.show()

""" Cas h = 0.002 tau = 0.002 """

Ui3, Ti3, Xi3, answ3 = SolutionImplicite(H[2], Tau[2], Mi2, N[2])

plt.plot(Xi3, Ui3[0, : ], label = 't = 0')
plt.plot(Xi3, answ3, label = 't = 1')
plt.plot(Xi3, Ui3[-1, : ], label = 't = 2')
plt.grid()
plt.legend()
plt.title('Implicite centré h = 0.002 tau = 0.002')
plt.show()

""" Cas h = 0.005 tau = 0.0002 """

Ui4, Ti4, Xi4, answ4 = SolutionImplicite(H[3], Tau[3], Mi3, N[3])

plt.plot(Xi4, Ui4[0, : ], label = 't = 0')
plt.plot(Xi4, answ4, label = 't = 1')
plt.plot(Xi4, Ui4[-1, : ], label = 't = 2')
plt.grid()
plt.legend()
plt.title('Implicite centré h = 0.005 tau = 0.0002')
plt.show()




### SCHEMA EXPLICITE DECENTRE AMONT

# Mda = matriceM(N, 1-c, 0, c)

Mda0 = matriceM(N[0], 1 - C[0], 0, C[0])
Mda1 = matriceM(N[1], 1 - C[1], 0, C[1])
Mda2 = matriceM(N[2], 1 - C[2], 0, C[2])
Mda3 = matriceM(N[3], 1 - C[3], 0, C[3])

def SolutionDecentreAmont(h, tau, Mda, N):
    """ Dans le schéma explicite décentré amont U_n+1 = Mda U_n """
    ntfinal = int(Tmax/tau)
    ndemi = int(ntfinal/2)
    X = np.linspace(0,1,N)
    T = np.arange(ntfinal + 1)*tau
    U = np.zeros((ntfinal, N))
    U[0, : ] = U0(X)
    for i in range(ntfinal-1):
        U[ i+1, : ] = Mda@U[ i , : ]
    ans = U[ndemi, : ]
    return U, T, X, ans


# Courbes pour le schéma décentré amont

""" Cas h = 0.02 tau = 0.01 """

Ud1, Td1, Xd1, ans1 = SolutionDecentreAmont(H[0], Tau[0], Mda0, N[0])

plt.plot(Xd1, Ud1[0, : ], 'x', label = 't = 0')
plt.plot(Xd1, ans1, label = 't = 1')
plt.plot(Xd1, Ud1[-1, : ], label = 't = 2')
plt.grid()
plt.legend()
plt.title('Explicite décentré amont h = 0.02 tau = 0.01')
plt.show()

""" Cas h = 0.002 tau = 0.005 """

Ud2, Td2, Xd2, ans2 = SolutionDecentreAmont(H[1], Tau[1], Mda1, N[1])

plt.plot(Xd2, Ud2[0, : ], label = 't = 0')
plt.plot(Xd2, ans2, label = 't = 1')
plt.plot(Xd2, Ud2[-1, : ], label = 't = 2')
plt.grid()
plt.legend()
plt.title('Explicite décentré amont h = 0.002 tau = 0.005')
plt.show()

""" Cas h = 0.002 tau = 0.002 """

Ud3, Td3, Xd3, ans3 = SolutionDecentreAmont(H[2], Tau[2], Mda2, N[2])

plt.plot(Xd3, Ud3[0, : ], label = 't = 0')
plt.plot(Xd3, ans3, label = 't = 1')
plt.plot(Xd3, Ud3[-1, : ], label = 't = 2')
plt.grid()
plt.legend()
plt.title('Explicite décentré amont h = 0.002 tau = 0.002')
plt.show()

""" Cas h = 0.005 tau = 0.0002 """

Ud4, Td4, Xd4, ans4 = SolutionDecentreAmont(H[3], Tau[3], Mda3, N[3])

plt.plot(Xd4, Ud4[0, : ], label = 't = 0')
plt.plot(Xd4, ans4, label = 't = 1')
plt.plot(Xd4, Ud4[-1, : ], label = 't = 2')
plt.grid()
plt.legend()
plt.title('Explicite décentré amont h = 0.005 tau = 0.0002')
plt.show()





### SCHEMA DE LAX-FRIEDRICHS

# M_LF = matriceM(N, 0, (1-c)/2, (1+c)/2)

M_LF0 = matriceM(N[0], 0, (1 - C[0])/2, (1+C[0])/2)
M_LF1 = matriceM(N[1], 0, (1 - C[1])/2, (1+C[1])/2)
M_LF2 = matriceM(N[2], 0, (1 - C[2])/2, (1+C[2])/2)
M_LF3 = matriceM(N[3], 0, (1 - C[3])/2, (1+C[3])/2)

def SolutionLF(h, tau, M_LF, N):
    """ Dans le schéma de Lax-Friedrichs U_n+1 = M_LF U_n """
    ntfinal = int(Tmax/tau)
    ntdemi = int(ntfinal/2)
    X = np.linspace(0,1,N)
    T = np.arange(ntfinal + 1)*tau
    U = np.zeros((ntfinal, N))
    U[0, : ] = U0(X)
    for i in range(ntfinal-1):
        U[ i+1, : ] = M_LF@U[ i , : ]
    demi = U[ntdemi, : ]
    return U, T, X, demi


# Courbes pour le schéma de Lax-Friedrichs

""" Cas h = 0.02 tau = 0.01 """

U_LF1, T_LF1, X_LF1, demi1 = SolutionLF(H[0], Tau[0], M_LF0, N[0])

plt.plot(X_LF1, U_LF1[0, : ], 'x', label = 't = 0', color ='b')
plt.plot(X_LF1, demi1, label = 't = 1', color = 'r')
plt.plot(X_LF1, U_LF1[-1, : ], label = 't = 2', color = 'g')
plt.grid()
plt.legend()
plt.title('Lax-Friedrichs h = 0.02 tau = 0.01')
plt.show()

""" Cas h = 0.002 tau = 0.005 """

U_LF2, T_LF2, X_LF2, demi2 = SolutionLF(H[1], Tau[1], M_LF1, N[1])

plt.plot(X_LF2, U_LF2[0, : ], label = 't = 0')
plt.plot(X_LF2, demi2, label = 't = 1')
plt.plot(X_LF2, U_LF2[-1, : ], label = 't = 2')
plt.grid()
plt.legend()
plt.title('Lax-Friedrichs h = 0.002 tau = 0.005')
plt.show()

""" Cas h = 0.002 tau = 0.002 """

U_LF3, T_LF3, X_LF3, demi3 = SolutionLF(H[2], Tau[2], M_LF2, N[2])

plt.plot(X_LF3, U_LF3[0, : ], label = 't = 0')
plt.plot(X_LF3, demi3, label = 't = 1')
plt.plot(X_LF3, U_LF3[-1, : ], label = 't = 2')
plt.grid()
plt.legend()
plt.title('Lax-Friedrichs h = 0.002 tau = 0.002')
plt.show()

""" Cas h = 0.005 tau = 0.0002 """

U_LF4, T_LF4, X_LF4, demi4 = SolutionLF(H[3], Tau[3], M_LF3, N[3])

plt.plot(X_LF4, U_LF4[0, : ], label = 't = 0')
plt.plot(X_LF4, demi4, label = 't = 1')
plt.plot(X_LF4, U_LF4[-1, : ], label = 't = 2')
plt.grid()
plt.legend()
plt.title('Lax-Friedrichs h = 0.005 tau = 0.0002')
plt.show()




### SCHEMA DE LAX-WENDROFF

# M_LW = matriceM(N, 1 - c**2, (c**2 - c)/2, (c + c**2)/2)

M_LW0 = matriceM(N[0], 1 - C[0]**2, (C[0]**2 - C[0])/2, (C[0] + C[0]**2)/2)
M_LW1 = matriceM(N[1], 1 - C[1]**2, (C[1]**2 - C[1])/2, (C[1] + C[1]**2)/2)
M_LW2 = matriceM(N[2], 1 - C[2]**2, (C[2]**2 - C[2])/2, (C[2] + C[2]**2)/2)
M_LW3 = matriceM(N[3], 1 - C[3]**2, (C[3]**2 - C[3])/2, (C[3] + C[3]**2)/2)

def SolutionLW(h, tau, M_LW, N):
    """ Dans le schéma de Lax-Wendroff U_n+1 = M_LW U_n """
    ntfinal = int(Tmax/tau)
    ndemi = int(ntfinal/2)
    X = np.linspace(0,1,N)
    T = np.arange(ntfinal + 1)*tau
    U = np.zeros((ntfinal, N))
    U[0, : ] = U0(X)
    for i in range(ntfinal-1):
        U[ i+1, : ] = M_LW@U[ i , : ]
    res = U[ndemi, : ]
    return U, T, X, res


# Courbes pour le schéma de Lax-Wendroff

""" Cas h = 0.02 tau = 0.01 """

U_LW1, T_LW1, X_LW1, res1 = SolutionLW(H[0], Tau[0], M_LW0, N[0])

plt.plot(X_LW1, U_LW1[0, : ], '>', label = 't = 0')
plt.plot(X_LW1, res1, label = 't = 1')
plt.plot(X_LW1, U_LW1[-1, : ], label = 't = 2')
plt.grid()
plt.legend()
plt.title('Lax-Wendroff h = 0.02 tau = 0.01')
plt.show()

""" Cas h = 0.002 tau = 0.005 """

U_LW2, T_LW2, X_LW2, res2 = SolutionLW(H[1], Tau[1], M_LW1, N[1])

plt.plot(X_LW2, U_LW2[0, : ], label = 't = 0')
plt.plot(X_LW2, res2, label = 't = 1')
plt.plot(X_LW2, U_LW2[-1, : ], label = 't = 2')
plt.grid()
plt.legend()
plt.title('Lax-Wendroff h = 0.002 tau = 0.005')
plt.show()

""" Cas h = 0.002 tau = 0.002 """

U_LW3, T_LW3, X_LW3, res3 = SolutionLW(H[2], Tau[2], M_LW2, N[2])

plt.plot(X_LW3, U_LW3[0, : ], label = 't = 0')
plt.plot(X_LW3, res3, label = 't = 1')
plt.plot(X_LW3, U_LW3[-1, : ], label = 't = 2')
plt.grid()
plt.legend()
plt.title('Lax-Wendroff h = 0.002 tau = 0.002')
plt.show()

""" Cas h = 0.005 tau = 0.0002 """

U_LW4, T_LW4, X_LW4, res4 = SolutionLW(H[3], Tau[3], M_LW3, N[3])

plt.plot(X_LW4, U_LW4[0, : ], label = 't = 0')
plt.plot(X_LW4, res4, label = 't = 1')
plt.plot(X_LW4, U_LW4[-1, : ], label = 't = 2')
plt.grid()
plt.legend()
plt.title('Lax-Wendroff h = 0.005 tau = 0.0002')
plt.show()





# ### SCHEMA EXPLICITE DECENTRE AVAL

# Md = matriceM(N, 1-c, -c, 0)

Md0 = matriceM(N[0], 1 - C[0], -C[0], 0)
Md1 = matriceM(N[1], 1 - C[1], -C[1], 0)
Md2 = matriceM(N[2], 1 - C[2], -C[2], 0)
Md3 = matriceM(N[3], 1 - C[3], -C[3], 0)

def SolutionDecentreAval(h, tau, Md, N):
    """ Dans le schéma explicite décentré amont U_n+1 = Mda U_n """
    ntfinal = int(Tmax/tau)
    ndemi = int(ntfinal/2)
    X = np.linspace(0,1,N)
    T = np.arange(ntfinal + 1)*tau
    U = np.zeros((ntfinal, N))
    U[0, : ] = U0(X)
    for i in range(ntfinal-1):
        U[ i+1, : ] = Md@U[ i , : ]
    ans = U[ndemi, : ]
    return U, T, X, ans


# Courbes pour le schéma décentré aval (nous ne testerons ce code que dans les cas où la CFL est respectée)

""" Cas h = 0.02 tau = 0.01 """

Uda1, Tda1, Xda1, ansa1 = SolutionDecentreAval(H[0], Tau[0], Md0, N[0])

plt.plot(Xda1, Uda1[0, : ], label = 't = 0')
plt.plot(Xda1, ansa1, label = 't = 1')
plt.plot(Xda1, Uda1[-1, : ], label = 't = 2')
plt.grid()
plt.legend()
plt.title('Explicite décentré aval h = 0.02 tau = 0.01')
plt.show()


""" Cas h = 0.005 tau = 0.0002 """

Uda4, Tda4, Xda4, ansa4 = SolutionDecentreAval(H[3], Tau[3], Md3, N[3])

#plt.plot(Xda4, Uda4[0, : ], label = 't = 0')
plt.plot(Xda4, ansa4, label = 't = 1')
plt.plot(Xda4, Uda4[-1, : ], label = 't = 2')
plt.grid()
plt.legend()
plt.title('Explicite décentré aval h = 0.005 tau = 0.0002')
plt.show()