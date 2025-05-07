from functools import partial
from scipy.special import legendre
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import math

def lobatto_quad(p):
    """Gauss Lobatto quadrature.
    Args:
        p (int) = order of quadrature

    Returns:
        nodal_pts (np.array) = nodal points of quadrature
        w (np.array) = correspodent weights of the quarature.
    """
    # nodes
    x_0 = np.cos(np.arange(1, p) / p * np.pi)
    nodal_pts = np.zeros((p + 1))
    # final and inital pt
    nodal_pts[0] = 1
    nodal_pts[-1] = -1
    # Newton method for root finding
    for i, ch_pt in enumerate(x_0):
        leg_p = partial(_legendre_prime_lobatto, n=p)
        leg_pp = partial(_legendre_double_prime, n=p)
        nodal_pts[i + 1] = _newton_method(leg_p, leg_pp, ch_pt, 100)

    # weights
    #print('T',p * (p + 1) * (legendre(p)(nodal_pts)) ** 2)
    weights = 2 / (p * (p + 1) * (legendre(p)(nodal_pts)) ** 2)
    return nodal_pts[::-1], weights

def _legendre_prime(x, n):
    """Calculate first derivative of the nth Legendre Polynomial recursively.
    Args:
        x (float,np.array) = domain.
        n (int) = degree of Legendre polynomial (L_n).
    Return:
        legendre_p (np.array) = value first derivative of L_n.
    """
    # P'_n+1 = (2n+1) P_n + P'_n-1
    # where P'_0 = 0 and P'_1 = 1
    # source: http://www.physicspages.com/2011/03/12/legendre-polynomials-recurrence-relations-ode/
    if n == 0:
        if isinstance(x, np.ndarray):
            return np.zeros(len(x))
        elif isinstance(x, (int, float)):
            return 0
    if n == 1:
        if isinstance(x, np.ndarray):
            return np.ones(len(x))
        elif isinstance(x, (int, float)):
            return 1
    legendre_p = (n * legendre(n - 1)(x) - n * x * legendre(n)(x)) / (1 - x ** 2)
    return legendre_p

def _legendre_prime_lobatto(x, n):
    return (1 - x ** 2) ** 2 * _legendre_prime(x, n)


def _legendre_double_prime(x, n):
    """Calculate second derivative legendre polynomial recursively.

    Args:
        x (float,np.array) = domain.
        n (int) = degree of Legendre polynomial (L_n).
    Return:
        legendre_pp (np.array) = value second derivative of L_n.
    """
    legendre_pp = 2 * x * _legendre_prime(x, n) - n * (n + 1) * legendre(n)(x)
    return legendre_pp * (1 - x ** 2)


def _newton_method(f, dfdx, x_0, n_max, min_error=np.finfo(float).eps * 10):
    """Newton method for rootfinding.

    It garantees quadratic convergence given f'(root) != 0 and abs(f'(Î¾)) < 1
    over the domain considered.

    Args:
        f (obj func) = function
        dfdx (obj func) = derivative of f
        x_0 (float) = starting point
        n_max (int) = max number of iterations
        min_error (float) = min allowed error

    Returns:
        x[-1] (float) = root of f
        x (np.array) = history of convergence
    """
    x = [x_0]
    for i in range(n_max - 1):
        x.append(x[i] - f(x[i]) / dfdx(x[i]))
        if abs(x[i + 1] - x[i]) < min_error: return x[-1]
    print('WARNING : Newton did not converge to machine precision \nRelative error : ',
          x[-1] - x[-2])
    return x[-1]



def compute_influence(xyzi, xyzj, n,):
    N = xyzi.shape[0]
    k = xyzj.shape[1]
    A = np.zeros((N, N))
    B = np.zeros_like(A)
    for i in range(N):
        pi = xyzi[i]
        ni = n[i]
        for j in range(N):
            contr = 0
            contrB = 0
            for m in range(k-1):
                b1 = xyzj[j, m]
                b2 = xyzj[j, m + 1]

                r1 = pi - b1
                r2 = pi - b2
                r0 = b2 - b1

                K = 1/(4*np.pi * np.linalg.norm(np.cross(r1, r2))**2) * np.dot(r0, r1/np.linalg.norm(r1) - r2/np.linalg.norm(r2))
                q12 = K * np.cross(r1, r2)

                contr += np.dot(q12, ni)
                if m!=1:
                    contrB += np.dot(q12, ni)

            A[i,j] = contr
            B[i,j] = contrB
    return A,B



@dataclass
class VortexFilament:
    b1: np.array
    b2: np.array
    Gamma: float

def biot_savart(p: np.array, filament: VortexFilament) -> np.array:
    r1 = p - filament.b1
    r2 = p - filament.b2
    r0 = filament.b2 - filament.b1
    cross = np.cross(r1, r2)
    norm_cross = np.linalg.norm(cross)
    coeff = filament.Gamma / (4 * np.pi * norm_cross**2)
    dot = np.dot(r0, r1/np.linalg.norm(r1) - r2/np.linalg.norm(r2))
    return coeff * cross * dot

class VortexRing:
    def __init__(self, segments: list[VortexFilament]):
        self.segments = segments

    def induced_velocity(self, p: np.array, )-> np.array:
        return sum(biot_savart(p, seg) for seg in self.segments)

    def update_gammas(filaments: list[VortexFilament], gammas: list[float]):
        for filament, gamma in zip(filaments, gammas):
            filament.Gamma = gamma

def construct_vortex_ring(points: list[np.ndarray], Gamma: float) -> VortexRing:
    segments = []
    for i in range(len(points)):
        start = points[i]
        end = points[(i + 1) % len(points)]
        segments.append(VortexFilament(b1=start, b2=end, Gamma=Gamma))
    return VortexRing(segments=segments)

class Mesh:
    def __init__(self, Nspan: int, Nwake: int):
        self.ci = np.zeros(Nspan)
        normals = 0


def iter(xyzi, xyzj, n, Qinf, AR, b, niter=200, tol=1e-6):
    c = b / AR
    CORE = 1e-5
    'iteration'
    umat = np.zeros((xyzi.shape[0], xyzi.shape[0]))
    vmat = np.zeros_like(umat)
    wmat = np.zeros_like(vmat)

    N = xyzi.shape[0]
    k = xyzj.shape[1]
    for i in range(N):
        pi = xyzi[i]
        ni = n[i]
        for j in range(N):
            contr = np.zeros(3)
            for m in range(k - 1):
                b1 = xyzj[j, m]
                b2 = xyzj[j, m + 1]

                #print(velocity_from_vortex_filament(1.0, b1, b2, pi))
                contr += velocity_3D_from_vortex_filament(1, b1, b2, pi, 1e-5)#velocity_from_vortex_filament(1.0, A=b1, B=b2, P=pi)
            #print(contr)
            umat[i, j] = contr[0]
            vmat[i, j] = contr[1]
            wmat[i, j] = contr[2]
    print(umat)
    gammas = np.zeros(xyzi.shape[0])
    GAMMAS_new = np.zeros_like(gammas)
    CLnew = np.zeros_like(gammas)
    ConvWeight=0.01
    for k in range(niter):
        gammas = GAMMAS_new.copy()
        for icp in range(gammas.size):
            u = v = w = 0

            for j in range(gammas.size):
                u = u + umat[icp,j]*gammas[j]
                v = v + vmat[icp,j]*gammas[j]
                w = w + wmat[icp,j]*gammas[j]
                #print(umat[icp,j], u, v, w)
            vel1 = Qinf + np.array([u, v, w])

            angle1 = np.arctan2(vel1[2], vel1[0])
            #print(angle1, np.sin(angle1))
            CLnew[icp] = 2 * np.pi * np.sin(angle1)
            #print(c)
            #print(Qinf[2], w, vel1[2])
            vmag = np.linalg.norm(vel1)
            GAMMAS_new[icp] = 0.5 * 1 * vmag * CLnew[icp]
        #refererror = np.max(np.abs(GAMMAS_new))
        #refererror =max(refererror, 0.001)
        #error = np.max(abs(GAMMAS_new-gammas))

        #error = error / refererror
        #ConvWeight = max((1 - error) * 0.3, 0.1)
        #print(ConvWeight)
        GAMMAS_new = (1-ConvWeight)*gammas + ConvWeight*GAMMAS_new

    #print(CLnew)
    plt.plot(CLnew)
    plt.ylim([0, 1.05 * np.max(CLnew)])
    plt.show()


def velocity_3D_from_vortex_filament(GAMMA, XV1, XV2, XVP1, CORE):
    X1, Y1, Z1 = XV1
    X2, Y2, Z2 = XV2
    XP, YP, ZP = XVP1

    R1 = math.sqrt((XP - X1)**2 + (YP - Y1)**2 + (ZP - Z1)**2)
    #print(XP, YP, ZP)
    R2 = math.sqrt((XP - X2)**2 + (YP - Y2)**2 + (ZP - Z2)**2)

    R1XR2_X = (YP - Y1)*(ZP - Z2) - (ZP - Z1)*(YP - Y2)
    R1XR2_Y = -((XP - X1)*(ZP - Z2) - (ZP - Z1)*(XP - X2))
    R1XR2_Z = (XP - X1)*(YP - Y2) - (YP - Y1)*(XP - X2)

    R1XR_SQR = R1XR2_X**2 + R1XR2_Y**2 + R1XR2_Z**2
    R0R1 = (X2 - X1)*(XP - X1) + (Y2 - Y1)*(YP - Y1) + (Z2 - Z1)*(ZP - Z1)
    R0R2 = (X2 - X1)*(XP - X2) + (Y2 - Y1)*(YP - Y2) + (Z2 - Z1)*(ZP - Z2)
    print(XVP1)
    if R1XR_SQR < CORE**2:
        R1XR_SQR = CORE**2
    if R1 < CORE:
        R1 = CORE
    if R2 < CORE:
        R2 = CORE

    K = GAMMA / (4 * math.pi * R1XR_SQR) * (R0R1 / R1 - R0R2 / R2)

    U = K * R1XR2_X
    V = K * R1XR2_Y
    W = K * R1XR2_Z

    return [U, V, W]



bounds = (1.-0.2)/2*lobatto_quad(11)[0]+1.2/2
print(1/2 * (bounds[1:]+bounds[:-1]))