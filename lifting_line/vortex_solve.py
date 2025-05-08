import  numpy as np
import  matplotlib.pyplot as plt
import  pandas as pd
from    abc import ABC, abstractmethod
from scipy.optimize import newton_krylov,root,fixed_point
np.set_printoptions(linewidth=7000)


def cosine_spacing(a, b, N):
    theta = np.linspace(0, np.pi, N + 1)
    x = 0.5 * (a + b) + 0.5 * (b - a) * np.cos(theta)
    return np.flip(x)


def planar_wing(N: int, c, b, Lwake, alpha, spacing: str = 'cosine', fcp = 3/4, fbound = 1/4):
    '''
    :param N: number of partitions
    :param c: chord length
    :param b: span
    :param Lwake: trailing vortex length
    :param alpha: [rad.] angle of attack
    :param spacing: (str) spacing type: cosine or other (uniform)
    :param fcp: c_{control point} / c
    :param fbound: c_{bound vortex} / c
    :return:
        xyzi: [N, 3], collocation node coordinates
        xyzj: [N, k, 3], vortex ring coordinates, k-1 := dimension of the ring
        ni:   [N, 3], normal vectors
    '''
    yi_elem_boundaries = cosine_spacing(0, b, N)
    if spacing != 'cosine': yi_elem_boundaries = np.linspace(0, b, N+1)
    yi = 1/2*(yi_elem_boundaries[:-1] + yi_elem_boundaries[1:])
    xyzi = np.stack((fcp*c*np.ones_like(yi), yi, np.zeros_like(yi))).T
    ni = np.tile([0,0,1], N).reshape(N,3)
    xyzj = np.array([
        np.array([
            [Lwake, y0, Lwake*np.sin(alpha)],
            [1.25*c, y0, 0],
            [c*fbound, y0, 0],
            [c*fbound, y1, 0],
            [1.25*c, y1, 0],
            [Lwake, y1, Lwake*np.sin(alpha)]
        ]) for y0, y1 in zip(yi_elem_boundaries[:-1], yi_elem_boundaries[1:])
    ])
    return xyzi, xyzj, ni


def rotor_blade(r_R):
    pitch = 2
    chord = 3 * (1 - r_R) + 1
    twist = -14 * (1 - r_R)
    return chord, np.radians(twist + pitch)


def rotor_wake(theta_array, ri_elem_boundaries, N: int, geom_func: callable, R, TSR, nblades: int, fcp=0.75, fbound = 1/4, plot: bool = False):
    '''
    :param N: Number of blade partitions
    :param k2: number of partitions per trailing vortex (i.e. k:=dim(ring_i) = 2*k_2)
    :param revolutions: number of revolutions of the wake system
    :param geom_func: returns: chord(r/R), blade_angle(r/R)
    :param R: blade radius
    :param TSR: tip speed ratio
    :param rotorbounds_r: (r_root/R, r_tip/R)
    :param nblades: number of blades
    :param fcp: (c_{control point}/c)
    :param fbound: (c_{bound vortex}/c)
    :param plot: plot system geometry
    :return: xyzi, xyzj, ni
    '''
    ri = 1 / 2 * (ri_elem_boundaries[:-1] + ri_elem_boundaries[1:])

    'reference blade'
    chord_ref, angle_ref = geom_func(ri/R)
    ni_ref = np.array([np.cos(angle_ref), np.zeros_like(angle_ref), -np.sin(angle_ref)]).T
    xyzi_ref = np.stack((fcp * chord_ref *np.sin(-angle_ref), ri, -fcp*chord_ref*np.cos(angle_ref))).T

    'control points'
    rotation_transform = np.arange(0, 2*np.pi, 2*np.pi/nblades)
    c = np.cos(rotation_transform)
    s = np.sin(rotation_transform)
    rot = np.stack([np.stack([np.ones_like(c), np.zeros_like(c), np.zeros_like(c)], axis=1),
        np.stack([np.zeros_like(c), c, -s], axis=1),
        np.stack([np.zeros_like(c), s, c], axis=1)], axis=1)
    ni = np.einsum('bj,akj->abk', ni_ref, rot).reshape(-1, 3)
    xyzi = np.einsum('bj,akj->abk', xyzi_ref, rot).reshape(-1, 3)

    'vortex rings'
    chord_ref, angle_ref = geom_func(ri_elem_boundaries/R)
    dys = np.cumsum(np.multiply.outer(np.diff(np.cos(-theta_array)),ri_elem_boundaries).T, axis=1)
    dzs = np.cumsum(np.multiply.outer(np.diff(np.sin(-theta_array)), ri_elem_boundaries).T, axis=1)
    dxs = np.cumsum(np.multiply.outer(np.diff(theta_array)/TSR*R, np.ones_like(ri_elem_boundaries)).T, axis=1)
    vortex_segments = np.stack([dxs, dys, dzs], axis=-1)
    xyzj_bound_ref = np.array([
        np.array([
            [chord_ref[i] *(1+fbound)* np.sin(-angle_ref[i]), r0, -chord_ref[i]*(1+fbound) * np.cos(angle_ref[i])],
            [chord_ref[i] * fbound* np.sin(-angle_ref[i]), r0, -chord_ref[i] *fbound* np.cos(angle_ref[i])],
            [chord_ref[i+1] * fbound* np.sin(-angle_ref[i+1]), r1, -chord_ref[i+1] *fbound* np.cos(angle_ref[i+1])],
            [chord_ref[i + 1]*(1+fbound) * np.sin(-angle_ref[i+1]), r1, -chord_ref[i+1]*(1+fbound)*np.cos(angle_ref[i+1])],
        ]) for i, (r0, r1) in enumerate(zip(ri_elem_boundaries[:-1], ri_elem_boundaries[1:]))
    ])
    xyzj_ref = np.concatenate((np.flip(vortex_segments[:-1,:,:], axis=1)+xyzj_bound_ref[:,[0],:],xyzj_bound_ref, xyzj_bound_ref[:,[-1],:]+vortex_segments[1:,:,:]), axis=1)
    xyzj = np.einsum('pbj,akj->apbk', xyzj_ref, rot)
    if plot:
        fig = plt.figure(figsize=(12,7))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_proj_type('persp', focal_length=0.2)
        ax.view_init(5, -125,0)
        ax.grid(False)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        blade_plot = xyzi.reshape(nblades, N, 3)
        for bi in range(nblades):
            ax.plot(blade_plot[bi, :,0],blade_plot[bi, :,1],blade_plot[bi, :,2], color='r',marker='none', zorder=10)
            for i in range(N):
                ax.plot(xyzj[ bi, i,:, 0], xyzj[bi,i,  :, 1], xyzj[ bi,i, :, 2], color='b', linewidth=.2, )#alpha=.6)
        plt.xlim([0, 4*R])
        plt.gca().set_aspect('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.gca().set_zlabel('z')
        plt.title(f'$\lambda={TSR}, N={N}, k_2={theta_array.size}$')
        plt.tight_layout()
        plt.show()
    return xyzi, xyzj.reshape(nblades * N, xyzj.shape[2], 3), ni, ri_elem_boundaries


def assemble_vortex_system(xyzi, xyzj, ni, Qinf, bound_idx = None, CORE = 1e-5, Omegavec = np.zeros(3)):
    '''
    N := number of collocation nodes
    k-1 := dimension of each filament ring
    :param xyzi: [N, 3]
    :param xyzj: [N, k, 3]
    :param ni: [N, 3]
    :param Qinf: [N, 3]
    :return: {GAMMA_i}_{i=1}^{N}

    NOTES:
        x_i     ∈ R3, i=1, ..., N
        x_{j,m} ∈ R3, j=1, ..., N, m=1, ..., k
        n_i     ∈ R3

        Let:
            r^1_{ijm} = x_i - x_{jm}
            r^2_{ijm} = x_i - x_{j,m+1}
            r^0_{jm}  = x_{j,m+1} - x_{jm}

        then:
            d_{ijm} = r^0_{jm} (r^1_{ijm}/||r^1_{ijm}|| - r^2_{ijm}/||r^2_{ijm}||)
            c_{ijm} = r^1_{ijm} x r^2_{ijm} ∈ R3,
            such that:
                c^a_{ijm} = ε_{abc} r^1_b r^2_c ∈ R3, where ε is the rank 3 Levi-Civita permutation

            u_{ijm} = d_{ijm}/(4π) * c_{ijm}/||c_{ijm}||
            a_{ij} = ∑_{m=1}^{k-1} u_{ijm}n_i
    '''
    if bound_idx is None: bound_idx = (xyzj.shape[1] - 1)//2; print(f'bound vortex idx = {bound_idx}')
    mask = np.ones(xyzj.shape[1]-1, dtype=bool)
    mask[bound_idx] = False

    'Relative Position Vectors'
    r1s = xyzi[:, None, None, :] - xyzj[None, :, :-1, :]
    r2s = xyzi[:, None, None, :] - xyzj[None, :, 1:, :]
    r0s = xyzj[:, 1:, :] - xyzj[:, :-1, :]

    'd'
    norm_r1s = np.linalg.norm(r1s, axis=3, keepdims=True)
    norm_r2s = np.linalg.norm(r2s, axis=3, keepdims=True)
    norm_r1s[norm_r1s < CORE] = CORE
    norm_r2s[norm_r2s < CORE] = CORE
    dot = np.einsum('ijk, pijk->pij', r0s, r1s / norm_r1s - r2s / norm_r2s)

    'c'
    eijk = np.zeros((3, 3, 3))
    eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
    eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
    cross = np.einsum('ijk, ...j, ...k', eijk, r1s, r2s)
    norm_cross_sq = np.linalg.norm(cross, axis=3)**2
    norm_cross_sq[norm_cross_sq < CORE**2] = CORE**2

    'induced velocity components'
    uvws = 1 / (4 * np.pi * norm_cross_sq[..., None]) * cross * dot[..., None]

    'projection onto normals'
    A = np.einsum('ijnk,ik->ij', uvws, ni)
    B = np.einsum('ijnk,ik->ij', uvws[:,:,mask,:], ni)
    vrot = np.einsum('ijk, ...j, ...k', eijk, Omegavec, xyzi)
    Qinf = Qinf[None, :] + vrot
    br = np.einsum('ki, ki->k', -Qinf, ni)
    return uvws, A, B, br


def direct_solve(A,B,br,xyzi, plot: bool = True):
    gammas = np.linalg.solve(A, br)
    w_ind = B @ gammas
    alphas = np.arctan2(Qinf[2] + w_ind, Qinf[0])
    cli = 2 * np.pi * np.sin(alphas)

    normGamma = np.linalg.norm(Qinf) ** 2 * np.pi / (nblades * TSR/R)
    plt.plot(np.sqrt(np.einsum('ij,ij->i',xyzi, xyzi)).reshape(nblades, int(gammas.size/nblades))[0]/R, np.average(gammas.reshape(nblades, int(gammas.size/nblades)), axis=0)/normGamma, marker='o')
    plt.show()
    if plot:
        plt.figure(figsize=(8, 3))
        plt.plot(xyzi[:,1], cli, marker='o')
        plt.ylim([0, 1.05 * np.max(cli)])
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    return gammas


def iter_solve_wing(uvws, Qinf, xyzi, niter=200, convweight=0.01, tol=1e-6, plot: bool = True):
    uvws = np.sum(uvws, axis=2)
    gammas = np.zeros(xyzi.shape[0])
    GAMMAS_new = np.zeros_like(gammas)
    CLnew = 0
    for k in range(niter):
        gammas = GAMMAS_new.copy()
        uvw = np.einsum('ijk, j -> ik', uvws, gammas)
        vel1s = Qinf[None,:] + uvw
        angle1s = np.arctan2(vel1s[:,2], vel1s[:,0])

        CLnew = 2 * np.pi * np.sin(angle1s)
        vmags = np.linalg.norm(vel1s, axis=1)
        GAMMAS_new = 0.5 * 1 * vmags * CLnew
        GAMMAS_new = (1-convweight)*gammas + convweight*GAMMAS_new
        error = max(abs(np.subtract(GAMMAS_new, gammas)))
        if error <= tol:  print(f'Iter. ended at k={k}'); break
    if plot:
        plt.figure(figsize=(8, 3))
        plt.plot(xyzi[:, 1], CLnew, marker='o')
        plt.ylim([0, 1.05 * np.max(CLnew)])
        plt.grid(True)
        plt.xlabel('y')
        plt.ylabel('$C_l(y)$')
        plt.tight_layout()
        plt.show()


def iter_solve_rotor_wake(uvws, Qinf, xyzi, Omega, Radius, polar_alpha, polar_cd, polar_cl, nblades: int, geomfunc: callable, convweight0:tuple, niter=600, tol=1e-4, plot: bool = True):
    uvws = np.sum(uvws, axis=2)
    gammas = np.zeros(xyzi.shape[0])
    GAMMAS_new = np.zeros_like(gammas)
    Omega_vec = np.array([-Omega, 0,0])

    eijk = np.zeros((3, 3, 3), dtype=int)
    eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
    eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
    #print(uvws[:,:,0])
    radial_positions = np.sqrt(np.einsum('ij, ij->i', xyzi, xyzi))
    orthogonals = np.array([-1/radial_positions, np.zeros_like(radial_positions),np.zeros_like(radial_positions)]).T
    vrot = np.einsum('ijk, ...j, ...k', eijk, Omega_vec, xyzi)
    azimdir = np.einsum('ijk, ...j, ...k', eijk, orthogonals, xyzi)
    chords, twists = geomfunc(radial_positions/Radius)
    temploads = np.zeros(1)
    aline = a = 0

    N = uvws.shape[0]
    A = uvws.reshape(3 * N, N)
    U, s, Vh = np.linalg.svd(A)
    spectral_radius = np.max(np.real(s))
    OMEGA = .9*2/(spectral_radius)
    print(spectral_radius, OMEGA)
    for k in range(niter):
        gammas = GAMMAS_new.copy()
        uvw = np.einsum('ijk, j -> ik', uvws, gammas)
        vel1s = Qinf[None, :] + uvw + vrot
        vazim = np.einsum('ij,ij->i', azimdir, vel1s)
        vaxial = vel1s[:,0]
        temploads = loadBladeElement(vaxial, vazim, chords, twists, polar_alpha, polar_cd, polar_cl)
        GAMMAS_new = temploads[2].copy()
        a = -(uvw[:, 0] + vrot[:, 0] / Qinf[0])
        aline = vazim / (radial_positions * Omega) - 1

        refererror = np.max(abs(GAMMAS_new))
        refererror = max(refererror, 0.001)
        error = np.max(abs(GAMMAS_new - gammas))
        error = error / refererror
        convweight = min(OMEGA, convweight0[1])#min(abs(1 - error) * convweight0[0], convweight0[1])
        print(k, error, refererror, convweight)
        GAMMAS_new = (1 - convweight) * gammas + convweight * GAMMAS_new
        if error <= tol:  print(f'Iter. ended at k={k}'); break
    r_R = radial_positions / Radius
    if plot:
        print(f'\nPlotting results:\nTSR={Omega*Radius}, N={uvws.shape[0]}\n')
        r_R = r_R.reshape(nblades, int(r_R.size/nblades))
        GAMMAS_new = GAMMAS_new.reshape(nblades, int(r_R.size/nblades))
        normGamma = np.linalg.norm(Qinf)**2 *np.pi / (nblades*Omega)
        Faxial = temploads[0].reshape(nblades, int(r_R.size/nblades))
        Fazim = temploads[1].reshape(nblades, int(r_R.size/nblades))
        normFax = .5*np.linalg.norm(Qinf)**2 * Radius
        a = a.reshape(nblades, int(r_R.size/nblades))
        aline = aline.reshape(nblades, int(r_R.size/nblades))

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(7, 8))
        axes[0].plot(r_R[0], np.average(GAMMAS_new, axis=0)/normGamma, marker='o')
        axes[0].grid()
        axes[0].set_ylim(bottom=0)
        axes[0].set_ylabel('$\\tilde{\Gamma}$')
        axes[1].plot(r_R[0], np.average(Faxial, axis=0) / normFax,  marker='o')
        axes[1].plot(r_R[0], np.average(Fazim, axis=0) / normFax,  marker='o')
        axes[1].set_ylim(bottom=0)
        axes[1].set_ylabel('$\\tilde{F}$')
        axes[1].grid()
        axes[2].plot(r_R[0], np.average(a, axis=0),  marker='o')
        axes[2].plot(r_R[0], np.average(aline, axis=0),  marker='o')
        axes[2].set_ylim(bottom=0)
        axes[2].grid()
        axes[2].set_xlabel('$r/R$')
        axes[2].set_ylabel('$a$')
        fig.tight_layout()
        plt.show()
    #print(gammas)
    return a, aline, r_R, temploads


def loadBladeElement(vnorm, vtan, chord, twist, polar_alpha, polar_cd, polar_cl):
    vmag2 = vnorm**2 + vtan**2
    inflowangle = np.arctan2(vnorm, vtan)
    alpha = twist + inflowangle
    cl = np.interp(alpha, polar_alpha, polar_cl)
    cd = np.interp(alpha, polar_alpha, polar_cd)*0
    lift = 0.5 * vmag2 * cl * chord
    drag = 0.5 * vmag2 * cd * chord
    fnorm = lift * np.cos(inflowangle) + drag * np.sin(inflowangle)
    ftan = lift * np.sin(inflowangle) - drag * np.cos(inflowangle)
    gamma = 0.5 * np.sqrt(vmag2) * cl * chord
    return fnorm, ftan, gamma, alpha, inflowangle


def read_polar(airfoil: str, plot: bool = False):
    data1 = pd.read_csv(airfoil, header=0, names=["alfa", "cl", "cd", "cm"], sep='\s+')
    polar_alpha = data1['alfa'][:].to_numpy().astype(float)
    polar_cl = data1['cl'][:].to_numpy().astype(float)
    polar_cd = data1['cd'][:].to_numpy().astype(float)
    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].plot(polar_alpha, polar_cl)
        axs[0].set_xlabel(r'$\alpha$')
        axs[0].set_ylabel(r'$C_l$')
        axs[0].grid()
        axs[1].plot(polar_cd, polar_cl)
        axs[1].set_xlabel(r'$C_d$')
        axs[1].grid()
        plt.show()
    return polar_alpha, polar_cl, polar_cd


def calculateCT_CProtor_CPflow(Fnorm,Ftan, Uinf, r_Rarray, Omega, Radius, NBlades):
    r_R_temp = 1/2*(r_Rarray[:-1]+r_Rarray[1:])
    drtemp = np.diff(r_Rarray)
    CTrotor = np.sum(drtemp*Fnorm*NBlades/(0.5*Uinf**2*np.pi*Radius))
    CProtor = np.sum(drtemp*Ftan*r_R_temp*Omega*NBlades/(0.5*Uinf**2*np.pi))
    return CTrotor, CProtor




if __name__ == "__main__":
    N = 11
    revolutions = 50
    TSR = 6.0
    R = 50
    nblades = 3
    aw = 0.2
    s_Array = np.arange(0, np.pi, np.pi / (N+1))
    s_Array[-1] = np.pi
    r_array = -1 * (np.cos(s_Array) - 1) / 2 * 0.8 + 0.2
    theta_array = np.arange(0, revolutions * 2 * np.pi, np.pi / 10)  # np.linspace(0, revolutions*2*np.pi, k2-1)
    ri_elem_boundaries = r_array*R# cosine_spacing(0.2 * R, R, N)#      #cosine_spacing(0.2 * R, R, N)# [:-1]*1.00705  # np.linspace(rotorbounds_r[0]*R, R, N+1)#[:-1] #
    print(1/2*(ri_elem_boundaries[:-1]+ri_elem_boundaries[1:])/R)
    Qinf = np.array([1, 0, 0])
    polar_alpha, polar_cl, polar_cd = read_polar(airfoil='../DU95W180.txt', plot=False)

    xyzi, xyzj, ni, ri = rotor_wake(theta_array=theta_array, ri_elem_boundaries=ri_elem_boundaries, N=N, geom_func=rotor_blade, R=R, TSR=TSR/(1-aw), nblades=nblades, plot=True, fbound=0., fcp=0.,)
    uvws, A, B, br = assemble_vortex_system(xyzi, xyzj, ni, Qinf, CORE=1e-5, Omegavec=np.array([-TSR/R,0,0]))

    a, aline, r_R, temploads = iter_solve_rotor_wake(tol = 0.01, niter=1200, uvws=uvws, Qinf=Qinf, xyzi=xyzi, Omega=TSR/R, Radius=R, convweight0=(0.1,0.1), geomfunc=rotor_blade, polar_alpha=np.radians(polar_alpha), polar_cd=polar_cd, polar_cl=polar_cl,  nblades=nblades)
    Faxial = temploads[0].reshape(nblades, int(r_R.size / nblades))
    Fazim = temploads[1].reshape(nblades, int(r_R.size / nblades))
    CT, CP = calculateCT_CProtor_CPflow(Fnorm=np.average(Faxial, axis=0), Ftan=np.average(Fazim, axis=0), Uinf=np.linalg.norm(Qinf), r_Rarray=ri_elem_boundaries / R, Omega=TSR/R, Radius=R, NBlades=nblades)
    print(f'(CT, CP) = ({CT:.2f},{CP:.2f})')

    #iter_solve_rotor_wake_2(niter=5000, uvws=uvws, Qinf=Qinf, xyzi=xyzi, Omega=TSR / R, Radius=R, convweight0=(0.9, .3), geomfunc=rotor_blade, polar_alpha=np.radians(polar_alpha), polar_cd=polar_cd, polar_cl=polar_cl, nblades=nblades)
    #print(direct_solve(A, B, br, xyzi, plot=False))
    print('--------------------------------')


    '''
    N = 31
    AR =61
    alpha = 5
    Qinf = np.array([1, 0, 1 * np.sin(np.radians(alpha))])  # Qinf/=np.linalg.norm(Qinf)

    xyzi, xyzj, ni = planar_wing(N=N, b=AR, Lwake=1000*AR, c=1, alpha=np.radians(alpha), spacing='cosine', fcp=0., fbound=0.)
    uvws, A, B, br = assemble_vortex_system(xyzi, xyzj, ni, Qinf)
    print(xyzi.shape, xyzj.shape, ni.shape)

    #direct_solve(A, B, br, xyzi, plot=True)
    iter_solve_wing(uvws, Qinf, xyzi, niter=600)
    #rq.iter(xyzi, xyzj, ni, Qinf, niter=600, b=AR, AR=AR)
    '''





