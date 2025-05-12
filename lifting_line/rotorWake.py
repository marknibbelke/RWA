import  numpy                as np
import  matplotlib.pyplot    as plt
import  pandas               as pd
from    abc                  import ABC, abstractmethod
from    scipy.optimize       import newton_krylov, root
np.set_printoptions(linewidth=7000)


class VortexSim(ABC):
    def __init__(self, xyzi, xyzj, ni, Qinf,):
        self.xyzi = xyzi
        self.xyzj = xyzj
        self.ni = ni
        self.Qinf = Qinf
        self.uvws, self.A, self.B, self.br = self._assemble_vortex_system(Qinf=Qinf,)

    @abstractmethod
    def iter_solve(self, *args, **kwargs):
        pass

    @abstractmethod
    def direct_solve(self, *args, **kwargs):
        pass

    def _assemble_vortex_system(self, Qinf, bound_idx=None, CORE=1e-5, Omegavec=np.zeros(3)):
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
        if bound_idx is None: bound_idx = (self.xyzj.shape[1] - 1) // 2; print(f'bound vortex idx = {bound_idx}')
        mask = np.ones(self.xyzj.shape[1] - 1, dtype=bool)
        mask[bound_idx] = False

        'Relative Position Vectors'
        r1s = self.xyzi[:, None, None, :] - self.xyzj[None, :, :-1, :]
        r2s = self.xyzi[:, None, None, :] - self.xyzj[None, :, 1:, :]
        r0s = self.xyzj[:, 1:, :] - self.xyzj[:, :-1, :]
        print(r1s.shape, r0s.shape)
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
        norm_cross_sq = np.linalg.norm(cross, axis=3) ** 2
        norm_cross_sq[norm_cross_sq < CORE ** 2] = CORE ** 2

        'induced velocity components'
        uvws = 1 / (4 * np.pi * norm_cross_sq[..., None]) * cross * dot[..., None]

        'projection onto normals'
        A = np.einsum('ijnk,ik->ij', uvws, self.ni)
        B = np.einsum('ijnk,ik->ij', uvws[:, :, mask, :], self.ni)
        vrot = np.einsum('ijk, ...j, ...k', eijk, Omegavec, self.xyzi)
        Qinf = Qinf[None, :] + vrot
        br = np.einsum('ki, ki->k', -Qinf, self.ni)

        return uvws, A, B, br

    def update(self, xyzi_new, xyzj_new, ni_new, Qinf_new, *args, **kwargs)->None:
        self.xyzi = xyzi_new
        self.xyzj = xyzj_new
        self.ni = ni_new
        self.Qinf = Qinf_new
        self.uvws, self.A, self.B, self.br = self._assemble_vortex_system(Qinf=Qinf_new, )
        self._post_update_hook(*args, **kwargs)

    def _post_update_hook(self, *args, **kwargs):
        pass


class WingSim(VortexSim):
    def __init__(self, xyzi, xyzj, ni, Qinf,):
        super().__init__(xyzi, xyzj, ni, Qinf)
        self.results = {}

    def direct_solve(self, *args, **kwargs):
        pass

    def iter_solve(self, niter=200, convweight=0.01, tol=1e-6, plot: bool = True)->None:
        uvws = np.sum(self.uvws, axis=2)
        gammas = np.zeros(self.xyzi.shape[0])
        GAMMAS_new = np.zeros_like(gammas)
        CLnew = 0
        for k in range(niter):
            gammas = GAMMAS_new.copy()
            uvw = np.einsum('ijk, j -> ik', uvws, gammas)
            vel1s = self.Qinf[None, :] + uvw
            angle1s = np.arctan2(vel1s[:, 2], vel1s[:, 0])
            CLnew = 2 * np.pi * np.sin(angle1s)
            vmags = np.linalg.norm(vel1s, axis=1)
            GAMMAS_new = 0.5 * 1 * vmags * CLnew
            GAMMAS_new = (1 - convweight) * gammas + convweight * GAMMAS_new
            error = max(abs(np.subtract(GAMMAS_new, gammas)))
            if error <= tol:  print(f'Iter. ended at k={k}'); break
        self.results['y'] = xyzi[:, 1]
        self.results['CL'] = CLnew
        if plot:
            plt.figure(figsize=(8, 3))
            plt.plot(self.results['y'] , self.results['CL'] , marker='o')
            plt.ylim([0, 1.05 * np.max(self.results['CL'])])
            plt.grid(True)
            plt.xlabel('y')
            plt.ylabel('$C_l(y)$')
            plt.tight_layout()
            plt.show()



class RotorWakeSim(VortexSim):
    def __init__(self, xyzi, xyzj, ni, elem_boundaries, Qinf, R, geomfunc:callable, nblades: int = 3, airfoil: str = '../DU95W180.txt'):
        super().__init__(xyzi, xyzj, ni, Qinf)
        self.elem_bounds = elem_boundaries
        self.R = R
        self.nblades = nblades
        self.geomfunc = geomfunc
        self.polar_alpha, self.polar_cl, self.polar_cd = self.read_polar(airfoil, plot=False)
        self.radial_positions = np.sqrt(np.einsum('ij, ij->i', xyzi, xyzi))
        self.chords, self.twists = geomfunc(self.radial_positions / R)
        self.results = {}

    def _post_update_hook(self, *args, **kwargs):
        self.radial_positions = np.sqrt(np.einsum('ij, ij->i', self.xyzi, self.xyzi))
        self.chords, self.twists = self.geomfunc(self.radial_positions / self.R)
        elem_bounds_new = kwargs.get("elem_bounds_new", None)
        if elem_bounds_new is not None:
            self.elem_bounds = elem_bounds_new

    def __add__(self, other):
        xyzi_combined = np.vstack((self.xyzi, other.xyzi))
        xyzj_combined = np.vstack((self.xyzj, other.xyzj))
        ni_combined = np.vstack((self.ni, other.ni))
        elem_bounds_combined = np.concatenate((self.elem_bounds, other.elem_bounds))
        R = (self.R, other.R)
        nblades = (self.nblades, other.nblades)
        return RotorWakeSim(xyzi_combined, xyzj_combined, ni_combined, elem_bounds_combined, self.Qinf, R, self.geomfunc, nblades, airfoil = '../DU95W180.txt')

    def direct_solve(self, *args, **kwargs):
        pass

    def plot_results(self, lw = 0.75, c='k')->None:
        TSR = self.results['TSR']
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 7))
        axes[0,0].plot(self.results['r_R'], self.results['Gamma'], marker='o', color=c,label='$\\tilde{\Gamma}$: '+f'TSR={TSR}', linewidth=lw)
        axes[0,0].grid()
        axes[0,0].set_ylim(bottom=0)
        axes[0,0].set_ylabel('$\\tilde{\Gamma}$')
        axes[0,0].legend()
        axes[0,1].plot(self.results['r_R'], self.results['Faxial'], marker='o', color=c, label='$\\tilde{F}_{axial}$: '+f'TSR={TSR}', linewidth=lw)
        axes[0,1].plot(self.results['r_R'], self.results['Fazim'], linestyle='--', marker='s',color=c, label='$\\tilde{F}_{azim}$: '+f'TSR={TSR}', linewidth=lw)
        axes[0,1].set_ylim(bottom=0)
        axes[0,1].set_ylabel('$\\tilde{F}$')
        axes[0,1].grid()
        axes[0,1].legend()
        axes[1,0].plot(self.results['r_R'], self.results['a'], marker='o', color=c,label='$a$: '+f'TSR={TSR}', linewidth=lw)
        axes[1,0].plot(self.results['r_R'], self.results['aline'], marker='s', linestyle='--', color=c,label='$a^{,}$: '+f'TSR={TSR}', linewidth=lw)
        axes[1,0].set_ylim(bottom=0)
        axes[1,0].grid()
        axes[1,0].legend()
        axes[1,0].set_ylabel('$a$')
        axes[1,1].set_xlabel('$r/R$')
        axes[1,1].plot(self.results['r_R'], self.results['alpha'], marker='o', color=c,label='$\\alpha$: '+f'TSR={TSR}', linewidth=lw)
        axes[1,1].plot(self.results['r_R'], self.results['inflow'], marker='o',linestyle='--', color=c, label='$\phi$: ' + f'TSR={TSR}', linewidth=lw)
        axes[1,1].grid()
        axes[1,1].set_ylim(bottom=0)
        axes[1,1].set_ylabel('$\\alpha, \phi$')
        axes[1,1].legend()
        axes[1,1].set_xlabel('$r/R$')
        fig.suptitle(f'$N={self.results["N"]}, n_b={self.nblades}, k={self.xyzj.shape[1]}$')
        fig.tight_layout()
        plt.show()

    def iter_solve(self, Omega, niter=600, tol=1e-6, plot: bool = True, method='broyden1', verbose: bool = True)->dict:
        uvws = np.sum(self.uvws, axis=2)
        Omega_vec = np.array([-Omega, 0, 0])
        N = self.xyzi.shape[0]
        gammas0 = np.zeros(N)

        eijk = np.zeros((3, 3, 3), dtype=int)
        eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
        eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
        orthogonals = np.array([-1 / self.radial_positions, np.zeros_like(self.radial_positions), np.zeros_like(self.radial_positions)]).T
        vrot = np.einsum('ijk, ...j, ...k', eijk, Omega_vec, self.xyzi)
        azimdir = np.einsum('ijk, ...j, ...k', eijk, orthogonals, self.xyzi)

        def _F(gammas):
            uvw = np.einsum('ijk, j -> ik', uvws, gammas)
            vel1s = self.Qinf[None, :] + uvw + vrot
            vazim = np.einsum('ij,ij->i', azimdir, vel1s)
            vaxial = vel1s[:, 0]
            temploads = self._loadBladeElement(vaxial, vazim,)
            gamma_new = temploads[2].copy()
            return gamma_new - gammas

        sol = root(_F, gammas0, method=method, tol=tol, options={'maxiter': niter, 'disp': verbose})
        print(sol.message)
        uvw = np.einsum('ijk,j->ik', uvws, sol.x)
        vel1s = self.Qinf[None, :] + uvw + vrot
        vazim = np.einsum('ij,ij->i', azimdir, vel1s)
        vaxial = vel1s[:, 0]
        temploads =  self._loadBladeElement(vaxial, vazim,)
        r_R = self.radial_positions / self.R
        r_R = r_R.reshape(self.nblades, int(r_R.size / self.nblades))
        aline = (vazim / (self.radial_positions * Omega) - 1).reshape(self.nblades, int(r_R.size / self.nblades))
        a = -(uvw[:, 0] + vrot[:, 0] / self.Qinf[0]).reshape(self.nblades, int(r_R.size / self.nblades))
        GAMMAS_new = sol.x.reshape(self.nblades, int(r_R.size / self.nblades))
        normGamma = np.linalg.norm(self.Qinf) ** 2 * np.pi / (self.nblades * Omega)
        Faxial = temploads[0].reshape(self.nblades, int(r_R.size / self.nblades))
        Fazim = temploads[1].reshape(self.nblades, int(r_R.size / self.nblades))
        normFax = .5 * np.linalg.norm(self.Qinf) ** 2 * self.R
        self.results['r_R'] = r_R[0]
        self.results['Gamma'] = np.average(GAMMAS_new, axis=0) / normGamma
        self.results['Faxial'] = np.average(Faxial, axis=0) / normFax
        self.results['Fazim'] = np.average(Fazim, axis=0) / normFax
        self.results['a'] = np.average(a, axis=0)
        self.results['aline'] = np.average(aline, axis=0)
        self.results['Omega'] = Omega
        self.results['TSR'] = Omega * self.R
        self.results['alpha'] = np.average(temploads[3].reshape(r_R.shape), axis=0)
        self.results['inflow'] = np.average(temploads[4].reshape(r_R.shape), axis=0)
        self.results['N'] = N
        self._calculateCT_CProtor_CPflow(np.average(Faxial, axis=0), np.average(Fazim, axis=0))
        if plot:
            print(f'\nPlotting results:\nTSR={Omega * self.R}, N={uvws.shape[0]}\n')
            self.plot_results()
        return self.results

    @staticmethod
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
        return np.radians(polar_alpha), polar_cl, polar_cd

    def _loadBladeElement(self, vnorm, vtan, ):
        vmag2 = vnorm ** 2 + vtan ** 2
        inflowangle = np.arctan2(vnorm, vtan)
        alpha = self.twists + inflowangle
        cl = np.interp(alpha, self.polar_alpha, self.polar_cl)
        cd = np.interp(alpha, self.polar_alpha, self.polar_cd) * 0
        lift = 0.5 * vmag2 * cl * self.chords
        drag = 0.5 * vmag2 * cd * self.chords
        fnorm = lift * np.cos(inflowangle) + drag * np.sin(inflowangle)
        ftan = lift * np.sin(inflowangle) - drag * np.cos(inflowangle)
        gamma = 0.5 * np.sqrt(vmag2) * cl * self.chords
        return fnorm, ftan, gamma, alpha, inflowangle

    def _calculateCT_CProtor_CPflow(self, Faxial, Fazim)->None:
        r_Rarray = self.elem_bounds/self.R
        r_R_temp = 1 / 2 * (r_Rarray[:-1] + r_Rarray[1:])
        drtemp = np.diff(r_Rarray)
        Uinf = np.linalg.norm(self.Qinf)
        self.results['CT'] = np.sum(drtemp * Faxial * self.nblades / (0.5 * Uinf ** 2 * np.pi * self.R))
        self.results['CP'] = np.sum(drtemp * Fazim  * r_R_temp * self.results['Omega']  * self.nblades / (0.5 * Uinf ** 2 * np.pi))


def rotor_blade(r_R):
    pitch = 2
    chord = 3 * (1 - r_R) + 1
    twist = -14 * (1 - r_R)
    return chord, np.radians(twist + pitch)

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

def double_rotor_wake():
    return


if __name__ == "__main__":
    'simulation parameters'
    N           = 31
    revolutions = 50
    TSR         = 6.0
    R           = 50
    nblades     = 3
    aw          = 0.2
    Qinf        = np.array([1, 0, 0])

    'generate geometry'
    s_Array            = np.arange(0, np.pi, np.pi / (N + 1))
    s_Array[-1]        = np.pi
    r_array            = -1 * (np.cos(s_Array) - 1) / 2 * 0.8 + 0.2
    theta_array        = np.arange(0, revolutions * 2 * np.pi, np.pi / 10)
    ri_elem_boundaries = cosine_spacing(0.2 * R, R, N) #r_array * R #
    xyzi, xyzj, ni, ri = rotor_wake(theta_array=theta_array, ri_elem_boundaries=ri_elem_boundaries, N=N, geom_func=rotor_blade, R=R, TSR=TSR / (1 - aw), nblades=nblades, plot=True, fbound=0., fcp=0., )
    print(1 / 2 * (ri_elem_boundaries[:-1] + ri_elem_boundaries[1:]) / R)

    'simulation'
    ROTORSIM = RotorWakeSim(xyzi=xyzi, xyzj=xyzj, ni=ni, Qinf=Qinf, R=R, geomfunc=rotor_blade, nblades=nblades, elem_boundaries=ri_elem_boundaries)
    ROTORSIM.iter_solve(Omega=TSR/R, niter=1200, tol=1e-8, plot=True, )     #method='excitingmixing')
    resLL    = ROTORSIM.results
    print(f'(CT, CP) = ({resLL["CT"]:.2f},{resLL["CP"]:.2f})')