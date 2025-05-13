import matplotlib.pyplot    as plt
import numpy                as np
from abc import ABC, abstractmethod
from   dataclasses          import dataclass, asdict, field
from   scipy.optimize       import root
import copy
import pandas               as pd
np.set_printoptions(linewidth=7000)


@dataclass
class Rotor:
    R: float
    geomfunc: callable
    nblades: int
    blade_bounds: tuple
    N: int = None
    dtheta: float = None
    TSR: float = None
    direction: int = +1
    results: dict = field(default_factory=dict)
    elem_boundaries: np.ndarray = None
    xyzi: np.ndarray = None
    xyzj: np.ndarray = None
    ni: np.ndarray = None
    center: np.ndarray = np.zeros(3)

    @staticmethod
    def from_dict(data: dict) -> 'Rotor':
        return Rotor(**data)

    def copy(self) -> 'Rotor':
        return Rotor(**asdict(self))

    def rotate(self, rotMat: np.ndarray) -> None:
        self.xyzi = np.einsum('bj,aj->ba', self.xyzi, rotMat)
        self.xyzj = np.einsum('bcj,aj->bca', self.xyzj, rotMat)
        self.ni = np.einsum('bj,aj->ba', self.ni, rotMat)

    def translate(self, displVec: np.ndarray) -> None:
        self.xyzi += displVec[None, :]
        self.xyzj += displVec[None, None, :]
        self.center = self.center + displVec


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



class MultiWakeSim(VortexSim):
    def __init__(self, ROTORs: list[Rotor], Qinf, airfoil: str = '../DU95W180.txt'):
        self.xyzi, self.xyzj, self.ni = self._compile(ROTORs)
        super().__init__(self.xyzi, self.xyzj, self.ni, Qinf)
        self.ROTORs=ROTORs
        self.polar_alpha, self.polar_cl, self.polar_cd = self.read_polar(airfoil, plot=False)
        self.twists, self.chords = [], []

    def _compile(self, ROTORs: list[Rotor]):
        xyzi = np.vstack(([ROTOR.xyzi for ROTOR in ROTORs]))
        xyzj = np.vstack(([ROTOR.xyzj for ROTOR in ROTORs]))
        ni = np.concatenate(([ROTOR.ni for ROTOR in ROTORs]))
        return xyzi, xyzj, ni

    def direct_solve(self):
        pass

    def plot_instantaneous(self, lw = 0.75, fs=5)->None:
        colors = ['k',  'b', 'g']
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 7))
        for r, rotor in enumerate(self.ROTORs):
            c = colors[r]
            blade_range = [0] if len(self.ROTORs) == 1 else range(rotor.nblades)
            for i in blade_range:
                leg_str = f'blade={i+1}, '+f'rotor={r+1}'
                axes[0,0].plot(rotor.results['r_R'][i], rotor.results['Gamma'][i], marker='o', color=c,label='$\\tilde{\Gamma}$: '+leg_str, linewidth=lw)
                axes[0,1].plot(rotor.results['r_R'][i], rotor.results['Faxial'][i], marker='o', color=c, label='$\\tilde{F}_{axial}$: '+leg_str, linewidth=lw)
                axes[0,1].plot(rotor.results['r_R'][i], rotor.results['Fazim'][i], linestyle='--', marker='s',color=c, label='$\\tilde{F}_{azim}$: '+leg_str, linewidth=lw)
                axes[1,0].plot(rotor.results['r_R'][i], rotor.results['a'][i], marker='o', color=c,label='$a$: '+leg_str, linewidth=lw)
                axes[1,0].plot(rotor.results['r_R'][i], rotor.results['aline'][i], marker='s', linestyle='--', color=c,label='$a^{,}$: '+leg_str, linewidth=lw)
                axes[1,1].plot(rotor.results['r_R'][i], rotor.results['alpha'][i], marker='o', color=c,label='$\\alpha$: '+leg_str, linewidth=lw)
                axes[1,1].plot(rotor.results['r_R'][i], rotor.results['inflow'][i], marker='o',linestyle='--', color=c, label='$\phi$: ' +leg_str, linewidth=lw)
        axes[0, 0].grid()
        axes[0, 0].set_ylim(bottom=0)
        axes[0, 0].set_ylabel('$\\tilde{\Gamma}$')
        axes[0, 0].legend(fontsize=fs)
        axes[0, 1].set_ylim(bottom=0)
        axes[0, 1].set_ylabel('$\\tilde{F}$')
        axes[0, 1].grid()
        axes[0, 1].legend(fontsize=fs)
        axes[1, 0].set_ylim(bottom=0)
        axes[1, 0].grid()
        axes[1, 0].legend(fontsize=fs)
        axes[1, 0].set_ylabel('$a$')
        axes[1, 1].set_xlabel('$r/R$')
        axes[1,1].set_ylim(bottom=0)
        axes[1,1].set_ylabel('$\\alpha, \phi$')
        axes[1,1].legend(fontsize=fs)
        axes[1, 1].grid()
        axes[1,1].set_xlabel('$r/R$')
        fig.tight_layout()
        plt.show()

    def iter_solve(self, niter=600, tol=1e-6, plot: bool = True, method='broyden1', verbose: bool = True)->None:
        ROTORs = self.ROTORs
        rotor_ids = np.concatenate([np.full(rotor.xyzi.shape[0], i) for i, rotor in enumerate(ROTORs)])
        uvws = np.sum(self.uvws, axis=2)
        Omegas = [rot.TSR/rot.R for rot in ROTORs]
        Omega_vecs = [np.array([-ROTORs[i].direction*Omegai, 0, 0]) for i, Omegai in enumerate(Omegas)]
        radial_positions = [np.sqrt(np.einsum('ij, ij->i', ROTOR.xyzi - ROTOR.center[None, :], ROTOR.xyzi - ROTOR.center[None, :])) for ROTOR in ROTORs]
        orthogonals = np.concatenate([np.array([-1 / radial_positions[i], np.zeros_like(radial_positions[i]), np.zeros_like(radial_positions[i])]).T for i in range(len(ROTORs))])
        self.chords, self.twists = map(lambda arrs: np.concatenate(arrs),zip(*[rot.geomfunc(radial_positions[i] / rot.R) for i, rot in enumerate(ROTORs)]))
        gammas0 = np.zeros(self.xyzi.shape[0])
        eijk = np.zeros((3, 3, 3), dtype=int)
        eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
        eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
        vrot = np.vstack([np.einsum('ijk, ...j, ...k', eijk, Omega_veci,ROTORs[i].xyzi-ROTORs[i].center[None,:]) for i, Omega_veci in enumerate(Omega_vecs)])
        azimdir = np.vstack([np.einsum('ijk, ...j, ...k', eijk, orthogonals[rotor_ids==i], ROTOR.xyzi - ROTOR.center[None, :]) for i, ROTOR in enumerate(ROTORs)])

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
        for i, ROTOR in enumerate(ROTORs):
            r_R = np.array(radial_positions[i] / ROTOR.R)
            r_R = r_R.reshape(ROTOR.nblades, int(r_R.size / ROTOR.nblades))
            alines = (vazim[rotor_ids == i] / (radial_positions[i] * Omegas[i]) - 1).reshape(ROTOR.nblades, int(r_R.size / ROTOR.nblades))
            a = -(uvw[:, 0] + vrot[:, 0] / self.Qinf[0])
            a_s = a[rotor_ids == i].reshape(ROTOR.nblades, int(r_R.size / ROTOR.nblades))
            gammas = sol.x[rotor_ids == i].reshape(ROTOR.nblades, int(r_R.size / ROTOR.nblades))
            norm_gammas = np.linalg.norm(self.Qinf) ** 2 * np.pi / (ROTOR.nblades * Omegas[i])
            Faxials = temploads[0][rotor_ids == i].reshape(ROTOR.nblades, int(r_R.size / ROTOR.nblades))
            Fazims = temploads[1][rotor_ids == i].reshape(ROTOR.nblades, int(r_R.size / ROTOR.nblades))
            normFaxs = .5 * np.linalg.norm(self.Qinf) ** 2 * ROTOR.R
            alphas = temploads[3][rotor_ids == i].reshape(ROTOR.nblades, int(r_R.size / ROTOR.nblades))
            inflows = temploads[4][rotor_ids == i].reshape(ROTOR.nblades, int(r_R.size / ROTOR.nblades))
            ROTOR.results.update({
                'r_R': r_R,
                'Gamma': gammas/norm_gammas,
                'a': a_s,
                'aline': alines,
                'Faxial': Faxials/ normFaxs,
                'Fazim': Fazims/ normFaxs,
                'norm_gamma': norm_gammas,
                'norm_Faxial': normFaxs,
                'Omega': Omegas[i],
                'TSR': Omegas[i]*ROTOR.R,
                'alpha':alphas,
                'inflow':inflows})
            CT, CP = self._calculateCT_CProtor_CPflow(ROTOR, np.average(Faxials, axis=0), np.average(Fazims, axis=0))
            ROTOR.results['CT'] = CT
            ROTOR.results['CP'] = CP
        if plot: self.plot_instantaneous()

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

    def _calculateCT_CProtor_CPflow(self, ROTOR: Rotor, Faxial, Fazim):
        r_Rarray = ROTOR.elem_boundaries/ ROTOR.R
        r_R_temp = 1 / 2 * (r_Rarray[:-1] + r_Rarray[1:])
        drtemp = np.diff(r_Rarray)
        Uinf = np.linalg.norm(self.Qinf)
        CT = np.sum(drtemp * Faxial * ROTOR.nblades / (0.5 * Uinf ** 2 * np.pi * ROTOR.R))
        CP = np.sum(drtemp * Fazim  * r_R_temp * ROTOR.results['Omega']  * ROTOR.nblades / (0.5 * Uinf ** 2 * np.pi))
        return CT, CP

def rotor_blade(r_R):
    pitch = 2
    chord = 3 * (1 - r_R) + 1
    twist = -14 * (1 - r_R)
    return chord, np.radians(twist + pitch)

def cosine_spacing(a, b, N):
    theta = np.linspace(0, np.pi, N + 1)
    x = 0.5 * (a + b) + 0.5 * (b - a) * np.cos(theta)
    return np.flip(x)


def rotor_wake(theta_array, ri_elem_boundaries, ROTOR: Rotor, aw=0.2, fcp=0.75, fbound = 1/4, plot: bool = False)->Rotor:
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
    chord_ref, angle_ref = ROTOR.geomfunc(ri/ROTOR.R)
    ni_ref = np.array([np.cos(angle_ref), np.zeros_like(angle_ref), -np.sin(angle_ref)]).T
    xyzi_ref = np.stack((fcp * chord_ref *np.sin(-angle_ref), ri, -fcp*chord_ref*np.cos(angle_ref))).T

    'control points'
    rotation_transform = np.arange(0, 2*np.pi, 2*np.pi/ROTOR.nblades)
    c = np.cos(rotation_transform)
    s = np.sin(rotation_transform)
    rot = np.stack([np.stack([np.ones_like(c), np.zeros_like(c), np.zeros_like(c)], axis=1),
        np.stack([np.zeros_like(c), c, -s], axis=1),
        np.stack([np.zeros_like(c), s, c], axis=1)], axis=1)
    ni = np.einsum('bj,akj->abk', ni_ref, rot).reshape(-1, 3)
    xyzi = np.einsum('bj,akj->abk', xyzi_ref, rot).reshape(-1, 3)

    'vortex rings'
    chord_ref, angle_ref = ROTOR.geomfunc(ri_elem_boundaries/ROTOR.R)
    dys = np.cumsum(np.multiply.outer(np.diff(np.cos(-theta_array)),ri_elem_boundaries).T, axis=1)
    dzs = np.cumsum(np.multiply.outer(np.diff(np.sin(-theta_array)), ri_elem_boundaries).T, axis=1)
    dxs = np.cumsum(np.multiply.outer(np.diff(theta_array)/(ROTOR.TSR/(1-aw))*ROTOR.R, np.ones_like(ri_elem_boundaries)).T, axis=1)
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
        blade_plot = xyzi.reshape(ROTOR.nblades, ROTOR.N, 3)

        for bi in range(ROTOR.nblades):
            ax.plot(blade_plot[bi, :,0],blade_plot[bi, :,1],blade_plot[bi, :,2], color='r',marker='none', zorder=10)
            for i in range(ROTOR.N):
                ax.plot(xyzj[ bi, i,:, 0], xyzj[bi,i,  :, 1], xyzj[ bi,i, :, 2], color='b', linewidth=.2, )#alpha=.6)
        plt.xlim([0, 4*ROTOR.R])
        plt.gca().set_aspect('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.gca().set_zlabel('z')
        plt.title(f'$\lambda={ROTOR.TSR}, N={ROTOR.N}, k_2={theta_array.size}$')
        plt.tight_layout()
        plt.show()
    ROTOR.xyzi = xyzi
    ROTOR.xyzj = xyzj.reshape(ROTOR.nblades * ROTOR.N, xyzj.shape[2], 3)
    ROTOR.elem_boundaries = ri_elem_boundaries
    ROTOR.ni = ni
    return ROTOR


class SingleRotorExperiment:
    def __init__(self, ROTOR: Rotor, Qinf,):
        self.ROTOR = ROTOR
        self.ROTOR.results={}
        self.Qinf = Qinf
        self.method_dict = {'cosine': self.cosine_spacing, 'linear': self.linear_spacing}

        self.geomfunc = rotor_wake
        ROTOR.xyzi = np.zeros((3, 3))
        ROTOR.ni = np.zeros((3, 3))
        ROTOR.xyzj = np.zeros((3, 3, 3))
        ROTOR.elem_boundaries = np.zeros(10)
        self.ROTORSIM = MultiWakeSim(ROTORs=[ROTOR], Qinf=Qinf,)
        #self.compiled_results = {}

    @staticmethod
    def cosine_spacing(a, b, N):
        return cosine_spacing(a, b, N)

    @staticmethod
    def linear_spacing(a, b, N):
        return np.linspace(a, b, N+1)

    def plot_compiled_convergence(self, lw=0.75, fs=8, c='k')->None:
        ls = ['--', '-',  ':']
        ms = ['o', 's', '<', 'x']
        keys = self.ROTOR.results['key']
        keyvalss = self.ROTOR.results['key_vals']
        CTs = []
        for keyval2 in keyvalss[1]:
            temp1 = []
            for keyval1 in keyvalss[0]:
                temp1.append(self.ROTOR.results['results'][(keyval1, keyval2)]['CT'])
            CTs.append(temp1)
        fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7, 6))
        fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(7, 6))
        errors = [[abs(CTs[i][j] - CTs[i][-1])/CTs[i][-1] for j in range(len(CTs[0])-1)] for i in range(len(keyvalss[1]))]
        for i in range(len(keyvalss[1])):
            kv1 = (f'={keyvalss[1][i]:.2f}' if isinstance(keyvalss[1][i], float) else f'={keyvalss[1][i]}')
            ax1.plot(keyvalss[0], CTs[i], lw=lw, color=c, linestyle=ls[i], marker=ms[i], label=f'{keys[1]}: '+f'{kv1}')
            ax2.loglog(keyvalss[0][:-1], errors[i], lw=lw, color=c, linestyle=ls[i], marker=ms[i], label=f'{keys[1]}: '+f'{kv1}')
        ax1.grid()
        ax1.legend(fontsize=fs)
        ax1.set_xlabel(keys[0])
        ax1.set_ylabel('$C_T$')
        ax2.grid()
        ax2.legend(fontsize=fs)
        ax2.set_xlabel(keys[0])
        ax2.set_ylabel('$-$')
        fixed_vars = self.ROTOR.results.get('fixed_vars', {})
        title_str = ', '.join(f'{k}={v:.2f}' if isinstance(v, float) else f'{k}={v}' for k, v in fixed_vars.items())
        ax1.set_title(title_str + f', $n_b$={self.ROTOR.nblades}')
        ax2.set_title(title_str + f', $n_b$={self.ROTOR.nblades}')
        fig1.tight_layout()
        fig2.tight_layout()
        plt.show()

    def plot_compiled_results(self, lw=0.75, fs=8)->None:
        colors = ['k', 'b', 'g', 'orange', 'r']
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 7))
        key = self.ROTOR.results['key']
        keyvals = self.ROTOR.results['key_vals']
        for i, keyval in enumerate(keyvals):
            kv = (f'{keyvals[i]:.2f}' if isinstance(keyvals[i], float) else f'{keyvals[i]}')
            results = self.ROTOR.results['results'][keyval]
            r_R = np.average(results['r_R'], axis=0)
            axes[0,0].plot(r_R, np.average(results['Gamma'], axis=0), marker='o', color=colors[i],label='$\\tilde{\Gamma}$: '+key+f'={kv}', linewidth=lw)
            axes[0,1].plot(r_R, np.average(results['Faxial'], axis=0), marker='o', color=colors[i], label='$\\tilde{F}_{axial}$: '+key+f'={kv}', linewidth=lw)
            axes[0,1].plot(r_R, np.average(results['Fazim'], axis=0), linestyle='--', marker='s',color=colors[i], label='$\\tilde{F}_{azim}$: '+key+f'={kv}', linewidth=lw)
            axes[1,0].plot(r_R, np.average(results['a'], axis=0), marker='o', color=colors[i],label='$a$: '+key+f'={kv}', linewidth=lw)
            axes[1,0].plot(r_R, np.average(results['aline'], axis=0), marker='s', linestyle='--', color=colors[i],label='$a^{,}$: '+key+f'={kv}', linewidth=lw)
            axes[1,1].plot(r_R, np.average(results['alpha'], axis=0), marker='o', color=colors[i],label='$\\alpha$: '+key+f'={kv}', linewidth=lw)
            axes[1,1].plot(r_R, np.average(results['inflow'], axis=0), marker='s',linestyle='--', color=colors[i], label='$\phi$: ' +key+ f'={kv}', linewidth=lw)
        axes[0, 0].grid()
        axes[0, 0].set_ylim(bottom=0)
        axes[0, 0].set_ylabel('$\\tilde{\Gamma}$')
        axes[0, 0].legend(fontsize=fs)
        axes[0, 1].set_ylim(bottom=0)
        axes[0, 1].set_ylabel('$\\tilde{F}$')
        axes[0, 1].grid()
        axes[0, 1].legend(fontsize=fs)
        axes[1, 0].set_ylim(bottom=0)
        axes[1, 0].grid()
        axes[1, 0].legend(fontsize=fs)
        axes[1, 0].set_ylabel('$a$')
        axes[1, 0].set_xlabel('$r/R$')
        axes[1, 1].grid()
        axes[1, 1].set_ylim(bottom=0)
        axes[1, 1].set_ylabel('$\\alpha, \phi$')
        axes[1, 1].legend(fontsize=fs)
        axes[1, 1].set_xlabel('$r/R$')
        fixed_vars = self.ROTOR.results.get('fixed_vars', {})
        title_str = ', '.join(f'{k}={v:.2f}' if isinstance(v, float) else f'{k}={v}' for k, v in fixed_vars.items())
        fig.suptitle(title_str+f', $n_b$={self.ROTOR.nblades}')
        fig.tight_layout()
        plt.show()

    def collect_convergence(self, monitor: bool = False, **kwargs,)->None:
        '''
        :param monitor: plot progress
        :param kwargs: 'TSR', 'aw', 'N', 'revolutions', 'dtheta', 'spacing', ONLY the sweep arg AND 'N' is of type np.ndarray. First occurrence is the x-axis
        '''
        sweep_keys = [key for key, value in kwargs.items() if isinstance(value, np.ndarray)]
        print(sweep_keys)
        self.ROTOR.results = {}
        self.ROTOR.results['key'] = sweep_keys
        self.ROTOR.results['key_vals'] = [kwargs[sweep_keys[i]] for i in range(len(sweep_keys))]
        self.ROTOR.results['results'] = {}
        self.ROTOR.results['fixed_vars'] = {k: v for k, v in kwargs.items() if k not in sweep_keys}
        print(100 * '-' + f'\nSWEEP KEYS: {sweep_keys}\n' + 100 * '-')
        for sweep_val_1 in kwargs[sweep_keys[0]]:
            for sweep_val_2 in kwargs[sweep_keys[1]]:
                loop_args = kwargs.copy()
                loop_args[sweep_keys[0]] = sweep_val_1
                loop_args[sweep_keys[1]] = sweep_val_2
                ri_elem_boundaries = self.method_dict[loop_args['spacing']](self.ROTOR.blade_bounds[0] * self.ROTOR.R, self.ROTOR.R, loop_args['N'])
                theta_array = np.arange(0, loop_args['revolutions'] * 2 * np.pi, loop_args['dtheta'])
                self.ROTOR.TSR = loop_args['TSR']
                self.ROTOR.N = loop_args['N']
                self.geomfunc(theta_array=theta_array, ri_elem_boundaries=ri_elem_boundaries, ROTOR=self.ROTOR, plot=monitor, fbound=0., fcp=0., aw=loop_args['aw'])
                self.ROTORSIM.update(xyzi_new=self.ROTOR.xyzi, xyzj_new=self.ROTOR.xyzj, ni_new=self.ROTOR.ni, Qinf_new=self.Qinf,)
                self.ROTORSIM.iter_solve(niter=1200, tol=1e-8, plot=monitor, verbose=False)
                self.ROTOR.results['results'][(sweep_val_1, sweep_val_2)] = self.ROTORSIM.ROTORs[0].results.copy()
        self.plot_compiled_convergence()

    def collect_variable(self, monitor: bool = False, **kwargs)->None:
        '''
        :param monitor: plot progress
        :param kwargs: 'TSR', 'aw', 'N', 'revolutions', 'dtheta', 'spacing', ONLY the sweep arg is of type np.ndarray
        '''
        print(kwargs.keys())
        sweep_key = next((key for key, value in kwargs.items() if isinstance(value, np.ndarray)), None)
        self.ROTOR.results = {}
        self.ROTOR.results['key'] = sweep_key
        self.ROTOR.results['key_vals'] = kwargs[sweep_key]
        self.ROTOR.results['results'] = {}
        self.ROTOR.results['fixed_vars'] = {k: v for k, v in kwargs.items() if k != sweep_key}
        print(100*'-'+f'\nSWEEP KEY: {sweep_key}\n'+100*'-')
        for sweep_val in kwargs[sweep_key]:
            loop_args = kwargs.copy()
            loop_args[sweep_key] = sweep_val
            ri_elem_boundaries = self.method_dict[loop_args['spacing']](self.ROTOR.blade_bounds[0] * self.ROTOR.R, self.ROTOR.R, loop_args['N'])
            theta_array = np.arange(0, loop_args['revolutions'] * 2 * np.pi, loop_args['dtheta'])
            self.ROTOR.TSR = loop_args['TSR']
            self.ROTOR.N = loop_args['N']
            self.geomfunc(theta_array=theta_array, ri_elem_boundaries=ri_elem_boundaries, ROTOR=self.ROTOR, plot=monitor, fbound=0., fcp=0., aw=loop_args['aw'])
            self.ROTORSIM.update(xyzi_new=self.ROTOR.xyzi, xyzj_new=self.ROTOR.xyzj, ni_new=self.ROTOR.ni, Qinf_new=self.Qinf,)
            self.ROTORSIM.iter_solve(niter=1200, tol=1e-8, plot=monitor, verbose=False)
            self.ROTOR.results['results'][sweep_val] = self.ROTORSIM.ROTORs[0].results.copy()
        self.plot_compiled_results()


class DualRotorExperiment:
    def __init__(self, ROTORS: tuple[Rotor, ...], Qinf, aw = 0.2, spacing='cosine'):
        self.spacing = spacing
        self.Qinf = Qinf
        self.method_dict = {'cosine': cosine_spacing, 'linear': self.linear_spacing}
        self.aw=aw
        wakerevs = [50, 50]
        self.ROTORS = [self._init_Rotor(ROTORS[i], wakerevs[i]) for i in range(len(ROTORS))]
        self.base_rotors = [ROT.copy() for ROT in self.ROTORS]
        self.nrotors = len(self.base_rotors)

    @staticmethod
    def linear_spacing(a, b, N):
        return np.linspace(a, b, N+1)

    def _reset_rotors(self)->None:
        self.ROTORS = self.base_rotors

    def _init_Rotor(self, ROTOR: Rotor, revolutions)->Rotor:
        theta_array = np.arange(0, revolutions * 2 * np.pi, ROTOR.dtheta)
        ri_elem_boundaries = self.method_dict[self.spacing](0.2 * ROTOR.R, ROTOR.R, ROTOR.N)
        rotor_wake(aw=self.aw, theta_array=theta_array, ri_elem_boundaries=ri_elem_boundaries, ROTOR=ROTOR, plot=False, fbound=0., fcp=0., )
        return ROTOR

    def simulate(self, delta_phi_0s: tuple[float, ...], delta_Ls: tuple[float, ...], plot: bool = True)->None:
        self._reset_rotors()
        for i in range(self.nrotors):
            rotMat = self.x_rotation_matrix(angle=delta_phi_0s[i])
            self.ROTORS[i].rotate(rotMat)
            self.ROTORS[i].translate(displVec=np.array([0, delta_Ls[i], 0]))
        rotorsim = MultiWakeSim(ROTORs=self.ROTORS, Qinf=self.Qinf, )
        self.rotor_ids = np.concatenate([np.full(rotor.xyzi.shape[0], i) for i, rotor in enumerate(self.ROTORS)])
        if plot:
            fig = plt.figure(figsize=(12, 7))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_proj_type('persp', focal_length=0.2)
            ax.view_init(5, -125, 0)
            ax.grid(False)
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            for r in range(self.nrotors):
                rotor_xyzj = self.ROTORS[r].xyzj
                rotor_xyzj = rotor_xyzj.reshape(self.ROTORS[r].nblades, self.ROTORS[r].N, rotor_xyzj.shape[1],3)
                for bi in range(self.ROTORS[r].nblades):
                    for i in range(self.ROTORS[r].N):
                        ax.plot(
                            rotor_xyzj[bi,i, :, 0],
                            rotor_xyzj[bi,i, :, 1],
                            rotor_xyzj[bi,i, :, 2],
                            color='b', linewidth=0.2)
            plt.xlim([0, 4 * self.ROTORS[0].R])
            plt.gca().set_aspect('equal')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.gca().set_zlabel('z')
            plt.tight_layout()
            plt.show()
        rotorsim.iter_solve(plot=plot)

    def x_rotation_matrix(self, angle):
        c = np.cos(angle)
        s = np.sin(angle)
        rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        return rot


if __name__ == "__main__":
    'simulation parameters'
    N           = 11
    revolutions = 50
    TSR         = 6.0
    R           = 50
    nblades     = 3
    aw          = 0.2
    Qinf        = np.array([1, 0, 0])

    'double rotor experiment'
    ROTOR1 = Rotor(R=50, dtheta=np.pi / 10, N=11, TSR=6, geomfunc=rotor_blade, nblades=3, blade_bounds=(0.2, 1), direction=1)
    ROTOR2 = Rotor(R=50, dtheta=np.pi / 10, N=8, TSR=8, geomfunc=rotor_blade, nblades=3, blade_bounds=(0.2, 1), direction=1)
    #DR = DualRotorExperiment(ROTORS=(ROTOR1, ROTOR2), Qinf=Qinf,)
    #DR.simulate(delta_phi_0s=(0, np.radians(30)), delta_Ls=(0, 200), plot=True)

    'single rotor experiment'
    ROTOR3 = Rotor(R=50, geomfunc=rotor_blade, nblades=3, blade_bounds=(0.2, 1), direction=+1)
    E = SingleRotorExperiment(ROTOR=ROTOR3, Qinf=Qinf)
    #E.collect_variable(TSR=6, aw=np.array([.05, .35, .65]), N=25, revolutions=50, dtheta=np.pi / 10, spacing='cosine')
    #E.collect_variable(TSR=np.array([4, 6, 8]), aw=0.2, N=25, revolutions=50, dtheta=np.pi / 10, spacing='cosine', monitor=False)
    #E.collect_variable(TSR=6, aw=0.2, N=25, revolutions=50, dtheta=np.linspace(np.pi/50,np.pi/2, 3), spacing='cosine')
    #E.collect_variable(TSR=6, aw=0.2, N=25, revolutions=np.array([.5, 5, 50]), dtheta=np.pi/10, spacing='cosine')
    #E.collect_variable(TSR=6, aw=0.2, N=40, revolutions=50, dtheta=np.pi / 10, spacing=np.array(['cosine', 'linear']))
    #E.collect_convergence(TSR=6, N=np.array([2**k for k in range(0, 7)]), aw=0.2, revolutions=50, dtheta=np.pi/10, spacing=np.array(['cosine', 'linear']), monitor=False)
    #E.collect_variable(TSR=6, aw=0.2, N=25, revolutions=np.array([.5, 5, 50]), dtheta=np.pi / 10, spacing='cosine')
    #E.collect_convergence(TSR=6, aw=0.2, N=11, revolutions=np.array([2**k for k in range(0, 10)]), dtheta=np.pi/10, spacing=np.array(['cosine', 'linear']))

    'generate and associate geometry'
    ROTOR4 = Rotor(N=N, TSR=TSR, R=R, nblades=nblades, geomfunc=rotor_blade, blade_bounds=(0.2, 1.0),)
    theta_array        = np.arange(0, revolutions * 2 * np.pi, np.pi / 10)
    ri_elem_boundaries = cosine_spacing(0.2 * R, R, N) #r_array * R #
    ROTOR4 = rotor_wake(theta_array=theta_array, ri_elem_boundaries=ri_elem_boundaries, aw=aw, ROTOR=ROTOR4, plot=True, fbound=0., fcp=0., )

    'simulation'
    ROTORSIM = MultiWakeSim(ROTORs=[ROTOR4], Qinf=Qinf,)
    ROTORSIM.iter_solve(niter=1200, tol=1e-8, plot=True, )
    resLL = ROTORSIM.ROTORs[0].results
    print(f'(CT, CP) = ({resLL["CT"]:.2f},{resLL["CP"]:.2f})')