import rotorWake as rw
import matplotlib.pyplot as plt
import numpy as np


class SingleRotorExperiment:
    def __init__(self, R, nblades: int, Qinf, bladefunc: callable = rw.rotor_blade, blade_bounds:tuple=(0.2, 1)):
        self.R = R
        self.nblades = nblades
        self.Qinf = Qinf
        self.bladefunc = bladefunc
        self.blade_bounds = blade_bounds

        self.geomfunc = rw.rotor_wake
        self.ROTORSIM = rw.RotorWakeSim(np.zeros((5, 3)), np.zeros((5,4,3)), np.zeros((5, 3)), np.zeros(6), Qinf, R, bladefunc, nblades)
        self.compiled_results = {}

    @staticmethod
    def cosine_spacing(a, b, N):
        return rw.cosine_spacing(a, b, N)

    @staticmethod
    def linear_spacing(a, b, N):
        return np.linspace(a, b, N+1)

    def plot_compiled_convergence(self, lw=0.75, fs=8, c='k')->None:

        ls = ['--', '-',  ':']
        ms = ['o', 's', '<', 'x']
        keys = self.compiled_results['key']
        keyvalss = self.compiled_results['key_vals']
        CTs = []
        for keyval2 in keyvalss[1]:
            temp1 = []
            for keyval1 in keyvalss[0]:
                temp1.append(self.compiled_results['results'][(keyval1, keyval2)]['CT'])
            CTs.append(temp1)

        fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7, 6))
        fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(7, 6))
        errors = [[abs(CTs[i][j] - CTs[i][-1]) for j in range(len(CTs[0]))] for i in range(len(keyvalss[1]))]
        for i in range(len(keyvalss[1])):
            ax1.plot(keyvalss[0], CTs[i], lw=lw, color=c, linestyle=ls[i], marker=ms[i], label=f'{keys[1]}: '+f'{keyvalss[1][i]}')
            ax2.loglog(keyvalss[0], errors[i], lw=lw, color=c, linestyle=ls[i], marker=ms[i], label=f'{keys[1]}: '+f'{keyvalss[1][i]}')
        ax1.grid()
        ax1.legend(fontsize=fs)
        ax1.set_xlabel('$N$')
        ax1.set_ylabel('$C_T$')
        ax2.grid()
        ax2.legend(fontsize=fs)
        ax2.set_xlabel('$N$')
        ax2.set_ylabel('$-$')
        plt.show()

    def plot_compiled_results(self, lw=0.75, fs=8)->None:
        colors = ['k', 'b', 'g', 'orange', 'r']
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 7))
        key = self.compiled_results['key']
        keyvals = self.compiled_results['key_vals']
        results=None
        for i, keyval in enumerate(keyvals):
            kv = keyvals[i]
            results = self.compiled_results['results'][keyval]
            axes[0,0].plot(results['r_R'], results['Gamma'], marker='o', color=colors[i],label='$\\tilde{\Gamma}$: '+key+f'={kv}', linewidth=lw)
            axes[0,1].plot(results['r_R'], results['Faxial'], marker='o', color=colors[i], label='$\\tilde{F}_{axial}$: '+key+f'={kv}', linewidth=lw)
            axes[0,1].plot(results['r_R'], results['Fazim'], linestyle='--', marker='s',color=colors[i], label='$\\tilde{F}_{azim}$: '+key+f'={kv}', linewidth=lw)
            axes[1,0].plot(results['r_R'], results['a'], marker='o', color=colors[i],label='$a$: '+key+f'={kv}', linewidth=lw)
            axes[1,0].plot(results['r_R'], results['aline'], marker='s', linestyle='--', color=colors[i],label='$a^{,}$: '+key+f'={kv}', linewidth=lw)
            axes[1,1].plot(results['r_R'], results['alpha'], marker='o', color=colors[i],label='$\\alpha$: '+key+f'={kv}', linewidth=lw)
            axes[1,1].plot(results['r_R'], results['inflow'], marker='o',linestyle='--', color=colors[i], label='$\phi$: ' +key+ f'={kv}', linewidth=lw)
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
        fig.suptitle(f'$N={results["N"]}, n_b={self.nblades}, k={self.ROTORSIM.xyzj.shape[1]}$')
        fig.tight_layout()
        plt.show()


    def collect_variable_TSR(self, TSRsweep: np.array, aw=0.2, N: int=11, revolutions=50, dtheta=np.pi/10, spacing:str='cosine', ):
        theta_array = np.arange(0, revolutions * 2 * np.pi, dtheta)
        ri_elem_boundaries = self.cosine_spacing(self.blade_bounds[0] * self.R, self.R, N)
        self.compiled_results = {}
        self.compiled_results['key'] = 'TSR'
        self.compiled_results['key_vals'] = TSRsweep
        self.compiled_results['results'] = {}
        for TSR in TSRsweep:
            xyzi, xyzj, ni, ri = self.geomfunc(theta_array=theta_array, ri_elem_boundaries=ri_elem_boundaries, N=N, geom_func=self.bladefunc, R=self.R, TSR=TSR / (1 - aw), nblades=self.nblades, plot=False, fbound=0., fcp=0., )
            self.ROTORSIM.update(xyzi_new=xyzi, xyzj_new=xyzj, ni_new=ni, Qinf_new=self.Qinf)
            self.ROTORSIM.elem_bounds = ri_elem_boundaries
            self.ROTORSIM.iter_solve(Omega=TSR/self.R, niter=1200, tol=1e-8, plot=False, verbose=False)
            self.compiled_results['results'][TSR] = self.ROTORSIM.results.copy()
        self.plot_compiled_results()

    def collect_variable_N(self, Nsweep: np.array, TSR = 6, aw=0.2, revolutions=50, dtheta=np.pi/10,):
        theta_array = np.arange(0, revolutions * 2 * np.pi, dtheta)
        self.compiled_results = {}
        self.compiled_results['key'] = ['N', 'spacing']
        self.compiled_results['key_vals'] = [Nsweep, ['cosine', 'linear']]
        self.compiled_results['results'] = {}
        methods = [self.cosine_spacing, self.linear_spacing]
        for N in Nsweep:
            for i, method in enumerate(methods):
                ri_elem_boundaries = method(self.blade_bounds[0] * self.R, self.R, N)
                xyzi, xyzj, ni, ri = self.geomfunc(theta_array=theta_array, ri_elem_boundaries=ri_elem_boundaries, N=N, geom_func=self.bladefunc, R=self.R, TSR=TSR / (1 - aw), nblades=self.nblades, plot=False, fbound=0., fcp=0., )
                self.ROTORSIM.update(xyzi_new=xyzi, xyzj_new=xyzj, ni_new=ni, Qinf_new=self.Qinf)
                self.ROTORSIM.elem_bounds = ri_elem_boundaries
                self.ROTORSIM.iter_solve(Omega=TSR / self.R, niter=1200, tol=1e-8, plot=False, verbose=False)
                self.compiled_results['results'][(N, self.compiled_results['key_vals'][1][i])] = self.ROTORSIM.results.copy()
        self.plot_compiled_convergence()

    def collect_variable_k(self):
        pass

    def collect_variable_aw(self):
        pass

    def collect_variable_rotations(self, TSRsweep: np.array, aw=0.2, N: int=11, revolutions=50, dtheta=np.pi/10, spacing:str='cosine',):
        pass

    def collect_variable_spacing(self, TSR = 6, aw=0.2, N: int=11, revolutions=50, dtheta=np.pi/10,):
        theta_array = np.arange(0, revolutions * 2 * np.pi, dtheta)
        self.compiled_results = {}
        self.compiled_results['key'] = 'spacing'
        self.compiled_results['key_vals'] = ['cosine', 'linear']
        self.compiled_results['results'] = {}
        methods = [self.cosine_spacing, self.linear_spacing]
        for i, spacing in enumerate(self.compiled_results['key_vals']):
            ri_elem_boundaries = methods[i](self.blade_bounds[0] * self.R, self.R, N)
            xyzi, xyzj, ni, ri = self.geomfunc(theta_array=theta_array, ri_elem_boundaries=ri_elem_boundaries, N=N, geom_func=self.bladefunc, R=self.R, TSR=TSR / (1 - aw), nblades=self.nblades, plot=False, fbound=0., fcp=0., )
            self.ROTORSIM.update(xyzi_new=xyzi, xyzj_new=xyzj, ni_new=ni, Qinf_new=self.Qinf)
            self.ROTORSIM.elem_bounds = ri_elem_boundaries
            self.ROTORSIM.iter_solve(Omega=TSR/self.R, niter=1200, tol=1e-8, plot=False, verbose=False)
            self.compiled_results['results'][spacing] = self.ROTORSIM.results.copy()
        self.plot_compiled_results()



if __name__ == '__main__':
    Qinf = np.array([1,0,0])
    E = SingleRotorExperiment(R=50, nblades=3, Qinf=Qinf)
    #E.collect_variable_TSR(TSRsweep=np.array([4, 6, 8]), N=25)
    #E.collect_variable_spacing(N=25)
    £E.collect_variable_N(Nsweep=np.array([2**k for k in range(0, 9)]), dtheta=np.pi/4, revolutions=3)
