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
        self.method_dict = {'cosine': self.cosine_spacing, 'linear': self.linear_spacing}

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
        fixed_vars = self.compiled_results.get('fixed_vars', {})
        title_str = ', '.join(f'{k}={v:.2f}' if isinstance(v, float) else f'{k}={v}' for k, v in fixed_vars.items())
        ax1.set_title(title_str + f', $n_b$={self.nblades}')
        ax2.set_title(title_str + f', $n_b$={self.nblades}')
        fig1.tight_layout()
        fig2.tight_layout()
        plt.show()

    def plot_compiled_results(self, lw=0.75, fs=8)->None:
        colors = ['k', 'b', 'g', 'orange', 'r']
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 7))
        key = self.compiled_results['key']
        keyvals = self.compiled_results['key_vals']
        for i, keyval in enumerate(keyvals):
            kv = (f'={keyvals[i]:.2f}' if isinstance(keyvals[i], float) else f'={keyvals[i]}')
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
        fixed_vars = self.compiled_results.get('fixed_vars', {})
        title_str = ', '.join(f'{k}={v:.2f}' if isinstance(v, float) else f'{k}={v}' for k, v in fixed_vars.items())
        fig.suptitle(title_str+f', $n_b$={self.nblades}')
        fig.tight_layout()
        plt.show()

    def collect_convergence(self, monitor: bool = False, **kwargs,)->None:
        '''
        :param monitor: plot progress
        :param kwargs: 'TSR', 'aw', 'N', 'revolutions', 'dtheta', 'spacing', ONLY the sweep arg AND 'N' is of type np.ndarray. First occurrence is the x-axis
        '''
        sweep_keys = [key for key, value in kwargs.items() if isinstance(value, np.ndarray)]
        print(sweep_keys)
        self.compiled_results = {}
        self.compiled_results['key'] = sweep_keys
        self.compiled_results['key_vals'] = [kwargs[sweep_keys[i]] for i in range(len(sweep_keys))]
        self.compiled_results['results'] = {}
        self.compiled_results['fixed_vars'] = {k: v for k, v in kwargs.items() if k not in sweep_keys}
        print(100 * '-' + f'\nSWEEP KEYS: {sweep_keys}\n' + 100 * '-')
        for sweep_val_1 in kwargs[sweep_keys[0]]:
            for sweep_val_2 in kwargs[sweep_keys[1]]:
                loop_args = kwargs.copy()
                loop_args[sweep_keys[0]] = sweep_val_1
                loop_args[sweep_keys[1]] = sweep_val_2
                ri_elem_boundaries = self.method_dict[loop_args['spacing']](self.blade_bounds[0] * self.R, self.R, loop_args['N'])
                theta_array = np.arange(0, loop_args['revolutions'] * 2 * np.pi, loop_args['dtheta'])
                xyzi, xyzj, ni, ri = self.geomfunc(theta_array=theta_array, ri_elem_boundaries=ri_elem_boundaries, N=loop_args['N'], geom_func=self.bladefunc, R=self.R, TSR=loop_args['TSR'] / (1 - loop_args['aw']), nblades=self.nblades, plot=monitor, fbound=0., fcp=0., )
                self.ROTORSIM.update(xyzi_new=xyzi, xyzj_new=xyzj, ni_new=ni, Qinf_new=self.Qinf, elem_bounds_new=ri_elem_boundaries)
                self.ROTORSIM.iter_solve(Omega=loop_args['TSR'] / self.R, niter=1200, tol=1e-8, plot=monitor, verbose=False)
                self.compiled_results['results'][(sweep_val_1, sweep_val_2)] = self.ROTORSIM.results.copy()
        self.plot_compiled_convergence()

    def collect_variable(self, monitor: bool = False, **kwargs)->None:
        '''
        :param monitor: plot progress
        :param kwargs: 'TSR', 'aw', 'N', 'revolutions', 'dtheta', 'spacing', ONLY the sweep arg is of type np.ndarray
        '''
        print(kwargs.keys())
        sweep_key = next((key for key, value in kwargs.items() if isinstance(value, np.ndarray)), None)
        self.compiled_results = {}
        self.compiled_results['key'] = sweep_key
        self.compiled_results['key_vals'] = kwargs[sweep_key]
        self.compiled_results['results'] = {}
        self.compiled_results['fixed_vars'] = {k: v for k, v in kwargs.items() if k != sweep_key}
        print(100*'-'+f'\nSWEEP KEY: {sweep_key}\n'+100*'-')
        for sweep_val in kwargs[sweep_key]:
            loop_args = kwargs.copy()
            loop_args[sweep_key] = sweep_val
            ri_elem_boundaries = self.method_dict[loop_args['spacing']](self.blade_bounds[0] * self.R, self.R, loop_args['N'])
            theta_array = np.arange(0, loop_args['revolutions'] * 2 * np.pi, loop_args['dtheta'])
            xyzi, xyzj, ni, ri = self.geomfunc(theta_array=theta_array, ri_elem_boundaries=ri_elem_boundaries, N=loop_args['N'], geom_func=self.bladefunc, R=self.R, TSR=loop_args['TSR'] / (1 - loop_args['aw']), nblades=self.nblades, plot=monitor, fbound=0., fcp=0., )
            self.ROTORSIM.update(xyzi_new=xyzi, xyzj_new=xyzj, ni_new=ni, Qinf_new=self.Qinf, elem_bounds_new=ri_elem_boundaries)
            self.ROTORSIM.iter_solve(Omega=loop_args['TSR']/self.R, niter=1200, tol=1e-8, plot=monitor, verbose=False)
            self.compiled_results['results'][sweep_val] = self.ROTORSIM.results.copy()
        self.plot_compiled_results()


if __name__ == '__main__':
    Qinf = np.array([1,0,0])
    E = SingleRotorExperiment(R=50, nblades=3, Qinf=Qinf)
    #E.collect_variable(TSR=6, aw=0.2, N=11, revolutions=50, dtheta=np.linspace(np.pi/50,np.pi/2, 3), spacing='cosine')
    E.collect_variable(TSR = np.array([4, 6, 8]), aw=0.2, N=11, revolutions=50, dtheta=np.pi/10, spacing='cosine')
    #E.collect_variable(TSR=6, aw=np.array([.05, .35, .65]), N=11, revolutions=50, dtheta=np.pi / 10, spacing='cosine')
    #E.collect_convergence(TSR=6, aw=0.2, N=np.array([2**k for k in range(1, 6)]), revolutions=50, dtheta=np.pi/10, spacing=np.array(['linear', 'cosine']))
    #E.collect_convergence(TSR=6, aw=0.2, N=np.array([2**k for k in range(1, 4)]), revolutions=50, dtheta=np.linspace(np.pi/2, np.pi/4, 2), spacing='cosine')

