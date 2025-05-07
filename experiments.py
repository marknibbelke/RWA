from BEM import *
import matplotlib.pyplot as plt



def run_annuli():
    delta_r_Rs = np.linspace(0.002, 0.075, 40)
    r_R_dict = {}


    # blade shape
    pitch = 2  # degrees

    # define flow conditions
    Uinf = 10  # unperturbed wind speed in m/s
    TSR = 8  # tip speed ratio
    Radius = 50
    NBlades = 3

    TipLocation_R = 1
    RootLocation_R = 0.2

    for delta_r in delta_r_Rs:
        N = int(0.8 / delta_r)


        i = np.arange(N)
        r_cosine = 0.5 * (RootLocation_R + TipLocation_R) + 0.5 * (TipLocation_R - RootLocation_R) * np.cos(np.pi * (1 - i / (N - 1)))#*.999

        r_R_dict[delta_r] = [np.linspace(r_cosine[0], r_cosine[-1], N)]
        r_R_dict[delta_r].append(r_cosine)

        #plt.plot(r_cosine, np.ones_like(r_cosine), marker='o')
        #plt.show()

    # CT, CP, CQ, results = run_bem(r_R=r_R, chord_distribution=chord_distribution, twist_distribution=twist_distribution, Uinf=Uinf, TSR=TSR, Radius=Radius, NBlades=NBlades, TipLocation_R=TipLocation_R, RootLocation_R=RootLocation_R, plot = True)
    # Plot vs TSRs:
    RESULTS = {}
    colors = ['k', 'g', 'r']
    CTS_linear = []
    CTS_cosine = []
    for delta_r in delta_r_Rs:
        r_R = r_R_dict[delta_r]

        chord_distribution = 3 * (1 - r_R[0]) + 1  # meters
        twist_distribution = -14 * (1 - r_R[0]) + pitch  # degrees
        CTilin, CPilin, CQilin, resultsilin = run_bem(r_R=r_R[0], chord_distribution=chord_distribution, twist_distribution=twist_distribution, Uinf=Uinf, TSR=TSR, Radius=Radius, NBlades=NBlades, TipLocation_R=TipLocation_R, RootLocation_R=RootLocation_R,  polar_alpha=polar_alpha, polar_cd=polar_cd, polar_cl=polar_cl, plot=False, verbose=False)
        CTS_linear.append(CTilin)


        chord_distribution = 3 * (1 - r_R[1]) + 1  # meters
        twist_distribution = -14 * (1 - r_R[1]) + pitch  # degrees

        CTicos, CPicos, CQicos, resultsicos = run_bem(r_R=r_R[1], chord_distribution=chord_distribution, twist_distribution=twist_distribution, Uinf=Uinf, TSR=TSR, Radius=Radius, NBlades=NBlades, TipLocation_R=TipLocation_R, RootLocation_R=RootLocation_R, polar_alpha=polar_alpha, polar_cd=polar_cd, polar_cl=polar_cl, plot=False, verbose=False)
        CTS_cosine.append(CTicos)

    plt.plot((TipLocation_R-RootLocation_R)/delta_r_Rs, CTS_linear, label='linear spacing', color='k', linestyle='-')
    plt.plot((TipLocation_R - RootLocation_R) / delta_r_Rs, CTS_cosine, label='cosine spacing', color='k', linestyle='--')
    plt.legend()
    plt.xlabel('$N = 1/(\Delta [r/R])$')
    plt.ylabel('$C_T$')
    plt.show()


    '''
    fig1 = plt.figure(figsize=(12, 6))
    plt.title('Axial and tangential induction')
    for i, t in enumerate(delta_r_Rs):
        plt.plot(RESULTS[t][:, 2], RESULTS[t][:, 0], color=colors[i], linestyle='-', label=f'$a, \Delta(r/R)={t}$', )
        plt.plot(RESULTS[t][:, 2], RESULTS[t][:, 1], color=colors[i], linestyle='--', label=f'$a^, \Delta(r/R)={t}$', )  # linewidth=1.2)
    plt.grid()
    plt.xlabel('r/R')
    plt.legend()
    plt.show()

    fig1 = plt.figure(figsize=(12, 6))
    plt.title(r'Normal and tagential force, non-dimensioned by $\frac{1}{2} \rho U_\infty^2 R$')
    for i, t in enumerate(delta_r_Rs):
        plt.plot(RESULTS[t][:, 2], RESULTS[t][:, 3] / (0.5 * Uinf ** 2 * Radius), color=colors[i], linestyle='-', label=f'Fnorm, $\Delta(r/R)={t}$', )  # linewidth=1.)
        plt.plot(RESULTS[t][:, 2], RESULTS[t][:, 4] / (0.5 * Uinf ** 2 * Radius), color=colors[i], linestyle='--', label=f'Ftan, $\Delta(r/R)={t}$', )  # linewidth=1.)
    plt.grid()
    plt.xlabel('r/R')
    plt.legend()
    plt.show()

    fig1 = plt.figure(figsize=(12, 6))
    plt.title(r'Distribution of AoA and inflow Angle')
    for i, t in enumerate(delta_r_Rs):
        plt.plot(RESULTS[t][:, 2], RESULTS[t][:, 6], color=colors[i], linestyle='-', label=f'$\\alpha$ [deg.], $\Delta(r/R)={t}$')
        plt.plot(RESULTS[t][:, 2], RESULTS[t][:, 7] * 180 / np.pi, color=colors[i], linestyle='--', label=f'$\phi$ [deg.], $\Delta(r/R)={t}$')
    plt.grid()
    plt.xlabel('r/R')
    plt.legend()
    plt.show()
    '''





if __name__ == '__main__':
    polar_alpha, polar_cl, polar_cd = read_polar(airfoil='DU95W180.txt', plot=False)
    run_annuli()
    #run_spacing_method()