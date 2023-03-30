"""The chain conformation-informed averaging characterization module for
composite uFJCs
"""

# import external modules
from __future__ import division
from composite_ufjc_scission import (
    CompositeuFJCScissionCharacterizer,
    RateIndependentScissionCompositeuFJC,
    latex_formatting_figure,
    save_current_figure
)
import numpy as np
import matplotlib.pyplot as plt
import sys


class ChainConformationInformedAveragingCharacterizer(
    CompositeuFJCScissionCharacterizer):
    """The characterization class assessing the chain
    conformation-informed averaging for composite uFJCs. It inherits all
    attributes and methods from
    the ``CompositeuFJCScissionCharacterizer`` class.
    """
    def __init__(self):
        """Initializes the
        ``ChainConformationInformedAveragingCharacterizer`` class by
        initializing and inheriting all attributes and methods from the
        ``CompositeuFJCScissionCharacterizer`` class.
        """
        CompositeuFJCScissionCharacterizer.__init__(self)
    
    def set_user_parameters(self):
        """Set user-defined parameters"""
        p = self.parameters

        p.characterizer.lmbda_c_eq_num_points = 1001
        p.characterizer.loginess_skew = 1.5
        p.characterizer.F_dot = 0.01 # 1/s
        p.characterizer.t_min = 0
        p.characterizer.t_max = 1
        p.characterizer.t_step = 0.2
        p.characterizer.t_step_chunk_num = 1
        
        p.characterizer.nu = 25
        p.characterizer.zeta_nu_char = 100
        p.characterizer.kappa_nu = 1000
    
    def prefix(self):
        """Set characterization prefix"""
        return "chain_conformation_informed_averaging"
    
    def characterization(self):
        """Define characterization routine"""

        cp = self.parameters.characterizer

        # Initialize applied deformation history of the chain
        # conformation space
        if cp.t_step > cp.t_max:
            sys.exit('Error: The time step is larger than the total deformation time!')

        # initialize the chunk counter and associated constants/lists
        chunk_counter  = 0
        chunk_indx_val = 0
        chunk_indx     = []

        # Initialization step: allocate time and stretch results, dependent upon the type of deformation being accounted for 
        t_val    = cp.t_min # initialize the time value at zero
        t        = [] # sec
        t_chunks = [] # sec

        # Append to appropriate lists
        t.append(t_val)
        t_chunks.append(t_val)
        chunk_indx.append(chunk_indx_val)

        # update the chunk iteration counter
        chunk_counter  += 1
        chunk_indx_val += 1

        # advance to the first time step
        t_val += cp.t_step

        while t_val <= (cp.t_max+1e-8):
            # Append to appropriate lists
            t.append(t_val)

            if chunk_counter == cp.t_step_chunk_num:
                # Append to appropriate lists
                t_chunks.append(t_val)
                chunk_indx.append(chunk_indx_val)

                # update the time step chunk iteration counter
                chunk_counter = 0

            # advance to the next time step
            t_val          += cp.t_step
            chunk_counter  += 1
            chunk_indx_val += 1
        
        # If the endpoint of the chunked applied deformation is not equal to the true endpoint of the applied deformation, then give the user the option to kill the simulation, or proceed on
        if chunk_indx[-1] != len(t)-1:
            terminal_statement = input('The endpoint of the chunked applied deformation is not equal to the endpoint of the actual applied deformation. Do you wish to kill the simulation here? If no, the simulation will proceed on. ')
            if terminal_statement.lower() == 'yes':
                sys.exit()
            else: pass
        
        # Begin time stepping
        # if statement with t=0 or else
        # chunk if statement

        # Initialize single chain
        # nu = 10, zeta_nu_char = 100, kappa_nu = 1000
        single_chain = RateIndependentScissionCompositeuFJC(nu=cp.nu,
                                                            zeta_nu_char=cp.zeta_nu_char,
                                                            kappa_nu=cp.kappa_nu)
        
        # Initial equilibrium chain stretch grid
        logspace_points = np.log10(np.linspace(0, (10**cp.loginess_skew)-1, cp.lmbda_c_eq_num_points)+1)
        lmbda_c_eq = (
            single_chain.lmbda_c_eq_crit
            - (single_chain.lmbda_c_eq_crit-single_chain.lmbda_c_eq_ref)
            / cp.loginess_skew * logspace_points
        )
        lmbda_c_eq = np.flip(lmbda_c_eq)
        lmbda_c_eq[0] = single_chain.lmbda_c_eq_ref

        # Calculate the equilibrium chain stretch time rate-of-change
        lmbda_c_eq_dot = cp.F_dot * lmbda_c_eq

        # Segment stretch
        lmbda_nu = np.asarray(
            [single_chain.lmbda_nu_func(lmbda_c_eq_val) for lmbda_c_eq_val
            in lmbda_c_eq]
        )

        # Rate-independent probability of segment scission
        p_nu_sci_hat = np.asarray(
            [single_chain.p_nu_sci_hat_func(lmbda_nu_val) for lmbda_nu_val
            in lmbda_nu]
        )

        # chain force, u_nu, s_cnu, psi_cnu, dissipated energy, etc.

        # Initial intact chain configuration partition function
        Z_intact = np.asarray(
            [single_chain.Z_intact_func(lmbda_c_eq_val) for lmbda_c_eq_val
            in lmbda_c_eq]
        )
        
        # Integrand of the zeroth moment of the initial intact chain
        # configuration equilibrium probability density distribution
        # without normalization
        I_0_intrgrnd = np.asarray(
            [single_chain.Z_intact_func(lmbda_c_eq_val) * lmbda_c_eq_val**2
            for lmbda_c_eq_val in lmbda_c_eq]
        )
        
        # Zeroth moment of the initial intact chain configuration
        # equilibrium probability density distribution without
        # normalization
        I_0 = np.trapz(I_0_intrgrnd, lmbda_c_eq)

        # Total configuration equilibrium partition function with
        # normalization (4*pi*exp(nu*zeta_nu_char)*(nu*l_nu^eq)**3))
        Z_eq_tot = (1.+single_chain.nu*np.exp(-single_chain.epsilon_nu_diss_hat_crit)) * I_0
        
        # Nondimensional intact chain configuration equilibrium
        # probability distribution
        # (P_intact_eq multiplied by
        # (4*pi*exp(nu*zeta_nu_char)*(nu*l_nu^eq)**3)))
        # Note: When performing integrals with this parameter, you only
        # need to perform a r^2 dr integration because the
        # nondimensional constant effectively is the angular directional
        # integrals to the 3D integral
        P_intact_nondim = Z_intact / Z_eq_tot

        # Chain conformation degradation
        Upsilon_c_chn_cnfrmtn_intrgrnd = np.asarray(
            [P_intact_nondim[lmbda_c_eq_indx] * lmbda_c_eq[lmbda_c_eq_indx]**2
            for lmbda_c_eq_indx in range(cp.lmbda_c_eq_num_points)]
        )

        Upsilon_c_chn_cnfrmtn = np.trapz(Upsilon_c_chn_cnfrmtn_intrgrnd, lmbda_c_eq)

        # Chain conformation equilibrium chain stretch
        lmbda_c_eq_chn_cnfrmtn_intrgrnd = np.asarray(
            [P_intact_nondim[lmbda_c_eq_indx] * lmbda_c_eq[lmbda_c_eq_indx]**4
            for lmbda_c_eq_indx in range(cp.lmbda_c_eq_num_points)]
        )

        lmbda_c_eq_chn_cnfrmtn = np.sqrt(np.trapz(lmbda_c_eq_chn_cnfrmtn_intrgrnd, lmbda_c_eq))

        # Calculate average chain force, u_nu, s_cnu, psi_cnu, dissipated energy at this initial state

        # Calculate rates-of-change
        partial_P_intact_nondim__partial_lmbda_c_eq = np.gradient(P_intact_nondim, lmbda_c_eq, edge_order=2)

        partial_P_intact_nondim__partial_t___chn_cnfrmtn = np.asarray(
            [-1*partial_P_intact_nondim__partial_lmbda_c_eq[lmbda_c_eq_indx]*lmbda_c_eq_dot[lmbda_c_eq_indx]
            for lmbda_c_eq_indx in range(cp.lmbda_c_eq_num_points)]
        )

        partial_P_rupture_nondim__partial_t = np.asarray(
            [single_chain.nu*p_nu_sci_hat[lmbda_c_eq_indx]*P_intact_nondim[lmbda_c_eq_indx]
            for lmbda_c_eq_indx in range(cp.lmbda_c_eq_num_points)]
        )

        partial_P_intact_nondim__partial_t = partial_P_intact_nondim__partial_t___chn_cnfrmtn - partial_P_rupture_nondim__partial_t

        # store average values in vectors



        self.single_chain = single_chain
        self.lmbda_c_eq = lmbda_c_eq
        self.P_intact_nondim = P_intact_nondim
        self.Upsilon_c_chn_cnfrmtn = Upsilon_c_chn_cnfrmtn
        self.lmbda_c_eq_chn_cnfrmtn = lmbda_c_eq_chn_cnfrmtn

    def finalization(self):
        """Define finalization analysis"""
        ppp = self.parameters.post_processing

        # plot results
        latex_formatting_figure(ppp)

        fig = plt.figure()
        plt.axvline(x=self.lmbda_c_eq_chn_cnfrmtn, linestyle=':',color='black', alpha=1, linewidth=1)
        plt.plot(self.lmbda_c_eq, self.P_intact_nondim,
                 linestyle='-', color='blue', alpha=1, linewidth=2.5)
        plt.xlim([self.lmbda_c_eq[0], self.lmbda_c_eq[-1]])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\lambda_c^{eq}$', 20, r'$\mathcal{P}^{intact} \times 4\pi e^{\nu\zeta_{\nu}^{char}}\left[\nu l_{\nu}^{eq}\right]^3$', 20,
            "P_intact_eq_nondim-vs-lmbda_c_eq")
        
        # fig = plt.figure()
        # plt.plot(self.lmbda_c_eq, self.partial_P_intact_eq_nondim__partial_lmbda_c_eq_edge_order_1,
        #          linestyle='-', color='black', alpha=1, linewidth=2.5)
        # plt.plot(self.lmbda_c_eq, self.partial_P_intact_eq_nondim__partial_lmbda_c_eq_edge_order_2,
        #          linestyle='-', color='red', alpha=1, linewidth=2.5)
        # plt.xlim([self.lmbda_c_eq[0], self.lmbda_c_eq[-1]])
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        # plt.grid(True, alpha=0.25)
        # save_current_figure(
        #     self.savedir, r'$\lambda_c^{eq}$', 20, r'$\frac{\partial\mathcal{P}_{eq}^{intact}}{\partial\lambda_c^{eq}} \times 4\pi e^{\nu\zeta_{\nu}^{char}}\left[\nu l_{\nu}^{eq}\right]^3$', 20,
        #     "partial_P_intact_eq_nondim__partial_lmbda_c_eq-vs-lmbda_c_eq")
        
        print(self.Upsilon_c_chn_cnfrmtn)

        percent_error_A_nu = np.abs(self.lmbda_c_eq_chn_cnfrmtn-self.single_chain.A_nu) / self.single_chain.A_nu * 100
        print(percent_error_A_nu)

if __name__ == '__main__':

    characterizer = ChainConformationInformedAveragingCharacterizer()
    characterizer.characterization()
    characterizer.finalization()