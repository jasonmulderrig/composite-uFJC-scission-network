"""The single-chain fracture toughness characterization module for
composite uFJCs that undergo scission
"""

# import external modules
from __future__ import division
from composite_ufjc_scission import (
    CompositeuFJCScissionCharacterizer,
    RateIndependentScissionCompositeuFJC,
    RateDependentScissionCompositeuFJC,
    latex_formatting_figure,
    save_current_figure,
    save_current_figure_no_labels,
    save_pickle_object,
    load_pickle_object
)
import numpy as np
from math import floor, log10
from scipy import constants
import matplotlib.pyplot as plt
from mpi4py import MPI


class FractureToughnessCharacterizer(CompositeuFJCScissionCharacterizer):
    """The characterization class assessing fracture toughness for
    composite uFJCs that undergo scission. It inherits all attributes
    and methods from the
    ``CompositeuFJCScissionCharacterizer`` class.
    """
    def __init__(self, paper_authors, chain, T):
        """Initializes the ``FractureToughnessCharacterizer`` class by
        initializing and inheriting all attributes and methods from the
        ``CompositeuFJCScissionCharacterizer`` class.
        """
        self.paper_authors = paper_authors
        self.chain = chain
        self.T = T

        self.comm = MPI.COMM_WORLD
        self.comm_rank = self.comm.Get_rank()
        self.comm_size = self.comm.Get_size()

        CompositeuFJCScissionCharacterizer.__init__(self)
    
    def set_user_parameters(self):
        """Set user-defined parameters"""
        p = self.parameters

        p.characterizer.chain_data_directory = (
            "./AFM_chain_tensile_test_curve_fit_results/"
        )

        p.characterizer.paper_authors2polymer_type_dict = {
            "al-maawali-et-al": "pdms",
            "hugel-et-al": "pva"
        }
        p.characterizer.paper_authors2polymer_type_label_dict = {
            "al-maawali-et-al": r'$\textrm{PDMS single chain data}$',
            "hugel-et-al": r'$\textrm{PVA single chain data}$'
        }
        p.characterizer.polymer_type_label2chain_backbone_bond_type_dict = {
            "pdms": "si-o",
            "pva": "c-c"
        }

        p.characterizer.LT_label = r'$u\textrm{FJC L\&T}$'
        p.characterizer.LT_inext_gaussian_label = (
            r'$\textrm{IGC L\&T}$'
        )
        p.characterizer.ufjc_label = r'$\textrm{c}u\textrm{FJC}$'

        p.characterizer.f_c_num_steps = 100001
        p.characterizer.r_nu_num_steps = 100001

        # nu = 1 -> nu = 10000, only 250 unique nu values exist here
        nu_list = np.unique(np.rint(np.logspace(0, 4, 351))) # 351 <-> 51
        nu_num = len(nu_list)
        nu_list_mpi_split = np.array_split(nu_list, self.comm_size)
        nu_num_list_mpi_split = [
            len(nu_list_mpi_split[proc_indx])
            for proc_indx in range(self.comm_size)
        ]
        p.characterizer.nu_list = nu_list
        p.characterizer.nu_num = nu_num
        p.characterizer.nu_list_mpi_split = nu_list_mpi_split
        p.characterizer.nu_num_list_mpi_split = nu_num_list_mpi_split

        tilde_xi_c_dot_list = np.logspace(-40, 10, 126)
        tilde_xi_c_dot_num = len(tilde_xi_c_dot_list)
        tilde_xi_c_dot_list_mpi_split = np.array_split(tilde_xi_c_dot_list, self.comm_size)
        tilde_xi_c_dot_num_list_mpi_split = [
            len(tilde_xi_c_dot_list_mpi_split[proc_indx])
            for proc_indx in range(self.comm_size)
        ]
        p.characterizer.tilde_xi_c_dot_list = tilde_xi_c_dot_list
        p.characterizer.tilde_xi_c_dot_num = tilde_xi_c_dot_num
        p.characterizer.tilde_xi_c_dot_list_mpi_split = tilde_xi_c_dot_list_mpi_split
        p.characterizer.tilde_xi_c_dot_num_list_mpi_split = tilde_xi_c_dot_num_list_mpi_split

        check_xi_c_dot_list = np.logspace(-40, 0, 101)
        check_xi_c_dot_num = len(check_xi_c_dot_list)
        check_xi_c_dot_list_mpi_split = np.array_split(check_xi_c_dot_list, self.comm_size)
        check_xi_c_dot_num_list_mpi_split = [
            len(check_xi_c_dot_list_mpi_split[proc_indx])
            for proc_indx in range(self.comm_size)
        ]
        p.characterizer.check_xi_c_dot_list = check_xi_c_dot_list
        p.characterizer.check_xi_c_dot_num = check_xi_c_dot_num
        p.characterizer.check_xi_c_dot_list_mpi_split = check_xi_c_dot_list_mpi_split
        p.characterizer.check_xi_c_dot_num_list_mpi_split = check_xi_c_dot_num_list_mpi_split

        # from DFT simulations on H_3C-CH_2-CH_3 (c-c) 
        # and H_3Si-O-CH_3 (si-o) by Beyer, J Chem. Phys., 2000
        chain_backbone_bond_type2beyer_2000_f_c_max_tau_b_dict = {
            "c-c": [6.05, 4.72, 3.81, 3.07, 2.45],
            "si-o": [4.77, 3.79, 3.14, 2.61, 2.17]
        } # nN
        beyer_2000_tau_b_list = [1e-12, 1e-6, 1e0, 1e5, 1e12] # sec
        beyer_2000_tau_b_num = len(beyer_2000_tau_b_list)
        beyer_2000_tau_b_exponent_list = [
            int(floor(log10(abs(beyer_2000_tau_b_list[tau_b_indx]))))
            for tau_b_indx in range(beyer_2000_tau_b_num)
        ]
        beyer_2000_tau_b_label_list = [
            r'$\textrm{WPCR},~\tau_{\nu}='+'10^{%g}~sec$' % (beyer_2000_tau_b_exponent_list[tau_b_indx])
            for tau_b_indx in range(beyer_2000_tau_b_num)
        ]
        beyer_2000_tau_b_color_list = ['black', 'black', 'black', 'black', 'black']
        beyer_2000_tau_b_linestyle_list = ['-', ':', '--', '-.', (0, (3, 1, 1, 1))]

        p.characterizer.chain_backbone_bond_type2beyer_2000_f_c_max_tau_b_dict = chain_backbone_bond_type2beyer_2000_f_c_max_tau_b_dict
        p.characterizer.beyer_2000_tau_b_list = beyer_2000_tau_b_list
        p.characterizer.beyer_2000_tau_b_num = beyer_2000_tau_b_num
        p.characterizer.beyer_2000_tau_b_exponent_list = beyer_2000_tau_b_exponent_list
        p.characterizer.beyer_2000_tau_b_label_list = beyer_2000_tau_b_label_list
        p.characterizer.beyer_2000_tau_b_color_list = beyer_2000_tau_b_color_list
        p.characterizer.beyer_2000_tau_b_linestyle_list = beyer_2000_tau_b_linestyle_list

        p.characterizer.AFM_exprmts_indx_list = [1, 2, 3]
        p.characterizer.typcl_AFM_exprmt_indx = 2

        f_c_dot_list = [1e1, 1e5, 1e9] # nN/sec
        f_c_dot_list = f_c_dot_list[::-1] # reverse order
        f_c_dot_num = len(f_c_dot_list)
        f_c_dot_exponent_list = [
            int(floor(log10(abs(f_c_dot_list[f_c_dot_indx]))))
            for f_c_dot_indx in range(f_c_dot_num)
        ]
        f_c_dot_label_list = [
            r'$\textrm{c}u\textrm{FJC},~\dot{f}_c='+'10^{%g}~nm/sec$' % (f_c_dot_exponent_list[f_c_dot_indx])
            for f_c_dot_indx in range(f_c_dot_num)
        ]
        f_c_dot_color_list = ['orange', 'purple', 'green']
        f_c_dot_color_list = f_c_dot_color_list[::-1]

        p.characterizer.f_c_dot_list          = f_c_dot_list
        p.characterizer.f_c_dot_num           = f_c_dot_num
        p.characterizer.f_c_dot_exponent_list = f_c_dot_exponent_list
        p.characterizer.f_c_dot_label_list    = f_c_dot_label_list
        p.characterizer.f_c_dot_color_list    = f_c_dot_color_list

        r_nu_dot_list = [1e1, 1e5, 1e9] # nm/sec
        r_nu_dot_list = r_nu_dot_list[::-1] # reverse order
        r_nu_dot_num = len(r_nu_dot_list)
        r_nu_dot_exponent_list = [
            int(floor(log10(abs(r_nu_dot_list[r_nu_dot_indx]))))
            for r_nu_dot_indx in range(r_nu_dot_num)
        ]
        r_nu_dot_label_list = [
            r'$\textrm{c}u\textrm{FJC},~\dot{r}_{\nu}='+'10^{%g}~nm/sec$' % (r_nu_dot_exponent_list[r_nu_dot_indx])
            for r_nu_dot_indx in range(r_nu_dot_num)
        ]
        r_nu_dot_color_list = ['orange', 'purple', 'green']
        r_nu_dot_color_list = r_nu_dot_color_list[::-1]

        p.characterizer.r_nu_dot_list          = r_nu_dot_list
        p.characterizer.r_nu_dot_num           = r_nu_dot_num
        p.characterizer.r_nu_dot_exponent_list = r_nu_dot_exponent_list
        p.characterizer.r_nu_dot_label_list    = r_nu_dot_label_list
        p.characterizer.r_nu_dot_color_list    = r_nu_dot_color_list

    def prefix(self):
        """Set characterization prefix"""
        return "fracture_toughness"
    
    def characterization(self):
        """Define characterization routine"""

        if self.comm_rank == 0:
            print(self.paper_authors+" "+self.chain+" characterization")

        k_B     = constants.value(u'Boltzmann constant') # J/K
        h       = constants.value(u'Planck constant') # J/Hz
        hbar    = h / (2*np.pi) # J*sec
        beta    = 1. / (k_B*self.T) # 1/J
        omega_0 = 1. / (beta*hbar) # J/(J*sec) = 1/sec

        beta = beta / (1e9*1e9) # 1/J = 1/(N*m) -> 1/(nN*m) -> 1/(nN*nm)

        cp = self.parameters.characterizer

        polymer_type = cp.paper_authors2polymer_type_dict[self.paper_authors]
        chain_backbone_bond_type = (
            cp.polymer_type_label2chain_backbone_bond_type_dict[polymer_type]
        )
        data_file_prefix = (
            self.paper_authors + '-' + polymer_type + '-'
            + chain_backbone_bond_type + '-' + self.chain
        )

        # unitless, unitless, unitless, nm, respectively
        nu = np.loadtxt(
            cp.chain_data_directory+data_file_prefix+'-composite-uFJC-curve-fit-intgr_nu'+'.txt')
        zeta_nu_char = np.loadtxt(
            cp.chain_data_directory+data_file_prefix+'-composite-uFJC-curve-fit-zeta_nu_char_intgr_nu'+'.txt')
        kappa_nu = np.loadtxt(
            cp.chain_data_directory+data_file_prefix+'-composite-uFJC-curve-fit-kappa_nu_intgr_nu'+'.txt')
        l_nu_eq = np.loadtxt(
            cp.chain_data_directory+data_file_prefix+'-composite-uFJC-curve-fit-l_nu_eq_intgr_nu'+'.txt')
        
        nu_list_mpi_scatter = self.comm.scatter(cp.nu_list_mpi_split, root=0)
        tilde_xi_c_dot_list_mpi_scatter = self.comm.scatter(cp.tilde_xi_c_dot_list_mpi_split, root=0)
        check_xi_c_dot_list_mpi_scatter = self.comm.scatter(cp.check_xi_c_dot_list_mpi_split, root=0)
        
        rate_independent_AFM_exprmt_single_chain = (
            RateIndependentScissionCompositeuFJC(nu=nu, zeta_nu_char=zeta_nu_char,
                                                 kappa_nu=kappa_nu)
        )
        rate_dependent_AFM_exprmt_single_chain = (
            RateDependentScissionCompositeuFJC(nu=nu, zeta_nu_char=zeta_nu_char,
                                               kappa_nu=kappa_nu, omega_0=omega_0)
        )
        
        beyer_2000_f_c_max_tau_b_list = (
            cp.chain_backbone_bond_type2beyer_2000_f_c_max_tau_b_dict[chain_backbone_bond_type]
        ) # nN
        
        # Rate-independent calculations
        if self.comm_rank == 0:
            print("Rate-independent calculations")
        
        A_nu__nu_chunk_list_mpi_scatter = []
        inext_gaussian_A_nu__nu_chunk_list_mpi_scatter = []
        inext_gaussian_A_nu_err__nu_chunk_list_mpi_scatter = []
        
        rate_independent_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter = []
        rate_independent_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter = []
        rate_independent_g_c_crit__nu_chunk_list_mpi_scatter = []
        rate_independent_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter = []
        rate_independent_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter = []
        rate_independent_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter = []
        rate_independent_overline_g_c_crit__nu_chunk_list_mpi_scatter = []
        rate_independent_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter = []
        
        rate_independent_LT_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter = []
        rate_independent_LT_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter = []
        rate_independent_LT_g_c_crit__nu_chunk_list_mpi_scatter = []
        rate_independent_LT_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter = []
        rate_independent_LT_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter = []
        rate_independent_LT_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter = []
        rate_independent_LT_overline_g_c_crit__nu_chunk_list_mpi_scatter = []
        rate_independent_LT_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter = []
        
        rate_independent_LT_inext_gaussian_g_c_crit__nu_chunk_list_mpi_scatter = []
        rate_independent_LT_inext_gaussian_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter = []
        rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_chunk_list_mpi_scatter = []
        rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter = []
        
        rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter = []
        rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter = []
        rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_chunk_list_mpi_scatter = []
        rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter = []
        rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter = []
        rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter = []
        rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_chunk_list_mpi_scatter = []
        rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter = []

        for nu_val in nu_list_mpi_scatter:
            rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_diss_hat_crit = []
            rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_diss_hat_crit = []
            rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit = []
            rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_squared = []
            rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_diss_hat_crit = []
            rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_diss_hat_crit = []
            rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit = []
            rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_squared = []

            rate_independent_single_chain = (
                RateIndependentScissionCompositeuFJC(nu=nu_val,
                                                     zeta_nu_char=zeta_nu_char,
                                                     kappa_nu=kappa_nu)
            )

            A_nu_val = rate_independent_single_chain.A_nu
            inext_gaussian_A_nu_val = 1 / np.sqrt(nu_val)
            inext_gaussian_A_nu_err_val = (
                np.abs((inext_gaussian_A_nu_val-A_nu_val)/A_nu_val) * 100
            )

            rate_independent_epsilon_cnu_diss_hat_crit_val = (
                rate_independent_single_chain.epsilon_cnu_diss_hat_crit
            )
            rate_independent_epsilon_c_diss_hat_crit_val = (
                nu_val * rate_independent_epsilon_cnu_diss_hat_crit_val
            )
            rate_independent_g_c_crit_val = (
                rate_independent_single_chain.g_c_crit
            )
            rate_independent_g_c_crit__nu_squared_val = (
                rate_independent_g_c_crit_val / nu_val**2
            )
            rate_independent_overline_epsilon_cnu_diss_hat_crit_val = (
                rate_independent_epsilon_cnu_diss_hat_crit_val / zeta_nu_char
            )
            rate_independent_overline_epsilon_c_diss_hat_crit_val = (
                nu_val * rate_independent_overline_epsilon_cnu_diss_hat_crit_val
            )
            rate_independent_overline_g_c_crit_val = (
                rate_independent_g_c_crit_val / zeta_nu_char
            )
            rate_independent_overline_g_c_crit__nu_squared_val = (
                rate_independent_overline_g_c_crit_val / nu_val**2
            )

            rate_independent_LT_epsilon_cnu_diss_hat_crit_val = zeta_nu_char
            rate_independent_LT_epsilon_c_diss_hat_crit_val = (
                nu_val * rate_independent_LT_epsilon_cnu_diss_hat_crit_val
            )
            rate_independent_LT_g_c_crit_val = (
                0.5 * A_nu_val * nu_val**2 * rate_independent_LT_epsilon_cnu_diss_hat_crit_val
            )
            rate_independent_LT_g_c_crit__nu_squared_val = (
                rate_independent_LT_g_c_crit_val / nu_val**2
            )
            rate_independent_LT_overline_epsilon_cnu_diss_hat_crit_val = 1.
            rate_independent_LT_overline_epsilon_c_diss_hat_crit_val = (
                nu_val * rate_independent_LT_overline_epsilon_cnu_diss_hat_crit_val
            )
            rate_independent_LT_overline_g_c_crit_val = (
                0.5 * A_nu_val * nu_val**2 * rate_independent_LT_overline_epsilon_cnu_diss_hat_crit_val
            )
            rate_independent_LT_overline_g_c_crit__nu_squared_val = (
                rate_independent_LT_overline_g_c_crit_val / nu_val**2
            )

            rate_independent_LT_inext_gaussian_g_c_crit_val = (
                0.5 * inext_gaussian_A_nu_val * nu_val**2 * rate_independent_LT_epsilon_cnu_diss_hat_crit_val
            )
            rate_independent_LT_inext_gaussian_g_c_crit__nu_squared_val = (
                rate_independent_LT_inext_gaussian_g_c_crit_val / nu_val**2
            )
            rate_independent_LT_inext_gaussian_overline_g_c_crit_val = (
                0.5 * inext_gaussian_A_nu_val * nu_val**2 * rate_independent_LT_overline_epsilon_cnu_diss_hat_crit_val
            )
            rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_squared_val = (
                rate_independent_LT_inext_gaussian_overline_g_c_crit_val / nu_val**2
            )

            for f_c_max_val in beyer_2000_f_c_max_tau_b_list:
                xi_c_max_val = f_c_max_val * beta * l_nu_eq
                lmbda_nu_xi_c_max_val = (
                    rate_independent_AFM_exprmt_single_chain.lmbda_nu_xi_c_hat_func(xi_c_max_val)
                )
                epsilon_cnu_diss_hat_crit_val = (
                    rate_independent_AFM_exprmt_single_chain.epsilon_cnu_sci_hat_func(lmbda_nu_xi_c_max_val)
                )
                epsilon_c_diss_hat_crit_val = (
                    nu_val * epsilon_cnu_diss_hat_crit_val
                )
                g_c_crit_val = 0.5 * A_nu_val * nu_val**2 * epsilon_cnu_diss_hat_crit_val
                g_c_crit__nu_squared_val = g_c_crit_val / nu_val**2
                overline_epsilon_cnu_diss_hat_crit_val = epsilon_cnu_diss_hat_crit_val / zeta_nu_char
                overline_epsilon_c_diss_hat_crit_val = (
                    nu_val * overline_epsilon_cnu_diss_hat_crit_val
                )
                overline_g_c_crit_val = 0.5 * A_nu_val * nu_val**2 * overline_epsilon_cnu_diss_hat_crit_val
                overline_g_c_crit__nu_squared_val = overline_g_c_crit_val / nu_val**2

                rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_diss_hat_crit.append(epsilon_cnu_diss_hat_crit_val)
                rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_diss_hat_crit.append(epsilon_c_diss_hat_crit_val)
                rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit.append(g_c_crit_val)
                rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_squared.append(g_c_crit__nu_squared_val)
                rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_diss_hat_crit.append(overline_epsilon_cnu_diss_hat_crit_val)
                rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_diss_hat_crit.append(overline_epsilon_c_diss_hat_crit_val)
                rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit.append(overline_g_c_crit_val)
                rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_squared.append(overline_g_c_crit__nu_squared_val)

            A_nu__nu_chunk_list_mpi_scatter.append(A_nu_val)
            inext_gaussian_A_nu__nu_chunk_list_mpi_scatter.append(inext_gaussian_A_nu_val)
            inext_gaussian_A_nu_err__nu_chunk_list_mpi_scatter.append(inext_gaussian_A_nu_err_val)

            rate_independent_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter.append(rate_independent_epsilon_cnu_diss_hat_crit_val)
            rate_independent_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter.append(rate_independent_epsilon_c_diss_hat_crit_val)
            rate_independent_g_c_crit__nu_chunk_list_mpi_scatter.append(rate_independent_g_c_crit_val)
            rate_independent_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter.append(rate_independent_g_c_crit__nu_squared_val)
            rate_independent_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter.append(rate_independent_overline_epsilon_cnu_diss_hat_crit_val)
            rate_independent_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter.append(rate_independent_overline_epsilon_c_diss_hat_crit_val)
            rate_independent_overline_g_c_crit__nu_chunk_list_mpi_scatter.append(rate_independent_overline_g_c_crit_val)
            rate_independent_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter.append(rate_independent_overline_g_c_crit__nu_squared_val)

            rate_independent_LT_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter.append(rate_independent_LT_epsilon_cnu_diss_hat_crit_val)
            rate_independent_LT_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter.append(rate_independent_LT_epsilon_c_diss_hat_crit_val)
            rate_independent_LT_g_c_crit__nu_chunk_list_mpi_scatter.append(rate_independent_LT_g_c_crit_val)
            rate_independent_LT_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter.append(rate_independent_LT_g_c_crit__nu_squared_val)
            rate_independent_LT_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter.append(rate_independent_LT_overline_epsilon_cnu_diss_hat_crit_val)
            rate_independent_LT_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter.append(rate_independent_LT_overline_epsilon_c_diss_hat_crit_val)
            rate_independent_LT_overline_g_c_crit__nu_chunk_list_mpi_scatter.append(rate_independent_LT_overline_g_c_crit_val)
            rate_independent_LT_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter.append(rate_independent_LT_overline_g_c_crit__nu_squared_val)

            rate_independent_LT_inext_gaussian_g_c_crit__nu_chunk_list_mpi_scatter.append(rate_independent_LT_inext_gaussian_g_c_crit_val)
            rate_independent_LT_inext_gaussian_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter.append(rate_independent_LT_inext_gaussian_g_c_crit__nu_squared_val)
            rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_chunk_list_mpi_scatter.append(rate_independent_LT_inext_gaussian_overline_g_c_crit_val)
            rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter.append(rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_squared_val)

            rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter.append(rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_diss_hat_crit)
            rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter.append(rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_diss_hat_crit)
            rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_chunk_list_mpi_scatter.append(rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit)
            rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter.append(rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_squared)
            rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter.append(rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_diss_hat_crit)
            rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter.append(rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_diss_hat_crit)
            rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_chunk_list_mpi_scatter.append(rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit)
            rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter.append(rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_squared)
        
        A_nu__nu_chunk_list_mpi_split = self.comm.gather(
            A_nu__nu_chunk_list_mpi_scatter, root=0
        )
        inext_gaussian_A_nu__nu_chunk_list_mpi_split = self.comm.gather(
            inext_gaussian_A_nu__nu_chunk_list_mpi_scatter, root=0
        )
        inext_gaussian_A_nu_err__nu_chunk_list_mpi_split = self.comm.gather(
            inext_gaussian_A_nu_err__nu_chunk_list_mpi_scatter, root=0
        )
        
        rate_independent_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_independent_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_independent_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_independent_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_independent_g_c_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_independent_g_c_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_independent_g_c_crit__nu_squared__nu_chunk_list_mpi_split = self.comm.gather(
            rate_independent_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter, root=0
        )
        rate_independent_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_independent_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_independent_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_independent_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_independent_overline_g_c_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_independent_overline_g_c_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_independent_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_split = self.comm.gather(
            rate_independent_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter, root=0
        )
        
        rate_independent_LT_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_independent_LT_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_independent_LT_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_independent_LT_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_independent_LT_g_c_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_independent_LT_g_c_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_independent_LT_g_c_crit__nu_squared__nu_chunk_list_mpi_split = self.comm.gather(
            rate_independent_LT_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter, root=0
        )
        rate_independent_LT_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_independent_LT_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_independent_LT_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_independent_LT_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_independent_LT_overline_g_c_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_independent_LT_overline_g_c_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_independent_LT_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_split = self.comm.gather(
            rate_independent_LT_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter, root=0
        )
        
        rate_independent_LT_inext_gaussian_g_c_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_independent_LT_inext_gaussian_g_c_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_independent_LT_inext_gaussian_g_c_crit__nu_squared__nu_chunk_list_mpi_split = self.comm.gather(
            rate_independent_LT_inext_gaussian_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter, root=0
        )
        rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_split = self.comm.gather(
            rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter, root=0
        )
        
        rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_squared__nu_chunk_list_mpi_split = self.comm.gather(
            rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter, root=0
        )
        rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_split = self.comm.gather(
            rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter, root=0
        )
        
        self.comm.Barrier()
        
        if self.comm_rank == 0:
            print("Post-processing rate-independent calculations")
            
            A_nu__nu_chunk_list = []
            inext_gaussian_A_nu__nu_chunk_list = []
            inext_gaussian_A_nu_err__nu_chunk_list = []
            
            rate_independent_epsilon_cnu_diss_hat_crit__nu_chunk_list = []
            rate_independent_epsilon_c_diss_hat_crit__nu_chunk_list = []
            rate_independent_g_c_crit__nu_chunk_list = []
            rate_independent_g_c_crit__nu_squared__nu_chunk_list = []
            rate_independent_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list = []
            rate_independent_overline_epsilon_c_diss_hat_crit__nu_chunk_list = []
            rate_independent_overline_g_c_crit__nu_chunk_list = []
            rate_independent_overline_g_c_crit__nu_squared__nu_chunk_list = []
            
            rate_independent_LT_epsilon_cnu_diss_hat_crit__nu_chunk_list = []
            rate_independent_LT_epsilon_c_diss_hat_crit__nu_chunk_list = []
            rate_independent_LT_g_c_crit__nu_chunk_list = []
            rate_independent_LT_g_c_crit__nu_squared__nu_chunk_list = []
            rate_independent_LT_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list = []
            rate_independent_LT_overline_epsilon_c_diss_hat_crit__nu_chunk_list = []
            rate_independent_LT_overline_g_c_crit__nu_chunk_list = []
            rate_independent_LT_overline_g_c_crit__nu_squared__nu_chunk_list = []
            
            rate_independent_LT_inext_gaussian_g_c_crit__nu_chunk_list = []
            rate_independent_LT_inext_gaussian_g_c_crit__nu_squared__nu_chunk_list = []
            rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_chunk_list = []
            rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_squared__nu_chunk_list = []
            
            rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_diss_hat_crit__nu_chunk_list = []
            rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_diss_hat_crit__nu_chunk_list = []
            rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_chunk_list = []
            rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_squared__nu_chunk_list = []
            rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list = []
            rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_diss_hat_crit__nu_chunk_list = []
            rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_chunk_list = []
            rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_squared__nu_chunk_list = []

            for proc_indx in range(self.comm_size):
                for nu_chunk_indx in range(cp.nu_num_list_mpi_split[proc_indx]):
                    A_nu__nu_chunk_val = A_nu__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    inext_gaussian_A_nu__nu_chunk_val = inext_gaussian_A_nu__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    inext_gaussian_A_nu_err__nu_chunk_val = inext_gaussian_A_nu_err__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]

                    rate_independent_epsilon_cnu_diss_hat_crit__nu_chunk_val = rate_independent_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_independent_epsilon_c_diss_hat_crit__nu_chunk_val = rate_independent_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_independent_g_c_crit__nu_chunk_val = rate_independent_g_c_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_independent_g_c_crit__nu_squared__nu_chunk_val = rate_independent_g_c_crit__nu_squared__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_independent_overline_epsilon_cnu_diss_hat_crit__nu_chunk_val = rate_independent_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_independent_overline_epsilon_c_diss_hat_crit__nu_chunk_val = rate_independent_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_independent_overline_g_c_crit__nu_chunk_val = rate_independent_overline_g_c_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_independent_overline_g_c_crit__nu_squared__nu_chunk_val = rate_independent_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]

                    rate_independent_LT_epsilon_cnu_diss_hat_crit__nu_chunk_val = rate_independent_LT_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_independent_LT_epsilon_c_diss_hat_crit__nu_chunk_val = rate_independent_LT_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_independent_LT_g_c_crit__nu_chunk_val = rate_independent_LT_g_c_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_independent_LT_g_c_crit__nu_squared__nu_chunk_val = rate_independent_LT_g_c_crit__nu_squared__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_independent_LT_overline_epsilon_cnu_diss_hat_crit__nu_chunk_val = rate_independent_LT_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_independent_LT_overline_epsilon_c_diss_hat_crit__nu_chunk_val = rate_independent_LT_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_independent_LT_overline_g_c_crit__nu_chunk_val = rate_independent_LT_overline_g_c_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_independent_LT_overline_g_c_crit__nu_squared__nu_chunk_val = rate_independent_LT_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]

                    rate_independent_LT_inext_gaussian_g_c_crit__nu_chunk_val = rate_independent_LT_inext_gaussian_g_c_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_independent_LT_inext_gaussian_g_c_crit__nu_squared__nu_chunk_val = rate_independent_LT_inext_gaussian_g_c_crit__nu_squared__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_chunk_val = rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_squared__nu_chunk_val = rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]

                    rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_diss_hat_crit__nu_chunk_val = rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_diss_hat_crit__nu_chunk_val = rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_chunk_val = rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_squared__nu_chunk_val = rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_squared__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_diss_hat_crit__nu_chunk_val = rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_diss_hat_crit__nu_chunk_val = rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_chunk_val = rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_squared__nu_chunk_val = rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]

                    A_nu__nu_chunk_list.append(A_nu__nu_chunk_val)
                    inext_gaussian_A_nu__nu_chunk_list.append(inext_gaussian_A_nu__nu_chunk_val)
                    inext_gaussian_A_nu_err__nu_chunk_list.append(inext_gaussian_A_nu_err__nu_chunk_val)

                    rate_independent_epsilon_cnu_diss_hat_crit__nu_chunk_list.append(rate_independent_epsilon_cnu_diss_hat_crit__nu_chunk_val)
                    rate_independent_epsilon_c_diss_hat_crit__nu_chunk_list.append(rate_independent_epsilon_c_diss_hat_crit__nu_chunk_val)
                    rate_independent_g_c_crit__nu_chunk_list.append(rate_independent_g_c_crit__nu_chunk_val)
                    rate_independent_g_c_crit__nu_squared__nu_chunk_list.append(rate_independent_g_c_crit__nu_squared__nu_chunk_val)
                    rate_independent_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list.append(rate_independent_overline_epsilon_cnu_diss_hat_crit__nu_chunk_val)
                    rate_independent_overline_epsilon_c_diss_hat_crit__nu_chunk_list.append(rate_independent_overline_epsilon_c_diss_hat_crit__nu_chunk_val)
                    rate_independent_overline_g_c_crit__nu_chunk_list.append(rate_independent_overline_g_c_crit__nu_chunk_val)
                    rate_independent_overline_g_c_crit__nu_squared__nu_chunk_list.append(rate_independent_overline_g_c_crit__nu_squared__nu_chunk_val)

                    rate_independent_LT_epsilon_cnu_diss_hat_crit__nu_chunk_list.append(rate_independent_LT_epsilon_cnu_diss_hat_crit__nu_chunk_val)
                    rate_independent_LT_epsilon_c_diss_hat_crit__nu_chunk_list.append(rate_independent_LT_epsilon_c_diss_hat_crit__nu_chunk_val)
                    rate_independent_LT_g_c_crit__nu_chunk_list.append(rate_independent_LT_g_c_crit__nu_chunk_val)
                    rate_independent_LT_g_c_crit__nu_squared__nu_chunk_list.append(rate_independent_LT_g_c_crit__nu_squared__nu_chunk_val)
                    rate_independent_LT_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list.append(rate_independent_LT_overline_epsilon_cnu_diss_hat_crit__nu_chunk_val)
                    rate_independent_LT_overline_epsilon_c_diss_hat_crit__nu_chunk_list.append(rate_independent_LT_overline_epsilon_c_diss_hat_crit__nu_chunk_val)
                    rate_independent_LT_overline_g_c_crit__nu_chunk_list.append(rate_independent_LT_overline_g_c_crit__nu_chunk_val)
                    rate_independent_LT_overline_g_c_crit__nu_squared__nu_chunk_list.append(rate_independent_LT_overline_g_c_crit__nu_squared__nu_chunk_val)

                    rate_independent_LT_inext_gaussian_g_c_crit__nu_chunk_list.append(rate_independent_LT_inext_gaussian_g_c_crit__nu_chunk_val)
                    rate_independent_LT_inext_gaussian_g_c_crit__nu_squared__nu_chunk_list.append(rate_independent_LT_inext_gaussian_g_c_crit__nu_squared__nu_chunk_val)
                    rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_chunk_list.append(rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_chunk_val)
                    rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_squared__nu_chunk_list.append(rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_squared__nu_chunk_val)

                    rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_diss_hat_crit__nu_chunk_list.append(rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_diss_hat_crit__nu_chunk_val)
                    rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_diss_hat_crit__nu_chunk_list.append(rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_diss_hat_crit__nu_chunk_val)
                    rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_chunk_list.append(rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_chunk_val)
                    rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_squared__nu_chunk_list.append(rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_squared__nu_chunk_val)
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list.append(rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_diss_hat_crit__nu_chunk_val)
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_diss_hat_crit__nu_chunk_list.append(rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_diss_hat_crit__nu_chunk_val)
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_chunk_list.append(rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_chunk_val)
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_squared__nu_chunk_list.append(rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_squared__nu_chunk_val)
            
            save_pickle_object(
                self.savedir, A_nu__nu_chunk_list,
                data_file_prefix+"-A_nu__nu_chunk_list")
            save_pickle_object(
                self.savedir, inext_gaussian_A_nu__nu_chunk_list,
                data_file_prefix+"-inext_gaussian_A_nu__nu_chunk_list")
            save_pickle_object(
                self.savedir, inext_gaussian_A_nu_err__nu_chunk_list,
                data_file_prefix+"-inext_gaussian_A_nu_err__nu_chunk_list")
            
            save_pickle_object(
                self.savedir, rate_independent_epsilon_cnu_diss_hat_crit__nu_chunk_list,
                data_file_prefix+"-rate_independent_epsilon_cnu_diss_hat_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir, rate_independent_epsilon_c_diss_hat_crit__nu_chunk_list,
                data_file_prefix+"-rate_independent_epsilon_c_diss_hat_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_independent_g_c_crit__nu_chunk_list,
                data_file_prefix+"-rate_independent_g_c_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir, rate_independent_g_c_crit__nu_squared__nu_chunk_list,
                data_file_prefix+"-rate_independent_g_c_crit__nu_squared__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_independent_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list,
                data_file_prefix+"-rate_independent_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_independent_overline_epsilon_c_diss_hat_crit__nu_chunk_list,
                data_file_prefix+"-rate_independent_overline_epsilon_c_diss_hat_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir, rate_independent_overline_g_c_crit__nu_chunk_list,
                data_file_prefix+"-rate_independent_overline_g_c_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_independent_overline_g_c_crit__nu_squared__nu_chunk_list,
                data_file_prefix+"-rate_independent_overline_g_c_crit__nu_squared__nu_chunk_list")
            
            save_pickle_object(
                self.savedir, rate_independent_LT_epsilon_cnu_diss_hat_crit__nu_chunk_list,
                data_file_prefix+"-rate_independent_LT_epsilon_cnu_diss_hat_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir, rate_independent_LT_epsilon_c_diss_hat_crit__nu_chunk_list,
                data_file_prefix+"-rate_independent_LT_epsilon_c_diss_hat_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir, rate_independent_LT_g_c_crit__nu_chunk_list,
                data_file_prefix+"-rate_independent_LT_g_c_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir, rate_independent_LT_g_c_crit__nu_squared__nu_chunk_list,
                data_file_prefix+"-rate_independent_LT_g_c_crit__nu_squared__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_independent_LT_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list,
                data_file_prefix+"-rate_independent_LT_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_independent_LT_overline_epsilon_c_diss_hat_crit__nu_chunk_list,
                data_file_prefix+"-rate_independent_LT_overline_epsilon_c_diss_hat_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir, rate_independent_LT_overline_g_c_crit__nu_chunk_list,
                data_file_prefix+"-rate_independent_LT_overline_g_c_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_independent_LT_overline_g_c_crit__nu_squared__nu_chunk_list,
                data_file_prefix+"-rate_independent_LT_overline_g_c_crit__nu_squared__nu_chunk_list")
            
            save_pickle_object(
                self.savedir, rate_independent_LT_inext_gaussian_g_c_crit__nu_chunk_list,
                data_file_prefix+"-rate_independent_LT_inext_gaussian_g_c_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_independent_LT_inext_gaussian_g_c_crit__nu_squared__nu_chunk_list,
                data_file_prefix+"-rate_independent_LT_inext_gaussian_g_c_crit__nu_squared__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_chunk_list,
                data_file_prefix+"-rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_squared__nu_chunk_list,
                data_file_prefix+"-rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_squared__nu_chunk_list")
            
            save_pickle_object(
                self.savedir, rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_diss_hat_crit__nu_chunk_list,
                data_file_prefix+"-rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_diss_hat_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir, rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_diss_hat_crit__nu_chunk_list,
                data_file_prefix+"-rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_diss_hat_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_chunk_list,
                data_file_prefix+"-rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir, rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_squared__nu_chunk_list,
                data_file_prefix+"-rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_squared__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list,
                data_file_prefix+"-rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_diss_hat_crit__nu_chunk_list,
                data_file_prefix+"-rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_diss_hat_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir, rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_chunk_list,
                data_file_prefix+"-rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_squared__nu_chunk_list,
                data_file_prefix+"-rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_squared__nu_chunk_list")
        
        self.comm.Barrier()
        
        # Rate-dependent calculations

        if self.comm_rank == 0:
            print("Rate-dependent calculations")
        
        if self.comm_rank == 0:
            print("Rate-dependent tilde_xi_c_dot sweep with the AFM experiment chain")

        rate_dependent_AFM_exprmt_epsilon_cnu_diss_hat_crit__tilde_xi_c_dot_chunk_list_mpi_scatter = []
        rate_dependent_AFM_exprmt_epsilon_c_diss_hat_crit__tilde_xi_c_dot_chunk_list_mpi_scatter = []
        rate_dependent_AFM_exprmt_g_c_crit__tilde_xi_c_dot_chunk_list_mpi_scatter = []
        rate_dependent_AFM_exprmt_g_c_crit__nu_squared__tilde_xi_c_dot_chunk_list_mpi_scatter = []
        rate_dependent_AFM_exprmt_overline_epsilon_cnu_diss_hat_crit__tilde_xi_c_dot_chunk_list_mpi_scatter = []
        rate_dependent_AFM_exprmt_overline_epsilon_c_diss_hat_crit__tilde_xi_c_dot_chunk_list_mpi_scatter = []
        rate_dependent_AFM_exprmt_overline_g_c_crit__tilde_xi_c_dot_chunk_list_mpi_scatter = []
        rate_dependent_AFM_exprmt_overline_g_c_crit__nu_squared__tilde_xi_c_dot_chunk_list_mpi_scatter = []

        for tilde_xi_c_dot_val in tilde_xi_c_dot_list_mpi_scatter:
            f_c_dot_val = tilde_xi_c_dot_val * omega_0 / (beta*l_nu_eq) # nN/sec
            A_nu_val = rate_dependent_AFM_exprmt_single_chain.A_nu
            f_c_crit = (
                rate_dependent_AFM_exprmt_single_chain.xi_c_crit / (beta*l_nu_eq)
            ) # (nN*nm)/nm = nN
            f_c_steps = np.linspace(0, f_c_crit, cp.f_c_num_steps) # nN
            t_steps = f_c_steps / f_c_dot_val # nN/(nN/sec) = sec

            # initialization
            p_nu_sci_hat_cmltv_intgrl_val       = 0.
            p_nu_sci_hat_cmltv_intgrl_val_prior = 0.
            p_nu_sci_hat_val                    = 0.
            p_nu_sci_hat_val_prior              = 0.
            epsilon_cnu_diss_hat_val            = 0.
            epsilon_cnu_diss_hat_val_prior      = 0.

            # Calculate results through applied chain force values
            for f_c_indx in range(cp.f_c_num_steps):
                t_val = t_steps[f_c_indx]
                xi_c_val = f_c_steps[f_c_indx] * beta * l_nu_eq # nN*nm/(nN*nm)
                lmbda_nu_val = (
                    rate_dependent_AFM_exprmt_single_chain.lmbda_nu_xi_c_hat_func(xi_c_val)
                )
                p_nu_sci_hat_val = (
                    rate_dependent_AFM_exprmt_single_chain.p_nu_sci_hat_func(lmbda_nu_val)
                )
                epsilon_cnu_sci_hat_val = (
                    rate_dependent_AFM_exprmt_single_chain.epsilon_cnu_sci_hat_func(
                        lmbda_nu_val)
                )

                if f_c_indx == 0:
                    pass
                else:
                    p_nu_sci_hat_cmltv_intgrl_val = (
                        rate_dependent_AFM_exprmt_single_chain.p_nu_sci_hat_cmltv_intgrl_func(
                            p_nu_sci_hat_val, t_val, p_nu_sci_hat_val_prior,
                            t_steps[f_c_indx-1],
                            p_nu_sci_hat_cmltv_intgrl_val_prior)
                    )
                    epsilon_cnu_diss_hat_val = (
                        rate_dependent_AFM_exprmt_single_chain.epsilon_cnu_diss_hat_func(
                            p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val,
                            epsilon_cnu_sci_hat_val, t_val, t_steps[f_c_indx-1],
                            epsilon_cnu_diss_hat_val_prior)
                    )
                
                p_nu_sci_hat_cmltv_intgrl_val_prior = (
                    p_nu_sci_hat_cmltv_intgrl_val
                )
                p_nu_sci_hat_val_prior = p_nu_sci_hat_val
                epsilon_cnu_diss_hat_val_prior = epsilon_cnu_diss_hat_val
            
            epsilon_cnu_diss_hat_crit_val = epsilon_cnu_diss_hat_val
            epsilon_c_diss_hat_crit_val = nu * epsilon_cnu_diss_hat_crit_val
            g_c_crit_val = (
                0.5 * A_nu_val * nu**2 * epsilon_cnu_diss_hat_crit_val
            )
            g_c_crit__nu_squared_val = (
                0.5 * A_nu_val * epsilon_cnu_diss_hat_crit_val
            )
            overline_epsilon_cnu_diss_hat_crit_val = (
                epsilon_cnu_diss_hat_crit_val / zeta_nu_char
            )
            overline_epsilon_c_diss_hat_crit_val = (
                nu * overline_epsilon_cnu_diss_hat_crit_val
            )
            overline_g_c_crit_val = (
                0.5 * A_nu_val * nu**2 * overline_epsilon_cnu_diss_hat_crit_val
            )
            overline_g_c_crit__nu_squared_val = (
                0.5 * A_nu_val * overline_epsilon_cnu_diss_hat_crit_val
            )

            rate_dependent_AFM_exprmt_epsilon_cnu_diss_hat_crit__tilde_xi_c_dot_chunk_list_mpi_scatter.append(epsilon_cnu_diss_hat_crit_val)
            rate_dependent_AFM_exprmt_epsilon_c_diss_hat_crit__tilde_xi_c_dot_chunk_list_mpi_scatter.append(epsilon_c_diss_hat_crit_val)
            rate_dependent_AFM_exprmt_g_c_crit__tilde_xi_c_dot_chunk_list_mpi_scatter.append(g_c_crit_val)
            rate_dependent_AFM_exprmt_g_c_crit__nu_squared__tilde_xi_c_dot_chunk_list_mpi_scatter.append(g_c_crit__nu_squared_val)
            rate_dependent_AFM_exprmt_overline_epsilon_cnu_diss_hat_crit__tilde_xi_c_dot_chunk_list_mpi_scatter.append(overline_epsilon_cnu_diss_hat_crit_val)
            rate_dependent_AFM_exprmt_overline_epsilon_c_diss_hat_crit__tilde_xi_c_dot_chunk_list_mpi_scatter.append(overline_epsilon_c_diss_hat_crit_val)
            rate_dependent_AFM_exprmt_overline_g_c_crit__tilde_xi_c_dot_chunk_list_mpi_scatter.append(overline_g_c_crit_val)
            rate_dependent_AFM_exprmt_overline_g_c_crit__nu_squared__tilde_xi_c_dot_chunk_list_mpi_scatter.append(overline_g_c_crit__nu_squared_val)
        
        rate_dependent_AFM_exprmt_epsilon_cnu_diss_hat_crit__tilde_xi_c_dot_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_AFM_exprmt_epsilon_cnu_diss_hat_crit__tilde_xi_c_dot_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_AFM_exprmt_epsilon_c_diss_hat_crit__tilde_xi_c_dot_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_AFM_exprmt_epsilon_c_diss_hat_crit__tilde_xi_c_dot_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_AFM_exprmt_g_c_crit__tilde_xi_c_dot_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_AFM_exprmt_g_c_crit__tilde_xi_c_dot_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_AFM_exprmt_g_c_crit__nu_squared__tilde_xi_c_dot_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_AFM_exprmt_g_c_crit__nu_squared__tilde_xi_c_dot_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_AFM_exprmt_overline_epsilon_cnu_diss_hat_crit__tilde_xi_c_dot_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_AFM_exprmt_overline_epsilon_cnu_diss_hat_crit__tilde_xi_c_dot_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_AFM_exprmt_overline_epsilon_c_diss_hat_crit__tilde_xi_c_dot_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_AFM_exprmt_overline_epsilon_c_diss_hat_crit__tilde_xi_c_dot_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_AFM_exprmt_overline_g_c_crit__tilde_xi_c_dot_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_AFM_exprmt_overline_g_c_crit__tilde_xi_c_dot_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_AFM_exprmt_overline_g_c_crit__nu_squared__tilde_xi_c_dot_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_AFM_exprmt_overline_g_c_crit__nu_squared__tilde_xi_c_dot_chunk_list_mpi_scatter, root=0
        )

        self.comm.Barrier()

        if self.comm_rank == 0:
            print("Rate-dependent check_xi_c_dot sweep with the AFM experiment chain")

        rate_dependent_AFM_exprmt_epsilon_cnu_diss_hat_crit__check_xi_c_dot_chunk_list_mpi_scatter = []
        rate_dependent_AFM_exprmt_epsilon_c_diss_hat_crit__check_xi_c_dot_chunk_list_mpi_scatter = []
        rate_dependent_AFM_exprmt_g_c_crit__check_xi_c_dot_chunk_list_mpi_scatter = []
        rate_dependent_AFM_exprmt_g_c_crit__nu_squared__check_xi_c_dot_chunk_list_mpi_scatter = []
        rate_dependent_AFM_exprmt_overline_epsilon_cnu_diss_hat_crit__check_xi_c_dot_chunk_list_mpi_scatter = []
        rate_dependent_AFM_exprmt_overline_epsilon_c_diss_hat_crit__check_xi_c_dot_chunk_list_mpi_scatter = []
        rate_dependent_AFM_exprmt_overline_g_c_crit__check_xi_c_dot_chunk_list_mpi_scatter = []
        rate_dependent_AFM_exprmt_overline_g_c_crit__nu_squared__check_xi_c_dot_chunk_list_mpi_scatter = []

        for check_xi_c_dot_val in check_xi_c_dot_list_mpi_scatter:
            f_c_dot_val = check_xi_c_dot_val * omega_0 * nu / (beta*l_nu_eq) # nN/sec
            A_nu_val = rate_dependent_AFM_exprmt_single_chain.A_nu
            f_c_crit = (
                rate_dependent_AFM_exprmt_single_chain.xi_c_crit / (beta*l_nu_eq)
            ) # (nN*nm)/nm = nN
            f_c_steps = np.linspace(0, f_c_crit, cp.f_c_num_steps) # nN
            t_steps = f_c_steps / f_c_dot_val # nN/(nN/sec) = sec

            # initialization
            p_nu_sci_hat_cmltv_intgrl_val       = 0.
            p_nu_sci_hat_cmltv_intgrl_val_prior = 0.
            p_nu_sci_hat_val                    = 0.
            p_nu_sci_hat_val_prior              = 0.
            epsilon_cnu_diss_hat_val            = 0.
            epsilon_cnu_diss_hat_val_prior      = 0.

            # Calculate results through applied chain force values
            for f_c_indx in range(cp.f_c_num_steps):
                t_val = t_steps[f_c_indx]
                xi_c_val = f_c_steps[f_c_indx] * beta * l_nu_eq # nN*nm/(nN*nm)
                lmbda_nu_val = (
                    rate_dependent_AFM_exprmt_single_chain.lmbda_nu_xi_c_hat_func(xi_c_val)
                )
                p_nu_sci_hat_val = (
                    rate_dependent_AFM_exprmt_single_chain.p_nu_sci_hat_func(lmbda_nu_val)
                )
                epsilon_cnu_sci_hat_val = (
                    rate_dependent_AFM_exprmt_single_chain.epsilon_cnu_sci_hat_func(
                        lmbda_nu_val)
                )

                if f_c_indx == 0:
                    pass
                else:
                    p_nu_sci_hat_cmltv_intgrl_val = (
                        rate_dependent_AFM_exprmt_single_chain.p_nu_sci_hat_cmltv_intgrl_func(
                            p_nu_sci_hat_val, t_val, p_nu_sci_hat_val_prior,
                            t_steps[f_c_indx-1],
                            p_nu_sci_hat_cmltv_intgrl_val_prior)
                    )
                    epsilon_cnu_diss_hat_val = (
                        rate_dependent_AFM_exprmt_single_chain.epsilon_cnu_diss_hat_func(
                            p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val,
                            epsilon_cnu_sci_hat_val, t_val, t_steps[f_c_indx-1],
                            epsilon_cnu_diss_hat_val_prior)
                    )
                
                p_nu_sci_hat_cmltv_intgrl_val_prior = (
                    p_nu_sci_hat_cmltv_intgrl_val
                )
                p_nu_sci_hat_val_prior = p_nu_sci_hat_val
                epsilon_cnu_diss_hat_val_prior = epsilon_cnu_diss_hat_val
            
            epsilon_cnu_diss_hat_crit_val = epsilon_cnu_diss_hat_val
            epsilon_c_diss_hat_crit_val = nu * epsilon_cnu_diss_hat_crit_val
            g_c_crit_val = (
                0.5 * A_nu_val * nu**2 * epsilon_cnu_diss_hat_crit_val
            )
            g_c_crit__nu_squared_val = (
                0.5 * A_nu_val * epsilon_cnu_diss_hat_crit_val
            )
            overline_epsilon_cnu_diss_hat_crit_val = (
                epsilon_cnu_diss_hat_crit_val / zeta_nu_char
            )
            overline_epsilon_c_diss_hat_crit_val = (
                nu * overline_epsilon_cnu_diss_hat_crit_val
            )
            overline_g_c_crit_val = (
                0.5 * A_nu_val * nu**2 * overline_epsilon_cnu_diss_hat_crit_val
            )
            overline_g_c_crit__nu_squared_val = (
                0.5 * A_nu_val * overline_epsilon_cnu_diss_hat_crit_val
            )

            rate_dependent_AFM_exprmt_epsilon_cnu_diss_hat_crit__check_xi_c_dot_chunk_list_mpi_scatter.append(epsilon_cnu_diss_hat_crit_val)
            rate_dependent_AFM_exprmt_epsilon_c_diss_hat_crit__check_xi_c_dot_chunk_list_mpi_scatter.append(epsilon_c_diss_hat_crit_val)
            rate_dependent_AFM_exprmt_g_c_crit__check_xi_c_dot_chunk_list_mpi_scatter.append(g_c_crit_val)
            rate_dependent_AFM_exprmt_g_c_crit__nu_squared__check_xi_c_dot_chunk_list_mpi_scatter.append(g_c_crit__nu_squared_val)
            rate_dependent_AFM_exprmt_overline_epsilon_cnu_diss_hat_crit__check_xi_c_dot_chunk_list_mpi_scatter.append(overline_epsilon_cnu_diss_hat_crit_val)
            rate_dependent_AFM_exprmt_overline_epsilon_c_diss_hat_crit__check_xi_c_dot_chunk_list_mpi_scatter.append(overline_epsilon_c_diss_hat_crit_val)
            rate_dependent_AFM_exprmt_overline_g_c_crit__check_xi_c_dot_chunk_list_mpi_scatter.append(overline_g_c_crit_val)
            rate_dependent_AFM_exprmt_overline_g_c_crit__nu_squared__check_xi_c_dot_chunk_list_mpi_scatter.append(overline_g_c_crit__nu_squared_val)
        
        rate_dependent_AFM_exprmt_epsilon_cnu_diss_hat_crit__check_xi_c_dot_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_AFM_exprmt_epsilon_cnu_diss_hat_crit__check_xi_c_dot_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_AFM_exprmt_epsilon_c_diss_hat_crit__check_xi_c_dot_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_AFM_exprmt_epsilon_c_diss_hat_crit__check_xi_c_dot_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_AFM_exprmt_g_c_crit__check_xi_c_dot_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_AFM_exprmt_g_c_crit__check_xi_c_dot_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_AFM_exprmt_g_c_crit__nu_squared__check_xi_c_dot_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_AFM_exprmt_g_c_crit__nu_squared__check_xi_c_dot_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_AFM_exprmt_overline_epsilon_cnu_diss_hat_crit__check_xi_c_dot_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_AFM_exprmt_overline_epsilon_cnu_diss_hat_crit__check_xi_c_dot_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_AFM_exprmt_overline_epsilon_c_diss_hat_crit__check_xi_c_dot_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_AFM_exprmt_overline_epsilon_c_diss_hat_crit__check_xi_c_dot_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_AFM_exprmt_overline_g_c_crit__check_xi_c_dot_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_AFM_exprmt_overline_g_c_crit__check_xi_c_dot_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_AFM_exprmt_overline_g_c_crit__nu_squared__check_xi_c_dot_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_AFM_exprmt_overline_g_c_crit__nu_squared__check_xi_c_dot_chunk_list_mpi_scatter, root=0
        )

        self.comm.Barrier()

        if self.comm_rank == 0:
            print("Rate-dependent tilde_xi_c_dot versus nu sweep")

        rate_dependent_tilde_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter = []
        rate_dependent_tilde_xi_c_dot_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter = []
        rate_dependent_tilde_xi_c_dot_g_c_crit__nu_chunk_list_mpi_scatter = []
        rate_dependent_tilde_xi_c_dot_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter = []
        rate_dependent_tilde_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter = []
        rate_dependent_tilde_xi_c_dot_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter = []
        rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_chunk_list_mpi_scatter = []
        rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter = []

        for nu_val in nu_list_mpi_scatter:
            rate_dependent_tilde_xi_c_dot_epsilon_cnu_diss_hat_crit = []
            rate_dependent_tilde_xi_c_dot_epsilon_c_diss_hat_crit = []
            rate_dependent_tilde_xi_c_dot_g_c_crit = []
            rate_dependent_tilde_xi_c_dot_g_c_crit__nu_squared = []
            rate_dependent_tilde_xi_c_dot_overline_epsilon_cnu_diss_hat_crit = []
            rate_dependent_tilde_xi_c_dot_overline_epsilon_c_diss_hat_crit = []
            rate_dependent_tilde_xi_c_dot_overline_g_c_crit = []
            rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_squared = []
            
            rate_dependent_single_chain = (
                RateDependentScissionCompositeuFJC(nu=nu_val,
                                                   zeta_nu_char=zeta_nu_char,
                                                   kappa_nu=kappa_nu,
                                                   omega_0=omega_0)
            )
            A_nu_val = rate_dependent_single_chain.A_nu
            f_c_crit_val = (
                rate_dependent_single_chain.xi_c_crit / (beta*l_nu_eq)
            ) # (nN*nm)/nm = nN
            f_c_steps = np.linspace(0, f_c_crit_val, cp.f_c_num_steps) # nN
            for tilde_xi_c_dot_val in cp.tilde_xi_c_dot_list:
                f_c_dot_val = (
                    tilde_xi_c_dot_val * omega_0 / (beta*l_nu_eq)
                ) # nN/sec
                t_steps = f_c_steps / f_c_dot_val # nN/(nN/sec) = sec

                # initialization
                p_nu_sci_hat_cmltv_intgrl_val       = 0.
                p_nu_sci_hat_cmltv_intgrl_val_prior = 0.
                p_nu_sci_hat_val                    = 0.
                p_nu_sci_hat_val_prior              = 0.
                epsilon_cnu_diss_hat_val            = 0.
                epsilon_cnu_diss_hat_val_prior      = 0.

                # Calculate results through applied chain force values
                for f_c_indx in range(cp.f_c_num_steps):
                    t_val = t_steps[f_c_indx]
                    xi_c_val = f_c_steps[f_c_indx] * beta * l_nu_eq # nN*nm/(nN*nm)
                    lmbda_nu_val = (
                        rate_dependent_single_chain.lmbda_nu_xi_c_hat_func(xi_c_val)
                    )
                    p_nu_sci_hat_val = (
                        rate_dependent_single_chain.p_nu_sci_hat_func(lmbda_nu_val)
                    )
                    epsilon_cnu_sci_hat_val = (
                        rate_dependent_single_chain.epsilon_cnu_sci_hat_func(
                            lmbda_nu_val)
                    )

                    if f_c_indx == 0:
                        pass
                    else:
                        p_nu_sci_hat_cmltv_intgrl_val = (
                            rate_dependent_single_chain.p_nu_sci_hat_cmltv_intgrl_func(
                                p_nu_sci_hat_val, t_val, p_nu_sci_hat_val_prior,
                                t_steps[f_c_indx-1],
                                p_nu_sci_hat_cmltv_intgrl_val_prior)
                        )
                        epsilon_cnu_diss_hat_val = (
                            rate_dependent_single_chain.epsilon_cnu_diss_hat_func(
                                p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val,
                                epsilon_cnu_sci_hat_val, t_val, t_steps[f_c_indx-1],
                                epsilon_cnu_diss_hat_val_prior)
                        )
                    
                    p_nu_sci_hat_cmltv_intgrl_val_prior = (
                        p_nu_sci_hat_cmltv_intgrl_val
                    )
                    p_nu_sci_hat_val_prior = p_nu_sci_hat_val
                    epsilon_cnu_diss_hat_val_prior = epsilon_cnu_diss_hat_val
                
                epsilon_cnu_diss_hat_crit_val = epsilon_cnu_diss_hat_val
                epsilon_c_diss_hat_crit_val = (
                    nu_val * epsilon_cnu_diss_hat_crit_val
                )
                g_c_crit_val = (
                    0.5 * A_nu_val * nu_val**2 * epsilon_cnu_diss_hat_crit_val
                )
                g_c_crit__nu_squared_val = (
                    0.5 * A_nu_val * epsilon_cnu_diss_hat_crit_val
                )
                overline_epsilon_cnu_diss_hat_crit_val = (
                    epsilon_cnu_diss_hat_crit_val / zeta_nu_char
                )
                overline_epsilon_c_diss_hat_crit_val = (
                    nu_val * overline_epsilon_cnu_diss_hat_crit_val
                )
                overline_g_c_crit_val = (
                    0.5 * A_nu_val * nu_val**2
                    * overline_epsilon_cnu_diss_hat_crit_val
                )
                overline_g_c_crit__nu_squared_val = (
                    0.5 * A_nu_val * overline_epsilon_cnu_diss_hat_crit_val
                )
                
                rate_dependent_tilde_xi_c_dot_epsilon_cnu_diss_hat_crit.append(
                    epsilon_cnu_diss_hat_crit_val
                )
                rate_dependent_tilde_xi_c_dot_epsilon_c_diss_hat_crit.append(
                    epsilon_c_diss_hat_crit_val
                )
                rate_dependent_tilde_xi_c_dot_g_c_crit.append(g_c_crit_val)
                rate_dependent_tilde_xi_c_dot_g_c_crit__nu_squared.append(
                    g_c_crit__nu_squared_val
                )
                rate_dependent_tilde_xi_c_dot_overline_epsilon_cnu_diss_hat_crit.append(
                    overline_epsilon_cnu_diss_hat_crit_val
                )
                rate_dependent_tilde_xi_c_dot_overline_epsilon_c_diss_hat_crit.append(
                    overline_epsilon_c_diss_hat_crit_val
                )
                rate_dependent_tilde_xi_c_dot_overline_g_c_crit.append(
                    overline_g_c_crit_val
                )
                rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_squared.append(
                    overline_g_c_crit__nu_squared_val
                )
            
            rate_dependent_tilde_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter.append(
                rate_dependent_tilde_xi_c_dot_epsilon_cnu_diss_hat_crit
            )
            rate_dependent_tilde_xi_c_dot_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter.append(
                rate_dependent_tilde_xi_c_dot_epsilon_c_diss_hat_crit
            )
            rate_dependent_tilde_xi_c_dot_g_c_crit__nu_chunk_list_mpi_scatter.append(
                rate_dependent_tilde_xi_c_dot_g_c_crit
            )
            rate_dependent_tilde_xi_c_dot_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter.append(
                rate_dependent_tilde_xi_c_dot_g_c_crit__nu_squared
            )
            rate_dependent_tilde_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter.append(
                rate_dependent_tilde_xi_c_dot_overline_epsilon_cnu_diss_hat_crit
            )
            rate_dependent_tilde_xi_c_dot_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter.append(
                rate_dependent_tilde_xi_c_dot_overline_epsilon_c_diss_hat_crit
            )
            rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_chunk_list_mpi_scatter.append(
                rate_dependent_tilde_xi_c_dot_overline_g_c_crit
            )
            rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter.append(
                rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_squared
            )
        
        rate_dependent_tilde_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_tilde_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_tilde_xi_c_dot_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_tilde_xi_c_dot_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_tilde_xi_c_dot_g_c_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_tilde_xi_c_dot_g_c_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_tilde_xi_c_dot_g_c_crit__nu_squared__nu_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_tilde_xi_c_dot_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_tilde_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_tilde_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_tilde_xi_c_dot_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_tilde_xi_c_dot_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter, root=0
        )
        
        self.comm.Barrier()

        if self.comm_rank == 0:
            print("Rate-dependent check_xi_c_dot versus nu sweep")

        rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter = []
        rate_dependent_check_xi_c_dot_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter = []
        rate_dependent_check_xi_c_dot_g_c_crit__nu_chunk_list_mpi_scatter = []
        rate_dependent_check_xi_c_dot_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter = []
        rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter = []
        rate_dependent_check_xi_c_dot_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter = []
        rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_chunk_list_mpi_scatter = []
        rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter = []

        for nu_val in nu_list_mpi_scatter:
            rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit = []
            rate_dependent_check_xi_c_dot_epsilon_c_diss_hat_crit = []
            rate_dependent_check_xi_c_dot_g_c_crit = []
            rate_dependent_check_xi_c_dot_g_c_crit__nu_squared = []
            rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit = []
            rate_dependent_check_xi_c_dot_overline_epsilon_c_diss_hat_crit = []
            rate_dependent_check_xi_c_dot_overline_g_c_crit = []
            rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_squared = []
            
            rate_dependent_single_chain = (
                RateDependentScissionCompositeuFJC(nu=nu_val,
                                                   zeta_nu_char=zeta_nu_char,
                                                   kappa_nu=kappa_nu,
                                                   omega_0=omega_0)
            )
            A_nu_val = rate_dependent_single_chain.A_nu
            f_c_crit_val = (
                rate_dependent_single_chain.xi_c_crit / (beta*l_nu_eq)
            ) # (nN*nm)/nm = nN
            f_c_steps = np.linspace(0, f_c_crit_val, cp.f_c_num_steps) # nN
            for check_xi_c_dot_val in cp.check_xi_c_dot_list:
                f_c_dot_val = (
                    check_xi_c_dot_val * omega_0 * nu_val / (beta*l_nu_eq)
                ) # nN/sec
                t_steps = f_c_steps / f_c_dot_val # nN/(nN/sec) = sec

                # initialization
                p_nu_sci_hat_cmltv_intgrl_val       = 0.
                p_nu_sci_hat_cmltv_intgrl_val_prior = 0.
                p_nu_sci_hat_val                    = 0.
                p_nu_sci_hat_val_prior              = 0.
                epsilon_cnu_diss_hat_val            = 0.
                epsilon_cnu_diss_hat_val_prior      = 0.

                # Calculate results through applied chain force values
                for f_c_indx in range(cp.f_c_num_steps):
                    t_val = t_steps[f_c_indx]
                    xi_c_val = f_c_steps[f_c_indx] * beta * l_nu_eq # nN*nm/(nN*nm)
                    lmbda_nu_val = (
                        rate_dependent_single_chain.lmbda_nu_xi_c_hat_func(xi_c_val)
                    )
                    p_nu_sci_hat_val = (
                        rate_dependent_single_chain.p_nu_sci_hat_func(lmbda_nu_val)
                    )
                    epsilon_cnu_sci_hat_val = (
                        rate_dependent_single_chain.epsilon_cnu_sci_hat_func(
                            lmbda_nu_val)
                    )

                    if f_c_indx == 0:
                        pass
                    else:
                        p_nu_sci_hat_cmltv_intgrl_val = (
                            rate_dependent_single_chain.p_nu_sci_hat_cmltv_intgrl_func(
                                p_nu_sci_hat_val, t_val, p_nu_sci_hat_val_prior,
                                t_steps[f_c_indx-1],
                                p_nu_sci_hat_cmltv_intgrl_val_prior)
                        )
                        epsilon_cnu_diss_hat_val = (
                            rate_dependent_single_chain.epsilon_cnu_diss_hat_func(
                                p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val,
                                epsilon_cnu_sci_hat_val, t_val, t_steps[f_c_indx-1],
                                epsilon_cnu_diss_hat_val_prior)
                        )
                    
                    p_nu_sci_hat_cmltv_intgrl_val_prior = (
                        p_nu_sci_hat_cmltv_intgrl_val
                    )
                    p_nu_sci_hat_val_prior = p_nu_sci_hat_val
                    epsilon_cnu_diss_hat_val_prior = epsilon_cnu_diss_hat_val
                
                epsilon_cnu_diss_hat_crit_val = epsilon_cnu_diss_hat_val
                epsilon_c_diss_hat_crit_val = (
                    nu_val * epsilon_cnu_diss_hat_crit_val
                )
                g_c_crit_val = (
                    0.5 * A_nu_val * nu_val**2 * epsilon_cnu_diss_hat_crit_val
                )
                g_c_crit__nu_squared_val = (
                    0.5 * A_nu_val * epsilon_cnu_diss_hat_crit_val
                )
                overline_epsilon_cnu_diss_hat_crit_val = (
                    epsilon_cnu_diss_hat_crit_val / zeta_nu_char
                )
                overline_epsilon_c_diss_hat_crit_val = (
                    nu_val * overline_epsilon_cnu_diss_hat_crit_val
                )
                overline_g_c_crit_val = (
                    0.5 * A_nu_val * nu_val**2
                    * overline_epsilon_cnu_diss_hat_crit_val
                )
                overline_g_c_crit__nu_squared_val = (
                    0.5 * A_nu_val * overline_epsilon_cnu_diss_hat_crit_val
                )
                
                rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit.append(
                    epsilon_cnu_diss_hat_crit_val
                )
                rate_dependent_check_xi_c_dot_epsilon_c_diss_hat_crit.append(
                    epsilon_c_diss_hat_crit_val
                )
                rate_dependent_check_xi_c_dot_g_c_crit.append(g_c_crit_val)
                rate_dependent_check_xi_c_dot_g_c_crit__nu_squared.append(
                    g_c_crit__nu_squared_val
                )
                rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit.append(
                    overline_epsilon_cnu_diss_hat_crit_val
                )
                rate_dependent_check_xi_c_dot_overline_epsilon_c_diss_hat_crit.append(
                    overline_epsilon_c_diss_hat_crit_val
                )
                rate_dependent_check_xi_c_dot_overline_g_c_crit.append(
                    overline_g_c_crit_val
                )
                rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_squared.append(
                    overline_g_c_crit__nu_squared_val
                )
            
            rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter.append(
                rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit
            )
            rate_dependent_check_xi_c_dot_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter.append(
                rate_dependent_check_xi_c_dot_epsilon_c_diss_hat_crit
            )
            rate_dependent_check_xi_c_dot_g_c_crit__nu_chunk_list_mpi_scatter.append(
                rate_dependent_check_xi_c_dot_g_c_crit
            )
            rate_dependent_check_xi_c_dot_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter.append(
                rate_dependent_check_xi_c_dot_g_c_crit__nu_squared
            )
            rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter.append(
                rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit
            )
            rate_dependent_check_xi_c_dot_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter.append(
                rate_dependent_check_xi_c_dot_overline_epsilon_c_diss_hat_crit
            )
            rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_chunk_list_mpi_scatter.append(
                rate_dependent_check_xi_c_dot_overline_g_c_crit
            )
            rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter.append(
                rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_squared
            )
        
        rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_check_xi_c_dot_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_check_xi_c_dot_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_check_xi_c_dot_g_c_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_check_xi_c_dot_g_c_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_check_xi_c_dot_g_c_crit__nu_squared__nu_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_check_xi_c_dot_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_check_xi_c_dot_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_check_xi_c_dot_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter, root=0
        )
        
        self.comm.Barrier()
        

        if self.comm_rank == 0:
            print("Rate-dependent force-controlled AFM experiments")
        
        rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter = []
        rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter = []
        rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit__nu_chunk_list_mpi_scatter = []
        rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter = []
        rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter = []
        rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter = []
        rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit__nu_chunk_list_mpi_scatter = []
        rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter = []

        for nu_val in nu_list_mpi_scatter:
            rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit = []
            rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit = []
            rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit = []
            rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit__nu_squared = []
            rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit = []
            rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit = []
            rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit = []
            rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared = []

            rate_dependent_single_chain = (
                RateDependentScissionCompositeuFJC(nu=nu_val,
                                                   zeta_nu_char=zeta_nu_char,
                                                   kappa_nu=kappa_nu,
                                                   omega_0=omega_0)
            )
            A_nu_val = rate_dependent_single_chain.A_nu
            f_c_crit_val = (
                rate_dependent_single_chain.xi_c_crit / (beta*l_nu_eq)
            ) # (nN*nm)/nm = nN
            f_c_steps = np.linspace(0, f_c_crit_val, cp.f_c_num_steps) # nN
            for f_c_dot_val in cp.f_c_dot_list:
                t_steps = f_c_steps / f_c_dot_val # nN/(nN/sec) = sec

                # initialization
                p_nu_sci_hat_cmltv_intgrl_val       = 0.
                p_nu_sci_hat_cmltv_intgrl_val_prior = 0.
                p_nu_sci_hat_val                    = 0.
                p_nu_sci_hat_val_prior              = 0.
                epsilon_cnu_diss_hat_val            = 0.
                epsilon_cnu_diss_hat_val_prior      = 0.

                # Calculate results through applied chain force values
                for f_c_indx in range(cp.f_c_num_steps):
                    t_val = t_steps[f_c_indx]
                    xi_c_val = f_c_steps[f_c_indx] * beta * l_nu_eq # nN*nm/(nN*nm)
                    lmbda_nu_val = (
                        rate_dependent_single_chain.lmbda_nu_xi_c_hat_func(xi_c_val)
                    )
                    p_nu_sci_hat_val = (
                        rate_dependent_single_chain.p_nu_sci_hat_func(lmbda_nu_val)
                    )
                    epsilon_cnu_sci_hat_val = (
                        rate_dependent_single_chain.epsilon_cnu_sci_hat_func(
                            lmbda_nu_val)
                    )

                    if f_c_indx == 0:
                        pass
                    else:
                        p_nu_sci_hat_cmltv_intgrl_val = (
                            rate_dependent_single_chain.p_nu_sci_hat_cmltv_intgrl_func(
                                p_nu_sci_hat_val, t_val, p_nu_sci_hat_val_prior,
                                t_steps[f_c_indx-1],
                                p_nu_sci_hat_cmltv_intgrl_val_prior)
                        )
                        epsilon_cnu_diss_hat_val = (
                            rate_dependent_single_chain.epsilon_cnu_diss_hat_func(
                                p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val,
                                epsilon_cnu_sci_hat_val, t_val, t_steps[f_c_indx-1],
                                epsilon_cnu_diss_hat_val_prior)
                        )
                    
                    p_nu_sci_hat_cmltv_intgrl_val_prior = (
                        p_nu_sci_hat_cmltv_intgrl_val
                    )
                    p_nu_sci_hat_val_prior = p_nu_sci_hat_val
                    epsilon_cnu_diss_hat_val_prior = epsilon_cnu_diss_hat_val
                
                epsilon_cnu_diss_hat_crit_val = epsilon_cnu_diss_hat_val
                epsilon_c_diss_hat_crit_val = (
                    nu_val * epsilon_cnu_diss_hat_crit_val
                )
                g_c_crit_val = (
                    0.5 * A_nu_val * nu_val**2 * epsilon_cnu_diss_hat_crit_val
                )
                g_c_crit__nu_squared_val = (
                    0.5 * A_nu_val * epsilon_cnu_diss_hat_crit_val
                )
                overline_epsilon_cnu_diss_hat_crit_val = (
                    epsilon_cnu_diss_hat_crit_val / zeta_nu_char
                )
                overline_epsilon_c_diss_hat_crit_val = (
                    nu_val * overline_epsilon_cnu_diss_hat_crit_val
                )
                overline_g_c_crit_val = (
                    0.5 * A_nu_val * nu_val**2
                    * overline_epsilon_cnu_diss_hat_crit_val
                )
                overline_g_c_crit__nu_squared_val = (
                    0.5 * A_nu_val * overline_epsilon_cnu_diss_hat_crit_val
                )
                
                rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit.append(
                    epsilon_cnu_diss_hat_crit_val
                )
                rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit.append(
                    epsilon_c_diss_hat_crit_val
                )
                rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit.append(g_c_crit_val)
                rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit__nu_squared.append(
                    g_c_crit__nu_squared_val
                )
                rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit.append(
                    overline_epsilon_cnu_diss_hat_crit_val
                )
                rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit.append(
                    overline_epsilon_c_diss_hat_crit_val
                )
                rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit.append(
                    overline_g_c_crit_val
                )
                rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared.append(
                    overline_g_c_crit__nu_squared_val
                )
            
            rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter.append(
                rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit
            )
            rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter.append(
                rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit
            )
            rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit__nu_chunk_list_mpi_scatter.append(
                rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit
            )
            rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter.append(
                rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit__nu_squared
            )
            rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter.append(
                rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit
            )
            rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter.append(
                rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit
            )
            rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit__nu_chunk_list_mpi_scatter.append(
                rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit
            )
            rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter.append(
                rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared
            )
        
        rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter, root=0
        )
        
        self.comm.Barrier()
        
        
        if self.comm_rank == 0:
            print("Rate-dependent displacement-controlled AFM experiments")
        
        rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter = []
        rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter = []
        rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit__nu_chunk_list_mpi_scatter = []
        rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter = []
        rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter = []
        rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter = []
        rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit__nu_chunk_list_mpi_scatter = []
        rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter = []

        for nu_val in nu_list_mpi_scatter:
            rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit = []
            rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit = []
            rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit = []
            rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit__nu_squared = []
            rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit = []
            rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit = []
            rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit = []
            rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared = []

            rate_dependent_single_chain = (
                RateDependentScissionCompositeuFJC(nu=nu_val,
                                                   zeta_nu_char=zeta_nu_char,
                                                   kappa_nu=kappa_nu,
                                                   omega_0=omega_0)
            )
            A_nu_val = rate_dependent_single_chain.A_nu
            lmbda_c_eq_crit_val = rate_dependent_single_chain.lmbda_c_eq_crit
            lmbda_c_eq_steps = np.linspace(0, lmbda_c_eq_crit_val, cp.r_nu_num_steps)
            for r_nu_dot_val in cp.r_nu_dot_list:
                lmbda_c_eq_dot_val = r_nu_dot_val / (nu_val*l_nu_eq) # (nm/sec)/nm = 1/sec
                t_steps = lmbda_c_eq_steps / lmbda_c_eq_dot_val # 1/(1/sec) = sec

                # initialization
                p_nu_sci_hat_cmltv_intgrl_val       = 0.
                p_nu_sci_hat_cmltv_intgrl_val_prior = 0.
                p_nu_sci_hat_val                    = 0.
                p_nu_sci_hat_val_prior              = 0.
                epsilon_cnu_diss_hat_val            = 0.
                epsilon_cnu_diss_hat_val_prior      = 0.

                # Calculate results through applied chain force values
                for r_nu_indx in range(cp.r_nu_num_steps):
                    t_val = t_steps[r_nu_indx]
                    lmbda_c_eq_val = lmbda_c_eq_steps[r_nu_indx]
                    lmbda_nu_val = (
                        rate_dependent_single_chain.lmbda_nu_func(lmbda_c_eq_val)
                    )
                    p_nu_sci_hat_val = (
                        rate_dependent_single_chain.p_nu_sci_hat_func(lmbda_nu_val)
                    )
                    epsilon_cnu_sci_hat_val = (
                        rate_dependent_single_chain.epsilon_cnu_sci_hat_func(
                            lmbda_nu_val)
                    )

                    if r_nu_indx == 0:
                        pass
                    else:
                        p_nu_sci_hat_cmltv_intgrl_val = (
                            rate_dependent_single_chain.p_nu_sci_hat_cmltv_intgrl_func(
                                p_nu_sci_hat_val, t_val, p_nu_sci_hat_val_prior,
                                t_steps[r_nu_indx-1],
                                p_nu_sci_hat_cmltv_intgrl_val_prior)
                        )
                        epsilon_cnu_diss_hat_val = (
                            rate_dependent_single_chain.epsilon_cnu_diss_hat_func(
                                p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val,
                                epsilon_cnu_sci_hat_val, t_val, t_steps[r_nu_indx-1],
                                epsilon_cnu_diss_hat_val_prior)
                        )
                    
                    p_nu_sci_hat_cmltv_intgrl_val_prior = (
                        p_nu_sci_hat_cmltv_intgrl_val
                    )
                    p_nu_sci_hat_val_prior = p_nu_sci_hat_val
                    epsilon_cnu_diss_hat_val_prior = epsilon_cnu_diss_hat_val
                
                epsilon_cnu_diss_hat_crit_val = epsilon_cnu_diss_hat_val
                epsilon_c_diss_hat_crit_val = (
                    nu_val * epsilon_cnu_diss_hat_crit_val
                )
                g_c_crit_val = (
                    0.5 * A_nu_val * nu_val**2 * epsilon_cnu_diss_hat_crit_val
                )
                g_c_crit__nu_squared_val = (
                    0.5 * A_nu_val * epsilon_cnu_diss_hat_crit_val
                )
                overline_epsilon_cnu_diss_hat_crit_val = (
                    epsilon_cnu_diss_hat_crit_val / zeta_nu_char
                )
                overline_epsilon_c_diss_hat_crit_val = (
                    nu_val * overline_epsilon_cnu_diss_hat_crit_val
                )
                overline_g_c_crit_val = (
                    0.5 * A_nu_val * nu_val**2
                    * overline_epsilon_cnu_diss_hat_crit_val
                )
                overline_g_c_crit__nu_squared_val = (
                    0.5 * A_nu_val * overline_epsilon_cnu_diss_hat_crit_val
                )
                
                rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit.append(
                    epsilon_cnu_diss_hat_crit_val
                )
                rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit.append(
                    epsilon_c_diss_hat_crit_val
                )
                rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit.append(g_c_crit_val)
                rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit__nu_squared.append(
                    g_c_crit__nu_squared_val
                )
                rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit.append(
                    overline_epsilon_cnu_diss_hat_crit_val
                )
                rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit.append(
                    overline_epsilon_c_diss_hat_crit_val
                )
                rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit.append(
                    overline_g_c_crit_val
                )
                rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared.append(
                    overline_g_c_crit__nu_squared_val
                )
            
            rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter.append(
                rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit
            )
            rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter.append(
                rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit
            )
            rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit__nu_chunk_list_mpi_scatter.append(
                rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit
            )
            rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter.append(
                rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit__nu_squared
            )
            rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter.append(
                rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit
            )
            rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter.append(
                rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit
            )
            rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit__nu_chunk_list_mpi_scatter.append(
                rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit
            )
            rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter.append(
                rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared
            )
        
        rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit__nu_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit__nu_chunk_list_mpi_scatter, root=0
        )
        rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_split = self.comm.gather(
            rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_scatter, root=0
        )
        
        self.comm.Barrier()
        
        if self.comm_rank == 0:
            print("Post-processing rate-dependent calculations")

            rate_dependent_AFM_exprmt_epsilon_cnu_diss_hat_crit__tilde_xi_c_dot_chunk_list = []
            rate_dependent_AFM_exprmt_epsilon_c_diss_hat_crit__tilde_xi_c_dot_chunk_list = []
            rate_dependent_AFM_exprmt_g_c_crit__tilde_xi_c_dot_chunk_list = []
            rate_dependent_AFM_exprmt_g_c_crit__nu_squared__tilde_xi_c_dot_chunk_list = []
            rate_dependent_AFM_exprmt_overline_epsilon_cnu_diss_hat_crit__tilde_xi_c_dot_chunk_list = []
            rate_dependent_AFM_exprmt_overline_epsilon_c_diss_hat_crit__tilde_xi_c_dot_chunk_list = []
            rate_dependent_AFM_exprmt_overline_g_c_crit__tilde_xi_c_dot_chunk_list = []
            rate_dependent_AFM_exprmt_overline_g_c_crit__nu_squared__tilde_xi_c_dot_chunk_list = []

            rate_dependent_AFM_exprmt_epsilon_cnu_diss_hat_crit__check_xi_c_dot_chunk_list = []
            rate_dependent_AFM_exprmt_epsilon_c_diss_hat_crit__check_xi_c_dot_chunk_list = []
            rate_dependent_AFM_exprmt_g_c_crit__check_xi_c_dot_chunk_list = []
            rate_dependent_AFM_exprmt_g_c_crit__nu_squared__check_xi_c_dot_chunk_list = []
            rate_dependent_AFM_exprmt_overline_epsilon_cnu_diss_hat_crit__check_xi_c_dot_chunk_list = []
            rate_dependent_AFM_exprmt_overline_epsilon_c_diss_hat_crit__check_xi_c_dot_chunk_list = []
            rate_dependent_AFM_exprmt_overline_g_c_crit__check_xi_c_dot_chunk_list = []
            rate_dependent_AFM_exprmt_overline_g_c_crit__nu_squared__check_xi_c_dot_chunk_list = []

            rate_dependent_tilde_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list = []
            rate_dependent_tilde_xi_c_dot_epsilon_c_diss_hat_crit__nu_chunk_list = []
            rate_dependent_tilde_xi_c_dot_g_c_crit__nu_chunk_list = []
            rate_dependent_tilde_xi_c_dot_g_c_crit__nu_squared__nu_chunk_list = []
            rate_dependent_tilde_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list = []
            rate_dependent_tilde_xi_c_dot_overline_epsilon_c_diss_hat_crit__nu_chunk_list = []
            rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_chunk_list = []
            rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_list = []

            rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list = []
            rate_dependent_check_xi_c_dot_epsilon_c_diss_hat_crit__nu_chunk_list = []
            rate_dependent_check_xi_c_dot_g_c_crit__nu_chunk_list = []
            rate_dependent_check_xi_c_dot_g_c_crit__nu_squared__nu_chunk_list = []
            rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list = []
            rate_dependent_check_xi_c_dot_overline_epsilon_c_diss_hat_crit__nu_chunk_list = []
            rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_chunk_list = []
            rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_list = []

            rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_list = []
            rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit__nu_chunk_list = []
            rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit__nu_chunk_list = []
            rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_list = []
            rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list = []
            rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit__nu_chunk_list = []
            rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit__nu_chunk_list = []
            rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_list = []

            rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_list = []
            rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit__nu_chunk_list = []
            rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit__nu_chunk_list = []
            rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_list = []
            rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list = []
            rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit__nu_chunk_list = []
            rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit__nu_chunk_list = []
            rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_list = []

            for proc_indx in range(self.comm_size):
                for tilde_xi_c_dot_chunk_indx in range(cp.tilde_xi_c_dot_num_list_mpi_split[proc_indx]):
                    rate_dependent_AFM_exprmt_epsilon_cnu_diss_hat_crit__tilde_xi_c_dot_chunk_val = rate_dependent_AFM_exprmt_epsilon_cnu_diss_hat_crit__tilde_xi_c_dot_chunk_list_mpi_split[proc_indx][tilde_xi_c_dot_chunk_indx]
                    rate_dependent_AFM_exprmt_epsilon_c_diss_hat_crit__tilde_xi_c_dot_chunk_val = rate_dependent_AFM_exprmt_epsilon_c_diss_hat_crit__tilde_xi_c_dot_chunk_list_mpi_split[proc_indx][tilde_xi_c_dot_chunk_indx]
                    rate_dependent_AFM_exprmt_g_c_crit__tilde_xi_c_dot_chunk_val = rate_dependent_AFM_exprmt_g_c_crit__tilde_xi_c_dot_chunk_list_mpi_split[proc_indx][tilde_xi_c_dot_chunk_indx]
                    rate_dependent_AFM_exprmt_g_c_crit__nu_squared__tilde_xi_c_dot_chunk_val = rate_dependent_AFM_exprmt_g_c_crit__nu_squared__tilde_xi_c_dot_chunk_list_mpi_split[proc_indx][tilde_xi_c_dot_chunk_indx]
                    rate_dependent_AFM_exprmt_overline_epsilon_cnu_diss_hat_crit__tilde_xi_c_dot_chunk_val = rate_dependent_AFM_exprmt_overline_epsilon_cnu_diss_hat_crit__tilde_xi_c_dot_chunk_list_mpi_split[proc_indx][tilde_xi_c_dot_chunk_indx]
                    rate_dependent_AFM_exprmt_overline_epsilon_c_diss_hat_crit__tilde_xi_c_dot_chunk_val = rate_dependent_AFM_exprmt_overline_epsilon_c_diss_hat_crit__tilde_xi_c_dot_chunk_list_mpi_split[proc_indx][tilde_xi_c_dot_chunk_indx]
                    rate_dependent_AFM_exprmt_overline_g_c_crit__tilde_xi_c_dot_chunk_val = rate_dependent_AFM_exprmt_overline_g_c_crit__tilde_xi_c_dot_chunk_list_mpi_split[proc_indx][tilde_xi_c_dot_chunk_indx]
                    rate_dependent_AFM_exprmt_overline_g_c_crit__nu_squared__tilde_xi_c_dot_chunk_val = rate_dependent_AFM_exprmt_overline_g_c_crit__nu_squared__tilde_xi_c_dot_chunk_list_mpi_split[proc_indx][tilde_xi_c_dot_chunk_indx]

                    rate_dependent_AFM_exprmt_epsilon_cnu_diss_hat_crit__tilde_xi_c_dot_chunk_list.append(rate_dependent_AFM_exprmt_epsilon_cnu_diss_hat_crit__tilde_xi_c_dot_chunk_val)
                    rate_dependent_AFM_exprmt_epsilon_c_diss_hat_crit__tilde_xi_c_dot_chunk_list.append(rate_dependent_AFM_exprmt_epsilon_c_diss_hat_crit__tilde_xi_c_dot_chunk_val)
                    rate_dependent_AFM_exprmt_g_c_crit__tilde_xi_c_dot_chunk_list.append(rate_dependent_AFM_exprmt_g_c_crit__tilde_xi_c_dot_chunk_val)
                    rate_dependent_AFM_exprmt_g_c_crit__nu_squared__tilde_xi_c_dot_chunk_list.append(rate_dependent_AFM_exprmt_g_c_crit__nu_squared__tilde_xi_c_dot_chunk_val)
                    rate_dependent_AFM_exprmt_overline_epsilon_cnu_diss_hat_crit__tilde_xi_c_dot_chunk_list.append(rate_dependent_AFM_exprmt_overline_epsilon_cnu_diss_hat_crit__tilde_xi_c_dot_chunk_val)
                    rate_dependent_AFM_exprmt_overline_epsilon_c_diss_hat_crit__tilde_xi_c_dot_chunk_list.append(rate_dependent_AFM_exprmt_overline_epsilon_c_diss_hat_crit__tilde_xi_c_dot_chunk_val)
                    rate_dependent_AFM_exprmt_overline_g_c_crit__tilde_xi_c_dot_chunk_list.append(rate_dependent_AFM_exprmt_overline_g_c_crit__tilde_xi_c_dot_chunk_val)
                    rate_dependent_AFM_exprmt_overline_g_c_crit__nu_squared__tilde_xi_c_dot_chunk_list.append(rate_dependent_AFM_exprmt_overline_g_c_crit__nu_squared__tilde_xi_c_dot_chunk_val)
                
                for check_xi_c_dot_chunk_indx in range(cp.check_xi_c_dot_num_list_mpi_split[proc_indx]):
                    rate_dependent_AFM_exprmt_epsilon_cnu_diss_hat_crit__check_xi_c_dot_chunk_val = rate_dependent_AFM_exprmt_epsilon_cnu_diss_hat_crit__check_xi_c_dot_chunk_list_mpi_split[proc_indx][check_xi_c_dot_chunk_indx]
                    rate_dependent_AFM_exprmt_epsilon_c_diss_hat_crit__check_xi_c_dot_chunk_val = rate_dependent_AFM_exprmt_epsilon_c_diss_hat_crit__check_xi_c_dot_chunk_list_mpi_split[proc_indx][check_xi_c_dot_chunk_indx]
                    rate_dependent_AFM_exprmt_g_c_crit__check_xi_c_dot_chunk_val = rate_dependent_AFM_exprmt_g_c_crit__check_xi_c_dot_chunk_list_mpi_split[proc_indx][check_xi_c_dot_chunk_indx]
                    rate_dependent_AFM_exprmt_g_c_crit__nu_squared__check_xi_c_dot_chunk_val = rate_dependent_AFM_exprmt_g_c_crit__nu_squared__check_xi_c_dot_chunk_list_mpi_split[proc_indx][check_xi_c_dot_chunk_indx]
                    rate_dependent_AFM_exprmt_overline_epsilon_cnu_diss_hat_crit__check_xi_c_dot_chunk_val = rate_dependent_AFM_exprmt_overline_epsilon_cnu_diss_hat_crit__check_xi_c_dot_chunk_list_mpi_split[proc_indx][check_xi_c_dot_chunk_indx]
                    rate_dependent_AFM_exprmt_overline_epsilon_c_diss_hat_crit__check_xi_c_dot_chunk_val = rate_dependent_AFM_exprmt_overline_epsilon_c_diss_hat_crit__check_xi_c_dot_chunk_list_mpi_split[proc_indx][check_xi_c_dot_chunk_indx]
                    rate_dependent_AFM_exprmt_overline_g_c_crit__check_xi_c_dot_chunk_val = rate_dependent_AFM_exprmt_overline_g_c_crit__check_xi_c_dot_chunk_list_mpi_split[proc_indx][check_xi_c_dot_chunk_indx]
                    rate_dependent_AFM_exprmt_overline_g_c_crit__nu_squared__check_xi_c_dot_chunk_val = rate_dependent_AFM_exprmt_overline_g_c_crit__nu_squared__check_xi_c_dot_chunk_list_mpi_split[proc_indx][check_xi_c_dot_chunk_indx]

                    rate_dependent_AFM_exprmt_epsilon_cnu_diss_hat_crit__check_xi_c_dot_chunk_list.append(rate_dependent_AFM_exprmt_epsilon_cnu_diss_hat_crit__check_xi_c_dot_chunk_val)
                    rate_dependent_AFM_exprmt_epsilon_c_diss_hat_crit__check_xi_c_dot_chunk_list.append(rate_dependent_AFM_exprmt_epsilon_c_diss_hat_crit__check_xi_c_dot_chunk_val)
                    rate_dependent_AFM_exprmt_g_c_crit__check_xi_c_dot_chunk_list.append(rate_dependent_AFM_exprmt_g_c_crit__check_xi_c_dot_chunk_val)
                    rate_dependent_AFM_exprmt_g_c_crit__nu_squared__check_xi_c_dot_chunk_list.append(rate_dependent_AFM_exprmt_g_c_crit__nu_squared__check_xi_c_dot_chunk_val)
                    rate_dependent_AFM_exprmt_overline_epsilon_cnu_diss_hat_crit__check_xi_c_dot_chunk_list.append(rate_dependent_AFM_exprmt_overline_epsilon_cnu_diss_hat_crit__check_xi_c_dot_chunk_val)
                    rate_dependent_AFM_exprmt_overline_epsilon_c_diss_hat_crit__check_xi_c_dot_chunk_list.append(rate_dependent_AFM_exprmt_overline_epsilon_c_diss_hat_crit__check_xi_c_dot_chunk_val)
                    rate_dependent_AFM_exprmt_overline_g_c_crit__check_xi_c_dot_chunk_list.append(rate_dependent_AFM_exprmt_overline_g_c_crit__check_xi_c_dot_chunk_val)
                    rate_dependent_AFM_exprmt_overline_g_c_crit__nu_squared__check_xi_c_dot_chunk_list.append(rate_dependent_AFM_exprmt_overline_g_c_crit__nu_squared__check_xi_c_dot_chunk_val)
                
                for nu_chunk_indx in range(cp.nu_num_list_mpi_split[proc_indx]):
                    rate_dependent_tilde_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_val = rate_dependent_tilde_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_dependent_tilde_xi_c_dot_epsilon_c_diss_hat_crit__nu_chunk_val = rate_dependent_tilde_xi_c_dot_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_dependent_tilde_xi_c_dot_g_c_crit__nu_chunk_val = rate_dependent_tilde_xi_c_dot_g_c_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_dependent_tilde_xi_c_dot_g_c_crit__nu_squared__nu_chunk_val = rate_dependent_tilde_xi_c_dot_g_c_crit__nu_squared__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_dependent_tilde_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_val = rate_dependent_tilde_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_dependent_tilde_xi_c_dot_overline_epsilon_c_diss_hat_crit__nu_chunk_val = rate_dependent_tilde_xi_c_dot_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_chunk_val = rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_val = rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]

                    rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_val = rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_dependent_check_xi_c_dot_epsilon_c_diss_hat_crit__nu_chunk_val = rate_dependent_check_xi_c_dot_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_dependent_check_xi_c_dot_g_c_crit__nu_chunk_val = rate_dependent_check_xi_c_dot_g_c_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_dependent_check_xi_c_dot_g_c_crit__nu_squared__nu_chunk_val = rate_dependent_check_xi_c_dot_g_c_crit__nu_squared__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_val = rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_dependent_check_xi_c_dot_overline_epsilon_c_diss_hat_crit__nu_chunk_val = rate_dependent_check_xi_c_dot_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_chunk_val = rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_val = rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]

                    rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_val = rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit__nu_chunk_val = rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit__nu_chunk_val = rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_val = rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_val = rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit__nu_chunk_val = rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit__nu_chunk_val = rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_val = rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]

                    rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_val = rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit__nu_chunk_val = rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit__nu_chunk_val = rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_val = rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_val = rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit__nu_chunk_val = rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit__nu_chunk_val = rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]
                    rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_val = rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_list_mpi_split[proc_indx][nu_chunk_indx]

                    rate_dependent_tilde_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list.append(rate_dependent_tilde_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_val)
                    rate_dependent_tilde_xi_c_dot_epsilon_c_diss_hat_crit__nu_chunk_list.append(rate_dependent_tilde_xi_c_dot_epsilon_c_diss_hat_crit__nu_chunk_val)
                    rate_dependent_tilde_xi_c_dot_g_c_crit__nu_chunk_list.append(rate_dependent_tilde_xi_c_dot_g_c_crit__nu_chunk_val)
                    rate_dependent_tilde_xi_c_dot_g_c_crit__nu_squared__nu_chunk_list.append(rate_dependent_tilde_xi_c_dot_g_c_crit__nu_squared__nu_chunk_val)
                    rate_dependent_tilde_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list.append(rate_dependent_tilde_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_val)
                    rate_dependent_tilde_xi_c_dot_overline_epsilon_c_diss_hat_crit__nu_chunk_list.append(rate_dependent_tilde_xi_c_dot_overline_epsilon_c_diss_hat_crit__nu_chunk_val)
                    rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_chunk_list.append(rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_chunk_val)
                    rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_list.append(rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_val)

                    rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list.append(rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_val)
                    rate_dependent_check_xi_c_dot_epsilon_c_diss_hat_crit__nu_chunk_list.append(rate_dependent_check_xi_c_dot_epsilon_c_diss_hat_crit__nu_chunk_val)
                    rate_dependent_check_xi_c_dot_g_c_crit__nu_chunk_list.append(rate_dependent_check_xi_c_dot_g_c_crit__nu_chunk_val)
                    rate_dependent_check_xi_c_dot_g_c_crit__nu_squared__nu_chunk_list.append(rate_dependent_check_xi_c_dot_g_c_crit__nu_squared__nu_chunk_val)
                    rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list.append(rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_val)
                    rate_dependent_check_xi_c_dot_overline_epsilon_c_diss_hat_crit__nu_chunk_list.append(rate_dependent_check_xi_c_dot_overline_epsilon_c_diss_hat_crit__nu_chunk_val)
                    rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_chunk_list.append(rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_chunk_val)
                    rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_list.append(rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_val)

                    rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_list.append(rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_val)
                    rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit__nu_chunk_list.append(rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit__nu_chunk_val)
                    rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit__nu_chunk_list.append(rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit__nu_chunk_val)
                    rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_list.append(rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_val)
                    rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list.append(rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_val)
                    rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit__nu_chunk_list.append(rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit__nu_chunk_val)
                    rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit__nu_chunk_list.append(rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit__nu_chunk_val)
                    rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_list.append(rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_val)

                    rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_list.append(rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_val)
                    rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit__nu_chunk_list.append(rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit__nu_chunk_val)
                    rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit__nu_chunk_list.append(rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit__nu_chunk_val)
                    rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_list.append(rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_val)
                    rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list.append(rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_val)
                    rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit__nu_chunk_list.append(rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit__nu_chunk_val)
                    rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit__nu_chunk_list.append(rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit__nu_chunk_val)
                    rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_list.append(rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_val)
            
            save_pickle_object(
                self.savedir,
                rate_dependent_AFM_exprmt_epsilon_cnu_diss_hat_crit__tilde_xi_c_dot_chunk_list,
                data_file_prefix+"-rate_dependent_AFM_exprmt_epsilon_cnu_diss_hat_crit__tilde_xi_c_dot_chunk_list"
            )
            save_pickle_object(
                self.savedir,
                rate_dependent_AFM_exprmt_epsilon_c_diss_hat_crit__tilde_xi_c_dot_chunk_list,
                data_file_prefix+"-rate_dependent_AFM_exprmt_epsilon_c_diss_hat_crit__tilde_xi_c_dot_chunk_list"
            )
            save_pickle_object(
                self.savedir,
                rate_dependent_AFM_exprmt_g_c_crit__tilde_xi_c_dot_chunk_list,
                data_file_prefix+"-rate_dependent_AFM_exprmt_g_c_crit__tilde_xi_c_dot_chunk_list"
            )
            save_pickle_object(
                self.savedir,
                rate_dependent_AFM_exprmt_g_c_crit__nu_squared__tilde_xi_c_dot_chunk_list,
                data_file_prefix+"-rate_dependent_AFM_exprmt_g_c_crit__nu_squared__tilde_xi_c_dot_chunk_list"
            )
            save_pickle_object(
                self.savedir,
                rate_dependent_AFM_exprmt_overline_epsilon_cnu_diss_hat_crit__tilde_xi_c_dot_chunk_list,
                data_file_prefix+"-rate_dependent_AFM_exprmt_overline_epsilon_cnu_diss_hat_crit__tilde_xi_c_dot_chunk_list"
            )
            save_pickle_object(
                self.savedir,
                rate_dependent_AFM_exprmt_overline_epsilon_c_diss_hat_crit__tilde_xi_c_dot_chunk_list,
                data_file_prefix+"-rate_dependent_AFM_exprmt_overline_epsilon_c_diss_hat_crit__tilde_xi_c_dot_chunk_list"
            )
            save_pickle_object(
                self.savedir,
                rate_dependent_AFM_exprmt_overline_g_c_crit__tilde_xi_c_dot_chunk_list,
                data_file_prefix+"-rate_dependent_AFM_exprmt_overline_g_c_crit__tilde_xi_c_dot_chunk_list"
            )
            save_pickle_object(
                self.savedir,
                rate_dependent_AFM_exprmt_overline_g_c_crit__nu_squared__tilde_xi_c_dot_chunk_list,
                data_file_prefix+"-rate_dependent_AFM_exprmt_overline_g_c_crit__nu_squared__tilde_xi_c_dot_chunk_list"
            )

            save_pickle_object(
                self.savedir,
                rate_dependent_AFM_exprmt_epsilon_cnu_diss_hat_crit__check_xi_c_dot_chunk_list,
                data_file_prefix+"-rate_dependent_AFM_exprmt_epsilon_cnu_diss_hat_crit__check_xi_c_dot_chunk_list"
            )
            save_pickle_object(
                self.savedir,
                rate_dependent_AFM_exprmt_epsilon_c_diss_hat_crit__check_xi_c_dot_chunk_list,
                data_file_prefix+"-rate_dependent_AFM_exprmt_epsilon_c_diss_hat_crit__check_xi_c_dot_chunk_list"
            )
            save_pickle_object(
                self.savedir,
                rate_dependent_AFM_exprmt_g_c_crit__check_xi_c_dot_chunk_list,
                data_file_prefix+"-rate_dependent_AFM_exprmt_g_c_crit__check_xi_c_dot_chunk_list"
            )
            save_pickle_object(
                self.savedir,
                rate_dependent_AFM_exprmt_g_c_crit__nu_squared__check_xi_c_dot_chunk_list,
                data_file_prefix+"-rate_dependent_AFM_exprmt_g_c_crit__nu_squared__check_xi_c_dot_chunk_list"
            )
            save_pickle_object(
                self.savedir,
                rate_dependent_AFM_exprmt_overline_epsilon_cnu_diss_hat_crit__check_xi_c_dot_chunk_list,
                data_file_prefix+"-rate_dependent_AFM_exprmt_overline_epsilon_cnu_diss_hat_crit__check_xi_c_dot_chunk_list"
            )
            save_pickle_object(
                self.savedir,
                rate_dependent_AFM_exprmt_overline_epsilon_c_diss_hat_crit__check_xi_c_dot_chunk_list,
                data_file_prefix+"-rate_dependent_AFM_exprmt_overline_epsilon_c_diss_hat_crit__check_xi_c_dot_chunk_list"
            )
            save_pickle_object(
                self.savedir,
                rate_dependent_AFM_exprmt_overline_g_c_crit__check_xi_c_dot_chunk_list,
                data_file_prefix+"-rate_dependent_AFM_exprmt_overline_g_c_crit__check_xi_c_dot_chunk_list"
            )
            save_pickle_object(
                self.savedir,
                rate_dependent_AFM_exprmt_overline_g_c_crit__nu_squared__check_xi_c_dot_chunk_list,
                data_file_prefix+"-rate_dependent_AFM_exprmt_overline_g_c_crit__nu_squared__check_xi_c_dot_chunk_list"
            )
            
            save_pickle_object(
                self.savedir,
                rate_dependent_tilde_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list,
                data_file_prefix+"-rate_dependent_tilde_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_tilde_xi_c_dot_epsilon_c_diss_hat_crit__nu_chunk_list,
                data_file_prefix+"-rate_dependent_tilde_xi_c_dot_epsilon_c_diss_hat_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir, rate_dependent_tilde_xi_c_dot_g_c_crit__nu_chunk_list,
                data_file_prefix+"-rate_dependent_tilde_xi_c_dot_g_c_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_tilde_xi_c_dot_g_c_crit__nu_squared__nu_chunk_list,
                data_file_prefix+"-rate_dependent_tilde_xi_c_dot_g_c_crit__nu_squared__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_tilde_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list,
                data_file_prefix+"-rate_dependent_tilde_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_tilde_xi_c_dot_overline_epsilon_c_diss_hat_crit__nu_chunk_list,
                data_file_prefix+"-rate_dependent_tilde_xi_c_dot_overline_epsilon_c_diss_hat_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_chunk_list,
                data_file_prefix+"-rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_list,
                data_file_prefix+"-rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_list")
            
            save_pickle_object(
                self.savedir,
                rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list,
                data_file_prefix+"-rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_check_xi_c_dot_epsilon_c_diss_hat_crit__nu_chunk_list,
                data_file_prefix+"-rate_dependent_check_xi_c_dot_epsilon_c_diss_hat_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir, rate_dependent_check_xi_c_dot_g_c_crit__nu_chunk_list,
                data_file_prefix+"-rate_dependent_check_xi_c_dot_g_c_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_check_xi_c_dot_g_c_crit__nu_squared__nu_chunk_list,
                data_file_prefix+"-rate_dependent_check_xi_c_dot_g_c_crit__nu_squared__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list,
                data_file_prefix+"-rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_check_xi_c_dot_overline_epsilon_c_diss_hat_crit__nu_chunk_list,
                data_file_prefix+"-rate_dependent_check_xi_c_dot_overline_epsilon_c_diss_hat_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_chunk_list,
                data_file_prefix+"-rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_list,
                data_file_prefix+"-rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_list")
            
            save_pickle_object(
                self.savedir,
                rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_list,
                data_file_prefix+"-rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit__nu_chunk_list,
                data_file_prefix+"-rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir, rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit__nu_chunk_list,
                data_file_prefix+"-rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_list,
                data_file_prefix+"-rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list,
                data_file_prefix+"-rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit__nu_chunk_list,
                data_file_prefix+"-rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit__nu_chunk_list,
                data_file_prefix+"-rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_list,
                data_file_prefix+"-rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_list")
            
            save_pickle_object(
                self.savedir,
                rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_list,
                data_file_prefix+"-rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit__nu_chunk_list,
                data_file_prefix+"-rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir, rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit__nu_chunk_list,
                data_file_prefix+"-rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_list,
                data_file_prefix+"-rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list,
                data_file_prefix+"-rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit__nu_chunk_list,
                data_file_prefix+"-rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit__nu_chunk_list,
                data_file_prefix+"-rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit__nu_chunk_list")
            save_pickle_object(
                self.savedir,
                rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_list,
                data_file_prefix+"-rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_list")


    def finalization(self):
        """Define finalization analysis"""

        if self.comm_rank == 0:
            print(self.paper_authors+" "+self.chain+" finalization")

        cp  = self.parameters.characterizer
        ppp = self.parameters.post_processing

        polymer_type = cp.paper_authors2polymer_type_dict[self.paper_authors]
        chain_backbone_bond_type = (
            cp.polymer_type_label2chain_backbone_bond_type_dict[polymer_type]
        )
        data_file_prefix = (
            self.paper_authors + '-' + polymer_type + '-'
            + chain_backbone_bond_type + '-' + self.chain
        )
        
        zeta_nu_char = np.loadtxt(
            cp.chain_data_directory+data_file_prefix+'-composite-uFJC-curve-fit-zeta_nu_char_intgr_nu'+'.txt')
        
        if self.comm_rank == 0:
            print("Plotting")

            A_nu__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-A_nu__nu_chunk_list")
            inext_gaussian_A_nu__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-inext_gaussian_A_nu__nu_chunk_list")
            inext_gaussian_A_nu_err__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-inext_gaussian_A_nu_err__nu_chunk_list")
            
            rate_independent_epsilon_cnu_diss_hat_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_epsilon_cnu_diss_hat_crit__nu_chunk_list")
            rate_independent_epsilon_c_diss_hat_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_epsilon_c_diss_hat_crit__nu_chunk_list")
            rate_independent_g_c_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_g_c_crit__nu_chunk_list")
            rate_independent_g_c_crit__nu_squared__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_g_c_crit__nu_squared__nu_chunk_list")
            rate_independent_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list")
            rate_independent_overline_epsilon_c_diss_hat_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_overline_epsilon_c_diss_hat_crit__nu_chunk_list")
            rate_independent_overline_g_c_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_overline_g_c_crit__nu_chunk_list")
            rate_independent_overline_g_c_crit__nu_squared__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_overline_g_c_crit__nu_squared__nu_chunk_list")
            
            rate_independent_LT_epsilon_cnu_diss_hat_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_LT_epsilon_cnu_diss_hat_crit__nu_chunk_list")
            rate_independent_LT_epsilon_c_diss_hat_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_LT_epsilon_c_diss_hat_crit__nu_chunk_list")
            rate_independent_LT_g_c_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_LT_g_c_crit__nu_chunk_list")
            rate_independent_LT_g_c_crit__nu_squared__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_LT_g_c_crit__nu_squared__nu_chunk_list")
            rate_independent_LT_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_LT_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list")
            rate_independent_LT_overline_epsilon_c_diss_hat_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_LT_overline_epsilon_c_diss_hat_crit__nu_chunk_list")
            rate_independent_LT_overline_g_c_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_LT_overline_g_c_crit__nu_chunk_list")
            rate_independent_LT_overline_g_c_crit__nu_squared__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_LT_overline_g_c_crit__nu_squared__nu_chunk_list")
            
            rate_independent_LT_inext_gaussian_g_c_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_LT_inext_gaussian_g_c_crit__nu_chunk_list")
            rate_independent_LT_inext_gaussian_g_c_crit__nu_squared__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_LT_inext_gaussian_g_c_crit__nu_squared__nu_chunk_list")
            rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_chunk_list")
            rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_squared__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_squared__nu_chunk_list")
            
            rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_diss_hat_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_diss_hat_crit__nu_chunk_list")
            rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_diss_hat_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_diss_hat_crit__nu_chunk_list")
            rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_chunk_list")
            rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_squared__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_squared__nu_chunk_list")
            rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list")
            rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_diss_hat_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_diss_hat_crit__nu_chunk_list")
            rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_chunk_list")
            rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_squared__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_squared__nu_chunk_list")


            rate_dependent_AFM_exprmt_epsilon_cnu_diss_hat_crit__tilde_xi_c_dot_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_AFM_exprmt_epsilon_cnu_diss_hat_crit__tilde_xi_c_dot_chunk_list"
            )
            rate_dependent_AFM_exprmt_epsilon_c_diss_hat_crit__tilde_xi_c_dot_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_AFM_exprmt_epsilon_c_diss_hat_crit__tilde_xi_c_dot_chunk_list"
            )
            rate_dependent_AFM_exprmt_g_c_crit__tilde_xi_c_dot_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_AFM_exprmt_g_c_crit__tilde_xi_c_dot_chunk_list"
            )
            rate_dependent_AFM_exprmt_g_c_crit__nu_squared__tilde_xi_c_dot_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_AFM_exprmt_g_c_crit__nu_squared__tilde_xi_c_dot_chunk_list"
            )
            rate_dependent_AFM_exprmt_overline_epsilon_cnu_diss_hat_crit__tilde_xi_c_dot_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_AFM_exprmt_overline_epsilon_cnu_diss_hat_crit__tilde_xi_c_dot_chunk_list"
            )
            rate_dependent_AFM_exprmt_overline_epsilon_c_diss_hat_crit__tilde_xi_c_dot_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_AFM_exprmt_overline_epsilon_c_diss_hat_crit__tilde_xi_c_dot_chunk_list"
            )
            rate_dependent_AFM_exprmt_overline_g_c_crit__tilde_xi_c_dot_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_AFM_exprmt_overline_g_c_crit__tilde_xi_c_dot_chunk_list"
            )
            rate_dependent_AFM_exprmt_overline_g_c_crit__nu_squared__tilde_xi_c_dot_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_AFM_exprmt_overline_g_c_crit__nu_squared__tilde_xi_c_dot_chunk_list"
            )

            rate_dependent_AFM_exprmt_epsilon_cnu_diss_hat_crit__check_xi_c_dot_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_AFM_exprmt_epsilon_cnu_diss_hat_crit__check_xi_c_dot_chunk_list"
            )
            rate_dependent_AFM_exprmt_epsilon_c_diss_hat_crit__check_xi_c_dot_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_AFM_exprmt_epsilon_c_diss_hat_crit__check_xi_c_dot_chunk_list"
            )
            rate_dependent_AFM_exprmt_g_c_crit__check_xi_c_dot_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_AFM_exprmt_g_c_crit__check_xi_c_dot_chunk_list"
            )
            rate_dependent_AFM_exprmt_g_c_crit__nu_squared__check_xi_c_dot_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_AFM_exprmt_g_c_crit__nu_squared__check_xi_c_dot_chunk_list"
            )
            rate_dependent_AFM_exprmt_overline_epsilon_cnu_diss_hat_crit__check_xi_c_dot_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_AFM_exprmt_overline_epsilon_cnu_diss_hat_crit__check_xi_c_dot_chunk_list"
            )
            rate_dependent_AFM_exprmt_overline_epsilon_c_diss_hat_crit__check_xi_c_dot_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_AFM_exprmt_overline_epsilon_c_diss_hat_crit__check_xi_c_dot_chunk_list"
            )
            rate_dependent_AFM_exprmt_overline_g_c_crit__check_xi_c_dot_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_AFM_exprmt_overline_g_c_crit__check_xi_c_dot_chunk_list"
            )
            rate_dependent_AFM_exprmt_overline_g_c_crit__nu_squared__check_xi_c_dot_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_AFM_exprmt_overline_g_c_crit__nu_squared__check_xi_c_dot_chunk_list"
            )
            
            rate_dependent_tilde_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_tilde_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list")
            rate_dependent_tilde_xi_c_dot_epsilon_c_diss_hat_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_tilde_xi_c_dot_epsilon_c_diss_hat_crit__nu_chunk_list")
            rate_dependent_tilde_xi_c_dot_g_c_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_tilde_xi_c_dot_g_c_crit__nu_chunk_list")
            rate_dependent_tilde_xi_c_dot_g_c_crit__nu_squared__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_tilde_xi_c_dot_g_c_crit__nu_squared__nu_chunk_list")
            rate_dependent_tilde_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_tilde_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list")
            rate_dependent_tilde_xi_c_dot_overline_epsilon_c_diss_hat_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_tilde_xi_c_dot_overline_epsilon_c_diss_hat_crit__nu_chunk_list")
            rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_chunk_list")
            rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_list")
            
            rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list")
            rate_dependent_check_xi_c_dot_epsilon_c_diss_hat_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_check_xi_c_dot_epsilon_c_diss_hat_crit__nu_chunk_list")
            rate_dependent_check_xi_c_dot_g_c_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_check_xi_c_dot_g_c_crit__nu_chunk_list")
            rate_dependent_check_xi_c_dot_g_c_crit__nu_squared__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_check_xi_c_dot_g_c_crit__nu_squared__nu_chunk_list")
            rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list")
            rate_dependent_check_xi_c_dot_overline_epsilon_c_diss_hat_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_check_xi_c_dot_overline_epsilon_c_diss_hat_crit__nu_chunk_list")
            rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_chunk_list")
            rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_list")
            
            rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_list")
            rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit__nu_chunk_list")
            rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit__nu_chunk_list")
            rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_list")
            rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list")
            rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit__nu_chunk_list")
            rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit__nu_chunk_list")
            rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_list")
            
            rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_list")
            rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit__nu_chunk_list")
            rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit__nu_chunk_list")
            rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_list")
            rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list")
            rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit__nu_chunk_list")
            rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit__nu_chunk_list")
            rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_list")


            # plot results
            latex_formatting_figure(ppp)

            fig, (ax1, ax2) = plt.subplots(
                2, 1, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
            
            ax1.semilogx(
                cp.nu_list, A_nu__nu_chunk_list, linestyle='-',
                color='blue', alpha=1, linewidth=2.5,
                label=r'$u\textrm{FJC}$')
            ax1.semilogx(
                cp.nu_list, inext_gaussian_A_nu__nu_chunk_list, linestyle='--',
                color='red', alpha=1, linewidth=2.5,
                label=r'$\textrm{inextensible Gaussian chain (IGC)}$')
            ax1.legend(loc='best', fontsize=14)
            ax1.tick_params(axis='y', labelsize=14)
            ax1.set_ylabel(r'$\mathcal{A}_{\nu}$', fontsize=20)
            ax1.grid(True, alpha=0.25)
            
            ax2.loglog(
                cp.nu_list, inext_gaussian_A_nu_err__nu_chunk_list,
                linestyle='-', color='blue', alpha=1, linewidth=2.5)
            ax2.tick_params(axis='y', labelsize=14)
            ax2.set_ylabel(r'$\%~\textrm{error}$', fontsize=20)
            ax2.grid(True, alpha=0.25)
            
            plt.xticks(fontsize=14)
            plt.xlabel(r'$\nu$', fontsize=20)
            save_current_figure_no_labels(
                self.savedir,
                data_file_prefix+"-A_nu-gen-ufjc-model-framework-and-inextensible-Gaussian-chain-comparison")

            fig = plt.figure()
            plt.semilogx(
                cp.nu_list, rate_independent_LT_epsilon_cnu_diss_hat_crit__nu_chunk_list,
                linestyle='-', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_label)
            rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_diss_hat_crit = [
                rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_diss_hat_crit__nu_chunk_list[nu_chunk_indx][cp.typcl_AFM_exprmt_indx]
                for nu_chunk_indx in range(cp.nu_num)
            ]
            plt.semilogx(
                cp.nu_list,
                rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_diss_hat_crit,
                linestyle=cp.beyer_2000_tau_b_linestyle_list[cp.typcl_AFM_exprmt_indx],
                color=cp.beyer_2000_tau_b_color_list[cp.typcl_AFM_exprmt_indx],
                alpha=1, linewidth=2.5,
                label=cp.beyer_2000_tau_b_label_list[cp.typcl_AFM_exprmt_indx])
            plt.semilogx(
                cp.nu_list, rate_independent_epsilon_cnu_diss_hat_crit__nu_chunk_list,
                linestyle='-', color='blue', alpha=1, linewidth=2.5,
                label=cp.ufjc_label)
            plt.legend(loc='best', fontsize=10)
            plt.ylim([-5, zeta_nu_char+5])
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 20,
                r'$\hat{\varepsilon}_{c\nu}^{diss}$', 20,
                data_file_prefix+"-rate-independent-nondimensional-dissipated-chain-scission-energy-per-segment-vs-nu")
            
            fig = plt.figure()
            plt.loglog(
                cp.nu_list, rate_independent_LT_epsilon_c_diss_hat_crit__nu_chunk_list,
                linestyle='-', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_label)
            rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_diss_hat_crit = [
                rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_diss_hat_crit__nu_chunk_list[nu_chunk_indx][cp.typcl_AFM_exprmt_indx]
                for nu_chunk_indx in range(cp.nu_num)
            ]
            plt.loglog(
                cp.nu_list,
                rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_diss_hat_crit,
                linestyle=cp.beyer_2000_tau_b_linestyle_list[cp.typcl_AFM_exprmt_indx],
                color=cp.beyer_2000_tau_b_color_list[cp.typcl_AFM_exprmt_indx],
                alpha=1, linewidth=2.5,
                label=cp.beyer_2000_tau_b_label_list[cp.typcl_AFM_exprmt_indx])
            plt.loglog(
                cp.nu_list, rate_independent_epsilon_c_diss_hat_crit__nu_chunk_list,
                linestyle='-', color='blue', alpha=1, linewidth=2.5,
                label=cp.ufjc_label)
            plt.legend(loc='best', fontsize=10)
            # plt.ylim([-5, zeta_nu_char+5])
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 20,
                r'$\hat{\varepsilon}_c^{diss}$', 20,
                data_file_prefix+"-rate-independent-nondimensional-dissipated-chain-scission-energy-vs-nu")
            
            fig = plt.figure()
            plt.semilogx(
                cp.nu_list, rate_independent_LT_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list,
                linestyle='-', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_label)
            rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_diss_hat_crit = [
                rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list[nu_chunk_indx][cp.typcl_AFM_exprmt_indx]
                for nu_chunk_indx in range(cp.nu_num)
            ]
            plt.semilogx(
                cp.nu_list,
                rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_diss_hat_crit,
                linestyle=cp.beyer_2000_tau_b_linestyle_list[cp.typcl_AFM_exprmt_indx],
                color=cp.beyer_2000_tau_b_color_list[cp.typcl_AFM_exprmt_indx],
                alpha=1, linewidth=2.5,
                label=cp.beyer_2000_tau_b_label_list[cp.typcl_AFM_exprmt_indx])
            plt.semilogx(
                cp.nu_list, rate_independent_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list,
                linestyle='-', color='blue', alpha=1, linewidth=2.5,
                label=cp.ufjc_label)
            plt.legend(loc='best', fontsize=10)
            plt.ylim([-0.05, 1.025])
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 20,
                r'$\overline{\hat{\varepsilon}_{c\nu}^{diss}}$', 20,
                data_file_prefix+"-rate-independent-nondimensional-scaled-dissipated-chain-scission-energy-per-segment-vs-nu")
            
            fig = plt.figure()
            plt.loglog(
                cp.nu_list, rate_independent_LT_overline_epsilon_c_diss_hat_crit__nu_chunk_list,
                linestyle='-', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_label)
            rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_diss_hat_crit = [
                rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_diss_hat_crit__nu_chunk_list[nu_chunk_indx][cp.typcl_AFM_exprmt_indx]
                for nu_chunk_indx in range(cp.nu_num)
            ]
            plt.loglog(
                cp.nu_list,
                rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_diss_hat_crit,
                linestyle=cp.beyer_2000_tau_b_linestyle_list[cp.typcl_AFM_exprmt_indx],
                color=cp.beyer_2000_tau_b_color_list[cp.typcl_AFM_exprmt_indx],
                alpha=1, linewidth=2.5,
                label=cp.beyer_2000_tau_b_label_list[cp.typcl_AFM_exprmt_indx])
            plt.loglog(
                cp.nu_list, rate_independent_overline_epsilon_c_diss_hat_crit__nu_chunk_list,
                linestyle='-', color='blue', alpha=1, linewidth=2.5,
                label=cp.ufjc_label)
            plt.legend(loc='best', fontsize=10)
            # plt.ylim([-0.05, 1.025])
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 20,
                r'$\overline{\hat{\varepsilon}_c^{diss}}$', 20,
                data_file_prefix+"-rate-independent-nondimensional-scaled-dissipated-chain-scission-energy-vs-nu")
            
            fig = plt.figure()
            plt.loglog(
                cp.nu_list, rate_independent_LT_g_c_crit__nu_chunk_list,
                linestyle='-', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_label)
            plt.loglog(
                cp.nu_list, rate_independent_LT_inext_gaussian_g_c_crit__nu_chunk_list,
                linestyle='--', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_inext_gaussian_label)
            rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit = [
                rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_chunk_list[nu_chunk_indx][cp.typcl_AFM_exprmt_indx]
                for nu_chunk_indx in range(cp.nu_num)
            ]
            plt.loglog(
                cp.nu_list,
                rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit,
                linestyle=cp.beyer_2000_tau_b_linestyle_list[cp.typcl_AFM_exprmt_indx],
                color=cp.beyer_2000_tau_b_color_list[cp.typcl_AFM_exprmt_indx],
                alpha=1, linewidth=2.5,
                label=cp.beyer_2000_tau_b_label_list[cp.typcl_AFM_exprmt_indx])
            plt.loglog(
                cp.nu_list, rate_independent_g_c_crit__nu_chunk_list,
                linestyle='-', color='blue', alpha=1, linewidth=2.5,
                label=cp.ufjc_label)
            plt.legend(loc='best', fontsize=10)
            # plt.ylim([-0.05, 1.025])
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 20,
                r'$\beta G_c/(\eta^{ref}l_{\nu}^{eq})$', 20,
                data_file_prefix+"-rate-independent-nondimensional-fracture-toughness-vs-nu")
            
            fig = plt.figure()
            plt.loglog(
                cp.nu_list, rate_independent_LT_g_c_crit__nu_squared__nu_chunk_list,
                linestyle='-', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_label)
            plt.loglog(
                cp.nu_list, rate_independent_LT_inext_gaussian_g_c_crit__nu_squared__nu_chunk_list,
                linestyle='--', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_inext_gaussian_label)
            rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_squared = [
                rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_squared__nu_chunk_list[nu_chunk_indx][cp.typcl_AFM_exprmt_indx]
                for nu_chunk_indx in range(cp.nu_num)
            ]
            plt.loglog(
                cp.nu_list,
                rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_squared,
                linestyle=cp.beyer_2000_tau_b_linestyle_list[cp.typcl_AFM_exprmt_indx],
                color=cp.beyer_2000_tau_b_color_list[cp.typcl_AFM_exprmt_indx],
                alpha=1, linewidth=2.5,
                label=cp.beyer_2000_tau_b_label_list[cp.typcl_AFM_exprmt_indx])
            plt.loglog(
                cp.nu_list, rate_independent_g_c_crit__nu_squared__nu_chunk_list,
                linestyle='-', color='blue', alpha=1, linewidth=2.5,
                label=cp.ufjc_label)
            plt.legend(loc='best', fontsize=10)
            # plt.ylim([-0.05, 1.025])
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 20,
                r'$\beta G_c/(\eta^{ref}l_{\nu}^{eq}\nu^2)$', 20,
                data_file_prefix+"-rate-independent-nondimensional-fracture-toughness-nu-squared-normalized-vs-nu")
            
            fig = plt.figure()
            plt.loglog(
                cp.nu_list, rate_independent_LT_overline_g_c_crit__nu_chunk_list,
                linestyle='-', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_label)
            plt.loglog(
                cp.nu_list, rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_chunk_list,
                linestyle='--', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_inext_gaussian_label)
            rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit = [
                rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_chunk_list[nu_chunk_indx][cp.typcl_AFM_exprmt_indx]
                for nu_chunk_indx in range(cp.nu_num)
            ]
            plt.loglog(
                cp.nu_list,
                rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit,
                linestyle=cp.beyer_2000_tau_b_linestyle_list[cp.typcl_AFM_exprmt_indx],
                color=cp.beyer_2000_tau_b_color_list[cp.typcl_AFM_exprmt_indx],
                alpha=1, linewidth=2.5,
                label=cp.beyer_2000_tau_b_label_list[cp.typcl_AFM_exprmt_indx])
            plt.loglog(
                cp.nu_list, rate_independent_overline_g_c_crit__nu_chunk_list,
                linestyle='-', color='blue', alpha=1, linewidth=2.5,
                label=cp.ufjc_label)
            plt.legend(loc='best', fontsize=10)
            # plt.ylim([-0.05, 1.025])
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 20,
                r'$\beta \overline{G_c}/(\eta^{ref}l_{\nu}^{eq})$', 20,
                data_file_prefix+"-rate-independent-nondimensional-scaled-fracture-toughness-vs-nu")
            
            fig = plt.figure()
            plt.loglog(
                cp.nu_list, rate_independent_LT_overline_g_c_crit__nu_squared__nu_chunk_list,
                linestyle='-', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_label)
            plt.loglog(
                cp.nu_list, rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_squared__nu_chunk_list,
                linestyle='--', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_inext_gaussian_label)
            rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_squared = [
                rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_squared__nu_chunk_list[nu_chunk_indx][cp.typcl_AFM_exprmt_indx]
                for nu_chunk_indx in range(cp.nu_num)
            ]
            plt.loglog(
                cp.nu_list,
                rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_squared,
                linestyle=cp.beyer_2000_tau_b_linestyle_list[cp.typcl_AFM_exprmt_indx],
                color=cp.beyer_2000_tau_b_color_list[cp.typcl_AFM_exprmt_indx],
                alpha=1, linewidth=2.5,
                label=cp.beyer_2000_tau_b_label_list[cp.typcl_AFM_exprmt_indx])
            plt.loglog(
                cp.nu_list, rate_independent_overline_g_c_crit__nu_squared__nu_chunk_list,
                linestyle='-', color='blue', alpha=1, linewidth=2.5,
                label=cp.ufjc_label)
            plt.legend(loc='best', fontsize=10)
            # plt.ylim([-0.05, 1.025])
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 20,
                r'$\beta \overline{G_c}/(\eta^{ref}l_{\nu}^{eq}\nu^2)$', 20,
                data_file_prefix+"-rate-independent-nondimensional-scaled-fracture-toughness-nu-squared-normalized-vs-nu")
            
            fig = plt.figure()
            plt.semilogx(
                cp.nu_list, rate_independent_LT_epsilon_cnu_diss_hat_crit__nu_chunk_list,
                linestyle='-', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_label)
            for AFM_expermts_indx in cp.AFM_exprmts_indx_list:
                rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_diss_hat_crit_list = [
                    rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_diss_hat_crit__nu_chunk_list[nu_chunk_indx][AFM_expermts_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_diss_hat_crit_list,
                    linestyle=cp.beyer_2000_tau_b_linestyle_list[AFM_expermts_indx],
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_label_list[AFM_expermts_indx])
            plt.semilogx(
                cp.nu_list, rate_independent_epsilon_cnu_diss_hat_crit__nu_chunk_list,
                linestyle='-', color='blue', alpha=1, linewidth=2.5,
                label=cp.ufjc_label)
            for f_c_dot_indx in range(cp.f_c_dot_num):
                rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit_list = [
                    rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_list[nu_chunk_indx][f_c_dot_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_dependent_frc_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit_list,
                    linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.f_c_dot_label_list[f_c_dot_indx])
            plt.legend(loc='best', fontsize=10)
            plt.ylim([-5, zeta_nu_char+5])
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 20,
                r'$\hat{\varepsilon}_{c\nu}^{diss}$', 20,
                data_file_prefix+"-rate-independent-and-rate-dependent-force-controlled-nondimensional-dissipated-chain-scission-energy-per-segment-vs-nu")
            
            fig = plt.figure()
            plt.semilogx(
                cp.nu_list, rate_independent_LT_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list,
                linestyle='-', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_label)
            for AFM_expermts_indx in cp.AFM_exprmts_indx_list:
                rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_diss_hat_crit_list = [
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list[nu_chunk_indx][AFM_expermts_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_diss_hat_crit_list,
                    linestyle=cp.beyer_2000_tau_b_linestyle_list[AFM_expermts_indx],
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_label_list[AFM_expermts_indx])
            plt.semilogx(
                cp.nu_list, rate_independent_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list,
                linestyle='-', color='blue', alpha=1, linewidth=2.5,
                label=cp.ufjc_label)
            for f_c_dot_indx in range(cp.f_c_dot_num):
                rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit_list = [
                    rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list[nu_chunk_indx][f_c_dot_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit_list,
                    linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.f_c_dot_label_list[f_c_dot_indx])
            plt.legend(loc='best', fontsize=10)
            plt.ylim([-0.05, 1.025])
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 20,
                r'$\overline{\hat{\varepsilon}_{c\nu}^{diss}}$', 20,
                data_file_prefix+"-rate-independent-and-rate-dependent-force-controlled-nondimensional-scaled-dissipated-chain-scission-energy-per-segment-vs-nu")
            
            fig = plt.figure()
            plt.loglog(
                cp.nu_list, rate_independent_LT_epsilon_c_diss_hat_crit__nu_chunk_list,
                linestyle='-', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_label)
            for AFM_expermts_indx in cp.AFM_exprmts_indx_list:
                rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_diss_hat_crit_list = [
                    rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_diss_hat_crit__nu_chunk_list[nu_chunk_indx][AFM_expermts_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_diss_hat_crit_list,
                    linestyle=cp.beyer_2000_tau_b_linestyle_list[AFM_expermts_indx],
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_label_list[AFM_expermts_indx])
            plt.loglog(
                cp.nu_list, rate_independent_epsilon_c_diss_hat_crit__nu_chunk_list,
                linestyle='-', color='blue', alpha=1, linewidth=2.5,
                label=cp.ufjc_label)
            for f_c_dot_indx in range(cp.f_c_dot_num):
                rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit_list = [
                    rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit__nu_chunk_list[nu_chunk_indx][f_c_dot_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.loglog(
                    cp.nu_list,
                    rate_dependent_frc_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit_list,
                    linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.f_c_dot_label_list[f_c_dot_indx])
            plt.legend(loc='best', fontsize=10)
            # plt.ylim([-5, zeta_nu_char+5])
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 20,
                r'$\hat{\varepsilon}_c^{diss}$', 20,
                data_file_prefix+"-rate-independent-and-rate-dependent-force-controlled-nondimensional-dissipated-chain-scission-energy-vs-nu")
            
            fig = plt.figure()
            plt.loglog(
                cp.nu_list, rate_independent_LT_overline_epsilon_c_diss_hat_crit__nu_chunk_list,
                linestyle='-', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_label)
            for AFM_expermts_indx in cp.AFM_exprmts_indx_list:
                rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_diss_hat_crit_list = [
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_diss_hat_crit__nu_chunk_list[nu_chunk_indx][AFM_expermts_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_diss_hat_crit_list,
                    linestyle=cp.beyer_2000_tau_b_linestyle_list[AFM_expermts_indx],
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_label_list[AFM_expermts_indx])
            plt.loglog(
                cp.nu_list, rate_independent_overline_epsilon_c_diss_hat_crit__nu_chunk_list,
                linestyle='-', color='blue', alpha=1, linewidth=2.5,
                label=cp.ufjc_label)
            for f_c_dot_indx in range(cp.f_c_dot_num):
                rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit_list = [
                    rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit__nu_chunk_list[nu_chunk_indx][f_c_dot_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.loglog(
                    cp.nu_list,
                    rate_dependent_frc_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit_list,
                    linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.f_c_dot_label_list[f_c_dot_indx])
            plt.legend(loc='best', fontsize=10)
            # plt.ylim([-0.05, 1.025])
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 20,
                r'$\overline{\hat{\varepsilon}_c^{diss}}$', 20,
                data_file_prefix+"-rate-independent-and-rate-dependent-force-controlled-nondimensional-scaled-dissipated-chain-scission-energy-vs-nu")
            
            fig = plt.figure()
            plt.loglog(
                cp.nu_list, rate_independent_LT_g_c_crit__nu_chunk_list,
                linestyle='-', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_label)
            plt.loglog(
                cp.nu_list, rate_independent_LT_inext_gaussian_g_c_crit__nu_chunk_list,
                linestyle='--', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_inext_gaussian_label)
            for AFM_expermts_indx in cp.AFM_exprmts_indx_list:
                rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit_list = [
                    rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_chunk_list[nu_chunk_indx][AFM_expermts_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit_list,
                    linestyle=cp.beyer_2000_tau_b_linestyle_list[AFM_expermts_indx],
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_label_list[AFM_expermts_indx])
            plt.loglog(
                cp.nu_list, rate_independent_g_c_crit__nu_chunk_list,
                linestyle='-', color='blue', alpha=1, linewidth=2.5,
                label=cp.ufjc_label)
            for f_c_dot_indx in range(cp.f_c_dot_num):
                rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit_list = [
                    rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit__nu_chunk_list[nu_chunk_indx][f_c_dot_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.loglog(
                    cp.nu_list,
                    rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit_list,
                    linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.f_c_dot_label_list[f_c_dot_indx])
            plt.legend(loc='best', fontsize=10)
            # plt.ylim([-0.05, 1.025])
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 20,
                r'$\beta G_c/(\eta^{ref}l_{\nu}^{eq})$', 20,
                data_file_prefix+"-rate-independent-and-rate-dependent-force-controlled-nondimensional-fracture-toughness-vs-nu")
            
            fig = plt.figure()
            plt.loglog(
                cp.nu_list, rate_independent_LT_g_c_crit__nu_squared__nu_chunk_list,
                linestyle='-', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_label)
            plt.loglog(
                cp.nu_list, rate_independent_LT_inext_gaussian_g_c_crit__nu_squared__nu_chunk_list,
                linestyle='--', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_inext_gaussian_label)
            for AFM_expermts_indx in cp.AFM_exprmts_indx_list:
                rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_squared_list = [
                    rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_squared__nu_chunk_list[nu_chunk_indx][AFM_expermts_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_squared_list,
                    linestyle=cp.beyer_2000_tau_b_linestyle_list[AFM_expermts_indx],
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_label_list[AFM_expermts_indx])
            plt.loglog(
                cp.nu_list, rate_independent_g_c_crit__nu_squared__nu_chunk_list,
                linestyle='-', color='blue', alpha=1, linewidth=2.5,
                label=cp.ufjc_label)
            for f_c_dot_indx in range(cp.f_c_dot_num):
                rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit__nu_squared_list = [
                    rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_list[nu_chunk_indx][f_c_dot_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.loglog(
                    cp.nu_list,
                    rate_dependent_frc_cntrld_AFM_exprmts_g_c_crit__nu_squared_list,
                    linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.f_c_dot_label_list[f_c_dot_indx])
            plt.legend(loc='best', fontsize=10)
            # plt.ylim([-0.05, 1.025])
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 20,
                r'$\beta G_c/(\eta^{ref}l_{\nu}^{eq}\nu^2)$', 20,
                data_file_prefix+"-rate-independent-and-rate-dependent-force-controlled-nondimensional-fracture-toughness-nu-squared-normalized-vs-nu")
            
            fig = plt.figure()
            plt.loglog(
                cp.nu_list, rate_independent_LT_overline_g_c_crit__nu_chunk_list,
                linestyle='-', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_label)
            plt.loglog(
                cp.nu_list, rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_chunk_list,
                linestyle='--', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_inext_gaussian_label)
            for AFM_expermts_indx in cp.AFM_exprmts_indx_list:
                rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit_list = [
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_chunk_list[nu_chunk_indx][AFM_expermts_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit_list,
                    linestyle=cp.beyer_2000_tau_b_linestyle_list[AFM_expermts_indx],
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_label_list[AFM_expermts_indx])
            plt.loglog(
                cp.nu_list, rate_independent_overline_g_c_crit__nu_chunk_list,
                linestyle='-', color='blue', alpha=1, linewidth=2.5,
                label=cp.ufjc_label)
            for f_c_dot_indx in range(cp.f_c_dot_num):
                rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit_list = [
                    rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit__nu_chunk_list[nu_chunk_indx][f_c_dot_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.loglog(
                    cp.nu_list,
                    rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit_list,
                    linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.f_c_dot_label_list[f_c_dot_indx])
            plt.legend(loc='best', fontsize=10)
            # plt.ylim([-0.05, 1.025])
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 20,
                r'$\beta \overline{G_c}/(\eta^{ref}l_{\nu}^{eq})$', 20,
                data_file_prefix+"-rate-independent-and-rate-dependent-force-controlled-nondimensional-scaled-fracture-toughness-vs-nu")
            
            fig = plt.figure()
            plt.loglog(
                cp.nu_list, rate_independent_LT_overline_g_c_crit__nu_squared__nu_chunk_list,
                linestyle='-', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_label)
            plt.loglog(
                cp.nu_list, rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_squared__nu_chunk_list,
                linestyle='--', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_inext_gaussian_label)
            for AFM_expermts_indx in cp.AFM_exprmts_indx_list:
                rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_squared_list = [
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_squared__nu_chunk_list[nu_chunk_indx][AFM_expermts_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_squared_list,
                    linestyle=cp.beyer_2000_tau_b_linestyle_list[AFM_expermts_indx],
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_label_list[AFM_expermts_indx])
            plt.loglog(
                cp.nu_list, rate_independent_overline_g_c_crit__nu_squared__nu_chunk_list,
                linestyle='-', color='blue', alpha=1, linewidth=2.5,
                label=cp.ufjc_label)
            for f_c_dot_indx in range(cp.f_c_dot_num):
                rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared_list = [
                    rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_list[nu_chunk_indx][f_c_dot_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.loglog(
                    cp.nu_list,
                    rate_dependent_frc_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared_list,
                    linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.f_c_dot_label_list[f_c_dot_indx])
            plt.legend(loc='best', fontsize=10)
            # plt.ylim([-0.05, 1.025])
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 20,
                r'$\beta \overline{G_c}/(\eta^{ref}l_{\nu}^{eq}\nu^2)$', 20,
                data_file_prefix+"-rate-independent-and-rate-dependent-force-controlled-nondimensional-scaled-fracture-toughness-nu-squared-normalized-vs-nu")
            
            fig = plt.figure()
            plt.semilogx(
                cp.nu_list, rate_independent_LT_epsilon_cnu_diss_hat_crit__nu_chunk_list,
                linestyle='-', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_label)
            for AFM_expermts_indx in cp.AFM_exprmts_indx_list:
                rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_diss_hat_crit_list = [
                    rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_diss_hat_crit__nu_chunk_list[nu_chunk_indx][AFM_expermts_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_independent_beyer_2000_f_c_max_tau_b_epsilon_cnu_diss_hat_crit_list,
                    linestyle=cp.beyer_2000_tau_b_linestyle_list[AFM_expermts_indx],
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_label_list[AFM_expermts_indx])
            plt.semilogx(
                cp.nu_list, rate_independent_epsilon_cnu_diss_hat_crit__nu_chunk_list,
                linestyle='-', color='blue', alpha=1, linewidth=2.5,
                label=cp.ufjc_label)
            for r_nu_dot_indx in range(cp.r_nu_dot_num):
                rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit_list = [
                    rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_list[nu_chunk_indx][r_nu_dot_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_dependent_strn_cntrld_AFM_exprmts_epsilon_cnu_diss_hat_crit_list,
                    linestyle='-', color=cp.r_nu_dot_color_list[r_nu_dot_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.r_nu_dot_label_list[r_nu_dot_indx])
            plt.legend(loc='best', fontsize=10)
            plt.ylim([-5, zeta_nu_char+5])
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 20,
                r'$\hat{\varepsilon}_{c\nu}^{diss}$', 20,
                data_file_prefix+"-rate-independent-and-rate-dependent-displacement-controlled-nondimensional-dissipated-chain-scission-energy-per-segment-vs-nu")
            
            fig = plt.figure()
            plt.semilogx(
                cp.nu_list, rate_independent_LT_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list,
                linestyle='-', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_label)
            for AFM_expermts_indx in cp.AFM_exprmts_indx_list:
                rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_diss_hat_crit_list = [
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list[nu_chunk_indx][AFM_expermts_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_cnu_diss_hat_crit_list,
                    linestyle=cp.beyer_2000_tau_b_linestyle_list[AFM_expermts_indx],
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_label_list[AFM_expermts_indx])
            plt.semilogx(
                cp.nu_list, rate_independent_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list,
                linestyle='-', color='blue', alpha=1, linewidth=2.5,
                label=cp.ufjc_label)
            for r_nu_dot_indx in range(cp.r_nu_dot_num):
                rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit_list = [
                    rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list[nu_chunk_indx][r_nu_dot_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit_list,
                    linestyle='-', color=cp.r_nu_dot_color_list[r_nu_dot_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.r_nu_dot_label_list[r_nu_dot_indx])
            plt.legend(loc='best', fontsize=10)
            plt.ylim([-0.05, 1.025])
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 20,
                r'$\overline{\hat{\varepsilon}_{c\nu}^{diss}}$', 20,
                data_file_prefix+"-rate-independent-and-rate-dependent-displacement-controlled-nondimensional-scaled-dissipated-chain-scission-energy-per-segment-vs-nu")
            
            fig = plt.figure()
            plt.loglog(
                cp.nu_list, rate_independent_LT_epsilon_c_diss_hat_crit__nu_chunk_list,
                linestyle='-', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_label)
            for AFM_expermts_indx in cp.AFM_exprmts_indx_list:
                rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_diss_hat_crit_list = [
                    rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_diss_hat_crit__nu_chunk_list[nu_chunk_indx][AFM_expermts_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_independent_beyer_2000_f_c_max_tau_b_epsilon_c_diss_hat_crit_list,
                    linestyle=cp.beyer_2000_tau_b_linestyle_list[AFM_expermts_indx],
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_label_list[AFM_expermts_indx])
            plt.loglog(
                cp.nu_list, rate_independent_epsilon_c_diss_hat_crit__nu_chunk_list,
                linestyle='-', color='blue', alpha=1, linewidth=2.5,
                label=cp.ufjc_label)
            for r_nu_dot_indx in range(cp.r_nu_dot_num):
                rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit_list = [
                    rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit__nu_chunk_list[nu_chunk_indx][r_nu_dot_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.loglog(
                    cp.nu_list,
                    rate_dependent_strn_cntrld_AFM_exprmts_epsilon_c_diss_hat_crit_list,
                    linestyle='-', color=cp.r_nu_dot_color_list[r_nu_dot_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.r_nu_dot_label_list[r_nu_dot_indx])
            plt.legend(loc='best', fontsize=10)
            # plt.ylim([-5, zeta_nu_char+5])
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 20,
                r'$\hat{\varepsilon}_c^{diss}$', 20,
                data_file_prefix+"-rate-independent-and-rate-dependent-displacement-controlled-nondimensional-dissipated-chain-scission-energy-vs-nu")
            
            fig = plt.figure()
            plt.loglog(
                cp.nu_list, rate_independent_LT_overline_epsilon_c_diss_hat_crit__nu_chunk_list,
                linestyle='-', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_label)
            for AFM_expermts_indx in cp.AFM_exprmts_indx_list:
                rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_diss_hat_crit_list = [
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_diss_hat_crit__nu_chunk_list[nu_chunk_indx][AFM_expermts_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_epsilon_c_diss_hat_crit_list,
                    linestyle=cp.beyer_2000_tau_b_linestyle_list[AFM_expermts_indx],
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_label_list[AFM_expermts_indx])
            plt.loglog(
                cp.nu_list, rate_independent_overline_epsilon_c_diss_hat_crit__nu_chunk_list,
                linestyle='-', color='blue', alpha=1, linewidth=2.5,
                label=cp.ufjc_label)
            for r_nu_dot_indx in range(cp.r_nu_dot_num):
                rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit_list = [
                    rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit__nu_chunk_list[nu_chunk_indx][r_nu_dot_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.loglog(
                    cp.nu_list,
                    rate_dependent_strn_cntrld_AFM_exprmts_overline_epsilon_c_diss_hat_crit_list,
                    linestyle='-', color=cp.r_nu_dot_color_list[r_nu_dot_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.r_nu_dot_label_list[r_nu_dot_indx])
            plt.legend(loc='best', fontsize=10)
            # plt.ylim([-0.05, 1.025])
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 20,
                r'$\overline{\hat{\varepsilon}_c^{diss}}$', 20,
                data_file_prefix+"-rate-independent-and-rate-dependent-displacement-controlled-nondimensional-scaled-dissipated-chain-scission-energy-vs-nu")
            
            fig = plt.figure()
            plt.loglog(
                cp.nu_list, rate_independent_LT_g_c_crit__nu_chunk_list,
                linestyle='-', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_label)
            plt.loglog(
                cp.nu_list, rate_independent_LT_inext_gaussian_g_c_crit__nu_chunk_list,
                linestyle='--', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_inext_gaussian_label)
            for AFM_expermts_indx in cp.AFM_exprmts_indx_list:
                rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit_list = [
                    rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_chunk_list[nu_chunk_indx][AFM_expermts_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit_list,
                    linestyle=cp.beyer_2000_tau_b_linestyle_list[AFM_expermts_indx],
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_label_list[AFM_expermts_indx])
            plt.loglog(
                cp.nu_list, rate_independent_g_c_crit__nu_chunk_list,
                linestyle='-', color='blue', alpha=1, linewidth=2.5,
                label=cp.ufjc_label)
            for r_nu_dot_indx in range(cp.r_nu_dot_num):
                rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit_list = [
                    rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit__nu_chunk_list[nu_chunk_indx][r_nu_dot_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.loglog(
                    cp.nu_list,
                    rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit_list,
                    linestyle='-', color=cp.r_nu_dot_color_list[r_nu_dot_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.r_nu_dot_label_list[r_nu_dot_indx])
            plt.legend(loc='best', fontsize=10)
            # plt.ylim([-0.05, 1.025])
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 20,
                r'$\beta G_c/(\eta^{ref}l_{\nu}^{eq})$', 20,
                data_file_prefix+"-rate-independent-and-rate-dependent-displacement-controlled-nondimensional-fracture-toughness-vs-nu")
            
            fig = plt.figure()
            plt.loglog(
                cp.nu_list, rate_independent_LT_g_c_crit__nu_squared__nu_chunk_list,
                linestyle='-', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_label)
            plt.loglog(
                cp.nu_list, rate_independent_LT_inext_gaussian_g_c_crit__nu_squared__nu_chunk_list,
                linestyle='--', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_inext_gaussian_label)
            for AFM_expermts_indx in cp.AFM_exprmts_indx_list:
                rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_squared_list = [
                    rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_squared__nu_chunk_list[nu_chunk_indx][AFM_expermts_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_independent_beyer_2000_f_c_max_tau_b_g_c_crit__nu_squared_list,
                    linestyle=cp.beyer_2000_tau_b_linestyle_list[AFM_expermts_indx],
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_label_list[AFM_expermts_indx])
            plt.loglog(
                cp.nu_list, rate_independent_g_c_crit__nu_squared__nu_chunk_list,
                linestyle='-', color='blue', alpha=1, linewidth=2.5,
                label=cp.ufjc_label)
            for r_nu_dot_indx in range(cp.r_nu_dot_num):
                rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit__nu_squared_list = [
                    rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_list[nu_chunk_indx][r_nu_dot_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.loglog(
                    cp.nu_list,
                    rate_dependent_strn_cntrld_AFM_exprmts_g_c_crit__nu_squared_list,
                    linestyle='-', color=cp.r_nu_dot_color_list[r_nu_dot_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.r_nu_dot_label_list[r_nu_dot_indx])
            plt.legend(loc='best', fontsize=10)
            # plt.ylim([-0.05, 1.025])
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 20,
                r'$\beta G_c/(\eta^{ref}l_{\nu}^{eq}\nu^2)$', 20,
                data_file_prefix+"-rate-independent-and-rate-dependent-displacement-controlled-nondimensional-fracture-toughness-nu-squared-normalized-vs-nu")
            
            fig = plt.figure()
            plt.loglog(
                cp.nu_list, rate_independent_LT_overline_g_c_crit__nu_chunk_list,
                linestyle='-', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_label)
            plt.loglog(
                cp.nu_list, rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_chunk_list,
                linestyle='--', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_inext_gaussian_label)
            for AFM_expermts_indx in cp.AFM_exprmts_indx_list:
                rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit_list = [
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_chunk_list[nu_chunk_indx][AFM_expermts_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit_list,
                    linestyle=cp.beyer_2000_tau_b_linestyle_list[AFM_expermts_indx],
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_label_list[AFM_expermts_indx])
            plt.loglog(
                cp.nu_list, rate_independent_overline_g_c_crit__nu_chunk_list,
                linestyle='-', color='blue', alpha=1, linewidth=2.5,
                label=cp.ufjc_label)
            for r_nu_dot_indx in range(cp.r_nu_dot_num):
                rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit_list = [
                    rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit__nu_chunk_list[nu_chunk_indx][r_nu_dot_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.loglog(
                    cp.nu_list,
                    rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit_list,
                    linestyle='-', color=cp.r_nu_dot_color_list[r_nu_dot_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.r_nu_dot_label_list[r_nu_dot_indx])
            plt.legend(loc='best', fontsize=10)
            # plt.ylim([-0.05, 1.025])
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 20,
                r'$\beta \overline{G_c}/(\eta^{ref}l_{\nu}^{eq})$', 20,
                data_file_prefix+"-rate-independent-and-rate-dependent-displacement-controlled-nondimensional-scaled-fracture-toughness-vs-nu")
            
            fig = plt.figure()
            plt.loglog(
                cp.nu_list, rate_independent_LT_overline_g_c_crit__nu_squared__nu_chunk_list,
                linestyle='-', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_label)
            plt.loglog(
                cp.nu_list, rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_squared__nu_chunk_list,
                linestyle='--', color='red', alpha=1, linewidth=2.5,
                label=cp.LT_inext_gaussian_label)
            for AFM_expermts_indx in cp.AFM_exprmts_indx_list:
                rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_squared_list = [
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_squared__nu_chunk_list[nu_chunk_indx][AFM_expermts_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.semilogx(
                    cp.nu_list,
                    rate_independent_beyer_2000_f_c_max_tau_b_overline_g_c_crit__nu_squared_list,
                    linestyle=cp.beyer_2000_tau_b_linestyle_list[AFM_expermts_indx],
                    color=cp.beyer_2000_tau_b_color_list[AFM_expermts_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.beyer_2000_tau_b_label_list[AFM_expermts_indx])
            plt.loglog(
                cp.nu_list, rate_independent_overline_g_c_crit__nu_squared__nu_chunk_list,
                linestyle='-', color='blue', alpha=1, linewidth=2.5,
                label=cp.ufjc_label)
            for r_nu_dot_indx in range(cp.r_nu_dot_num):
                rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared_list = [
                    rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_list[nu_chunk_indx][r_nu_dot_indx]
                    for nu_chunk_indx in range(cp.nu_num)
                ]
                plt.loglog(
                    cp.nu_list,
                    rate_dependent_strn_cntrld_AFM_exprmts_overline_g_c_crit__nu_squared_list,
                    linestyle='-', color=cp.r_nu_dot_color_list[r_nu_dot_indx],
                    alpha=1, linewidth=2.5,
                    label=cp.r_nu_dot_label_list[r_nu_dot_indx])
            plt.legend(loc='best', fontsize=10)
            # plt.ylim([-0.05, 1.025])
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            plt.grid(True, alpha=0.25)
            save_current_figure(
                self.savedir, r'$\nu$', 20,
                r'$\beta \overline{G_c}/(\eta^{ref}l_{\nu}^{eq}\nu^2)$', 20,
                data_file_prefix+"-rate-independent-and-rate-dependent-displacement-controlled-nondimensional-scaled-fracture-toughness-nu-squared-normalized-vs-nu")

            contourf_levels_num = 101
            contourf_levels = np.linspace(0, zeta_nu_char, contourf_levels_num)

            contour_levels_num = 26
            contour_levels = np.linspace(0, zeta_nu_char, contour_levels_num)

            nu_list_meshgrid, tilde_xi_c_dot_list_meshgrid = np.meshgrid(
                cp.nu_list, cp.tilde_xi_c_dot_list)

            rate_dependent_tilde_xi_c_dot_epsilon_cnu_diss_hat_crit_list = np.transpose(
                np.asarray(rate_dependent_tilde_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list))

            fig, ax1 = plt.subplots()

            filled_contour_plot = ax1.contourf(
                nu_list_meshgrid, tilde_xi_c_dot_list_meshgrid,
                rate_dependent_tilde_xi_c_dot_epsilon_cnu_diss_hat_crit_list,
                levels=contourf_levels, cmap=plt.cm.hsv)
            
            for fcp in filled_contour_plot.collections:
                fcp.set_edgecolor('face')
            
            ax1.set_xlabel(r'$\nu$', fontsize=20)
            ax1.set_ylabel(r'$\tilde{\dot{\xi}}_c$', fontsize=20)
            ax1.set_ylim(1e-40, 1e1)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            # ax1.tick_params(axis='both', labelsize=16)

            labeled_contour_plot = ax1.contour(
                nu_list_meshgrid, tilde_xi_c_dot_list_meshgrid,
                rate_dependent_tilde_xi_c_dot_epsilon_cnu_diss_hat_crit_list,
                levels=contour_levels, colors=('gray',), linewidths=0.25) # linewidths=0.5
            ax1.clabel(labeled_contour_plot, fmt='%3.1f', colors='gray', fontsize=6) # fontsize=8

            cbar = fig.colorbar(filled_contour_plot)
            cbar.ax.set_ylabel(r'$\hat{\varepsilon}_{c\nu}^{diss}$', fontsize=20)
            cbar.ax.tick_params(axis='y', labelsize=14)
            
            plt.yticks(fontsize=14)
            plt.xticks(fontsize=14)
            
            save_current_figure_no_labels(
                self.savedir,
                data_file_prefix+"-nondimensional-dissipated-chain-scission-energy-per-segment-filled-contour-tilde_xi_c_dot-vs-nu")

            nu_list_meshgrid, check_xi_c_dot_list_meshgrid = np.meshgrid(
                cp.nu_list, cp.check_xi_c_dot_list)

            rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit_list = np.transpose(
                np.asarray(rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list))

            fig, ax1 = plt.subplots()

            filled_contour_plot = ax1.contourf(
                nu_list_meshgrid, check_xi_c_dot_list_meshgrid,
                rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit_list,
                levels=contourf_levels, cmap=plt.cm.hsv)
            
            for fcp in filled_contour_plot.collections:
                fcp.set_edgecolor('face')
            
            ax1.set_xlabel(r'$\nu$', fontsize=20)
            ax1.set_ylabel(r'$\check{\dot{\xi}}_c$', fontsize=20)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            # ax1.tick_params(axis='both', labelsize=16)

            labeled_contour_plot = ax1.contour(
                nu_list_meshgrid, check_xi_c_dot_list_meshgrid,
                rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit_list,
                levels=contour_levels, colors=('gray',), linewidths=0.25) # linewidths=0.5
            ax1.clabel(labeled_contour_plot, fmt='%3.1f', colors='gray', fontsize=6) # fontsize=8

            cbar = fig.colorbar(filled_contour_plot)
            cbar.ax.set_ylabel(r'$\hat{\varepsilon}_{c\nu}^{diss}$', fontsize=20)
            cbar.ax.tick_params(axis='y', labelsize=14)
            
            plt.yticks(fontsize=14)
            plt.xticks(fontsize=14)
            
            save_current_figure_no_labels(
                self.savedir,
                data_file_prefix+"-nondimensional-dissipated-chain-scission-energy-per-segment-filled-contour-check_xi_c_dot-vs-nu")


if __name__ == '__main__':

    T = 298 # absolute room temperature, K

    AFM_chain_tensile_tests_dict = {
        "al-maawali-et-al": "chain-a", "hugel-et-al": "chain-a"
    }

    al_maawali_et_al_fracture_toughness_characterizer = (
        FractureToughnessCharacterizer(
            paper_authors="al-maawali-et-al", chain="chain-a", T=T,)
    )
    # al_maawali_et_al_fracture_toughness_characterizer.characterization()
    al_maawali_et_al_fracture_toughness_characterizer.finalization()

    hugel_et_al_fracture_toughness_characterizer = (
        FractureToughnessCharacterizer(
            paper_authors="hugel-et-al", chain="chain-a", T=T)
    )
    # hugel_et_al_fracture_toughness_characterizer.characterization()
    hugel_et_al_fracture_toughness_characterizer.finalization()