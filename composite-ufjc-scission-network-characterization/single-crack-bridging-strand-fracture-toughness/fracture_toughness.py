"""The single-chain fracture toughness characterization module for
composite uFJCs that undergo scission
"""

# import external modules
from __future__ import division
from composite_ufjc_scission import (CompositeuFJCScissionCharacterizer,
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
        # from DFT simulations on H_3C-CH_2-CH_3 (c-c) 
        # and H_3Si-O-CH_3 (si-o) by Beyer, J Chem. Phys., 2000
        p.characterizer.chain_backbone_bond_type2f_c_max_dict = {
            "c-c": 6.92,
            "si-o": 5.20
        } # nN
        # from DFT simulations on H_3C-CH_2-CH_3 (c-c) 
        # and H_3Si-O-CH_3 (si-o) by Beyer, J Chem. Phys., 2000
        p.characterizer.chain_backbone_bond_type2typcl_AFM_exprmt_f_c_max_dict = {
            "c-c": 3.81,
            "si-o": 3.14
        } # nN

        p.characterizer.f_c_num_steps = 100001

        # nu = 1 -> nu = 10000, only 250 unique nu values exist here
        nu_list = np.unique(np.rint(np.logspace(0, 4, 351)))
        tilde_xi_c_dot_list = np.logspace(-40, 10, 126)
        check_xi_c_dot_list = np.logspace(-40, 0, 101)

        p.characterizer.nu_list = nu_list
        p.characterizer.tilde_xi_c_dot_list = tilde_xi_c_dot_list
        p.characterizer.check_xi_c_dot_list = check_xi_c_dot_list

        f_c_dot_list = [1e1, 1e5, 1e9] # nN/sec
        f_c_dot_exponent_list = [
            int(floor(log10(abs(f_c_dot_list[i]))))
            for i in range(len(f_c_dot_list))
        ]
        f_c_dot_label_list = [
            r'$\textrm{composite}~u\textrm{FJC scission},~\dot{f}_c='+'10^{0:d}'.format(f_c_dot_exponent_list[i])+'~nN/sec$'
            for i in range(len(f_c_dot_list))
        ]
        f_c_dot_color_list = ['orange', 'purple', 'green']

        p.characterizer.f_c_dot_list          = f_c_dot_list
        p.characterizer.f_c_dot_exponent_list = f_c_dot_exponent_list
        p.characterizer.f_c_dot_label_list    = f_c_dot_label_list
        p.characterizer.f_c_dot_color_list    = f_c_dot_color_list

    def prefix(self):
        """Set characterization prefix"""
        return "fracture_toughness"
    
    def characterization(self):
        """Define characterization routine"""
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
        
        f_c_max = (
            cp.chain_backbone_bond_type2f_c_max_dict[chain_backbone_bond_type]
        ) # nN
        typcl_AFM_exprmt_f_c_max = (
            cp.chain_backbone_bond_type2typcl_AFM_exprmt_f_c_max_dict[chain_backbone_bond_type]
        ) # nN
        xi_c_max = f_c_max * beta * l_nu_eq
        typcl_AFM_exprmt_xi_c_max = typcl_AFM_exprmt_f_c_max * beta * l_nu_eq

        if chain_backbone_bond_type == "c-c":
            intrmdt_AFM_exprmt_f_c_max = 4.5 # nN
            intrmdt_AFM_exprmt_xi_c_max = (
                intrmdt_AFM_exprmt_f_c_max * beta * l_nu_eq
            )
        
        
        # # Rate-independent calculations
        
        
        # single_chain_list = [
        #     RateIndependentScissionCompositeuFJC(
        #         nu=nu_val, zeta_nu_char=zeta_nu_char, kappa_nu=kappa_nu)
        #     for nu_val in cp.nu_list
        # ]
        
        
        # A_nu_list = [single_chain.A_nu for single_chain in single_chain_list]
        
        # inext_gaussian_A_nu_list = [1/np.sqrt(nu_val) for nu_val in cp.nu_list]
        
        # inext_gaussian_A_nu_err_list = [
        #     np.abs((inext_gaussian_A_nu_val-A_nu_val)/A_nu_val)*100
        #     for inext_gaussian_A_nu_val, A_nu_val
        #     in zip(inext_gaussian_A_nu_list, A_nu_list)
        # ]
        
        
        # rate_independent_epsilon_cnu_diss_hat_crit_list = [
        #     single_chain.epsilon_cnu_diss_hat_crit
        #     for single_chain in single_chain_list
        # ]
        # rate_independent_g_c_crit_list = [
        #     single_chain.g_c_crit for single_chain in single_chain_list
        # ]
        # rate_independent_g_c_crit__nu_squared_list = [
        #     g_c_crit_val / nu_val**2 for g_c_crit_val, nu_val
        #     in zip(rate_independent_g_c_crit_list, cp.nu_list)
        # ]
        # rate_independent_overline_epsilon_cnu_diss_hat_crit_list = [
        #     epsilon_cnu_diss_hat_crit_val / zeta_nu_char
        #     for epsilon_cnu_diss_hat_crit_val
        #     in rate_independent_epsilon_cnu_diss_hat_crit_list
        # ]
        # rate_independent_overline_g_c_crit_list = [
        #     g_c_crit_val / zeta_nu_char for g_c_crit_val
        #     in rate_independent_g_c_crit_list
        # ]
        # rate_independent_overline_g_c_crit__nu_squared_list = [
        #     overline_g_c_crit_val / nu_val**2 for overline_g_c_crit_val, nu_val
        #     in zip(rate_independent_overline_g_c_crit_list, cp.nu_list)
        # ]
        
        
        # rate_independent_LT_epsilon_cnu_diss_hat_crit_list = (
        #     [zeta_nu_char] * len(cp.nu_list)
        # )
        # rate_independent_LT_g_c_crit_list = [
        #     0.5 * A_nu_val * nu_val**2 * epsilon_cnu_diss_hat_crit_val
        #     for A_nu_val, nu_val, epsilon_cnu_diss_hat_crit_val
        #     in zip(
        #         A_nu_list, cp.nu_list,
        #         rate_independent_LT_epsilon_cnu_diss_hat_crit_list)
        # ]
        # rate_independent_LT_g_c_crit__nu_squared_list = [
        #     LT_g_c_crit_val / nu_val**2 for LT_g_c_crit_val, nu_val
        #     in zip(rate_independent_LT_g_c_crit_list, cp.nu_list)
        # ]
        # rate_independent_LT_overline_epsilon_cnu_diss_hat_crit_list = (
        #     [1] * len(cp.nu_list)
        # )
        # rate_independent_LT_overline_g_c_crit_list = [
        #     0.5 * A_nu_val * nu_val**2 * overline_epsilon_cnu_diss_hat_crit_val
        #     for A_nu_val, nu_val, overline_epsilon_cnu_diss_hat_crit_val
        #     in zip(
        #         A_nu_list, cp.nu_list,
        #         rate_independent_LT_overline_epsilon_cnu_diss_hat_crit_list)
        # ]
        # rate_independent_LT_overline_g_c_crit__nu_squared_list = [
        #     LT_overline_g_c_crit_val / nu_val**2
        #     for LT_overline_g_c_crit_val, nu_val
        #     in zip(rate_independent_LT_overline_g_c_crit_list, cp.nu_list)
        # ]
        
        
        # rate_independent_LT_inext_gaussian_g_c_crit_list = [
        #     0.5 * A_nu_val * nu_val**2 * epsilon_cnu_diss_hat_crit_val
        #     for A_nu_val, nu_val, epsilon_cnu_diss_hat_crit_val
        #     in zip(
        #         inext_gaussian_A_nu_list, cp.nu_list,
        #         rate_independent_LT_epsilon_cnu_diss_hat_crit_list)
        # ]
        # rate_independent_LT_inext_gaussian_g_c_crit__nu_squared_list = [
        #     LT_inext_gaussian_g_c_crit_val / nu_val**2
        #     for LT_inext_gaussian_g_c_crit_val, nu_val
        #     in zip(rate_independent_LT_inext_gaussian_g_c_crit_list, cp.nu_list)
        # ]
        # rate_independent_LT_inext_gaussian_overline_g_c_crit_list = [
        #     0.5 * A_nu_val * nu_val**2 * overline_epsilon_cnu_diss_hat_crit_val
        #     for A_nu_val, nu_val, overline_epsilon_cnu_diss_hat_crit_val
        #     in zip(
        #         inext_gaussian_A_nu_list, cp.nu_list,
        #         rate_independent_LT_overline_epsilon_cnu_diss_hat_crit_list)
        # ]
        # rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_squared_list = [
        #     LT_inext_gaussian_overline_g_c_crit_val / nu_val**2
        #     for LT_inext_gaussian_overline_g_c_crit_val, nu_val
        #     in zip(
        #         rate_independent_LT_inext_gaussian_overline_g_c_crit_list,
        #         cp.nu_list)
        # ]
        
        
        # single_chain = RateIndependentScissionCompositeuFJC(
        #     nu=nu, zeta_nu_char=zeta_nu_char, kappa_nu=kappa_nu)
        
        
        # CR_typcl_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_val = (
        #     single_chain.epsilon_cnu_sci_hat_func(
        #         single_chain.lmbda_nu_xi_c_hat_func(typcl_AFM_exprmt_xi_c_max))
        # )
        # rate_independent_CR_typcl_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list = (
        #     [CR_typcl_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_val]
        #     * len(cp.nu_list)
        # )
        # rate_independent_CR_typcl_AFM_exprmt_xi_c_max_g_c_crit_list = [
        #     0.5 * A_nu_val * nu_val**2 * epsilon_cnu_diss_hat_crit_val
        #     for A_nu_val, nu_val, epsilon_cnu_diss_hat_crit_val
        #     in zip(
        #         A_nu_list, cp.nu_list,
        #         rate_independent_CR_typcl_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list)
        # ]
        # rate_independent_CR_typcl_AFM_exprmt_xi_c_max_g_c_crit__nu_squared_list = [
        #     CR_typcl_AFM_exprmt_xi_c_max_g_c_crit_val / nu_val**2
        #     for CR_typcl_AFM_exprmt_xi_c_max_g_c_crit_val, nu_val
        #     in zip(
        #         rate_independent_CR_typcl_AFM_exprmt_xi_c_max_g_c_crit_list,
        #         cp.nu_list)
        # ]
        # rate_independent_CR_typcl_AFM_exprmt_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list = [
        #     CR_typcl_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_val/zeta_nu_char
        #     for CR_typcl_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_val
        #     in rate_independent_CR_typcl_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list
        # ]
        # rate_independent_CR_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit_list = [
        #     0.5 * A_nu_val * nu_val**2 * overline_epsilon_cnu_diss_hat_crit_val
        #     for A_nu_val, nu_val, overline_epsilon_cnu_diss_hat_crit_val
        #     in zip(
        #         A_nu_list, cp.nu_list,
        #         rate_independent_CR_typcl_AFM_exprmt_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list)
        # ]
        # rate_independent_CR_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit__nu_squared_list = [
        #     CR_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit_val / nu_val**2
        #     for CR_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit_val, nu_val
        #     in zip(
        #         rate_independent_CR_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit_list,
        #         cp.nu_list)
        # ]

        # if chain_backbone_bond_type == "c-c":
        #     CR_intrmdt_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_val = (
        #     single_chain.epsilon_cnu_sci_hat_func(
        #         single_chain.lmbda_nu_xi_c_hat_func(intrmdt_AFM_exprmt_xi_c_max))
        #     )
        #     rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list = (
        #         [CR_intrmdt_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_val]
        #         * len(cp.nu_list)
        #     )
        #     rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_g_c_crit_list = [
        #         0.5 * A_nu_val * nu_val**2 * epsilon_cnu_diss_hat_crit_val
        #         for A_nu_val, nu_val, epsilon_cnu_diss_hat_crit_val
        #         in zip(
        #             A_nu_list, cp.nu_list,
        #             rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list)
        #     ]
        #     rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_g_c_crit__nu_squared_list = [
        #         CR_intrmdt_AFM_exprmt_xi_c_max_g_c_crit_val / nu_val**2
        #         for CR_intrmdt_AFM_exprmt_xi_c_max_g_c_crit_val, nu_val
        #         in zip(
        #             rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_g_c_crit_list,
        #             cp.nu_list)
        #     ]
        #     rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list = [
        #         CR_intrmdt_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_val/zeta_nu_char
        #         for CR_intrmdt_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_val
        #         in rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list
        #     ]
        #     rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_overline_g_c_crit_list = [
        #         0.5 * A_nu_val * nu_val**2 * overline_epsilon_cnu_diss_hat_crit_val
        #         for A_nu_val, nu_val, overline_epsilon_cnu_diss_hat_crit_val
        #         in zip(
        #             A_nu_list, cp.nu_list,
        #             rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list)
        #     ]
        #     rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_overline_g_c_crit__nu_squared_list = [
        #         CR_intrmdt_AFM_exprmt_xi_c_max_overline_g_c_crit_val / nu_val**2
        #         for CR_intrmdt_AFM_exprmt_xi_c_max_overline_g_c_crit_val, nu_val
        #         in zip(
        #             rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_overline_g_c_crit_list,
        #             cp.nu_list)
        #     ]
        
        
        # save_pickle_object(
        #     self.savedir, A_nu_list, data_file_prefix+"-A_nu_list")
        # save_pickle_object(
        #     self.savedir, inext_gaussian_A_nu_list,
        #     data_file_prefix+"-inext_gaussian_A_nu_list")
        # save_pickle_object(
        #     self.savedir, inext_gaussian_A_nu_err_list,
        #     data_file_prefix+"-inext_gaussian_A_nu_err_list")
        
        # save_pickle_object(
        #     self.savedir, rate_independent_epsilon_cnu_diss_hat_crit_list,
        #     data_file_prefix+"-rate_independent_epsilon_cnu_diss_hat_crit_list")
        # save_pickle_object(
        #     self.savedir,
        #     rate_independent_g_c_crit_list,
        #     data_file_prefix+"-rate_independent_g_c_crit_list")
        # save_pickle_object(
        #     self.savedir, rate_independent_g_c_crit__nu_squared_list,
        #     data_file_prefix+"-rate_independent_g_c_crit__nu_squared_list")
        # save_pickle_object(
        #     self.savedir,
        #     rate_independent_overline_epsilon_cnu_diss_hat_crit_list,
        #     data_file_prefix+"-rate_independent_overline_epsilon_cnu_diss_hat_crit_list")
        # save_pickle_object(
        #     self.savedir, rate_independent_overline_g_c_crit_list,
        #     data_file_prefix+"-rate_independent_overline_g_c_crit_list")
        # save_pickle_object(
        #     self.savedir,
        #     rate_independent_overline_g_c_crit__nu_squared_list,
        #     data_file_prefix+"-rate_independent_overline_g_c_crit__nu_squared_list")
        
        # save_pickle_object(
        #     self.savedir, rate_independent_LT_epsilon_cnu_diss_hat_crit_list,
        #     data_file_prefix+"-rate_independent_LT_epsilon_cnu_diss_hat_crit_list")
        # save_pickle_object(
        #     self.savedir, rate_independent_LT_g_c_crit_list,
        #     data_file_prefix+"-rate_independent_LT_g_c_crit_list")
        # save_pickle_object(
        #     self.savedir, rate_independent_LT_g_c_crit__nu_squared_list,
        #     data_file_prefix+"-rate_independent_LT_g_c_crit__nu_squared_list")
        # save_pickle_object(
        #     self.savedir,
        #     rate_independent_LT_overline_epsilon_cnu_diss_hat_crit_list,
        #     data_file_prefix+"-rate_independent_LT_overline_epsilon_cnu_diss_hat_crit_list")
        # save_pickle_object(
        #     self.savedir, rate_independent_LT_overline_g_c_crit_list,
        #     data_file_prefix+"-rate_independent_LT_overline_g_c_crit_list")
        # save_pickle_object(
        #     self.savedir,
        #     rate_independent_LT_overline_g_c_crit__nu_squared_list,
        #     data_file_prefix+"-rate_independent_LT_overline_g_c_crit__nu_squared_list")
        
        # save_pickle_object(
        #     self.savedir, rate_independent_LT_inext_gaussian_g_c_crit_list,
        # data_file_prefix+"-rate_independent_LT_inext_gaussian_g_c_crit_list")
        # save_pickle_object(
        #     self.savedir,
        #     rate_independent_LT_inext_gaussian_g_c_crit__nu_squared_list,
        #     data_file_prefix+"-rate_independent_LT_inext_gaussian_g_c_crit__nu_squared_list")
        # save_pickle_object(
        #     self.savedir,
        #     rate_independent_LT_inext_gaussian_overline_g_c_crit_list,
        #     data_file_prefix+"-rate_independent_LT_inext_gaussian_overline_g_c_crit_list")
        # save_pickle_object(
        #     self.savedir,
        #     rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_squared_list,
        #     data_file_prefix+"-rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_squared_list")
        
        # save_pickle_object(
        #     self.savedir,
        #     rate_independent_CR_typcl_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list,
        #     data_file_prefix+"-rate_independent_CR_typcl_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list")
        # save_pickle_object(
        #     self.savedir,
        #     rate_independent_CR_typcl_AFM_exprmt_xi_c_max_g_c_crit_list,
        #     data_file_prefix+"-rate_independent_CR_typcl_AFM_exprmt_xi_c_max_g_c_crit_list")
        # save_pickle_object(
        #     self.savedir,
        #     rate_independent_CR_typcl_AFM_exprmt_xi_c_max_g_c_crit__nu_squared_list,
        #     data_file_prefix+"-rate_independent_CR_typcl_AFM_exprmt_xi_c_max_g_c_crit__nu_squared_list")
        # save_pickle_object(
        #     self.savedir,
        #     rate_independent_CR_typcl_AFM_exprmt_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list,
        #     data_file_prefix+"-rate_independent_CR_typcl_AFM_exprmt_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list")
        # save_pickle_object(
        #     self.savedir, rate_independent_CR_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit_list,
        #     data_file_prefix+"-rate_independent_CR_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit_list")
        # save_pickle_object(
        #     self.savedir,
        #     rate_independent_CR_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit__nu_squared_list,
        #     data_file_prefix+"-rate_independent_CR_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit__nu_squared_list")
        
        # if chain_backbone_bond_type == "c-c":
        #     save_pickle_object(
        #         self.savedir,
        #         rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list,
        #         data_file_prefix+"-rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list")
        #     save_pickle_object(
        #         self.savedir,
        #         rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_g_c_crit_list,
        #         data_file_prefix+"-rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_g_c_crit_list")
        #     save_pickle_object(
        #         self.savedir,
        #         rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_g_c_crit__nu_squared_list,
        #         data_file_prefix+"-rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_g_c_crit__nu_squared_list")
        #     save_pickle_object(
        #         self.savedir,
        #         rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list,
        #         data_file_prefix+"-rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list")
        #     save_pickle_object(
        #         self.savedir,
        #         rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_overline_g_c_crit_list,
        #         data_file_prefix+"-rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_overline_g_c_crit_list")
        #     save_pickle_object(
        #         self.savedir,
        #         rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_overline_g_c_crit__nu_squared_list,
        #         data_file_prefix+"-rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_overline_g_c_crit__nu_squared_list")
        
        
        # Rate-dependent calculations


        rate_dependent_tilde_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]
        rate_dependent_tilde_xi_c_dot_g_c_crit__nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]
        rate_dependent_tilde_xi_c_dot_g_c_crit__nu_squared__nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]
        rate_dependent_tilde_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]
        rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]
        rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]

        for nu_indx in range(len(cp.nu_list)):
            rate_dependent_tilde_xi_c_dot_epsilon_cnu_diss_hat_crit = []
            rate_dependent_tilde_xi_c_dot_g_c_crit = []
            rate_dependent_tilde_xi_c_dot_g_c_crit__nu_squared = []
            rate_dependent_tilde_xi_c_dot_overline_epsilon_cnu_diss_hat_crit = []
            rate_dependent_tilde_xi_c_dot_overline_g_c_crit = []
            rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_squared = []

            nu_val = cp.nu_list[nu_indx]
            rate_dependent_single_chain = (
                RateDependentScissionCompositeuFJC(
                    nu=nu_val, zeta_nu_char=zeta_nu_char, kappa_nu=kappa_nu,
                    omega_0=omega_0)
            )
            A_nu_val = rate_dependent_single_chain.A_nu
            f_c_crit = (
                rate_dependent_single_chain.xi_c_crit / (beta*l_nu_eq)
            ) # (nN*nm)/nm = nN
            f_c_steps = np.linspace(0, f_c_crit, cp.f_c_num_steps) # nN
            for tilde_xi_c_dot_indx in range(len(cp.tilde_xi_c_dot_list)):
                tilde_xi_c_dot_val = cp.tilde_xi_c_dot_list[tilde_xi_c_dot_indx]
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
                    xi_c_val = (
                        f_c_steps[f_c_indx] * beta * l_nu_eq
                    ) # nN*nm/(nN*nm)
                    lmbda_nu_val = (
                        rate_dependent_single_chain.lmbda_nu_xi_c_hat_func(
                            xi_c_val)
                    )
                    p_nu_sci_hat_val = (
                        rate_dependent_single_chain.p_nu_sci_hat_func(
                            lmbda_nu_val)
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
                                epsilon_cnu_sci_hat_val, t_val,
                                t_steps[f_c_indx-1],
                                epsilon_cnu_diss_hat_val_prior)
                        )
                    
                    p_nu_sci_hat_cmltv_intgrl_val_prior = (
                        p_nu_sci_hat_cmltv_intgrl_val
                    )
                    p_nu_sci_hat_val_prior = p_nu_sci_hat_val
                    epsilon_cnu_diss_hat_val_prior = epsilon_cnu_diss_hat_val
                
                epsilon_cnu_diss_hat_crit_val = epsilon_cnu_diss_hat_val
                g_c_crit_val = (
                    0.5 * A_nu_val * nu_val**2 * epsilon_cnu_diss_hat_crit_val
                )
                g_c_crit__nu_squared_val = (
                    0.5 * A_nu_val * epsilon_cnu_diss_hat_crit_val
                )
                overline_epsilon_cnu_diss_hat_crit_val = (
                    epsilon_cnu_diss_hat_crit_val / zeta_nu_char
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
                rate_dependent_tilde_xi_c_dot_g_c_crit.append(g_c_crit_val)
                rate_dependent_tilde_xi_c_dot_g_c_crit__nu_squared.append(
                    g_c_crit__nu_squared_val
                )
                rate_dependent_tilde_xi_c_dot_overline_epsilon_cnu_diss_hat_crit.append(
                    overline_epsilon_cnu_diss_hat_crit_val
                )
                rate_dependent_tilde_xi_c_dot_overline_g_c_crit.append(overline_g_c_crit_val)
                rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_squared.append(
                    overline_g_c_crit__nu_squared_val
                )
            
            rate_dependent_tilde_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list[nu_indx] = (
                rate_dependent_tilde_xi_c_dot_epsilon_cnu_diss_hat_crit
            )
            rate_dependent_tilde_xi_c_dot_g_c_crit__nu_chunk_list[nu_indx] = (
                rate_dependent_tilde_xi_c_dot_g_c_crit
            )
            rate_dependent_tilde_xi_c_dot_g_c_crit__nu_squared__nu_chunk_list[nu_indx] = (
                rate_dependent_tilde_xi_c_dot_g_c_crit__nu_squared
            )
            rate_dependent_tilde_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list[nu_indx] = (
                rate_dependent_tilde_xi_c_dot_overline_epsilon_cnu_diss_hat_crit
            )
            rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_chunk_list[nu_indx] = (
                rate_dependent_tilde_xi_c_dot_overline_g_c_crit
            )
            rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_list[nu_indx] = (
                rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_squared
            )
        
        
        # rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list = [
        #     0. for nu_indx in range(len(cp.nu_list))
        # ]
        # rate_dependent_check_xi_c_dot_g_c_crit__nu_chunk_list = [
        #     0. for nu_indx in range(len(cp.nu_list))
        # ]
        # rate_dependent_check_xi_c_dot_g_c_crit__nu_squared__nu_chunk_list = [
        #     0. for nu_indx in range(len(cp.nu_list))
        # ]
        # rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list = [
        #     0. for nu_indx in range(len(cp.nu_list))
        # ]
        # rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_chunk_list = [
        #     0. for nu_indx in range(len(cp.nu_list))
        # ]
        # rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_list = [
        #     0. for nu_indx in range(len(cp.nu_list))
        # ]

        # for nu_indx in range(len(cp.nu_list)):
        #     rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit = []
        #     rate_dependent_check_xi_c_dot_g_c_crit = []
        #     rate_dependent_check_xi_c_dot_g_c_crit__nu_squared = []
        #     rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit = []
        #     rate_dependent_check_xi_c_dot_overline_g_c_crit = []
        #     rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_squared = []

        #     nu_val = cp.nu_list[nu_indx]
        #     rate_dependent_single_chain = (
        #         RateDependentScissionCompositeuFJC(
        #             nu=nu_val, zeta_nu_char=zeta_nu_char, kappa_nu=kappa_nu,
        #             omega_0=omega_0)
        #     )
        #     A_nu_val = rate_dependent_single_chain.A_nu
        #     f_c_crit = (
        #         rate_dependent_single_chain.xi_c_crit / (beta*l_nu_eq)
        #     ) # (nN*nm)/nm = nN
        #     f_c_steps = np.linspace(0, f_c_crit, cp.f_c_num_steps) # nN
        #     for check_xi_c_dot_indx in range(len(cp.check_xi_c_dot_list)):
        #         check_xi_c_dot_val = cp.check_xi_c_dot_list[check_xi_c_dot_indx]
        #         f_c_dot_val = (
        #             check_xi_c_dot_val * omega_0 * nu_val / (beta*l_nu_eq)
        #         ) # nN/sec
                
        #         t_steps = f_c_steps / f_c_dot_val # nN/(nN/sec) = sec
                
        #         # initialization
        #         p_nu_sci_hat_cmltv_intgrl_val       = 0.
        #         p_nu_sci_hat_cmltv_intgrl_val_prior = 0.
        #         p_nu_sci_hat_val                    = 0.
        #         p_nu_sci_hat_val_prior              = 0.
        #         epsilon_cnu_diss_hat_val            = 0.
        #         epsilon_cnu_diss_hat_val_prior      = 0.
                
        #         # Calculate results through applied chain force values
        #         for f_c_indx in range(cp.f_c_num_steps):
        #             t_val = t_steps[f_c_indx]
        #             xi_c_val = (
        #                 f_c_steps[f_c_indx] * beta * l_nu_eq
        #             ) # nN*nm/(nN*nm)
        #             lmbda_nu_val = (
        #                 rate_dependent_single_chain.lmbda_nu_xi_c_hat_func(
        #                     xi_c_val)
        #             )
        #             p_nu_sci_hat_val = (
        #                 rate_dependent_single_chain.p_nu_sci_hat_func(
        #                     lmbda_nu_val)
        #             )
        #             epsilon_cnu_sci_hat_val = (
        #                 rate_dependent_single_chain.epsilon_cnu_sci_hat_func(
        #                     lmbda_nu_val)
        #             )

        #             if f_c_indx == 0:
        #                 pass
        #             else:
        #                 p_nu_sci_hat_cmltv_intgrl_val = (
        #                     rate_dependent_single_chain.p_nu_sci_hat_cmltv_intgrl_func(
        #                         p_nu_sci_hat_val, t_val, p_nu_sci_hat_val_prior,
        #                         t_steps[f_c_indx-1],
        #                         p_nu_sci_hat_cmltv_intgrl_val_prior)
        #                 )
        #                 epsilon_cnu_diss_hat_val = (
        #                     rate_dependent_single_chain.epsilon_cnu_diss_hat_func(
        #                         p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val,
        #                         epsilon_cnu_sci_hat_val, t_val,
        #                         t_steps[f_c_indx-1],
        #                         epsilon_cnu_diss_hat_val_prior)
        #                 )
                    
        #             p_nu_sci_hat_cmltv_intgrl_val_prior = (
        #                 p_nu_sci_hat_cmltv_intgrl_val
        #             )
        #             p_nu_sci_hat_val_prior = p_nu_sci_hat_val
        #             epsilon_cnu_diss_hat_val_prior = epsilon_cnu_diss_hat_val
                
        #         epsilon_cnu_diss_hat_crit_val = epsilon_cnu_diss_hat_val
        #         g_c_crit_val = (
        #             0.5 * A_nu_val * nu_val**2 * epsilon_cnu_diss_hat_crit_val
        #         )
        #         g_c_crit__nu_squared_val = (
        #             0.5 * A_nu_val * epsilon_cnu_diss_hat_crit_val
        #         )
        #         overline_epsilon_cnu_diss_hat_crit_val = (
        #             epsilon_cnu_diss_hat_crit_val / zeta_nu_char
        #         )
        #         overline_g_c_crit_val = (
        #             0.5 * A_nu_val * nu_val**2
        #             * overline_epsilon_cnu_diss_hat_crit_val
        #         )
        #         overline_g_c_crit__nu_squared_val = (
        #             0.5 * A_nu_val * overline_epsilon_cnu_diss_hat_crit_val
        #         )
                
        #         rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit.append(
        #             epsilon_cnu_diss_hat_crit_val
        #         )
        #         rate_dependent_check_xi_c_dot_g_c_crit.append(g_c_crit_val)
        #         rate_dependent_check_xi_c_dot_g_c_crit__nu_squared.append(
        #             g_c_crit__nu_squared_val
        #         )
        #         rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit.append(
        #             overline_epsilon_cnu_diss_hat_crit_val
        #         )
        #         rate_dependent_check_xi_c_dot_overline_g_c_crit.append(overline_g_c_crit_val)
        #         rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_squared.append(
        #             overline_g_c_crit__nu_squared_val
        #         )
            
        #     rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list[nu_indx] = (
        #         rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit
        #     )
        #     rate_dependent_check_xi_c_dot_g_c_crit__nu_chunk_list[nu_indx] = (
        #         rate_dependent_check_xi_c_dot_g_c_crit
        #     )
        #     rate_dependent_check_xi_c_dot_g_c_crit__nu_squared__nu_chunk_list[nu_indx] = (
        #         rate_dependent_check_xi_c_dot_g_c_crit__nu_squared
        #     )
        #     rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list[nu_indx] = (
        #         rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit
        #     )
        #     rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_chunk_list[nu_indx] = (
        #         rate_dependent_check_xi_c_dot_overline_g_c_crit
        #     )
        #     rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_list[nu_indx] = (
        #         rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_squared
        #     )
        
        
        # rate_dependent_nu_fit_epsilon_cnu_diss_hat_crit_list = []
        # rate_dependent_nu_fit_g_c_crit_list = []
        # rate_dependent_nu_fit_g_c_crit__nu_squared_list = []
        # rate_dependent_nu_fit_overline_epsilon_cnu_diss_hat_crit_list = []
        # rate_dependent_nu_fit_overline_g_c_crit_list = []
        # rate_dependent_nu_fit_overline_g_c_crit__nu_squared_list = []

        # for check_xi_c_dot_indx in range(len(cp.check_xi_c_dot_list)):
        #     check_xi_c_dot_val = cp.check_xi_c_dot_list[check_xi_c_dot_indx]
        #     f_c_dot_val = (
        #         check_xi_c_dot_val * omega_0 * nu / (beta*l_nu_eq)
        #     ) # nN/sec

        #     rate_dependent_single_chain = (
        #         RateDependentScissionCompositeuFJC(
        #             nu=nu, zeta_nu_char=zeta_nu_char, kappa_nu=kappa_nu,
        #             omega_0=omega_0)
        #     )

        #     A_nu_val = rate_dependent_single_chain.A_nu
        #     f_c_crit = (
        #         rate_dependent_single_chain.xi_c_crit / (beta*l_nu_eq)
        #     ) # (nN*nm)/nm = nN
        #     f_c_steps = np.linspace(0, f_c_crit, cp.f_c_num_steps) # nN
        #     t_steps = f_c_steps / f_c_dot_val # nN/(nN/sec) = sec

        #     # initialization
        #     p_nu_sci_hat_cmltv_intgrl_val       = 0.
        #     p_nu_sci_hat_cmltv_intgrl_val_prior = 0.
        #     p_nu_sci_hat_val                    = 0.
        #     p_nu_sci_hat_val_prior              = 0.
        #     epsilon_cnu_diss_hat_val            = 0.
        #     epsilon_cnu_diss_hat_val_prior      = 0.

        #     # Calculate results through applied chain force values
        #     for f_c_indx in range(cp.f_c_num_steps):
        #         t_val = t_steps[f_c_indx]
        #         xi_c_val = f_c_steps[f_c_indx] * beta * l_nu_eq # nN*nm/(nN*nm)
        #         lmbda_nu_val = (
        #             rate_dependent_single_chain.lmbda_nu_xi_c_hat_func(xi_c_val)
        #         )
        #         p_nu_sci_hat_val = (
        #             rate_dependent_single_chain.p_nu_sci_hat_func(lmbda_nu_val)
        #         )
        #         epsilon_cnu_sci_hat_val = (
        #             rate_dependent_single_chain.epsilon_cnu_sci_hat_func(
        #                 lmbda_nu_val)
        #         )

        #         if f_c_indx == 0:
        #             pass
        #         else:
        #             p_nu_sci_hat_cmltv_intgrl_val = (
        #                 rate_dependent_single_chain.p_nu_sci_hat_cmltv_intgrl_func(
        #                     p_nu_sci_hat_val, t_val, p_nu_sci_hat_val_prior,
        #                     t_steps[f_c_indx-1],
        #                     p_nu_sci_hat_cmltv_intgrl_val_prior)
        #             )
        #             epsilon_cnu_diss_hat_val = (
        #                 rate_dependent_single_chain.epsilon_cnu_diss_hat_func(
        #                     p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val,
        #                     epsilon_cnu_sci_hat_val, t_val, t_steps[f_c_indx-1],
        #                     epsilon_cnu_diss_hat_val_prior)
        #             )
                
        #         p_nu_sci_hat_cmltv_intgrl_val_prior = (
        #             p_nu_sci_hat_cmltv_intgrl_val
        #         )
        #         p_nu_sci_hat_val_prior = p_nu_sci_hat_val
        #         epsilon_cnu_diss_hat_val_prior = epsilon_cnu_diss_hat_val
            
        #     epsilon_cnu_diss_hat_crit_val = epsilon_cnu_diss_hat_val
        #     g_c_crit_val = (
        #         0.5 * A_nu_val * nu**2 * epsilon_cnu_diss_hat_crit_val
        #     )
        #     g_c_crit__nu_squared_val = (
        #         0.5 * A_nu_val * epsilon_cnu_diss_hat_crit_val
        #     )
        #     overline_epsilon_cnu_diss_hat_crit_val = (
        #         epsilon_cnu_diss_hat_crit_val / zeta_nu_char
        #     )
        #     overline_g_c_crit_val = (
        #         0.5 * A_nu_val * nu**2 * overline_epsilon_cnu_diss_hat_crit_val
        #     )
        #     overline_g_c_crit__nu_squared_val = (
        #         0.5 * A_nu_val * overline_epsilon_cnu_diss_hat_crit_val
        #     )
            
            
        #     rate_dependent_nu_fit_epsilon_cnu_diss_hat_crit_list.append(
        #         epsilon_cnu_diss_hat_crit_val
        #     )
        #     rate_dependent_nu_fit_g_c_crit_list.append(g_c_crit_val)
        #     rate_dependent_nu_fit_g_c_crit__nu_squared_list.append(
        #         g_c_crit__nu_squared_val
        #     )
        #     rate_dependent_nu_fit_overline_epsilon_cnu_diss_hat_crit_list.append(
        #         overline_epsilon_cnu_diss_hat_crit_val
        #     )
        #     rate_dependent_nu_fit_overline_g_c_crit_list.append(
        #         overline_g_c_crit_val
        #     )
        #     rate_dependent_nu_fit_overline_g_c_crit__nu_squared_list.append(
        #         overline_g_c_crit__nu_squared_val
        #     )


        # rate_dependent_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_list = [
        #     0. for nu_indx in range(len(cp.nu_list))
        # ]
        # rate_dependent_AFM_exprmts_g_c_crit__nu_chunk_list = [
        #     0. for nu_indx in range(len(cp.nu_list))
        # ]
        # rate_dependent_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_list = [
        #     0. for nu_indx in range(len(cp.nu_list))
        # ]
        # rate_dependent_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list = [
        #     0. for nu_indx in range(len(cp.nu_list))
        # ]
        # rate_dependent_AFM_exprmts_overline_g_c_crit__nu_chunk_list = [
        #     0. for nu_indx in range(len(cp.nu_list))
        # ]
        # rate_dependent_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_list = [
        #     0. for nu_indx in range(len(cp.nu_list))
        # ]

        # for nu_indx in range(len(cp.nu_list)):
        #     rate_dependent_AFM_exprmts_epsilon_cnu_diss_hat_crit = []
        #     rate_dependent_AFM_exprmts_g_c_crit = []
        #     rate_dependent_AFM_exprmts_g_c_crit__nu_squared = []
        #     rate_dependent_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit = []
        #     rate_dependent_AFM_exprmts_overline_g_c_crit = []
        #     rate_dependent_AFM_exprmts_overline_g_c_crit__nu_squared = []

        #     nu_val = cp.nu_list[nu_indx]
        #     rate_dependent_single_chain = (
        #         RateDependentScissionCompositeuFJC(
        #             nu=nu_val, zeta_nu_char=zeta_nu_char, kappa_nu=kappa_nu,
        #             omega_0=omega_0)
        #     )
        #     A_nu_val = rate_dependent_single_chain.A_nu
        #     f_c_crit = (
        #         rate_dependent_single_chain.xi_c_crit / (beta*l_nu_eq)
        #     ) # (nN*nm)/nm = nN
        #     f_c_steps = np.linspace(0, f_c_crit, cp.f_c_num_steps) # nN
        #     for f_c_dot_indx in range(len(cp.f_c_dot_list)):
        #         f_c_dot_val = cp.f_c_dot_list[f_c_dot_indx] # nN/sec
        #         t_steps = f_c_steps / f_c_dot_val # nN/(nN/sec) = sec

        #         # initialization
        #         p_nu_sci_hat_cmltv_intgrl_val       = 0.
        #         p_nu_sci_hat_cmltv_intgrl_val_prior = 0.
        #         p_nu_sci_hat_val                    = 0.
        #         p_nu_sci_hat_val_prior              = 0.
        #         epsilon_cnu_diss_hat_val            = 0.
        #         epsilon_cnu_diss_hat_val_prior      = 0.

        #         # Calculate results through applied chain force values
        #         for f_c_indx in range(cp.f_c_num_steps):
        #             t_val = t_steps[f_c_indx]
        #             xi_c_val = f_c_steps[f_c_indx] * beta * l_nu_eq # nN*nm/(nN*nm)
        #             lmbda_nu_val = (
        #                 rate_dependent_single_chain.lmbda_nu_xi_c_hat_func(xi_c_val)
        #             )
        #             p_nu_sci_hat_val = (
        #                 rate_dependent_single_chain.p_nu_sci_hat_func(lmbda_nu_val)
        #             )
        #             epsilon_cnu_sci_hat_val = (
        #                 rate_dependent_single_chain.epsilon_cnu_sci_hat_func(
        #                     lmbda_nu_val)
        #             )

        #             if f_c_indx == 0:
        #                 pass
        #             else:
        #                 p_nu_sci_hat_cmltv_intgrl_val = (
        #                     rate_dependent_single_chain.p_nu_sci_hat_cmltv_intgrl_func(
        #                         p_nu_sci_hat_val, t_val, p_nu_sci_hat_val_prior,
        #                         t_steps[f_c_indx-1],
        #                         p_nu_sci_hat_cmltv_intgrl_val_prior)
        #                 )
        #                 epsilon_cnu_diss_hat_val = (
        #                     rate_dependent_single_chain.epsilon_cnu_diss_hat_func(
        #                         p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val,
        #                         epsilon_cnu_sci_hat_val, t_val, t_steps[f_c_indx-1],
        #                         epsilon_cnu_diss_hat_val_prior)
        #                 )
                    
        #             p_nu_sci_hat_cmltv_intgrl_val_prior = (
        #                 p_nu_sci_hat_cmltv_intgrl_val
        #             )
        #             p_nu_sci_hat_val_prior = p_nu_sci_hat_val
        #             epsilon_cnu_diss_hat_val_prior = epsilon_cnu_diss_hat_val
                
        #         epsilon_cnu_diss_hat_crit_val = epsilon_cnu_diss_hat_val
        #         g_c_crit_val = (
        #             0.5 * A_nu_val * nu_val**2 * epsilon_cnu_diss_hat_crit_val
        #         )
        #         g_c_crit__nu_squared_val = (
        #             0.5 * A_nu_val * epsilon_cnu_diss_hat_crit_val
        #         )
        #         overline_epsilon_cnu_diss_hat_crit_val = (
        #             epsilon_cnu_diss_hat_crit_val / zeta_nu_char
        #         )
        #         overline_g_c_crit_val = (
        #             0.5 * A_nu_val * nu_val**2
        #             * overline_epsilon_cnu_diss_hat_crit_val
        #         )
        #         overline_g_c_crit__nu_squared_val = (
        #             0.5 * A_nu_val * overline_epsilon_cnu_diss_hat_crit_val
        #         )
                
        #         rate_dependent_AFM_exprmts_epsilon_cnu_diss_hat_crit.append(
        #             epsilon_cnu_diss_hat_crit_val
        #         )
        #         rate_dependent_AFM_exprmts_g_c_crit.append(g_c_crit_val)
        #         rate_dependent_AFM_exprmts_g_c_crit__nu_squared.append(
        #             g_c_crit__nu_squared_val
        #         )
        #         rate_dependent_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit.append(
        #             overline_epsilon_cnu_diss_hat_crit_val
        #         )
        #         rate_dependent_AFM_exprmts_overline_g_c_crit.append(
        #             overline_g_c_crit_val
        #         )
        #         rate_dependent_AFM_exprmts_overline_g_c_crit__nu_squared.append(
        #             overline_g_c_crit__nu_squared_val
        #         )
            
        #     rate_dependent_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_list[nu_indx] = (
        #         rate_dependent_AFM_exprmts_epsilon_cnu_diss_hat_crit
        #     )
        #     rate_dependent_AFM_exprmts_g_c_crit__nu_chunk_list[nu_indx] = (
        #         rate_dependent_AFM_exprmts_g_c_crit
        #     )
        #     rate_dependent_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_list[nu_indx] = (
        #         rate_dependent_AFM_exprmts_g_c_crit__nu_squared
        #     )
        #     rate_dependent_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list[nu_indx] = (
        #         rate_dependent_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit
        #     )
        #     rate_dependent_AFM_exprmts_overline_g_c_crit__nu_chunk_list[nu_indx] = (
        #         rate_dependent_AFM_exprmts_overline_g_c_crit
        #     )
        #     rate_dependent_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_list[nu_indx] = (
        #         rate_dependent_AFM_exprmts_overline_g_c_crit__nu_squared
        #     )
        
        
        save_pickle_object(
            self.savedir, rate_dependent_tilde_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list,
            data_file_prefix+"-rate_dependent_tilde_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list")
        save_pickle_object(
            self.savedir, rate_dependent_tilde_xi_c_dot_g_c_crit__nu_chunk_list,
            data_file_prefix+"-rate_dependent_tilde_xi_c_dot_g_c_crit__nu_chunk_list")
        save_pickle_object(
            self.savedir, rate_dependent_tilde_xi_c_dot_g_c_crit__nu_squared__nu_chunk_list,
            data_file_prefix+"-rate_dependent_tilde_xi_c_dot_g_c_crit__nu_squared__nu_chunk_list")
        save_pickle_object(
            self.savedir,
            rate_dependent_tilde_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list,
            data_file_prefix+"-rate_dependent_tilde_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list")
        save_pickle_object(
            self.savedir,  rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_chunk_list,
            data_file_prefix+"-rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_chunk_list")
        save_pickle_object(
            self.savedir,  rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_list,
            data_file_prefix+"-rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_list")
        
        # save_pickle_object(
        #     self.savedir, rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list,
        #     data_file_prefix+"-rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list")
        # save_pickle_object(
        #     self.savedir, rate_dependent_check_xi_c_dot_g_c_crit__nu_chunk_list,
        #     data_file_prefix+"-rate_dependent_check_xi_c_dot_g_c_crit__nu_chunk_list")
        # save_pickle_object(
        #     self.savedir, rate_dependent_check_xi_c_dot_g_c_crit__nu_squared__nu_chunk_list,
        #     data_file_prefix+"-rate_dependent_check_xi_c_dot_g_c_crit__nu_squared__nu_chunk_list")
        # save_pickle_object(
        #     self.savedir,
        #     rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list,
        #     data_file_prefix+"-rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list")
        # save_pickle_object(
        #     self.savedir,  rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_chunk_list,
        #     data_file_prefix+"- rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_chunk_list")
        # save_pickle_object(
        #     self.savedir,  rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_list,
        #     data_file_prefix+"- rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_list")
        
        # save_pickle_object(
        #     self.savedir, rate_dependent_nu_fit_epsilon_cnu_diss_hat_crit_list,
        #     data_file_prefix+"-rate_dependent_nu_fit_epsilon_cnu_diss_hat_crit_list")
        # save_pickle_object(
        #     self.savedir, rate_dependent_nu_fit_g_c_crit_list,
        #     data_file_prefix+"-rate_dependent_nu_fit_g_c_crit_list")
        # save_pickle_object(
        #     self.savedir, rate_dependent_nu_fit_g_c_crit__nu_squared_list,
        #     data_file_prefix+"-rate_dependent_nu_fit_g_c_crit__nu_squared_list")
        # save_pickle_object(
        #     self.savedir,
        #     rate_dependent_nu_fit_overline_epsilon_cnu_diss_hat_crit_list,
        #     data_file_prefix+"-rate_dependent_nu_fit_overline_epsilon_cnu_diss_hat_crit_list")
        # save_pickle_object(
        #     self.savedir, rate_dependent_nu_fit_overline_g_c_crit_list,
        #     data_file_prefix+"-rate_dependent_nu_fit_overline_g_c_crit_list")
        # save_pickle_object(
        #     self.savedir, rate_dependent_nu_fit_overline_g_c_crit__nu_squared_list,
        #     data_file_prefix+"-rate_dependent_nu_fit_overline_g_c_crit__nu_squared_list")

        # save_pickle_object(
        #     self.savedir,
        #     rate_dependent_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_list,
        #     data_file_prefix+"-rate_dependent_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_list")
        # save_pickle_object(
        #     self.savedir, rate_dependent_AFM_exprmts_g_c_crit__nu_chunk_list,
        #     data_file_prefix+"-rate_dependent_AFM_exprmts_g_c_crit__nu_chunk_list")
        # save_pickle_object(
        #     self.savedir,
        #     rate_dependent_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_list,
        #     data_file_prefix+"-rate_dependent_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_list")
        # save_pickle_object(
        #     self.savedir,
        #     rate_dependent_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list,
        #     data_file_prefix+"-rate_dependent_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list")
        # save_pickle_object(
        #     self.savedir,
        #     rate_dependent_AFM_exprmts_overline_g_c_crit__nu_chunk_list,
        #     data_file_prefix+"-rate_dependent_AFM_exprmts_overline_g_c_crit__nu_chunk_list")
        # save_pickle_object(
        #     self.savedir,
        #     rate_dependent_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_list,
        #     data_file_prefix+"-rate_dependent_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_list")
        
        

    def finalization(self):
        """Define finalization analysis"""
        cp  = self.parameters.characterizer
        ppp = self.parameters.post_processing

        k_B     = constants.value(u'Boltzmann constant') # J/K
        h       = constants.value(u'Planck constant') # J/Hz
        hbar    = h / (2*np.pi) # J*sec
        beta    = 1. / (k_B*self.T) # 1/J

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

        zeta_nu_char = np.loadtxt(
            cp.chain_data_directory+data_file_prefix+'-composite-uFJC-curve-fit-zeta_nu_char_intgr_nu'+'.txt')
        
        typcl_AFM_exprmt_f_c_max = (
            cp.chain_backbone_bond_type2typcl_AFM_exprmt_f_c_max_dict[chain_backbone_bond_type]
        ) # nN
        typcl_AFM_exprmt_f_c_max_label = (
            r'$\textrm{Wang et al. (2019)},~f_c^{max,DFT}='+'{0:.2f}'.format(typcl_AFM_exprmt_f_c_max)+'~nN$'
        )

        if chain_backbone_bond_type == "c-c":
            intrmdt_AFM_exprmt_f_c_max = 4.5 # nN
            intrmdt_AFM_exprmt_f_c_max_label = (
                r'$\textrm{Wang et al. (2019)},~f_c^{max}='+'{0:.2f}'.format(intrmdt_AFM_exprmt_f_c_max)+'~nN$'
            )
        
        LT_label = r'$u\textrm{FJC Lake and Thomas (1967)}$'
        LT_inext_gaussian_label = (
            r'$\textrm{IGC Lake and Thomas (1967)}$'
        )
        ufjc_label = r'$\textrm{composite}~u\textrm{FJC scission}$'
        
        A_nu_list = load_pickle_object(
            self.savedir, data_file_prefix+"-A_nu_list")
        inext_gaussian_A_nu_list = load_pickle_object(
            self.savedir, data_file_prefix+"-inext_gaussian_A_nu_list")
        inext_gaussian_A_nu_err_list = load_pickle_object(
            self.savedir, data_file_prefix+"-inext_gaussian_A_nu_err_list")

        rate_independent_epsilon_cnu_diss_hat_crit_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-rate_independent_epsilon_cnu_diss_hat_crit_list")
        rate_independent_g_c_crit_list = load_pickle_object(
            self.savedir, data_file_prefix+"-rate_independent_g_c_crit_list")
        rate_independent_g_c_crit__nu_squared_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-rate_independent_g_c_crit__nu_squared_list")
        rate_independent_overline_epsilon_cnu_diss_hat_crit_list = (
            load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_overline_epsilon_cnu_diss_hat_crit_list")
        )
        rate_independent_overline_g_c_crit_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-rate_independent_overline_g_c_crit_list")
        rate_independent_overline_g_c_crit__nu_squared_list = (
            load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_overline_g_c_crit__nu_squared_list")
        )

        rate_independent_LT_epsilon_cnu_diss_hat_crit_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-rate_independent_LT_epsilon_cnu_diss_hat_crit_list")
        rate_independent_LT_g_c_crit_list = load_pickle_object(
            self.savedir, data_file_prefix+"-rate_independent_LT_g_c_crit_list")
        rate_independent_LT_g_c_crit__nu_squared_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-rate_independent_LT_g_c_crit__nu_squared_list")
        rate_independent_LT_overline_epsilon_cnu_diss_hat_crit_list = (
            load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_LT_overline_epsilon_cnu_diss_hat_crit_list")
        )
        rate_independent_LT_overline_g_c_crit_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-rate_independent_LT_overline_g_c_crit_list")
        rate_independent_LT_overline_g_c_crit__nu_squared_list = (
            load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_LT_overline_g_c_crit__nu_squared_list")
        )

        rate_independent_LT_inext_gaussian_g_c_crit_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-rate_independent_LT_inext_gaussian_g_c_crit_list")
        rate_independent_LT_inext_gaussian_g_c_crit__nu_squared_list = (
            load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_LT_inext_gaussian_g_c_crit__nu_squared_list")
        )
        rate_independent_LT_inext_gaussian_overline_g_c_crit_list = (
            load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_LT_inext_gaussian_overline_g_c_crit_list")
        )
        rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_squared_list = (
            load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_squared_list")
        )

        rate_independent_CR_typcl_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list = (
            load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_CR_typcl_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list")
        )
        rate_independent_CR_typcl_AFM_exprmt_xi_c_max_g_c_crit_list = (
            load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_CR_typcl_AFM_exprmt_xi_c_max_g_c_crit_list")
        )
        rate_independent_CR_typcl_AFM_exprmt_xi_c_max_g_c_crit__nu_squared_list = (
            load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_CR_typcl_AFM_exprmt_xi_c_max_g_c_crit__nu_squared_list")
        )
        rate_independent_CR_typcl_AFM_exprmt_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list = (
            load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_CR_typcl_AFM_exprmt_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list")
        )
        rate_independent_CR_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit_list = (
            load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_CR_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit_list")
        )
        rate_independent_CR_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit__nu_squared_list = (
            load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_independent_CR_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit__nu_squared_list")
        )
        
        if chain_backbone_bond_type == "c-c":
            rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list")
            )
            rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_g_c_crit_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_g_c_crit_list")
            )
            rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_g_c_crit__nu_squared_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_g_c_crit__nu_squared_list")
            )
            rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list")
            )
            rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_overline_g_c_crit_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_overline_g_c_crit_list")
            )
            rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_overline_g_c_crit__nu_squared_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_overline_g_c_crit__nu_squared_list")
            )


        # plot results
        latex_formatting_figure(ppp)

        fig, (ax1, ax2) = plt.subplots(
            2, 1, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
        
        ax1.semilogx(
            cp.nu_list, A_nu_list, linestyle='-',
            color='blue', alpha=1, linewidth=2.5,
            label=r'$u\textrm{FJC}$')
        ax1.semilogx(
            cp.nu_list, inext_gaussian_A_nu_list, linestyle='--',
            color='red', alpha=1, linewidth=2.5,
            label=r'$\textrm{inextensible Gaussian chain (IGC)}$')
        ax1.legend(loc='best', fontsize=14)
        ax1.tick_params(axis='y', labelsize=14)
        ax1.set_ylabel(r'$\mathcal{A}_{\nu}$', fontsize=20)
        ax1.grid(True, alpha=0.25)
        
        ax2.loglog(
            cp.nu_list, inext_gaussian_A_nu_err_list,
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
            cp.nu_list, rate_independent_LT_epsilon_cnu_diss_hat_crit_list,
            linestyle='-', color='red', alpha=1, linewidth=2.5,
            label=LT_label)
        plt.semilogx(
            cp.nu_list,
            rate_independent_CR_typcl_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list,
            linestyle=':', color='black', alpha=1, linewidth=2.5,
            label=typcl_AFM_exprmt_f_c_max_label)
        # if chain_backbone_bond_type == "c-c":
        #     plt.semilogx(
        #         cp.nu_list,
        #         rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list,
        #         linestyle='-.', color='black', alpha=1, linewidth=2.5,
        #         label=intrmdt_AFM_exprmt_f_c_max_label)
        plt.semilogx(
            cp.nu_list, rate_independent_epsilon_cnu_diss_hat_crit_list,
            linestyle='-', color='blue', alpha=1, linewidth=2.5,
            label=ufjc_label)
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
        plt.semilogx(
            cp.nu_list, rate_independent_LT_overline_epsilon_cnu_diss_hat_crit_list,
            linestyle='-', color='red', alpha=1, linewidth=2.5,
            label=LT_label)
        plt.semilogx(
            cp.nu_list,
            rate_independent_CR_typcl_AFM_exprmt_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list,
            linestyle=':', color='black', alpha=1, linewidth=2.5,
            label=typcl_AFM_exprmt_f_c_max_label)
        # if chain_backbone_bond_type == "c-c":
        #     plt.semilogx(
        #         cp.nu_list,
        #         rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list,
        #         linestyle='-.', color='black', alpha=1, linewidth=2.5,
        #         label=intrmdt_AFM_exprmt_f_c_max_label)
        plt.semilogx(
            cp.nu_list, rate_independent_overline_epsilon_cnu_diss_hat_crit_list,
            linestyle='-', color='blue', alpha=1, linewidth=2.5,
            label=ufjc_label)
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
            cp.nu_list, rate_independent_LT_g_c_crit_list,
            linestyle='-', color='red', alpha=1, linewidth=2.5,
            label=LT_label)
        plt.loglog(
            cp.nu_list, rate_independent_LT_inext_gaussian_g_c_crit_list,
            linestyle='--', color='red', alpha=1, linewidth=2.5,
            label=LT_inext_gaussian_label)
        plt.loglog(
            cp.nu_list,
            rate_independent_CR_typcl_AFM_exprmt_xi_c_max_g_c_crit_list,
            linestyle=':', color='black', alpha=1, linewidth=2.5,
            label=typcl_AFM_exprmt_f_c_max_label)
        # if chain_backbone_bond_type == "c-c":
        #     plt.loglog(
        #         cp.nu_list,
        #         rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_g_c_crit_list,
        #         linestyle='-.', color='black', alpha=1, linewidth=2.5,
        #         label=intrmdt_AFM_exprmt_f_c_max_label)
        plt.loglog(
            cp.nu_list, rate_independent_g_c_crit_list,
            linestyle='-', color='blue', alpha=1, linewidth=2.5,
            label=ufjc_label)
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
            cp.nu_list, rate_independent_LT_g_c_crit__nu_squared_list,
            linestyle='-', color='red', alpha=1, linewidth=2.5,
            label=LT_label)
        plt.loglog(
            cp.nu_list, rate_independent_LT_inext_gaussian_g_c_crit__nu_squared_list,
            linestyle='--', color='red', alpha=1, linewidth=2.5,
            label=LT_inext_gaussian_label)
        plt.loglog(
            cp.nu_list,
            rate_independent_CR_typcl_AFM_exprmt_xi_c_max_g_c_crit__nu_squared_list,
            linestyle=':', color='black', alpha=1, linewidth=2.5,
            label=typcl_AFM_exprmt_f_c_max_label)
        # if chain_backbone_bond_type == "c-c":
        #     plt.loglog(
        #         cp.nu_list,
        #         rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_g_c_crit__nu_squared_list,
        #         linestyle='-.', color='black', alpha=1, linewidth=2.5,
        #         label=intrmdt_AFM_exprmt_f_c_max_label)
        plt.loglog(
            cp.nu_list, rate_independent_g_c_crit__nu_squared_list,
            linestyle='-', color='blue', alpha=1, linewidth=2.5,
            label=ufjc_label)
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
            cp.nu_list, rate_independent_LT_overline_g_c_crit_list,
            linestyle='-', color='red', alpha=1, linewidth=2.5,
            label=LT_label)
        plt.loglog(
            cp.nu_list, rate_independent_LT_inext_gaussian_overline_g_c_crit_list,
            linestyle='--', color='red', alpha=1, linewidth=2.5,
            label=LT_inext_gaussian_label)
        plt.loglog(
            cp.nu_list,
            rate_independent_CR_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit_list,
            linestyle=':', color='black', alpha=1, linewidth=2.5,
            label=typcl_AFM_exprmt_f_c_max_label)
        # if chain_backbone_bond_type == "c-c":
        #     plt.loglog(
        #         cp.nu_list,
        #         rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_overline_g_c_crit_list,
        #         linestyle='-.', color='black', alpha=1, linewidth=2.5,
        #         label=intrmdt_AFM_exprmt_f_c_max_label)
        plt.loglog(
            cp.nu_list, rate_independent_overline_g_c_crit_list,
            linestyle='-', color='blue', alpha=1, linewidth=2.5,
            label=ufjc_label)
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
            cp.nu_list, rate_independent_LT_overline_g_c_crit__nu_squared_list,
            linestyle='-', color='red', alpha=1, linewidth=2.5,
            label=LT_label)
        plt.loglog(
            cp.nu_list, rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_squared_list,
            linestyle='--', color='red', alpha=1, linewidth=2.5,
            label=LT_inext_gaussian_label)
        plt.loglog(
            cp.nu_list,
            rate_independent_CR_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit__nu_squared_list,
            linestyle=':', color='black', alpha=1, linewidth=2.5,
            label=typcl_AFM_exprmt_f_c_max_label)
        # if chain_backbone_bond_type == "c-c":
        #     plt.loglog(
        #         cp.nu_list,
        #         rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_overline_g_c_crit__nu_squared_list,
        #         linestyle='-.', color='black', alpha=1, linewidth=2.5,
        #         label=intrmdt_AFM_exprmt_f_c_max_label)
        plt.loglog(
            cp.nu_list, rate_independent_overline_g_c_crit__nu_squared_list,
            linestyle='-', color='blue', alpha=1, linewidth=2.5,
            label=ufjc_label)
        plt.legend(loc='best', fontsize=10)
        # plt.ylim([-0.05, 1.025])
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\nu$', 20,
            r'$\beta \overline{G_c}/(\eta^{ref}l_{\nu}^{eq}\nu^2)$', 20,
            data_file_prefix+"-rate-independent-nondimensional-scaled-fracture-toughness-nu-squared-normalized-vs-nu")

        rate_dependent_tilde_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-rate_dependent_tilde_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list"
        )
        rate_dependent_tilde_xi_c_dot_g_c_crit__nu_chunk_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-rate_dependent_tilde_xi_c_dot_g_c_crit__nu_chunk_list"
        )
        rate_dependent_tilde_xi_c_dot_g_c_crit__nu_squared__nu_chunk_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-rate_dependent_tilde_xi_c_dot_g_c_crit__nu_squared__nu_chunk_list"
        )
        rate_dependent_tilde_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list = (
            load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_tilde_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list")
        )
        rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_chunk_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_chunk_list"
        )
        rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-rate_dependent_tilde_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_list"
        )
        
        rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list"
        )
        rate_dependent_check_xi_c_dot_g_c_crit__nu_chunk_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-rate_dependent_check_xi_c_dot_g_c_crit__nu_chunk_list"
        )
        rate_dependent_check_xi_c_dot_g_c_crit__nu_squared__nu_chunk_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-rate_dependent_check_xi_c_dot_g_c_crit__nu_squared__nu_chunk_list"
        )
        rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list = (
            load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_check_xi_c_dot_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list")
        )
        rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_chunk_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_chunk_list"
        )
        rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-rate_dependent_check_xi_c_dot_overline_g_c_crit__nu_squared__nu_chunk_list"
        )
        
        rate_dependent_nu_fit_epsilon_cnu_diss_hat_crit_list = (
            load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_nu_fit_epsilon_cnu_diss_hat_crit_list")
        )
        rate_dependent_nu_fit_g_c_crit_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-rate_dependent_nu_fit_g_c_crit_list")
        rate_dependent_nu_fit_g_c_crit__nu_squared_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-rate_dependent_nu_fit_g_c_crit__nu_squared_list")
        rate_dependent_nu_fit_overline_epsilon_cnu_diss_hat_crit_list = (
            load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_nu_fit_overline_epsilon_cnu_diss_hat_crit_list")
        )
        rate_dependent_nu_fit_overline_g_c_crit_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-rate_dependent_nu_fit_overline_g_c_crit_list")
        rate_dependent_nu_fit_overline_g_c_crit__nu_squared_list = (
            load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_nu_fit_overline_g_c_crit__nu_squared_list")
        )
        
        rate_dependent_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_list = (
            load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_list")
        )
        rate_dependent_AFM_exprmts_g_c_crit__nu_chunk_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-rate_dependent_AFM_exprmts_g_c_crit__nu_chunk_list")
        rate_dependent_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_list = (
            load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_list")
        )
        rate_dependent_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list = (
            load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list")
        )
        rate_dependent_AFM_exprmts_overline_g_c_crit__nu_chunk_list = (
            load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_AFM_exprmts_overline_g_c_crit__nu_chunk_list")
        )
        rate_dependent_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_list = (
            load_pickle_object(
                self.savedir,
                data_file_prefix+"-rate_dependent_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_list")
        )

        # check_xi_c_dot_indx_list = [0, 25, 50, 75, 100]
        # check_xi_c_dot_indx_list = check_xi_c_dot_indx_list[::-1]

        # check_xi_c_dot_significand_list = [
        #     cp.check_xi_c_dot_list[check_xi_c_dot_indx_list[i]]/10**int(floor(log10(abs(cp.check_xi_c_dot_list[check_xi_c_dot_indx_list[i]]))))
        #     for i in range(len(check_xi_c_dot_indx_list))
        # ]
        # check_xi_c_dot_exponent_list = [
        #     int(floor(log10(abs(cp.check_xi_c_dot_list[check_xi_c_dot_indx_list[i]]))))
        #     for i in range(len(check_xi_c_dot_indx_list))
        # ]
        # check_xi_c_dot_label_list = [
        #     r'$\textrm{composite}~u\textrm{FJC scission},~\check{\dot{\xi}}_c='+'{0:.2f}'.format(check_xi_c_dot_significand_list[i])+'*'+'10^{0:d}'.format(check_xi_c_dot_exponent_list[i])+'$'
        #     for i in range(len(check_xi_c_dot_indx_list))
        # ]
        # check_xi_c_dot_color_list = ['cyan', 'orange', 'purple', 'green', 'brown']


        fig = plt.figure()
        plt.semilogx(
            cp.nu_list, rate_independent_LT_epsilon_cnu_diss_hat_crit_list,
            linestyle='-', color='red', alpha=1, linewidth=2.5,
            label=LT_label)
        plt.semilogx(
            cp.nu_list,
            rate_independent_CR_typcl_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list,
            linestyle=':', color='black', alpha=1, linewidth=2.5,
            label=typcl_AFM_exprmt_f_c_max_label)
        # if chain_backbone_bond_type == "c-c":
        #     plt.semilogx(
        #         cp.nu_list,
        #         rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list,
        #         linestyle='-.', color='black', alpha=1, linewidth=2.5,
        #         label=intrmdt_AFM_exprmt_f_c_max_label)
        plt.semilogx(
            cp.nu_list, rate_independent_epsilon_cnu_diss_hat_crit_list,
            linestyle='-', color='blue', alpha=1, linewidth=2.5,
            label=ufjc_label)
        for f_c_dot_indx in range(len(cp.f_c_dot_list)):
            rate_dependent_AFM_exprmts_epsilon_cnu_diss_hat_crit = [
                rate_dependent_AFM_exprmts_epsilon_cnu_diss_hat_crit__nu_chunk_list[i][f_c_dot_indx]
                for i in range(len(cp.nu_list))
            ]
            plt.semilogx(
                cp.nu_list,
                rate_dependent_AFM_exprmts_epsilon_cnu_diss_hat_crit,
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
            data_file_prefix+"-rate-independent-and-rate-dependent-nondimensional-dissipated-chain-scission-energy-per-segment-vs-nu")
        
        fig = plt.figure()
        plt.semilogx(
            cp.nu_list, rate_independent_LT_overline_epsilon_cnu_diss_hat_crit_list,
            linestyle='-', color='red', alpha=1, linewidth=2.5,
            label=LT_label)
        plt.semilogx(
            cp.nu_list,
            rate_independent_CR_typcl_AFM_exprmt_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list,
            linestyle=':', color='black', alpha=1, linewidth=2.5,
            label=typcl_AFM_exprmt_f_c_max_label)
        # if chain_backbone_bond_type == "c-c":
        #     plt.semilogx(
        #         cp.nu_list,
        #         rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list,
        #         linestyle='-.', color='black', alpha=1, linewidth=2.5,
        #         label=intrmdt_AFM_exprmt_f_c_max_label)
        plt.semilogx(
            cp.nu_list, rate_independent_overline_epsilon_cnu_diss_hat_crit_list,
            linestyle='-', color='blue', alpha=1, linewidth=2.5,
            label=ufjc_label)
        for f_c_dot_indx in range(len(cp.f_c_dot_list)):
            rate_dependent_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit = [
                rate_dependent_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit__nu_chunk_list[i][f_c_dot_indx]
                for i in range(len(cp.nu_list))
            ]
            plt.semilogx(
                cp.nu_list,
                rate_dependent_AFM_exprmts_overline_epsilon_cnu_diss_hat_crit,
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
            data_file_prefix+"-rate-independent-and-rate-dependent-nondimensional-scaled-dissipated-chain-scission-energy-per-segment-vs-nu")
        
        fig = plt.figure()
        plt.loglog(
            cp.nu_list, rate_independent_LT_g_c_crit_list,
            linestyle='-', color='red', alpha=1, linewidth=2.5,
            label=LT_label)
        plt.loglog(
            cp.nu_list, rate_independent_LT_inext_gaussian_g_c_crit_list,
            linestyle='--', color='red', alpha=1, linewidth=2.5,
            label=LT_inext_gaussian_label)
        plt.loglog(
            cp.nu_list,
            rate_independent_CR_typcl_AFM_exprmt_xi_c_max_g_c_crit_list,
            linestyle=':', color='black', alpha=1, linewidth=2.5,
            label=typcl_AFM_exprmt_f_c_max_label)
        # if chain_backbone_bond_type == "c-c":
        #     plt.loglog(
        #         cp.nu_list,
        #         rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_g_c_crit_list,
        #         linestyle='-.', color='black', alpha=1, linewidth=2.5,
        #         label=intrmdt_AFM_exprmt_f_c_max_label)
        plt.loglog(
            cp.nu_list, rate_independent_g_c_crit_list,
            linestyle='-', color='blue', alpha=1, linewidth=2.5,
            label=ufjc_label)
        for f_c_dot_indx in range(len(cp.f_c_dot_list)):
            rate_dependent_AFM_exprmts_g_c_crit = [
                rate_dependent_AFM_exprmts_g_c_crit__nu_chunk_list[i][f_c_dot_indx]
                for i in range(len(cp.nu_list))
            ]
            plt.semilogx(
                cp.nu_list, rate_dependent_AFM_exprmts_g_c_crit,
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
            data_file_prefix+"-rate-independent-and-rate-dependent-nondimensional-fracture-toughness-vs-nu")
        
        fig = plt.figure()
        plt.loglog(
            cp.nu_list, rate_independent_LT_g_c_crit__nu_squared_list,
            linestyle='-', color='red', alpha=1, linewidth=2.5,
            label=LT_label)
        plt.loglog(
            cp.nu_list, rate_independent_LT_inext_gaussian_g_c_crit__nu_squared_list,
            linestyle='--', color='red', alpha=1, linewidth=2.5,
            label=LT_inext_gaussian_label)
        plt.loglog(
            cp.nu_list,
            rate_independent_CR_typcl_AFM_exprmt_xi_c_max_g_c_crit__nu_squared_list,
            linestyle=':', color='black', alpha=1, linewidth=2.5,
            label=typcl_AFM_exprmt_f_c_max_label)
        # if chain_backbone_bond_type == "c-c":
        #     plt.loglog(
        #         cp.nu_list,
        #         rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_g_c_crit__nu_squared_list,
        #         linestyle='-.', color='black', alpha=1, linewidth=2.5,
        #         label=intrmdt_AFM_exprmt_f_c_max_label)
        plt.loglog(
            cp.nu_list, rate_independent_g_c_crit__nu_squared_list,
            linestyle='-', color='blue', alpha=1, linewidth=2.5,
            label=ufjc_label)
        for f_c_dot_indx in range(len(cp.f_c_dot_list)):
            rate_dependent_AFM_exprmts_g_c_crit__nu_squared = [
                rate_dependent_AFM_exprmts_g_c_crit__nu_squared__nu_chunk_list[i][f_c_dot_indx]
                for i in range(len(cp.nu_list))
            ]
            plt.semilogx(
                cp.nu_list, rate_dependent_AFM_exprmts_g_c_crit__nu_squared,
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
            data_file_prefix+"-rate-independent-and-rate-dependent-nondimensional-fracture-toughness-nu-squared-normalized-vs-nu")
        
        fig = plt.figure()
        plt.loglog(
            cp.nu_list, rate_independent_LT_overline_g_c_crit_list,
            linestyle='-', color='red', alpha=1, linewidth=2.5,
            label=LT_label)
        plt.loglog(
            cp.nu_list, rate_independent_LT_inext_gaussian_overline_g_c_crit_list,
            linestyle='--', color='red', alpha=1, linewidth=2.5,
            label=LT_inext_gaussian_label)
        plt.loglog(
            cp.nu_list,
            rate_independent_CR_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit_list,
            linestyle=':', color='black', alpha=1, linewidth=2.5,
            label=typcl_AFM_exprmt_f_c_max_label)
        # if chain_backbone_bond_type == "c-c":
        #     plt.loglog(
        #         cp.nu_list,
        #         rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_overline_g_c_crit_list,
        #         linestyle='-.', color='black', alpha=1, linewidth=2.5,
        #         label=intrmdt_AFM_exprmt_f_c_max_label)
        plt.loglog(
            cp.nu_list, rate_independent_overline_g_c_crit_list,
            linestyle='-', color='blue', alpha=1, linewidth=2.5,
            label=ufjc_label)
        for f_c_dot_indx in range(len(cp.f_c_dot_list)):
            rate_dependent_AFM_exprmts_overline_g_c_crit = [
                rate_dependent_AFM_exprmts_overline_g_c_crit__nu_chunk_list[i][f_c_dot_indx]
                for i in range(len(cp.nu_list))
            ]
            plt.semilogx(
                cp.nu_list, rate_dependent_AFM_exprmts_overline_g_c_crit,
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
            data_file_prefix+"-rate-independent-and-rate-dependent-nondimensional-scaled-fracture-toughness-vs-nu")
        
        fig = plt.figure()
        plt.loglog(
            cp.nu_list, rate_independent_LT_overline_g_c_crit__nu_squared_list,
            linestyle='-', color='red', alpha=1, linewidth=2.5,
            label=LT_label)
        plt.loglog(
            cp.nu_list, rate_independent_LT_inext_gaussian_overline_g_c_crit__nu_squared_list,
            linestyle='--', color='red', alpha=1, linewidth=2.5,
            label=LT_inext_gaussian_label)
        plt.loglog(
            cp.nu_list,
            rate_independent_CR_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit__nu_squared_list,
            linestyle=':', color='black', alpha=1, linewidth=2.5,
            label=typcl_AFM_exprmt_f_c_max_label)
        # if chain_backbone_bond_type == "c-c":
        #     plt.loglog(
        #         cp.nu_list,
        #         rate_independent_CR_intrmdt_AFM_exprmt_xi_c_max_overline_g_c_crit__nu_squared_list,
        #         linestyle='-.', color='black', alpha=1, linewidth=2.5,
        #         label=intrmdt_AFM_exprmt_f_c_max_label)
        plt.loglog(
            cp.nu_list, rate_independent_overline_g_c_crit__nu_squared_list,
            linestyle='-', color='blue', alpha=1, linewidth=2.5,
            label=ufjc_label)
        for f_c_dot_indx in range(len(cp.f_c_dot_list)):
            rate_dependent_AFM_exprmts_overline_g_c_crit__nu_squared = [
                rate_dependent_AFM_exprmts_overline_g_c_crit__nu_squared__nu_chunk_list[i][f_c_dot_indx]
                for i in range(len(cp.nu_list))
            ]
            plt.semilogx(
                cp.nu_list,
                rate_dependent_AFM_exprmts_overline_g_c_crit__nu_squared,
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
            data_file_prefix+"-rate-independent-and-rate-dependent-nondimensional-scaled-fracture-toughness-nu-squared-normalized-vs-nu")

        nu_list_meshgrid, check_xi_c_dot_list_meshgrid = np.meshgrid(
            cp.nu_list, cp.check_xi_c_dot_list)

        rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit_list = np.transpose(
            np.asarray(rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list))

        fig, ax1 = plt.subplots()

        filled_contour_plot = ax1.contourf(
            nu_list_meshgrid, check_xi_c_dot_list_meshgrid,
            rate_dependent_check_xi_c_dot_epsilon_cnu_diss_hat_crit_list, 100,
            cmap=plt.cm.cividis)
        
        for fcp in filled_contour_plot.collections:
            fcp.set_edgecolor('face')
        
        ax1.set_xlabel(r'$\nu$', fontsize=20)
        ax1.set_ylabel(r'$\check{\dot{\xi}}_c$', fontsize=20)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        # ax1.tick_params(axis='both', labelsize=16)

        cbar = fig.colorbar(filled_contour_plot)
        cbar.ax.set_ylabel(r'$\hat{\varepsilon}_{c\nu}^{diss}$', fontsize=20)
        cbar.ax.tick_params(axis='y', labelsize=14)
        
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        
        save_current_figure_no_labels(
            self.savedir,
            data_file_prefix+"-nondimensional-dissipated-chain-scission-energy-per-segment-filled-contour-check_xi_c_dot-vs-nu")
        

        nu_list_meshgrid, tilde_xi_c_dot_list_meshgrid = np.meshgrid(
            cp.nu_list, cp.tilde_xi_c_dot_list)

        rate_dependent_tilde_xi_c_dot_epsilon_cnu_diss_hat_crit_list = np.transpose(
            np.asarray(rate_dependent_tilde_xi_c_dot_epsilon_cnu_diss_hat_crit__nu_chunk_list))

        fig, ax1 = plt.subplots()

        filled_contour_plot = ax1.contourf(
            nu_list_meshgrid, tilde_xi_c_dot_list_meshgrid,
            rate_dependent_tilde_xi_c_dot_epsilon_cnu_diss_hat_crit_list, 100,
            cmap=plt.cm.cividis)
        
        for fcp in filled_contour_plot.collections:
            fcp.set_edgecolor('face')
        
        ax1.set_xlabel(r'$\nu$', fontsize=20)
        ax1.set_ylabel(r'$\tilde{\dot{\xi}}_c$', fontsize=20)
        ax1.set_ylim(1e-40, 1e1)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        # ax1.tick_params(axis='both', labelsize=16)

        cbar = fig.colorbar(filled_contour_plot)
        cbar.ax.set_ylabel(r'$\hat{\varepsilon}_{c\nu}^{diss}$', fontsize=20)
        cbar.ax.tick_params(axis='y', labelsize=14)
        
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        
        save_current_figure_no_labels(
            self.savedir,
            data_file_prefix+"-nondimensional-dissipated-chain-scission-energy-per-segment-filled-contour-tilde_xi_c_dot-vs-nu")

if __name__ == '__main__':

    T = 298 # absolute room temperature, K

    AFM_chain_tensile_tests_dict = {
        "al-maawali-et-al": "chain-a", "hugel-et-al": "chain-a"
    }

    al_maawali_et_al_fracture_toughness_characterizer = (
        FractureToughnessCharacterizer(
            paper_authors="al-maawali-et-al", chain="chain-a", T=T)
    )
    # al_maawali_et_al_fracture_toughness_characterizer.characterization()
    al_maawali_et_al_fracture_toughness_characterizer.finalization()

    hugel_et_al_fracture_toughness_characterizer = (
        FractureToughnessCharacterizer(
            paper_authors="hugel-et-al", chain="chain-a", T=T)
    )
    # hugel_et_al_fracture_toughness_characterizer.characterization()
    hugel_et_al_fracture_toughness_characterizer.finalization()