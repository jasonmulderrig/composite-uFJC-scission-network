"""The single-chain rate-dependent fracture toughness characterization
module for composite uFJCs that undergo scission
"""

# import external modules
from __future__ import division
from composite_ufjc_scission import (CompositeuFJCScissionCharacterizer,
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


class RateDependentFractureToughnessCharacterizer(
        CompositeuFJCScissionCharacterizer):
    """The characterization class assessing rate-dependent fracture
    toughness for composite uFJCs that undergo scission. It inherits all
    attributes and methods from the
    ``CompositeuFJCScissionCharacterizer`` class.
    """
    def __init__(self, paper_authors, chain, T):
        """Initializes the
        ``RateDependentFractureToughnessCharacterizer`` class by
        initializing and inheriting all attributes and methods from the
        ``CompositeuFJCScissionCharacterizer`` class.
        """
        self.paper_authors = paper_authors
        self.chain = chain
        self.T = T

        CompositeuFJCScissionCharacterizer.__init__(self)
    
    def set_user_parameters(self):
        """Set user-defined parameters"""
        k_B     = constants.value(u'Boltzmann constant') # J/K
        h       = constants.value(u'Planck constant') # J/Hz
        hbar    = h / (2*np.pi) # J*sec
        beta    = 1. / (k_B*self.T) # 1/J
        omega_0 = 1. / (beta*hbar) # J/(J*sec) = 1/sec

        beta = beta / (1e9*1e9) # 1/J = 1/(N*m) -> 1/(nN*m) -> 1/(nN*nm)

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
        # (c-c) from the CRC Handbook of Chemistry and Physics for
        # CH_3-C_2H_5 citing Luo, Y.R., Comprehensive Handbook of
        # Chemical Bond Energies, CRC Press, 2007
        # (si-o) from Schwaderer et al., Langmuir, 2008, citing Holleman
        # and Wilberg, Inorganic Chemistry, 2001
        chain_backbone_bond_type2epsilon_b_char_dict = {
            "c-c": 370.3,
            "si-o": 444
        } # kJ/mol
        p.characterizer.chain_backbone_bond_type2zeta_b_char_dict = {
            chain_backbone_bond_type_key: epsilon_b_char_val/N_A*1000*beta
            for chain_backbone_bond_type_key, epsilon_b_char_val
            in chain_backbone_bond_type2epsilon_b_char_dict.items()
        } # (kJ/mol -> kJ -> J)*1/J
        # (c-c) from the CRC Handbook of Chemistry and Physics for the
        # C-C bond in C#-H_2C-CH_2-C#
        # (si-o) from the CRC Handbook of Chemistry and Physics for the
        # Si-O bond in X_3-Si-O-C#
        chain_backbone_bond_type2l_b_eq_dict = {
            "c-c": 1.524,
            "si-o": 1.645
        } # Angstroms
        p.characterizer.chain_backbone_bond_type2l_b_eq_dict = {
            chain_backbone_bond_type_key: l_b_eq_val/10
            for chain_backbone_bond_type_key, l_b_eq_val
            in chain_backbone_bond_type2l_b_eq_dict.items()
        } # Angstroms -> nm

        p.characterizer.f_c_num_steps = 100001

        f_c_dot_list = [1e1, 1e5, 1e9] # nN/sec
        f_c_dot_exponent_list = [
            int(floor(log10(abs(f_c_dot_list[i]))))
            for i in range(len(f_c_dot_list))
        ]
        f_c_dot_label_list = [
            r'$\dot{f}_c='+'10^{0:d}'.format(f_c_dot_exponent_list[i])+'~nN/sec$'
            for i in range(len(f_c_dot_list))
        ]
        f_c_dot_color_list = ['orange', 'purple', 'green']

        p.characterizer.f_c_dot_list          = f_c_dot_list
        p.characterizer.f_c_dot_exponent_list = f_c_dot_exponent_list
        p.characterizer.f_c_dot_label_list    = f_c_dot_label_list
        p.characterizer.f_c_dot_color_list    = f_c_dot_color_list

        # nu = 1 -> nu = 5000
        nu_list = [i for i in range(1, 5001)]

        p.characterizer.nu_list = nu_list

    def prefix(self):
        """Set characterization prefix"""
        return "rate_independent_fracture_toughness"
    
    def characterization(self):
        """Define characterization routine"""
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
        
        
        single_chain_list = [
            RateIndependentScissionCompositeuFJC(
                nu=nu_val, zeta_nu_char=zeta_nu_char, kappa_nu=kappa_nu)
            for nu_val in cp.nu_list
        ]
        
        
        A_nu_list = [single_chain.A_nu for single_chain in single_chain_list]
        
        inext_gaussian_A_nu_list = [1/np.sqrt(nu_val) for nu_val in cp.nu_list]
        
        inext_gaussian_A_nu_err_list = [
            np.abs((inext_gaussian_A_nu_val-A_nu_val)/A_nu_val)*100
            for inext_gaussian_A_nu_val, A_nu_val
            in zip(inext_gaussian_A_nu_list, A_nu_list)
        ]
        
        
        epsilon_cnu_diss_hat_crit_list = [
            single_chain.epsilon_cnu_diss_hat_crit
            for single_chain in single_chain_list
        ]
        g_c_crit_list = [
            single_chain.g_c_crit for single_chain in single_chain_list
        ]
        g_c_crit__nu_squared_list = [
            g_c_crit_val / nu_val**2 for g_c_crit_val, nu_val
            in zip(g_c_crit_list, cp.nu_list)
        ]
        overline_epsilon_cnu_diss_hat_crit_list = [
            epsilon_cnu_diss_hat_crit_val/zeta_nu_char
            for epsilon_cnu_diss_hat_crit_val in epsilon_cnu_diss_hat_crit_list
        ]
        overline_g_c_crit_list = [
            g_c_crit_val/zeta_nu_char for g_c_crit_val in g_c_crit_list
        ]
        overline_g_c_crit__nu_squared_list = [
            overline_g_c_crit_val / nu_val**2 for overline_g_c_crit_val, nu_val
            in zip(overline_g_c_crit_list, cp.nu_list)
        ]
        
        
        LT_epsilon_cnu_diss_hat_crit_list = [zeta_nu_char] * len(cp.nu_list)
        LT_g_c_crit_list = [
            0.5 * A_nu_val * nu_val**2 * epsilon_cnu_diss_hat_crit_val
            for A_nu_val, nu_val, epsilon_cnu_diss_hat_crit_val
            in zip(
                A_nu_list, cp.nu_list,
                LT_epsilon_cnu_diss_hat_crit_list)
        ]
        LT_g_c_crit__nu_squared_list = [
            LT_g_c_crit_val / nu_val**2 for LT_g_c_crit_val, nu_val
            in zip(LT_g_c_crit_list, cp.nu_list)
        ]
        LT_overline_epsilon_cnu_diss_hat_crit_list = [1] * len(cp.nu_list)
        LT_overline_g_c_crit_list = [
            0.5 * A_nu_val * nu_val**2 * overline_epsilon_cnu_diss_hat_crit_val
            for A_nu_val, nu_val, overline_epsilon_cnu_diss_hat_crit_val
            in zip(A_nu_list, cp.nu_list,
                LT_overline_epsilon_cnu_diss_hat_crit_list)
        ]
        LT_overline_g_c_crit__nu_squared_list = [
            LT_overline_g_c_crit_val / nu_val**2
            for LT_overline_g_c_crit_val, nu_val
            in zip(LT_overline_g_c_crit_list, cp.nu_list)
        ]
        
        
        LT_inext_gaussian_g_c_crit_list = [
            0.5 * A_nu_val * nu_val**2 * epsilon_cnu_diss_hat_crit_val
            for A_nu_val, nu_val, epsilon_cnu_diss_hat_crit_val
            in zip(inext_gaussian_A_nu_list, cp.nu_list,
                LT_epsilon_cnu_diss_hat_crit_list)
        ]
        LT_inext_gaussian_g_c_crit__nu_squared_list = [
            LT_inext_gaussian_g_c_crit_val / nu_val**2
            for LT_inext_gaussian_g_c_crit_val, nu_val
            in zip(LT_inext_gaussian_g_c_crit_list, cp.nu_list)
        ]
        LT_inext_gaussian_overline_g_c_crit_list = [
            0.5 * A_nu_val * nu_val**2 * overline_epsilon_cnu_diss_hat_crit_val
            for A_nu_val, nu_val, overline_epsilon_cnu_diss_hat_crit_val
            in zip(inext_gaussian_A_nu_list, cp.nu_list,
                LT_overline_epsilon_cnu_diss_hat_crit_list)
        ]
        LT_inext_gaussian_overline_g_c_crit__nu_squared_list = [
            LT_inext_gaussian_overline_g_c_crit_val / nu_val**2
            for LT_inext_gaussian_overline_g_c_crit_val, nu_val
            in zip(LT_inext_gaussian_overline_g_c_crit_list, cp.nu_list)
        ]
        
        
        single_chain = RateIndependentScissionCompositeuFJC(
            nu=nu, zeta_nu_char=zeta_nu_char, kappa_nu=kappa_nu)
        
        
        CR_xi_c_max_epsilon_cnu_diss_hat_crit_val = (
            single_chain.epsilon_cnu_sci_hat_func(
                single_chain.lmbda_nu_xi_c_hat_func(xi_c_max))
        )
        CR_xi_c_max_epsilon_cnu_diss_hat_crit_list = (
            [CR_xi_c_max_epsilon_cnu_diss_hat_crit_val] * len(cp.nu_list)
        )
        CR_xi_c_max_g_c_crit_list = [
            0.5 * A_nu_val * nu_val**2 * epsilon_cnu_diss_hat_crit_val
            for A_nu_val, nu_val, epsilon_cnu_diss_hat_crit_val
            in zip(A_nu_list, cp.nu_list,
                CR_xi_c_max_epsilon_cnu_diss_hat_crit_list)
        ]
        CR_xi_c_max_g_c_crit__nu_squared_list = [
            CR_xi_c_max_g_c_crit_val / nu_val**2
            for CR_xi_c_max_g_c_crit_val, nu_val
            in zip(CR_xi_c_max_g_c_crit_list, cp.nu_list)
        ]
        CR_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list = [
            CR_xi_c_max_epsilon_cnu_diss_hat_crit_val/zeta_nu_char
            for CR_xi_c_max_epsilon_cnu_diss_hat_crit_val
            in CR_xi_c_max_epsilon_cnu_diss_hat_crit_list
        ]
        CR_xi_c_max_overline_g_c_crit_list = [
            0.5 * A_nu_val * nu_val**2 * overline_epsilon_cnu_diss_hat_crit_val
            for A_nu_val, nu_val, overline_epsilon_cnu_diss_hat_crit_val
            in zip(A_nu_list, cp.nu_list,
                CR_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list)
        ]
        CR_xi_c_max_overline_g_c_crit__nu_squared_list = [
            CR_xi_c_max_overline_g_c_crit_val / nu_val**2
            for CR_xi_c_max_overline_g_c_crit_val, nu_val
            in zip(CR_xi_c_max_overline_g_c_crit_list, cp.nu_list)
        ]
        
        
        CR_typcl_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_val = (
            single_chain.epsilon_cnu_sci_hat_func(
                single_chain.lmbda_nu_xi_c_hat_func(typcl_AFM_exprmt_xi_c_max))
        )
        CR_typcl_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list = (
            [CR_typcl_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_val]
            * len(cp.nu_list)
        )
        CR_typcl_AFM_exprmt_xi_c_max_g_c_crit_list = [
            0.5 * A_nu_val * nu_val**2 * epsilon_cnu_diss_hat_crit_val
            for A_nu_val, nu_val, epsilon_cnu_diss_hat_crit_val
            in zip(A_nu_list, cp.nu_list,
                CR_typcl_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list)
        ]
        CR_typcl_AFM_exprmt_xi_c_max_g_c_crit__nu_squared_list = [
            CR_typcl_AFM_exprmt_xi_c_max_g_c_crit_val / nu_val**2
            for CR_typcl_AFM_exprmt_xi_c_max_g_c_crit_val, nu_val
            in zip(CR_typcl_AFM_exprmt_xi_c_max_g_c_crit_list, cp.nu_list)
        ]
        CR_typcl_AFM_exprmt_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list = [
            CR_typcl_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_val/zeta_nu_char
            for CR_typcl_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_val
            in CR_typcl_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list
        ]
        CR_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit_list = [
            0.5 * A_nu_val * nu_val**2 * overline_epsilon_cnu_diss_hat_crit_val
            for A_nu_val, nu_val, overline_epsilon_cnu_diss_hat_crit_val
            in zip(A_nu_list, cp.nu_list,
                CR_typcl_AFM_exprmt_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list)
        ]
        CR_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit__nu_squared_list = [
            CR_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit_val / nu_val**2
            for CR_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit_val, nu_val
            in zip(CR_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit_list, cp.nu_list)
        ]

        if chain_backbone_bond_type == "c-c":
            CR_intrmdt_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_val = (
            single_chain.epsilon_cnu_sci_hat_func(
                single_chain.lmbda_nu_xi_c_hat_func(intrmdt_AFM_exprmt_xi_c_max))
            )
            CR_intrmdt_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list = (
                [CR_intrmdt_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_val]
                * len(cp.nu_list)
            )
            CR_intrmdt_AFM_exprmt_xi_c_max_g_c_crit_list = [
                0.5 * A_nu_val * nu_val**2 * epsilon_cnu_diss_hat_crit_val
                for A_nu_val, nu_val, epsilon_cnu_diss_hat_crit_val
                in zip(A_nu_list, cp.nu_list,
                    CR_intrmdt_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list)
            ]
            CR_intrmdt_AFM_exprmt_xi_c_max_g_c_crit__nu_squared_list = [
                CR_intrmdt_AFM_exprmt_xi_c_max_g_c_crit_val / nu_val**2
                for CR_intrmdt_AFM_exprmt_xi_c_max_g_c_crit_val, nu_val
                in zip(CR_intrmdt_AFM_exprmt_xi_c_max_g_c_crit_list, cp.nu_list)
            ]
            CR_intrmdt_AFM_exprmt_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list = [
                CR_intrmdt_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_val/zeta_nu_char
                for CR_intrmdt_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_val
                in CR_intrmdt_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list
            ]
            CR_intrmdt_AFM_exprmt_xi_c_max_overline_g_c_crit_list = [
                0.5 * A_nu_val * nu_val**2 * overline_epsilon_cnu_diss_hat_crit_val
                for A_nu_val, nu_val, overline_epsilon_cnu_diss_hat_crit_val
                in zip(A_nu_list, cp.nu_list,
                    CR_intrmdt_AFM_exprmt_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list)
            ]
            CR_intrmdt_AFM_exprmt_xi_c_max_overline_g_c_crit__nu_squared_list = [
                CR_intrmdt_AFM_exprmt_xi_c_max_overline_g_c_crit_val / nu_val**2
                for CR_intrmdt_AFM_exprmt_xi_c_max_overline_g_c_crit_val, nu_val
                in zip(
                    CR_intrmdt_AFM_exprmt_xi_c_max_overline_g_c_crit_list, cp.nu_list)
            ]
        
        
        save_pickle_object(
            self.savedir, A_nu_list, data_file_prefix+"-A_nu_list")
        save_pickle_object(
            self.savedir, inext_gaussian_A_nu_list,
            data_file_prefix+"-inext_gaussian_A_nu_list")
        save_pickle_object(
            self.savedir, inext_gaussian_A_nu_err_list,
            data_file_prefix+"-inext_gaussian_A_nu_err_list")
        
        save_pickle_object(
            self.savedir, epsilon_cnu_diss_hat_crit_list,
            data_file_prefix+"-epsilon_cnu_diss_hat_crit_list")
        save_pickle_object(
            self.savedir, g_c_crit_list, data_file_prefix+"-g_c_crit_list")
        save_pickle_object(
            self.savedir, g_c_crit__nu_squared_list,
            data_file_prefix+"-g_c_crit__nu_squared_list")
        save_pickle_object(
            self.savedir, overline_epsilon_cnu_diss_hat_crit_list,
            data_file_prefix+"-overline_epsilon_cnu_diss_hat_crit_list")
        save_pickle_object(
            self.savedir, overline_g_c_crit_list,
            data_file_prefix+"-overline_g_c_crit_list")
        save_pickle_object(
            self.savedir, overline_g_c_crit__nu_squared_list,
            data_file_prefix+"-overline_g_c_crit__nu_squared_list")
        
        save_pickle_object(
            self.savedir, LT_epsilon_cnu_diss_hat_crit_list,
            data_file_prefix+"-LT_epsilon_cnu_diss_hat_crit_list")
        save_pickle_object(
            self.savedir, LT_g_c_crit_list,
            data_file_prefix+"-LT_g_c_crit_list")
        save_pickle_object(
            self.savedir, LT_g_c_crit__nu_squared_list,
            data_file_prefix+"-LT_g_c_crit__nu_squared_list")
        save_pickle_object(
            self.savedir, LT_overline_epsilon_cnu_diss_hat_crit_list,
            data_file_prefix+"-LT_overline_epsilon_cnu_diss_hat_crit_list")
        save_pickle_object(
            self.savedir, LT_overline_g_c_crit_list,
            data_file_prefix+"-LT_overline_g_c_crit_list")
        save_pickle_object(
            self.savedir, LT_overline_g_c_crit__nu_squared_list,
            data_file_prefix+"-LT_overline_g_c_crit__nu_squared_list")
        
        save_pickle_object(
            self.savedir, LT_inext_gaussian_g_c_crit_list, 
        data_file_prefix+"-LT_inext_gaussian_g_c_crit_list")
        save_pickle_object(
            self.savedir, LT_inext_gaussian_g_c_crit__nu_squared_list, 
        data_file_prefix+"-LT_inext_gaussian_g_c_crit__nu_squared_list")
        save_pickle_object(
            self.savedir, LT_inext_gaussian_overline_g_c_crit_list, 
        data_file_prefix+"-LT_inext_gaussian_overline_g_c_crit_list")
        save_pickle_object(
            self.savedir, LT_inext_gaussian_overline_g_c_crit__nu_squared_list, 
        data_file_prefix+"-LT_inext_gaussian_overline_g_c_crit__nu_squared_list")
        
        save_pickle_object(
            self.savedir, CR_xi_c_max_epsilon_cnu_diss_hat_crit_list,
            data_file_prefix+"-CR_xi_c_max_epsilon_cnu_diss_hat_crit_list")
        save_pickle_object(
            self.savedir, CR_xi_c_max_g_c_crit_list,
            data_file_prefix+"-CR_xi_c_max_g_c_crit_list")
        save_pickle_object(
            self.savedir, CR_xi_c_max_g_c_crit__nu_squared_list,
            data_file_prefix+"-CR_xi_c_max_g_c_crit__nu_squared_list")
        save_pickle_object(
            self.savedir, CR_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list,
            data_file_prefix+"-CR_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list")
        save_pickle_object(
            self.savedir, CR_xi_c_max_overline_g_c_crit_list,
            data_file_prefix+"-CR_xi_c_max_overline_g_c_crit_list")
        save_pickle_object(
            self.savedir, CR_xi_c_max_overline_g_c_crit__nu_squared_list,
            data_file_prefix+"-CR_xi_c_max_overline_g_c_crit__nu_squared_list")
        
        save_pickle_object(
            self.savedir,
            CR_typcl_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list,
            data_file_prefix+"-CR_typcl_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list")
        save_pickle_object(
            self.savedir, CR_typcl_AFM_exprmt_xi_c_max_g_c_crit_list,
            data_file_prefix+"-CR_typcl_AFM_exprmt_xi_c_max_g_c_crit_list")
        save_pickle_object(
            self.savedir,
            CR_typcl_AFM_exprmt_xi_c_max_g_c_crit__nu_squared_list,
            data_file_prefix+"-CR_typcl_AFM_exprmt_xi_c_max_g_c_crit__nu_squared_list")
        save_pickle_object(
            self.savedir,
            CR_typcl_AFM_exprmt_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list,
            data_file_prefix+"-CR_typcl_AFM_exprmt_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list")
        save_pickle_object(
            self.savedir, CR_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit_list,
            data_file_prefix+"-CR_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit_list")
        save_pickle_object(
            self.savedir,
            CR_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit__nu_squared_list,
            data_file_prefix+"-CR_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit__nu_squared_list")
        
        if chain_backbone_bond_type == "c-c":
            save_pickle_object(
                self.savedir,
                CR_intrmdt_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list,
                data_file_prefix+"-CR_intrmdt_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list")
            save_pickle_object(
                self.savedir, CR_intrmdt_AFM_exprmt_xi_c_max_g_c_crit_list,
                data_file_prefix+"-CR_intrmdt_AFM_exprmt_xi_c_max_g_c_crit_list")
            save_pickle_object(
                self.savedir,
                CR_intrmdt_AFM_exprmt_xi_c_max_g_c_crit__nu_squared_list,
                data_file_prefix+"-CR_intrmdt_AFM_exprmt_xi_c_max_g_c_crit__nu_squared_list")
            save_pickle_object(
                self.savedir,
                CR_intrmdt_AFM_exprmt_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list,
                data_file_prefix+"-CR_intrmdt_AFM_exprmt_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list")
            save_pickle_object(
                self.savedir,
                CR_intrmdt_AFM_exprmt_xi_c_max_overline_g_c_crit_list,
                data_file_prefix+"-CR_intrmdt_AFM_exprmt_xi_c_max_overline_g_c_crit_list")
            save_pickle_object(
                self.savedir,
                CR_intrmdt_AFM_exprmt_xi_c_max_overline_g_c_crit__nu_squared_list,
                data_file_prefix+"-CR_intrmdt_AFM_exprmt_xi_c_max_overline_g_c_crit__nu_squared_list")

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
        
        f_c_max = (
            cp.chain_backbone_bond_type2f_c_max_dict[chain_backbone_bond_type]
        ) # nN
        f_c_max_label = r'$\textrm{Wang et al. (2019)},~f_c^{max}='+'{0:.2f}'.format(f_c_max)+'~nN$'
        typcl_AFM_exprmt_f_c_max = (
            cp.chain_backbone_bond_type2typcl_AFM_exprmt_f_c_max_dict[chain_backbone_bond_type]
        ) # nN
        typcl_AFM_exprmt_f_c_max_label = (
            r'$\textrm{Wang et al. (2019)},~f_c^{max,typical}='+'{0:.2f}'.format(typcl_AFM_exprmt_f_c_max)+'~nN$'
        )

        if chain_backbone_bond_type == "c-c":
            intrmdt_AFM_exprmt_f_c_max = 4.5 # nN
            intrmdt_AFM_exprmt_f_c_max_label = (
                r'$\textrm{Wang et al. (2019)},~f_c^{max,typical}='+'{0:.2f}'.format(intrmdt_AFM_exprmt_f_c_max)+'~nN$'
            )
        
        LT_label = r'$\textrm{Lake and Thomas (1967)}$'
        LT_inext_gaussian_label = (
            r'$\textrm{Lake and Thomas (1967) inextensible Gaussian chains}$'
        )
        ufjc_label = r'$\textrm{composite}~u\textrm{FJC scission}$'
        
        A_nu_list = load_pickle_object(
            self.savedir, data_file_prefix+"-A_nu_list")
        inext_gaussian_A_nu_list = load_pickle_object(
            self.savedir, data_file_prefix+"-inext_gaussian_A_nu_list")
        inext_gaussian_A_nu_err_list = load_pickle_object(
            self.savedir, data_file_prefix+"-inext_gaussian_A_nu_err_list")

        epsilon_cnu_diss_hat_crit_list = load_pickle_object(
            self.savedir, data_file_prefix+"-epsilon_cnu_diss_hat_crit_list")
        g_c_crit_list = load_pickle_object(
            self.savedir, data_file_prefix+"-g_c_crit_list")
        g_c_crit__nu_squared_list = load_pickle_object(
            self.savedir, data_file_prefix+"-g_c_crit__nu_squared_list")
        overline_epsilon_cnu_diss_hat_crit_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-overline_epsilon_cnu_diss_hat_crit_list")
        overline_g_c_crit_list = load_pickle_object(
            self.savedir, data_file_prefix+"-overline_g_c_crit_list")
        overline_g_c_crit__nu_squared_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-overline_g_c_crit__nu_squared_list")

        LT_epsilon_cnu_diss_hat_crit_list = load_pickle_object(
            self.savedir, data_file_prefix+"-LT_epsilon_cnu_diss_hat_crit_list")
        LT_g_c_crit_list = load_pickle_object(
            self.savedir, data_file_prefix+"-LT_g_c_crit_list")
        LT_g_c_crit__nu_squared_list = load_pickle_object(
            self.savedir, data_file_prefix+"-LT_g_c_crit__nu_squared_list")
        LT_overline_epsilon_cnu_diss_hat_crit_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-LT_overline_epsilon_cnu_diss_hat_crit_list")
        LT_overline_g_c_crit_list = load_pickle_object(
            self.savedir, data_file_prefix+"-LT_overline_g_c_crit_list")
        LT_overline_g_c_crit__nu_squared_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-LT_overline_g_c_crit__nu_squared_list")

        LT_inext_gaussian_g_c_crit_list = load_pickle_object(
            self.savedir, data_file_prefix+"-LT_inext_gaussian_g_c_crit_list")
        LT_inext_gaussian_g_c_crit__nu_squared_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-LT_inext_gaussian_g_c_crit__nu_squared_list")
        LT_inext_gaussian_overline_g_c_crit_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-LT_inext_gaussian_overline_g_c_crit_list")
        LT_inext_gaussian_overline_g_c_crit__nu_squared_list = (
            load_pickle_object(
                self.savedir,
                data_file_prefix+"-LT_inext_gaussian_overline_g_c_crit__nu_squared_list")
        )

        CR_xi_c_max_epsilon_cnu_diss_hat_crit_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-CR_xi_c_max_epsilon_cnu_diss_hat_crit_list")
        CR_xi_c_max_g_c_crit_list = load_pickle_object(
            self.savedir, data_file_prefix+"-CR_xi_c_max_g_c_crit_list")
        CR_xi_c_max_g_c_crit__nu_squared_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-CR_xi_c_max_g_c_crit__nu_squared_list")
        CR_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list = (
            load_pickle_object(
                self.savedir,
                data_file_prefix+"-CR_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list")
        )
        CR_xi_c_max_overline_g_c_crit_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-CR_xi_c_max_overline_g_c_crit_list")
        CR_xi_c_max_overline_g_c_crit__nu_squared_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-CR_xi_c_max_overline_g_c_crit__nu_squared_list")

        CR_typcl_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list = (
            load_pickle_object(
                self.savedir,
                data_file_prefix+"-CR_typcl_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list")
        )
        CR_typcl_AFM_exprmt_xi_c_max_g_c_crit_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-CR_typcl_AFM_exprmt_xi_c_max_g_c_crit_list")
        CR_typcl_AFM_exprmt_xi_c_max_g_c_crit__nu_squared_list = (
            load_pickle_object(
                self.savedir,
                data_file_prefix+"-CR_typcl_AFM_exprmt_xi_c_max_g_c_crit__nu_squared_list")
        )
        CR_typcl_AFM_exprmt_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list = (
            load_pickle_object(
                self.savedir,
                data_file_prefix+"-CR_typcl_AFM_exprmt_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list")
        )
        CR_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit_list = (
            load_pickle_object(
                self.savedir,
                data_file_prefix+"-CR_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit_list")
        )
        CR_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit__nu_squared_list = (
            load_pickle_object(
                self.savedir,
                data_file_prefix+"-CR_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit__nu_squared_list")
        )
        
        if chain_backbone_bond_type == "c-c":
            CR_intrmdt_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-CR_intrmdt_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list")
            )
            CR_intrmdt_AFM_exprmt_xi_c_max_g_c_crit_list = load_pickle_object(
                self.savedir,
                data_file_prefix+"-CR_intrmdt_AFM_exprmt_xi_c_max_g_c_crit_list")
            CR_intrmdt_AFM_exprmt_xi_c_max_g_c_crit__nu_squared_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-CR_intrmdt_AFM_exprmt_xi_c_max_g_c_crit__nu_squared_list")
            )
            CR_intrmdt_AFM_exprmt_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-CR_intrmdt_AFM_exprmt_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list")
            )
            CR_intrmdt_AFM_exprmt_xi_c_max_overline_g_c_crit_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-CR_intrmdt_AFM_exprmt_xi_c_max_overline_g_c_crit_list")
            )
            CR_intrmdt_AFM_exprmt_xi_c_max_overline_g_c_crit__nu_squared_list = (
                load_pickle_object(
                    self.savedir,
                    data_file_prefix+"-CR_intrmdt_AFM_exprmt_xi_c_max_overline_g_c_crit__nu_squared_list")
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
            label=r'$\textrm{inextensible Gaussian chain}$')
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
            cp.nu_list, LT_epsilon_cnu_diss_hat_crit_list,
            linestyle='-', color='red', alpha=1, linewidth=2.5,
            label=LT_label)
        plt.semilogx(
            cp.nu_list, CR_xi_c_max_epsilon_cnu_diss_hat_crit_list,
            linestyle='--', color='black', alpha=1, linewidth=2.5,
            label=f_c_max_label)
        plt.semilogx(
            cp.nu_list,
            CR_typcl_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list,
            linestyle=':', color='black', alpha=1, linewidth=2.5,
            label=typcl_AFM_exprmt_f_c_max_label)
        if chain_backbone_bond_type == "c-c":
            plt.semilogx(
                cp.nu_list,
                CR_intrmdt_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list,
                linestyle='-.', color='black', alpha=1, linewidth=2.5,
                label=intrmdt_AFM_exprmt_f_c_max_label)
        plt.semilogx(
            cp.nu_list, epsilon_cnu_diss_hat_crit_list,
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
            cp.nu_list, LT_overline_epsilon_cnu_diss_hat_crit_list,
            linestyle='-', color='red', alpha=1, linewidth=2.5,
            label=LT_label)
        plt.semilogx(
            cp.nu_list, CR_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list,
            linestyle='--', color='black', alpha=1, linewidth=2.5,
            label=f_c_max_label)
        plt.semilogx(
            cp.nu_list,
            CR_typcl_AFM_exprmt_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list,
            linestyle=':', color='black', alpha=1, linewidth=2.5,
            label=typcl_AFM_exprmt_f_c_max_label)
        if chain_backbone_bond_type == "c-c":
            plt.semilogx(
                cp.nu_list,
                CR_intrmdt_AFM_exprmt_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list,
                linestyle='-.', color='black', alpha=1, linewidth=2.5,
                label=intrmdt_AFM_exprmt_f_c_max_label)
        plt.semilogx(
            cp.nu_list, overline_epsilon_cnu_diss_hat_crit_list,
            linestyle='-', color='blue', alpha=1, linewidth=2.5,
            label=ufjc_label)
        plt.legend(loc='best', fontsize=10)
        plt.ylim([-0.05, 1.05])
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\nu$', 20,
            r'$\overline{\hat{\varepsilon}_{c\nu}^{diss}}$', 20,
            data_file_prefix+"-rate-independent-nondimensional-scaled-dissipated-chain-scission-energy-per-segment-vs-nu")
        
        fig = plt.figure()
        plt.loglog(
            cp.nu_list, LT_g_c_crit_list,
            linestyle='-', color='red', alpha=1, linewidth=2.5,
            label=LT_label)
        plt.loglog(
            cp.nu_list, LT_inext_gaussian_g_c_crit_list,
            linestyle='--', color='red', alpha=1, linewidth=2.5,
            label=LT_inext_gaussian_label)
        plt.loglog(
            cp.nu_list, CR_xi_c_max_g_c_crit_list,
            linestyle='--', color='black', alpha=1, linewidth=2.5,
            label=f_c_max_label)
        plt.loglog(
            cp.nu_list,
            CR_typcl_AFM_exprmt_xi_c_max_g_c_crit_list,
            linestyle=':', color='black', alpha=1, linewidth=2.5,
            label=typcl_AFM_exprmt_f_c_max_label)
        if chain_backbone_bond_type == "c-c":
            plt.loglog(
                cp.nu_list,
                CR_intrmdt_AFM_exprmt_xi_c_max_g_c_crit_list,
                linestyle='-.', color='black', alpha=1, linewidth=2.5,
                label=intrmdt_AFM_exprmt_f_c_max_label)
        plt.loglog(
            cp.nu_list, g_c_crit_list,
            linestyle='-', color='blue', alpha=1, linewidth=2.5,
            label=ufjc_label)
        plt.legend(loc='best', fontsize=10)
        # plt.ylim([-0.05, 1.05])
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\nu$', 20,
            r'$\beta G_c/(\eta^{ref}l_{\nu}^{eq})$', 20,
            data_file_prefix+"-rate-independent-nondimensional-fracture-toughness-vs-nu")
        
        fig = plt.figure()
        plt.loglog(
            cp.nu_list, LT_g_c_crit__nu_squared_list,
            linestyle='-', color='red', alpha=1, linewidth=2.5,
            label=LT_label)
        plt.loglog(
            cp.nu_list, LT_inext_gaussian_g_c_crit__nu_squared_list,
            linestyle='--', color='red', alpha=1, linewidth=2.5,
            label=LT_inext_gaussian_label)
        plt.loglog(
            cp.nu_list, CR_xi_c_max_g_c_crit__nu_squared_list,
            linestyle='--', color='black', alpha=1, linewidth=2.5,
            label=f_c_max_label)
        plt.loglog(
            cp.nu_list,
            CR_typcl_AFM_exprmt_xi_c_max_g_c_crit__nu_squared_list,
            linestyle=':', color='black', alpha=1, linewidth=2.5,
            label=typcl_AFM_exprmt_f_c_max_label)
        if chain_backbone_bond_type == "c-c":
            plt.loglog(
                cp.nu_list,
                CR_intrmdt_AFM_exprmt_xi_c_max_g_c_crit__nu_squared_list,
                linestyle='-.', color='black', alpha=1, linewidth=2.5,
                label=intrmdt_AFM_exprmt_f_c_max_label)
        plt.loglog(
            cp.nu_list, g_c_crit__nu_squared_list,
            linestyle='-', color='blue', alpha=1, linewidth=2.5,
            label=ufjc_label)
        plt.legend(loc='best', fontsize=10)
        # plt.ylim([-0.05, 1.05])
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\nu$', 20,
            r'$\beta G_c/(\eta^{ref}l_{\nu}^{eq}\nu^2)$', 20,
            data_file_prefix+"-rate-independent-nondimensional-fracture-toughness-nu-squared-normalized-vs-nu")
        
        fig = plt.figure()
        plt.loglog(
            cp.nu_list, LT_overline_g_c_crit_list,
            linestyle='-', color='red', alpha=1, linewidth=2.5,
            label=LT_label)
        plt.loglog(
            cp.nu_list, LT_inext_gaussian_overline_g_c_crit_list,
            linestyle='--', color='red', alpha=1, linewidth=2.5,
            label=LT_inext_gaussian_label)
        plt.loglog(
            cp.nu_list, CR_xi_c_max_overline_g_c_crit_list,
            linestyle='--', color='black', alpha=1, linewidth=2.5,
            label=f_c_max_label)
        plt.loglog(
            cp.nu_list,
            CR_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit_list,
            linestyle=':', color='black', alpha=1, linewidth=2.5,
            label=typcl_AFM_exprmt_f_c_max_label)
        if chain_backbone_bond_type == "c-c":
            plt.loglog(
                cp.nu_list,
                CR_intrmdt_AFM_exprmt_xi_c_max_overline_g_c_crit_list,
                linestyle='-.', color='black', alpha=1, linewidth=2.5,
                label=intrmdt_AFM_exprmt_f_c_max_label)
        plt.loglog(
            cp.nu_list, overline_g_c_crit_list,
            linestyle='-', color='blue', alpha=1, linewidth=2.5,
            label=ufjc_label)
        plt.legend(loc='best', fontsize=10)
        # plt.ylim([-0.05, 1.05])
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\nu$', 20,
            r'$\beta \overline{G_c}/(\eta^{ref}l_{\nu}^{eq})$', 20,
            data_file_prefix+"-rate-independent-nondimensional-scaled-fracture-toughness-vs-nu")
        
        fig = plt.figure()
        plt.loglog(
            cp.nu_list, LT_overline_g_c_crit__nu_squared_list,
            linestyle='-', color='red', alpha=1, linewidth=2.5,
            label=LT_label)
        plt.loglog(
            cp.nu_list, LT_inext_gaussian_overline_g_c_crit__nu_squared_list,
            linestyle='--', color='red', alpha=1, linewidth=2.5,
            label=LT_inext_gaussian_label)
        plt.loglog(
            cp.nu_list, CR_xi_c_max_overline_g_c_crit__nu_squared_list,
            linestyle='--', color='black', alpha=1, linewidth=2.5,
            label=f_c_max_label)
        plt.loglog(
            cp.nu_list,
            CR_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit__nu_squared_list,
            linestyle=':', color='black', alpha=1, linewidth=2.5,
            label=typcl_AFM_exprmt_f_c_max_label)
        if chain_backbone_bond_type == "c-c":
            plt.loglog(
                cp.nu_list,
                CR_intrmdt_AFM_exprmt_xi_c_max_overline_g_c_crit__nu_squared_list,
                linestyle='-.', color='black', alpha=1, linewidth=2.5,
                label=intrmdt_AFM_exprmt_f_c_max_label)
        plt.loglog(
            cp.nu_list, overline_g_c_crit__nu_squared_list,
            linestyle='-', color='blue', alpha=1, linewidth=2.5,
            label=ufjc_label)
        plt.legend(loc='best', fontsize=10)
        # plt.ylim([-0.05, 1.05])
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\nu$', 20,
            r'$\beta \overline{G_c}/(\eta^{ref}l_{\nu}^{eq}\nu^2)$', 20,
            data_file_prefix+"-rate-independent-nondimensional-scaled-fracture-toughness-nu-squared-normalized-vs-nu")

if __name__ == '__main__':

    T = 298 # absolute room temperature, K

    AFM_chain_tensile_tests_dict = {
        "al-maawali-et-al": "chain-a", "hugel-et-al": "chain-a"
    }

    al_maawali_et_al_rate_independent_characterizer = (
        RateDependentFractureToughnessCharacterizer(
            paper_authors="al-maawali-et-al", chain="chain-a", T=T)
    )
    # al_maawali_et_al_rate_independent_characterizer.characterization()
    al_maawali_et_al_rate_independent_characterizer.finalization()

    hugel_et_al_rate_independent_characterizer = (
        RateDependentFractureToughnessCharacterizer(
            paper_authors="hugel-et-al", chain="chain-a", T=T)
    )
    # hugel_et_al_rate_independent_characterizer.characterization()
    hugel_et_al_rate_independent_characterizer.finalization()