"""The intrinsic fracture toughness characterization module for
composite uFJCs that undergo scission
"""

# import external modules
from __future__ import division
from composite_ufjc_scission import (CompositeuFJCScissionCharacterizer,
    RateIndependentScissionCompositeuFJC,
    latex_formatting_figure,
    save_current_figure,
    save_current_figure_no_labels,
    save_pickle_object,
    load_pickle_object
)
import numpy as np
from math import isinf, floor, log10
from scipy import constants
import matplotlib.pyplot as plt
from matplotlib import ticker


class IntrinsicFractureToughnessCharacterizer(
        CompositeuFJCScissionCharacterizer):
    """The characterization class assessing intrinsic fracture toughness
    for composite uFJCs that undergo scission. It inherits all
    attributes and methods from the
    ``CompositeuFJCScissionCharacterizer`` class.
    """
    def __init__(self, paper_authors, chain, T):
        """Initializes the ``IntrinsicFractureToughnessCharacterizer``
        class by initializing and inheriting all attributes and methods
        from the ``CompositeuFJCScissionCharacterizer`` class.
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

        # nu = 1 -> nu = 10000
        nu_list = np.unique(np.rint(np.logspace(0, 4, 501)))

        p.characterizer.nu_list = nu_list

        nu_slice_list = [1e0, 1e1, 1e2, 1e3, 1e4] # nN/sec
        nu_slice_exponent_list = [
            int(floor(log10(abs(nu_slice_list[i]))))
            for i in range(len(nu_slice_list))
        ]
        nu_slice_label_list = [
            r'$\nu='+'10^{0:d}'.format(nu_slice_exponent_list[i])+'$'
            for i in range(len(nu_slice_list))
        ]
        nu_slice_color_list = ['orange', 'purple', 'green', 'cyan', 'brown']

        nu_slice_indx_list = [
            np.where(nu_list == nu_slice_list[i])[0][0]
            for i in range(len(nu_slice_list))
        ]

        p.characterizer.nu_slice_list = nu_slice_list
        p.characterizer.nu_slice_exponent_list = nu_slice_exponent_list
        p.characterizer.nu_slice_label_list = nu_slice_label_list
        p.characterizer.nu_slice_color_list = nu_slice_color_list
        p.characterizer.nu_slice_indx_list = nu_slice_indx_list

        p.characterizer.lmbda_c_star_num = 10001
        p.characterizer.lmbda_nu_hat_star_num = 10001
        p.characterizer.xi_c_star_num = 10001

    def prefix(self):
        """Set characterization prefix"""
        return "intrinsic_fracture_toughness"
    
    def characterization(self):
        """Define characterization routine"""

        def CN_fail_func(single_chain, lmbda_nu_hat_star_val):
            """Failure fatigue cycle number
            
            This function returns the fatigue cycle number needed for
            the number of chains crossing the crack unit surface area to
            vanish. It is a function of a CompositeUFJC single chain
            object and the maximum segment stretch that the fatigue
            cycle loading scheme applies
            """
            if lmbda_nu_hat_star_val >= single_chain.lmbda_nu_crit:
                return 1
            else:
                p_c_sur_hat_star_val = (
                    single_chain.p_c_sur_hat_func(lmbda_nu_hat_star_val)
                )
                if p_c_sur_hat_star_val == 0:
                    return 1
                else:
                    CN_fail_val = (
                        np.log(single_chain.eps_val) / np.log(p_c_sur_hat_star_val)
                    )
                    if isinf(CN_fail_val):
                        return np.nan
                    else:
                        return int(np.ceil(CN_fail_val))

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
        zeta_nu_char = np.loadtxt(
            cp.chain_data_directory+data_file_prefix+'-composite-uFJC-curve-fit-zeta_nu_char_intgr_nu'+'.txt')
        kappa_nu = np.loadtxt(
            cp.chain_data_directory+data_file_prefix+'-composite-uFJC-curve-fit-kappa_nu_intgr_nu'+'.txt')
        
        # Rate-independent calculations
        
        single_chain_list = [
            RateIndependentScissionCompositeuFJC(
                nu=nu_val, zeta_nu_char=zeta_nu_char, kappa_nu=kappa_nu)
            for nu_val in cp.nu_list
        ]
        lmbda_c_crit_list = [
            single_chain.lmbda_c_eq_crit / single_chain.A_nu
            for single_chain in single_chain_list
        ]
        lmbda_c_star_steps = np.linspace(
            0., np.amax(lmbda_c_crit_list), cp.lmbda_c_star_num)

        CN_fail_lmbda_c_star___nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]
        p_c_sci_hat_lmbda_c_star___nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]
        epsilon_cnu_diss_hat_lmbda_c_star___nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]
        g_c_lmbda_c_star___nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]
        g_c__nu_squared_lmbda_c_star___nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]
        overline_epsilon_cnu_diss_hat_lmbda_c_star___nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]
        overline_g_c_lmbda_c_star___nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]
        overline_g_c__nu_squared_lmbda_c_star___nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]

        for nu_indx in range(len(cp.nu_list)):
            lmbda_nu_hat_star = []
            lmbda_nu_hat_star_max = []
            CN_fail_lmbda_c_star = []
            p_c_sci_hat_lmbda_c_star = []
            epsilon_cnu_diss_hat_lmbda_c_star = []
            
            lmbda_nu_hat_star_max_val = 0.
            epsilon_cnu_diss_hat_lmbda_c_star_val = 0.
            
            single_chain = single_chain_list[nu_indx]
            lmbda_c_eq_star_steps = [
                lmbda_c_star_val * single_chain.A_nu
                for lmbda_c_star_val in lmbda_c_star_steps
            ]
            lmbda_nu_hat_star_steps = [
                single_chain.lmbda_nu_func(lmbda_c_eq_star_val)
                for lmbda_c_eq_star_val in lmbda_c_eq_star_steps
            ]

            for lmbda_nu_hat_star_indx in range(len(lmbda_nu_hat_star_steps)):
                lmbda_nu_hat_star_val = (
                    lmbda_nu_hat_star_steps[lmbda_nu_hat_star_indx]
                )
                lmbda_nu_hat_star_max_val = (
                    max([lmbda_nu_hat_star_max_val, lmbda_nu_hat_star_val])
                )
                CN_fail_lmbda_c_star_val = (
                    CN_fail_func(single_chain, lmbda_nu_hat_star_val)
                )
                p_c_sci_hat_lmbda_c_star_val = single_chain.p_c_sci_hat_func(
                    lmbda_nu_hat_star_val
                )
                if lmbda_nu_hat_star_indx == 0:
                    pass
                elif lmbda_nu_hat_star_val > single_chain.lmbda_nu_crit:
                    epsilon_cnu_diss_hat_lmbda_c_star_val = (
                        single_chain.epsilon_cnu_diss_hat_crit
                    )
                else:
                    epsilon_cnu_diss_hat_lmbda_c_star_val = (
                        single_chain.epsilon_cnu_diss_hat_func(
                            lmbda_nu_hat_star_max_val,
                            lmbda_nu_hat_star_max[lmbda_nu_hat_star_indx-1],
                            lmbda_nu_hat_star_val,
                            lmbda_nu_hat_star[lmbda_nu_hat_star_indx-1],
                            epsilon_cnu_diss_hat_lmbda_c_star[lmbda_nu_hat_star_indx-1])
                    )
                if epsilon_cnu_diss_hat_lmbda_c_star_val > single_chain.epsilon_cnu_diss_hat_crit:
                    epsilon_cnu_diss_hat_lmbda_c_star_val = (
                        single_chain.epsilon_cnu_diss_hat_crit
                    )
                
                lmbda_nu_hat_star.append(lmbda_nu_hat_star_val)
                lmbda_nu_hat_star_max.append(lmbda_nu_hat_star_max_val)
                CN_fail_lmbda_c_star.append(CN_fail_lmbda_c_star_val)
                p_c_sci_hat_lmbda_c_star.append(p_c_sci_hat_lmbda_c_star_val)
                epsilon_cnu_diss_hat_lmbda_c_star.append(epsilon_cnu_diss_hat_lmbda_c_star_val)
            
            g_c_lmbda_c_star = [
                0.5 * single_chain.A_nu * single_chain.nu**2 * epsilon_cnu_diss_hat_lmbda_c_star_val
                for epsilon_cnu_diss_hat_lmbda_c_star_val
                in epsilon_cnu_diss_hat_lmbda_c_star
            ]
            g_c__nu_squared_lmbda_c_star = [
                g_c_lmbda_c_star_val / single_chain.nu**2
                for g_c_lmbda_c_star_val in g_c_lmbda_c_star
            ]
            overline_epsilon_cnu_diss_hat_lmbda_c_star = [
                epsilon_cnu_diss_hat_lmbda_c_star_val / single_chain.zeta_nu_char
                for epsilon_cnu_diss_hat_lmbda_c_star_val
                in epsilon_cnu_diss_hat_lmbda_c_star
            ]
            overline_g_c_lmbda_c_star = [
                0.5 * single_chain.A_nu * single_chain.nu**2 * overline_epsilon_cnu_diss_hat_lmbda_c_star_val
                for overline_epsilon_cnu_diss_hat_lmbda_c_star_val
                in overline_epsilon_cnu_diss_hat_lmbda_c_star
            ]
            overline_g_c__nu_squared_lmbda_c_star = [
                overline_g_c_lmbda_c_star_val / single_chain.nu**2
                for overline_g_c_lmbda_c_star_val in overline_g_c_lmbda_c_star
            ]
            
            CN_fail_lmbda_c_star___nu_chunk_list[nu_indx] = CN_fail_lmbda_c_star
            p_c_sci_hat_lmbda_c_star___nu_chunk_list[nu_indx] = p_c_sci_hat_lmbda_c_star
            epsilon_cnu_diss_hat_lmbda_c_star___nu_chunk_list[nu_indx] = (
                epsilon_cnu_diss_hat_lmbda_c_star
            )
            g_c_lmbda_c_star___nu_chunk_list[nu_indx] = g_c_lmbda_c_star
            g_c__nu_squared_lmbda_c_star___nu_chunk_list[nu_indx] = (
                g_c__nu_squared_lmbda_c_star
            )
            overline_epsilon_cnu_diss_hat_lmbda_c_star___nu_chunk_list[nu_indx] = (
                overline_epsilon_cnu_diss_hat_lmbda_c_star
            )
            overline_g_c_lmbda_c_star___nu_chunk_list[nu_indx] = (
                overline_g_c_lmbda_c_star
            )
            overline_g_c__nu_squared_lmbda_c_star___nu_chunk_list[nu_indx] = (
                overline_g_c__nu_squared_lmbda_c_star
            )
        
        save_pickle_object(
            self.savedir, lmbda_c_star_steps,
            data_file_prefix+"-lmbda_c_star_steps")
        save_pickle_object(
            self.savedir, CN_fail_lmbda_c_star___nu_chunk_list,
            data_file_prefix+"-CN_fail_lmbda_c_star___nu_chunk_list")
        save_pickle_object(
            self.savedir, p_c_sci_hat_lmbda_c_star___nu_chunk_list,
            data_file_prefix+"-p_c_sci_hat_lmbda_c_star___nu_chunk_list")
        save_pickle_object(
            self.savedir, epsilon_cnu_diss_hat_lmbda_c_star___nu_chunk_list,
            data_file_prefix+"-epsilon_cnu_diss_hat_lmbda_c_star___nu_chunk_list")
        save_pickle_object(
            self.savedir, g_c_lmbda_c_star___nu_chunk_list,
            data_file_prefix+"-g_c_lmbda_c_star___nu_chunk_list")
        save_pickle_object(
            self.savedir, g_c__nu_squared_lmbda_c_star___nu_chunk_list,
            data_file_prefix+"-g_c__nu_squared_lmbda_c_star___nu_chunk_list")
        save_pickle_object(
            self.savedir, overline_epsilon_cnu_diss_hat_lmbda_c_star___nu_chunk_list,
            data_file_prefix+"-overline_epsilon_cnu_diss_hat_lmbda_c_star___nu_chunk_list")
        save_pickle_object(
            self.savedir, overline_g_c_lmbda_c_star___nu_chunk_list,
            data_file_prefix+"-overline_g_c_lmbda_c_star___nu_chunk_list")
        save_pickle_object(
            self.savedir, overline_g_c__nu_squared_lmbda_c_star___nu_chunk_list,
            data_file_prefix+"-overline_g_c__nu_squared_lmbda_c_star___nu_chunk_list")
        

        CN_fail_lmbda_c_star__lmbda_c_crit___nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]
        p_c_sci_hat_lmbda_c_star__lmbda_c_crit___nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]
        epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit___nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]
        g_c_lmbda_c_star__lmbda_c_crit___nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]
        g_c__nu_squared_lmbda_c_star__lmbda_c_crit___nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]
        overline_epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit___nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]
        overline_g_c_lmbda_c_star__lmbda_c_crit___nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]
        overline_g_c__nu_squared_lmbda_c_star__lmbda_c_crit___nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]

        for nu_indx in range(len(cp.nu_list)):
            lmbda_nu_hat_star = []
            lmbda_nu_hat_star_max = []
            CN_fail_lmbda_c_star__lmbda_c_crit = []
            p_c_sci_hat_lmbda_c_star__lmbda_c_crit = []
            epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit = []
            
            lmbda_nu_hat_star_max_val = 0.
            epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit_val = 0.
            
            single_chain = single_chain_list[nu_indx]
            lmbda_c_eq_star_steps = np.linspace(
                0., single_chain.lmbda_c_eq_crit, cp.lmbda_c_star_num)
            
            lmbda_nu_hat_star_steps = [
                single_chain.lmbda_nu_func(lmbda_c_eq_star_val)
                for lmbda_c_eq_star_val in lmbda_c_eq_star_steps
            ]

            for lmbda_nu_hat_star_indx in range(len(lmbda_nu_hat_star_steps)):
                lmbda_nu_hat_star_val = (
                    lmbda_nu_hat_star_steps[lmbda_nu_hat_star_indx]
                )
                lmbda_nu_hat_star_max_val = (
                    max([lmbda_nu_hat_star_max_val, lmbda_nu_hat_star_val])
                )
                CN_fail_lmbda_c_star__lmbda_c_crit_val = (
                    CN_fail_func(single_chain, lmbda_nu_hat_star_val)
                )
                p_c_sci_hat_lmbda_c_star__lmbda_c_crit_val = (
                    single_chain.p_c_sci_hat_func(lmbda_nu_hat_star_val)
                )
                if lmbda_nu_hat_star_indx == 0:
                    pass
                else:
                    epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit_val = (
                        single_chain.epsilon_cnu_diss_hat_func(
                            lmbda_nu_hat_star_max_val,
                            lmbda_nu_hat_star_max[lmbda_nu_hat_star_indx-1],
                            lmbda_nu_hat_star_val,
                            lmbda_nu_hat_star[lmbda_nu_hat_star_indx-1],
                            epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit[lmbda_nu_hat_star_indx-1])
                    )
                
                lmbda_nu_hat_star.append(lmbda_nu_hat_star_val)
                lmbda_nu_hat_star_max.append(lmbda_nu_hat_star_max_val)
                CN_fail_lmbda_c_star__lmbda_c_crit.append(CN_fail_lmbda_c_star__lmbda_c_crit_val)
                p_c_sci_hat_lmbda_c_star__lmbda_c_crit.append(p_c_sci_hat_lmbda_c_star__lmbda_c_crit_val)
                epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit.append(epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit_val)
            
            g_c_lmbda_c_star__lmbda_c_crit = [
                0.5 * single_chain.A_nu * single_chain.nu**2 * epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit_val
                for epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit_val
                in epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit
            ]
            g_c__nu_squared_lmbda_c_star__lmbda_c_crit = [
                g_c_lmbda_c_star__lmbda_c_crit_val / single_chain.nu**2
                for g_c_lmbda_c_star__lmbda_c_crit_val in g_c_lmbda_c_star__lmbda_c_crit
            ]
            overline_epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit = [
                epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit_val / single_chain.zeta_nu_char
                for epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit_val
                in epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit
            ]
            overline_g_c_lmbda_c_star__lmbda_c_crit = [
                0.5 * single_chain.A_nu * single_chain.nu**2 * overline_epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit_val
                for overline_epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit_val
                in overline_epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit
            ]
            overline_g_c__nu_squared_lmbda_c_star__lmbda_c_crit = [
                overline_g_c_lmbda_c_star__lmbda_c_crit_val / single_chain.nu**2
                for overline_g_c_lmbda_c_star__lmbda_c_crit_val in overline_g_c_lmbda_c_star__lmbda_c_crit
            ]
            
            CN_fail_lmbda_c_star__lmbda_c_crit___nu_chunk_list[nu_indx] = CN_fail_lmbda_c_star__lmbda_c_crit
            p_c_sci_hat_lmbda_c_star__lmbda_c_crit___nu_chunk_list[nu_indx] = p_c_sci_hat_lmbda_c_star__lmbda_c_crit
            epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit___nu_chunk_list[nu_indx] = (
                epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit
            )
            g_c_lmbda_c_star__lmbda_c_crit___nu_chunk_list[nu_indx] = g_c_lmbda_c_star__lmbda_c_crit
            g_c__nu_squared_lmbda_c_star__lmbda_c_crit___nu_chunk_list[nu_indx] = (
                g_c__nu_squared_lmbda_c_star__lmbda_c_crit
            )
            overline_epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit___nu_chunk_list[nu_indx] = (
                overline_epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit
            )
            overline_g_c_lmbda_c_star__lmbda_c_crit___nu_chunk_list[nu_indx] = (
                overline_g_c_lmbda_c_star__lmbda_c_crit
            )
            overline_g_c__nu_squared_lmbda_c_star__lmbda_c_crit___nu_chunk_list[nu_indx] = (
                overline_g_c__nu_squared_lmbda_c_star__lmbda_c_crit
            )
        
        save_pickle_object(
            self.savedir, CN_fail_lmbda_c_star__lmbda_c_crit___nu_chunk_list,
            data_file_prefix+"-CN_fail_lmbda_c_star__lmbda_c_crit___nu_chunk_list")
        save_pickle_object(
            self.savedir, p_c_sci_hat_lmbda_c_star__lmbda_c_crit___nu_chunk_list,
            data_file_prefix+"-p_c_sci_hat_lmbda_c_star__lmbda_c_crit___nu_chunk_list")
        save_pickle_object(
            self.savedir, epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit___nu_chunk_list,
            data_file_prefix+"-epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit___nu_chunk_list")
        save_pickle_object(
            self.savedir, g_c_lmbda_c_star__lmbda_c_crit___nu_chunk_list,
            data_file_prefix+"-g_c_lmbda_c_star__lmbda_c_crit___nu_chunk_list")
        save_pickle_object(
            self.savedir, g_c__nu_squared_lmbda_c_star__lmbda_c_crit___nu_chunk_list,
            data_file_prefix+"-g_c__nu_squared_lmbda_c_star__lmbda_c_crit___nu_chunk_list")
        save_pickle_object(
            self.savedir, overline_epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit___nu_chunk_list,
            data_file_prefix+"-overline_epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit___nu_chunk_list")
        save_pickle_object(
            self.savedir, overline_g_c_lmbda_c_star__lmbda_c_crit___nu_chunk_list,
            data_file_prefix+"-overline_g_c_lmbda_c_star__lmbda_c_crit___nu_chunk_list")
        save_pickle_object(
            self.savedir, overline_g_c__nu_squared_lmbda_c_star__lmbda_c_crit___nu_chunk_list,
            data_file_prefix+"-overline_g_c__nu_squared_lmbda_c_star__lmbda_c_crit___nu_chunk_list")
        

        CN_fail_xi_c_star__xi_c_crit___nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]
        p_c_sci_hat_xi_c_star__xi_c_crit___nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]
        epsilon_cnu_diss_hat_xi_c_star__xi_c_crit___nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]
        g_c_xi_c_star__xi_c_crit___nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]
        g_c__nu_squared_xi_c_star__xi_c_crit___nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]
        overline_epsilon_cnu_diss_hat_xi_c_star__xi_c_crit___nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]
        overline_g_c_xi_c_star__xi_c_crit___nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]
        overline_g_c__nu_squared_xi_c_star__xi_c_crit___nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]

        for nu_indx in range(len(cp.nu_list)):
            lmbda_nu_hat_star = []
            lmbda_nu_hat_star_max = []
            CN_fail_xi_c_star__xi_c_crit = []
            p_c_sci_hat_xi_c_star__xi_c_crit = []
            epsilon_cnu_diss_hat_xi_c_star__xi_c_crit = []
            
            lmbda_nu_hat_star_max_val = 0.
            epsilon_cnu_diss_hat_xi_c_star__xi_c_crit_val = 0.
            
            single_chain = single_chain_list[nu_indx]
            xi_c_star_steps = np.linspace(
                0., single_chain.xi_c_crit, cp.xi_c_star_num)
            
            lmbda_nu_hat_star_steps = [
                single_chain.lmbda_nu_xi_c_hat_func(xi_c_star_val)
                for xi_c_star_val in xi_c_star_steps
            ]

            for lmbda_nu_hat_star_indx in range(len(lmbda_nu_hat_star_steps)):
                lmbda_nu_hat_star_val = (
                    lmbda_nu_hat_star_steps[lmbda_nu_hat_star_indx]
                )
                lmbda_nu_hat_star_max_val = (
                    max([lmbda_nu_hat_star_max_val, lmbda_nu_hat_star_val])
                )
                CN_fail_xi_c_star__xi_c_crit_val = (
                    CN_fail_func(single_chain, lmbda_nu_hat_star_val)
                )
                p_c_sci_hat_xi_c_star__xi_c_crit_val = (
                    single_chain.p_c_sci_hat_func(lmbda_nu_hat_star_val)
                )
                if lmbda_nu_hat_star_indx == 0:
                    pass
                else:
                    epsilon_cnu_diss_hat_xi_c_star__xi_c_crit_val = (
                        single_chain.epsilon_cnu_diss_hat_func(
                            lmbda_nu_hat_star_max_val,
                            lmbda_nu_hat_star_max[lmbda_nu_hat_star_indx-1],
                            lmbda_nu_hat_star_val,
                            lmbda_nu_hat_star[lmbda_nu_hat_star_indx-1],
                            epsilon_cnu_diss_hat_xi_c_star__xi_c_crit[lmbda_nu_hat_star_indx-1])
                    )
                
                lmbda_nu_hat_star.append(lmbda_nu_hat_star_val)
                lmbda_nu_hat_star_max.append(lmbda_nu_hat_star_max_val)
                CN_fail_xi_c_star__xi_c_crit.append(CN_fail_xi_c_star__xi_c_crit_val)
                p_c_sci_hat_xi_c_star__xi_c_crit.append(p_c_sci_hat_xi_c_star__xi_c_crit_val)
                epsilon_cnu_diss_hat_xi_c_star__xi_c_crit.append(epsilon_cnu_diss_hat_xi_c_star__xi_c_crit_val)
            
            g_c_xi_c_star__xi_c_crit = [
                0.5 * single_chain.A_nu * single_chain.nu**2 * epsilon_cnu_diss_hat_xi_c_star__xi_c_crit_val
                for epsilon_cnu_diss_hat_xi_c_star__xi_c_crit_val
                in epsilon_cnu_diss_hat_xi_c_star__xi_c_crit
            ]
            g_c__nu_squared_xi_c_star__xi_c_crit = [
                g_c_xi_c_star__xi_c_crit_val / single_chain.nu**2
                for g_c_xi_c_star__xi_c_crit_val in g_c_xi_c_star__xi_c_crit
            ]
            overline_epsilon_cnu_diss_hat_xi_c_star__xi_c_crit = [
                epsilon_cnu_diss_hat_xi_c_star__xi_c_crit_val / single_chain.zeta_nu_char
                for epsilon_cnu_diss_hat_xi_c_star__xi_c_crit_val
                in epsilon_cnu_diss_hat_xi_c_star__xi_c_crit
            ]
            overline_g_c_xi_c_star__xi_c_crit = [
                0.5 * single_chain.A_nu * single_chain.nu**2 * overline_epsilon_cnu_diss_hat_xi_c_star__xi_c_crit_val
                for overline_epsilon_cnu_diss_hat_xi_c_star__xi_c_crit_val
                in overline_epsilon_cnu_diss_hat_xi_c_star__xi_c_crit
            ]
            overline_g_c__nu_squared_xi_c_star__xi_c_crit = [
                overline_g_c_xi_c_star__xi_c_crit_val / single_chain.nu**2
                for overline_g_c_xi_c_star__xi_c_crit_val in overline_g_c_xi_c_star__xi_c_crit
            ]
            
            CN_fail_xi_c_star__xi_c_crit___nu_chunk_list[nu_indx] = CN_fail_xi_c_star__xi_c_crit
            p_c_sci_hat_xi_c_star__xi_c_crit___nu_chunk_list[nu_indx] = p_c_sci_hat_xi_c_star__xi_c_crit
            epsilon_cnu_diss_hat_xi_c_star__xi_c_crit___nu_chunk_list[nu_indx] = (
                epsilon_cnu_diss_hat_xi_c_star__xi_c_crit
            )
            g_c_xi_c_star__xi_c_crit___nu_chunk_list[nu_indx] = g_c_xi_c_star__xi_c_crit
            g_c__nu_squared_xi_c_star__xi_c_crit___nu_chunk_list[nu_indx] = (
                g_c__nu_squared_xi_c_star__xi_c_crit
            )
            overline_epsilon_cnu_diss_hat_xi_c_star__xi_c_crit___nu_chunk_list[nu_indx] = (
                overline_epsilon_cnu_diss_hat_xi_c_star__xi_c_crit
            )
            overline_g_c_xi_c_star__xi_c_crit___nu_chunk_list[nu_indx] = (
                overline_g_c_xi_c_star__xi_c_crit
            )
            overline_g_c__nu_squared_xi_c_star__xi_c_crit___nu_chunk_list[nu_indx] = (
                overline_g_c__nu_squared_xi_c_star__xi_c_crit
            )
        
        save_pickle_object(
            self.savedir, CN_fail_xi_c_star__xi_c_crit___nu_chunk_list,
            data_file_prefix+"-CN_fail_xi_c_star__xi_c_crit___nu_chunk_list")
        save_pickle_object(
            self.savedir, p_c_sci_hat_xi_c_star__xi_c_crit___nu_chunk_list,
            data_file_prefix+"-p_c_sci_hat_xi_c_star__xi_c_crit___nu_chunk_list")
        save_pickle_object(
            self.savedir, epsilon_cnu_diss_hat_xi_c_star__xi_c_crit___nu_chunk_list,
            data_file_prefix+"-epsilon_cnu_diss_hat_xi_c_star__xi_c_crit___nu_chunk_list")
        save_pickle_object(
            self.savedir, g_c_xi_c_star__xi_c_crit___nu_chunk_list,
            data_file_prefix+"-g_c_xi_c_star__xi_c_crit___nu_chunk_list")
        save_pickle_object(
            self.savedir, g_c__nu_squared_xi_c_star__xi_c_crit___nu_chunk_list,
            data_file_prefix+"-g_c__nu_squared_xi_c_star__xi_c_crit___nu_chunk_list")
        save_pickle_object(
            self.savedir, overline_epsilon_cnu_diss_hat_xi_c_star__xi_c_crit___nu_chunk_list,
            data_file_prefix+"-overline_epsilon_cnu_diss_hat_xi_c_star__xi_c_crit___nu_chunk_list")
        save_pickle_object(
            self.savedir, overline_g_c_xi_c_star__xi_c_crit___nu_chunk_list,
            data_file_prefix+"-overline_g_c_xi_c_star__xi_c_crit___nu_chunk_list")
        save_pickle_object(
            self.savedir, overline_g_c__nu_squared_xi_c_star__xi_c_crit___nu_chunk_list,
            data_file_prefix+"-overline_g_c__nu_squared_xi_c_star__xi_c_crit___nu_chunk_list")
        
        
        CN_fail_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]
        p_c_sci_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]
        epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]
        g_c_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]
        g_c__nu_squared_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]
        overline_epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]
        overline_g_c_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]
        overline_g_c__nu_squared_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list = [
            0. for nu_indx in range(len(cp.nu_list))
        ]

        for nu_indx in range(len(cp.nu_list)):
            lmbda_nu_hat_star = []
            lmbda_nu_hat_star_max = []
            CN_fail_lmbda_nu_hat_star__lmbda_nu_hat_crit = []
            p_c_sci_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit = []
            epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit = []
            
            lmbda_nu_hat_star_max_val = 0.
            epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit_val = 0.
            
            single_chain = single_chain_list[nu_indx]
            lmbda_nu_hat_star_steps = np.linspace(
                1., single_chain.lmbda_nu_crit, cp.lmbda_nu_hat_star_num)

            for lmbda_nu_hat_star_indx in range(len(lmbda_nu_hat_star_steps)):
                lmbda_nu_hat_star_val = (
                    lmbda_nu_hat_star_steps[lmbda_nu_hat_star_indx]
                )
                lmbda_nu_hat_star_max_val = (
                    max([lmbda_nu_hat_star_max_val, lmbda_nu_hat_star_val])
                )
                CN_fail_lmbda_nu_hat_star__lmbda_nu_hat_crit_val = (
                    CN_fail_func(single_chain, lmbda_nu_hat_star_val)
                )
                p_c_sci_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit_val = (
                    single_chain.p_c_sci_hat_func(lmbda_nu_hat_star_val)
                )
                if lmbda_nu_hat_star_indx == 0:
                    pass
                else:
                    epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit_val = (
                        single_chain.epsilon_cnu_diss_hat_func(
                            lmbda_nu_hat_star_max_val,
                            lmbda_nu_hat_star_max[lmbda_nu_hat_star_indx-1],
                            lmbda_nu_hat_star_val,
                            lmbda_nu_hat_star[lmbda_nu_hat_star_indx-1],
                            epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit[lmbda_nu_hat_star_indx-1])
                    )
                
                lmbda_nu_hat_star.append(lmbda_nu_hat_star_val)
                lmbda_nu_hat_star_max.append(lmbda_nu_hat_star_max_val)
                CN_fail_lmbda_nu_hat_star__lmbda_nu_hat_crit.append(CN_fail_lmbda_nu_hat_star__lmbda_nu_hat_crit_val)
                p_c_sci_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit.append(p_c_sci_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit_val)
                epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit.append(epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit_val)
            
            g_c_lmbda_nu_hat_star__lmbda_nu_hat_crit = [
                0.5 * single_chain.A_nu * single_chain.nu**2 * epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit_val
                for epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit_val
                in epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit
            ]
            g_c__nu_squared_lmbda_nu_hat_star__lmbda_nu_hat_crit = [
                g_c_lmbda_nu_hat_star__lmbda_nu_hat_crit_val / single_chain.nu**2
                for g_c_lmbda_nu_hat_star__lmbda_nu_hat_crit_val in g_c_lmbda_nu_hat_star__lmbda_nu_hat_crit
            ]
            overline_epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit = [
                epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit_val / single_chain.zeta_nu_char
                for epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit_val
                in epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit
            ]
            overline_g_c_lmbda_nu_hat_star__lmbda_nu_hat_crit = [
                0.5 * single_chain.A_nu * single_chain.nu**2 * overline_epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit_val
                for overline_epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit_val
                in overline_epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit
            ]
            overline_g_c__nu_squared_lmbda_nu_hat_star__lmbda_nu_hat_crit = [
                overline_g_c_lmbda_nu_hat_star__lmbda_nu_hat_crit_val / single_chain.nu**2
                for overline_g_c_lmbda_nu_hat_star__lmbda_nu_hat_crit_val in overline_g_c_lmbda_nu_hat_star__lmbda_nu_hat_crit
            ]
            
            CN_fail_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list[nu_indx] = CN_fail_lmbda_nu_hat_star__lmbda_nu_hat_crit
            p_c_sci_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list[nu_indx] = p_c_sci_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit
            epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list[nu_indx] = (
                epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit
            )
            g_c_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list[nu_indx] = g_c_lmbda_nu_hat_star__lmbda_nu_hat_crit
            g_c__nu_squared_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list[nu_indx] = (
                g_c__nu_squared_lmbda_nu_hat_star__lmbda_nu_hat_crit
            )
            overline_epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list[nu_indx] = (
                overline_epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit
            )
            overline_g_c_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list[nu_indx] = (
                overline_g_c_lmbda_nu_hat_star__lmbda_nu_hat_crit
            )
            overline_g_c__nu_squared_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list[nu_indx] = (
                overline_g_c__nu_squared_lmbda_nu_hat_star__lmbda_nu_hat_crit
            )
        
        save_pickle_object(
            self.savedir, CN_fail_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list,
            data_file_prefix+"-CN_fail_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list")
        save_pickle_object(
            self.savedir, p_c_sci_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list,
            data_file_prefix+"-p_c_sci_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list")
        save_pickle_object(
            self.savedir, epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list,
            data_file_prefix+"-epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list")
        save_pickle_object(
            self.savedir, g_c_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list,
            data_file_prefix+"-g_c_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list")
        save_pickle_object(
            self.savedir, g_c__nu_squared_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list,
            data_file_prefix+"-g_c__nu_squared_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list")
        save_pickle_object(
            self.savedir, overline_epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list,
            data_file_prefix+"-overline_epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list")
        save_pickle_object(
            self.savedir, overline_g_c_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list,
            data_file_prefix+"-overline_g_c_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list")
        save_pickle_object(
            self.savedir, overline_g_c__nu_squared_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list,
            data_file_prefix+"-overline_g_c__nu_squared_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list")


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

        # plot results
        latex_formatting_figure(ppp)
        
        lmbda_c_star_steps = load_pickle_object(
            self.savedir, data_file_prefix+"-lmbda_c_star_steps")
        CN_fail_lmbda_c_star___nu_chunk_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-CN_fail_lmbda_c_star___nu_chunk_list")
        p_c_sci_hat_lmbda_c_star___nu_chunk_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-p_c_sci_hat_lmbda_c_star___nu_chunk_list")
        epsilon_cnu_diss_hat_lmbda_c_star___nu_chunk_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-epsilon_cnu_diss_hat_lmbda_c_star___nu_chunk_list")
        # g_c_lmbda_c_star___nu_chunk_list = load_pickle_object(
        #     self.savedir, data_file_prefix+"-g_c_lmbda_c_star___nu_chunk_list")
        # g_c__nu_squared_lmbda_c_star___nu_chunk_list = load_pickle_object(
        #     self.savedir,
        #     data_file_prefix+"-g_c__nu_squared_lmbda_c_star___nu_chunk_list")
        overline_epsilon_cnu_diss_hat_lmbda_c_star___nu_chunk_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-overline_epsilon_cnu_diss_hat_lmbda_c_star___nu_chunk_list")
        # overline_g_c_lmbda_c_star___nu_chunk_list = load_pickle_object(
        #     self.savedir,
        #     data_file_prefix+"-overline_g_c_lmbda_c_star___nu_chunk_list")
        # overline_g_c__nu_squared_lmbda_c_star___nu_chunk_list = load_pickle_object(
        #     self.savedir,
        #     data_file_prefix+"-overline_g_c__nu_squared_lmbda_c_star___nu_chunk_list")

        lmbda_c_star_steps_meshgrid, nu_list_meshgrid = np.meshgrid(
            lmbda_c_star_steps, cp.nu_list)

        CN_fail_lmbda_c_star_list = np.asarray(
            CN_fail_lmbda_c_star___nu_chunk_list)
        
        p_c_sci_hat_lmbda_c_star_list = np.asarray(
            p_c_sci_hat_lmbda_c_star___nu_chunk_list)
        
        epsilon_cnu_diss_hat_lmbda_c_star_list = np.asarray(
            epsilon_cnu_diss_hat_lmbda_c_star___nu_chunk_list)
        
        overline_epsilon_cnu_diss_hat_lmbda_c_star_list = np.asarray(
            overline_epsilon_cnu_diss_hat_lmbda_c_star___nu_chunk_list)
        

        fig, ax1 = plt.subplots()

        ax1.set_facecolor("gray")

        filled_contour_plot = ax1.contourf(
            lmbda_c_star_steps_meshgrid, nu_list_meshgrid,
            CN_fail_lmbda_c_star_list, locator=ticker.LogLocator(),
            cmap=plt.cm.cividis)
        
        for fcp in filled_contour_plot.collections:
            fcp.set_edgecolor('face')
        
        ax1.set_xlabel(r'$\lambda_c^{\star}$', fontsize=20)
        ax1.set_ylabel(r'$\nu$', fontsize=20)
        # ax1.set_xscale('log')
        ax1.set_yscale('log')
        # ax1.tick_params(axis='both', labelsize=16)

        cbar = fig.colorbar(filled_contour_plot)
        cbar.ax.set_ylabel(r'$\textrm{CN}^{\textrm{fail}}$', fontsize=20)
        cbar.ax.tick_params(axis='y', labelsize=14)
        
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)

        plt.tight_layout()
        plt.savefig(self.savedir+data_file_prefix+"-rate-independent-CN-fail-filled-contour-nu-vs-lmbda_c_star"+".pdf")
        plt.close()


        fig = plt.figure()
        for nu_slice_indx in range(len(cp.nu_slice_list)):
            plt.semilogy(
                lmbda_c_star_steps,
                CN_fail_lmbda_c_star___nu_chunk_list[cp.nu_slice_indx_list[nu_slice_indx]],
                linestyle='-', color=cp.nu_slice_color_list[nu_slice_indx],
                alpha=1, linewidth=2.5,
                label=cp.nu_slice_label_list[nu_slice_indx])
        plt.legend(loc='best', fontsize=10)
        plt.ylim([1e0, 1e11])
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\lambda_c^{\star}$', 20,
            r'$\textrm{CN}^{\textrm{fail}}$', 20,
            data_file_prefix+"-rate-independent-CN-fail-vs-lmbda_c_star")
        
        
        fig, ax1 = plt.subplots()

        ax1.set_facecolor("gray")

        filled_contour_plot = ax1.contourf(
            lmbda_c_star_steps_meshgrid, nu_list_meshgrid,
            p_c_sci_hat_lmbda_c_star_list, locator=ticker.LogLocator(),
            cmap=plt.cm.cividis)
        
        for fcp in filled_contour_plot.collections:
            fcp.set_edgecolor('face')
        
        ax1.set_xlabel(r'$\lambda_c^{\star}$', fontsize=20)
        ax1.set_ylabel(r'$\nu$', fontsize=20)
        # ax1.set_xscale('log')
        ax1.set_yscale('log')
        # ax1.tick_params(axis='both', labelsize=16)

        cbar = fig.colorbar(filled_contour_plot)
        cbar.ax.set_ylabel(r'$d\overline{\mathcal{N}}_{\Gamma}/dN$', fontsize=20)
        cbar.ax.tick_params(axis='y', labelsize=14)
        
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)

        plt.tight_layout()
        plt.savefig(self.savedir+data_file_prefix+"-rate-independent-d_mathcal_N_Gamma__dN-filled-contour-nu-vs-lmbda_c_star"+".pdf")
        plt.close()


        fig = plt.figure()
        for nu_slice_indx in range(len(cp.nu_slice_list)):
            plt.plot(
                lmbda_c_star_steps,
                p_c_sci_hat_lmbda_c_star___nu_chunk_list[cp.nu_slice_indx_list[nu_slice_indx]],
                linestyle='-', color=cp.nu_slice_color_list[nu_slice_indx],
                alpha=1, linewidth=2.5,
                label=cp.nu_slice_label_list[nu_slice_indx])
        plt.legend(loc='best', fontsize=10)
        plt.ylim([-0.05, 1.05])
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\lambda_c^{\star}$', 20,
            r'$d\overline{\mathcal{N}}_{\Gamma}/dN$', 20,
            data_file_prefix+"-rate-independent-d_mathcal_N_Gamma__dN-vs-lmbda_c_star")
        
        fig = plt.figure()
        for nu_slice_indx in range(len(cp.nu_slice_list)):
            plt.semilogy(
                lmbda_c_star_steps,
                p_c_sci_hat_lmbda_c_star___nu_chunk_list[cp.nu_slice_indx_list[nu_slice_indx]],
                linestyle='-', color=cp.nu_slice_color_list[nu_slice_indx],
                alpha=1, linewidth=2.5,
                label=cp.nu_slice_label_list[nu_slice_indx])
        plt.legend(loc='best', fontsize=10)
        # plt.ylim([1e0, 1e11])
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\lambda_c^{\star}$', 20,
            r'$d\overline{\mathcal{N}}_{\Gamma}/dN$', 20,
            data_file_prefix+"-rate-independent-d_mathcal_N_Gamma__dN-semilogy-vs-lmbda_c_star")

        
        fig, ax1 = plt.subplots()

        filled_contour_plot = ax1.contourf(
            lmbda_c_star_steps_meshgrid, nu_list_meshgrid,
            epsilon_cnu_diss_hat_lmbda_c_star_list, 100,
            cmap=plt.cm.cividis)
        
        for fcp in filled_contour_plot.collections:
            fcp.set_edgecolor('face')
        
        ax1.set_xlabel(r'$\lambda_c^{\star}$', fontsize=20)
        ax1.set_ylabel(r'$\nu$', fontsize=20)
        # ax1.set_xscale('log')
        ax1.set_yscale('log')
        # ax1.tick_params(axis='both', labelsize=16)

        cbar = fig.colorbar(filled_contour_plot)
        cbar.ax.set_ylabel(r'$\hat{\varepsilon}_{c\nu}^{diss}$', fontsize=20)
        cbar.ax.tick_params(axis='y', labelsize=14)
        
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        
        save_current_figure_no_labels(
            self.savedir,
            data_file_prefix+"-rate-independent-nondimensional-dissipated-chain-scission-energy-per-segment-filled-contour-nu-vs-lmbda_c_star")
        

        fig = plt.figure()
        for nu_slice_indx in range(len(cp.nu_slice_list)):
            plt.plot(
                lmbda_c_star_steps,
                epsilon_cnu_diss_hat_lmbda_c_star___nu_chunk_list[cp.nu_slice_indx_list[nu_slice_indx]],
                linestyle='-', color=cp.nu_slice_color_list[nu_slice_indx],
                alpha=1, linewidth=2.5,
                label=cp.nu_slice_label_list[nu_slice_indx])
        plt.legend(loc='best', fontsize=10)
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\lambda_c^{\star}$', 20,
            r'$\hat{\varepsilon}_{c\nu}^{diss}$', 20,
            data_file_prefix+"-rate-independent-nondimensional-dissipated-chain-scission-energy-per-segment-vs-lmbda_c_star")
        
        
        fig, ax1 = plt.subplots()

        filled_contour_plot = ax1.contourf(
            lmbda_c_star_steps_meshgrid, nu_list_meshgrid,
            overline_epsilon_cnu_diss_hat_lmbda_c_star_list, 100,
            cmap=plt.cm.cividis)
        
        for fcp in filled_contour_plot.collections:
            fcp.set_edgecolor('face')
        
        ax1.set_xlabel(r'$\lambda_c^{\star}$', fontsize=20)
        ax1.set_ylabel(r'$\nu$', fontsize=20)
        # ax1.set_xscale('log')
        ax1.set_yscale('log')
        # ax1.tick_params(axis='both', labelsize=16)

        cbar = fig.colorbar(filled_contour_plot)
        cbar.ax.set_ylabel(r'$\overline{\hat{\varepsilon}_{c\nu}^{diss}}$', fontsize=20)
        cbar.ax.tick_params(axis='y', labelsize=14)
        
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        
        save_current_figure_no_labels(
            self.savedir,
            data_file_prefix+"-rate-independent-nondimensional-scaled-dissipated-chain-scission-energy-per-segment-filled-contour-nu-vs-lmbda_c_star")
        
        
        fig = plt.figure()
        for nu_slice_indx in range(len(cp.nu_slice_list)):
            plt.plot(
                lmbda_c_star_steps,
                overline_epsilon_cnu_diss_hat_lmbda_c_star___nu_chunk_list[cp.nu_slice_indx_list[nu_slice_indx]],
                linestyle='-', color=cp.nu_slice_color_list[nu_slice_indx],
                alpha=1, linewidth=2.5,
                label=cp.nu_slice_label_list[nu_slice_indx])
        plt.legend(loc='best', fontsize=10)
        # plt.ylim([1e0, 1e11])
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\lambda_c^{\star}$', 20,
            r'$\overline{\hat{\varepsilon}_{c\nu}^{diss}}$', 20,
            data_file_prefix+"-rate-independent-nondimensional-scaled-dissipated-chain-scission-energy-per-segment-vs-lmbda_c_star")
        

        CN_fail_lmbda_c_star__lmbda_c_crit___nu_chunk_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-CN_fail_lmbda_c_star__lmbda_c_crit___nu_chunk_list")
        p_c_sci_hat_lmbda_c_star__lmbda_c_crit___nu_chunk_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-p_c_sci_hat_lmbda_c_star__lmbda_c_crit___nu_chunk_list")
        epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit___nu_chunk_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit___nu_chunk_list")
        # g_c_lmbda_c_star__lmbda_c_crit___nu_chunk_list = load_pickle_object(
        #     self.savedir, data_file_prefix+"-g_c_lmbda_c_star__lmbda_c_crit___nu_chunk_list")
        # g_c__nu_squared_lmbda_c_star__lmbda_c_crit___nu_chunk_list = load_pickle_object(
        #     self.savedir,
        #     data_file_prefix+"-g_c__nu_squared_lmbda_c_star__lmbda_c_crit___nu_chunk_list")
        overline_epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit___nu_chunk_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-overline_epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit___nu_chunk_list")
        # overline_g_c_lmbda_c_star__lmbda_c_crit___nu_chunk_list = load_pickle_object(
        #     self.savedir,
        #     data_file_prefix+"-overline_g_c_lmbda_c_star__lmbda_c_crit___nu_chunk_list")
        # overline_g_c__nu_squared_lmbda_c_star__lmbda_c_crit___nu_chunk_list = load_pickle_object(
        #     self.savedir,
        #     data_file_prefix+"-overline_g_c__nu_squared_lmbda_c_star__lmbda_c_crit___nu_chunk_list")
        
        lmbda_c_star__lmbda_c_crit_steps = np.linspace(0., 1., cp.lmbda_c_star_num)

        lmbda_c_star__lmbda_c_crit_steps_meshgrid, nu_list_meshgrid = np.meshgrid(
            lmbda_c_star__lmbda_c_crit_steps, cp.nu_list)

        CN_fail_lmbda_c_star__lmbda_c_crit_list = np.asarray(
            CN_fail_lmbda_c_star__lmbda_c_crit___nu_chunk_list)
        
        p_c_sci_hat_lmbda_c_star__lmbda_c_crit_list = np.asarray(
            p_c_sci_hat_lmbda_c_star__lmbda_c_crit___nu_chunk_list)
        
        epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit_list = np.asarray(
            epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit___nu_chunk_list)
        
        overline_epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit_list = np.asarray(
            overline_epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit___nu_chunk_list)
        

        fig, ax1 = plt.subplots()

        ax1.set_facecolor("gray")

        filled_contour_plot = ax1.contourf(
            lmbda_c_star__lmbda_c_crit_steps_meshgrid, nu_list_meshgrid,
            CN_fail_lmbda_c_star__lmbda_c_crit_list, locator=ticker.LogLocator(),
            cmap=plt.cm.cividis)
        
        for fcp in filled_contour_plot.collections:
            fcp.set_edgecolor('face')
        
        ax1.set_xlim(0.8, 1.)
        ax1.set_xlabel(r'$\lambda_c^{\star}/\lambda_c^{crit}$', fontsize=20)
        ax1.set_ylabel(r'$\nu$', fontsize=20)
        # ax1.set_xscale('log')
        ax1.set_yscale('log')
        # ax1.tick_params(axis='both', labelsize=16)

        cbar = fig.colorbar(filled_contour_plot)
        cbar.ax.set_ylabel(r'$\textrm{CN}^{\textrm{fail}}$', fontsize=20)
        cbar.ax.tick_params(axis='y', labelsize=14)
        
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)

        plt.tight_layout()
        plt.savefig(self.savedir+data_file_prefix+"-rate-independent-CN-fail-filled-contour-nu-vs-lmbda_c_star__lmbda_c_crit"+".pdf")
        plt.close()


        fig = plt.figure()
        for nu_slice_indx in range(len(cp.nu_slice_list)):
            plt.semilogy(
                lmbda_c_star__lmbda_c_crit_steps,
                CN_fail_lmbda_c_star__lmbda_c_crit___nu_chunk_list[cp.nu_slice_indx_list[nu_slice_indx]],
                linestyle='-', color=cp.nu_slice_color_list[nu_slice_indx],
                alpha=1, linewidth=2.5,
                label=cp.nu_slice_label_list[nu_slice_indx])
        plt.legend(loc='best', fontsize=10)
        plt.xlim([0.8, 1.])
        plt.ylim([1e0, 1e11])
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\lambda_c^{\star}/\lambda_c^{crit}$', 20,
            r'$\textrm{CN}^{\textrm{fail}}$', 20,
            data_file_prefix+"-rate-independent-CN-fail-vs-lmbda_c_star__lmbda_c_crit")
        

        fig, ax1 = plt.subplots()

        ax1.set_facecolor("gray")

        filled_contour_plot = ax1.contourf(
            lmbda_c_star__lmbda_c_crit_steps_meshgrid, nu_list_meshgrid,
            p_c_sci_hat_lmbda_c_star__lmbda_c_crit_list, locator=ticker.LogLocator(),
            cmap=plt.cm.cividis)
        
        for fcp in filled_contour_plot.collections:
            fcp.set_edgecolor('face')
        
        ax1.set_xlim(0.8, 1.)
        ax1.set_xlabel(r'$\lambda_c^{\star}/\lambda_c^{crit}$', fontsize=20)
        ax1.set_ylabel(r'$\nu$', fontsize=20)
        # ax1.set_xscale('log')
        ax1.set_yscale('log')
        # ax1.tick_params(axis='both', labelsize=16)

        cbar = fig.colorbar(filled_contour_plot)
        cbar.ax.set_ylabel(r'$d\overline{\mathcal{N}}_{\Gamma}/dN$', fontsize=20)
        cbar.ax.tick_params(axis='y', labelsize=14)
        
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)

        plt.tight_layout()
        plt.savefig(self.savedir+data_file_prefix+"-rate-independent-d_mathcal_N_Gamma__dN-filled-contour-nu-vs-lmbda_c_star__lmbda_c_crit"+".pdf")
        plt.close()


        fig = plt.figure()
        for nu_slice_indx in range(len(cp.nu_slice_list)):
            plt.plot(
                lmbda_c_star__lmbda_c_crit_steps,
                p_c_sci_hat_lmbda_c_star__lmbda_c_crit___nu_chunk_list[cp.nu_slice_indx_list[nu_slice_indx]],
                linestyle='-', color=cp.nu_slice_color_list[nu_slice_indx],
                alpha=1, linewidth=2.5,
                label=cp.nu_slice_label_list[nu_slice_indx])
        plt.legend(loc='best', fontsize=10)
        plt.xlim([0.8, 1.])
        plt.ylim([-0.05, 1.05])
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\lambda_c^{\star}/\lambda_c^{crit}$', 20,
            r'$d\overline{\mathcal{N}}_{\Gamma}/dN$', 20,
            data_file_prefix+"-rate-independent-d_mathcal_N_Gamma__dN-vs-lmbda_c_star__lmbda_c_crit")
        
        fig = plt.figure()
        for nu_slice_indx in range(len(cp.nu_slice_list)):
            plt.semilogy(
                lmbda_c_star__lmbda_c_crit_steps,
                p_c_sci_hat_lmbda_c_star__lmbda_c_crit___nu_chunk_list[cp.nu_slice_indx_list[nu_slice_indx]],
                linestyle='-', color=cp.nu_slice_color_list[nu_slice_indx],
                alpha=1, linewidth=2.5,
                label=cp.nu_slice_label_list[nu_slice_indx])
        plt.legend(loc='best', fontsize=10)
        plt.xlim([0.8, 1.])
        # plt.ylim([-0.05, 1.05])
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\lambda_c^{\star}/\lambda_c^{crit}$', 20,
            r'$d\overline{\mathcal{N}}_{\Gamma}/dN$', 20,
            data_file_prefix+"-rate-independent-d_mathcal_N_Gamma__dN-semilogy-vs-lmbda_c_star__lmbda_c_crit")
        
        fig = plt.figure()
        for nu_slice_indx in range(len(cp.nu_slice_list)):
            plt.loglog(
                lmbda_c_star__lmbda_c_crit_steps,
                p_c_sci_hat_lmbda_c_star__lmbda_c_crit___nu_chunk_list[cp.nu_slice_indx_list[nu_slice_indx]],
                linestyle='-', color=cp.nu_slice_color_list[nu_slice_indx],
                alpha=1, linewidth=2.5,
                label=cp.nu_slice_label_list[nu_slice_indx])
        plt.legend(loc='best', fontsize=10)
        plt.xlim([0.8, 1.])
        # plt.ylim([-0.05, 1.05])
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\lambda_c^{\star}/\lambda_c^{crit}$', 20,
            r'$d\overline{\mathcal{N}}_{\Gamma}/dN$', 20,
            data_file_prefix+"-rate-independent-d_mathcal_N_Gamma__dN-vs-lmbda_c_star__lmbda_c_crit-loglog")

        
        fig, ax1 = plt.subplots()

        filled_contour_plot = ax1.contourf(
            lmbda_c_star__lmbda_c_crit_steps_meshgrid, nu_list_meshgrid,
            epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit_list, 100,
            cmap=plt.cm.cividis)
        
        for fcp in filled_contour_plot.collections:
            fcp.set_edgecolor('face')
        
        ax1.set_xlim(0.8, 1.)
        ax1.set_xlabel(r'$\lambda_c^{\star}/\lambda_c^{crit}$', fontsize=20)
        ax1.set_ylabel(r'$\nu$', fontsize=20)
        # ax1.set_xscale('log')
        ax1.set_yscale('log')
        # ax1.tick_params(axis='both', labelsize=16)

        cbar = fig.colorbar(filled_contour_plot)
        cbar.ax.set_ylabel(r'$\hat{\varepsilon}_{c\nu}^{diss}$', fontsize=20)
        cbar.ax.tick_params(axis='y', labelsize=14)
        
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        
        save_current_figure_no_labels(
            self.savedir,
            data_file_prefix+"-rate-independent-nondimensional-dissipated-chain-scission-energy-per-segment-filled-contour-nu-vs-lmbda_c_star__lmbda_c_crit")
        

        fig = plt.figure()
        for nu_slice_indx in range(len(cp.nu_slice_list)):
            plt.plot(
                lmbda_c_star__lmbda_c_crit_steps,
                epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit___nu_chunk_list[cp.nu_slice_indx_list[nu_slice_indx]],
                linestyle='-', color=cp.nu_slice_color_list[nu_slice_indx],
                alpha=1, linewidth=2.5,
                label=cp.nu_slice_label_list[nu_slice_indx])
        plt.legend(loc='best', fontsize=10)
        plt.xlim([0.8, 1.])
        # plt.ylim([1e0, 1e11])
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\lambda_c^{\star}/\lambda_c^{crit}$', 20,
            r'$\hat{\varepsilon}_{c\nu}^{diss}$', 20,
            data_file_prefix+"-rate-independent-nondimensional-dissipated-chain-scission-energy-per-segment-vs-lmbda_c_star__lmbda_c_crit")
        
        
        fig, ax1 = plt.subplots()

        filled_contour_plot = ax1.contourf(
            lmbda_c_star__lmbda_c_crit_steps_meshgrid, nu_list_meshgrid,
            overline_epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit_list, 100,
            cmap=plt.cm.cividis)
        
        for fcp in filled_contour_plot.collections:
            fcp.set_edgecolor('face')
        
        ax1.set_xlim(0.8, 1.)
        ax1.set_xlabel(r'$\lambda_c^{\star}/\lambda_c^{crit}$', fontsize=20)
        ax1.set_ylabel(r'$\nu$', fontsize=20)
        # ax1.set_xscale('log')
        ax1.set_yscale('log')
        # ax1.tick_params(axis='both', labelsize=16)

        cbar = fig.colorbar(filled_contour_plot)
        cbar.ax.set_ylabel(r'$\overline{\hat{\varepsilon}_{c\nu}^{diss}}$', fontsize=20)
        cbar.ax.tick_params(axis='y', labelsize=14)
        
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        
        save_current_figure_no_labels(
            self.savedir,
            data_file_prefix+"-rate-independent-nondimensional-scaled-dissipated-chain-scission-energy-per-segment-filled-contour-nu-vs-lmbda_c_star__lmbda_c_crit")
        

        fig = plt.figure()
        for nu_slice_indx in range(len(cp.nu_slice_list)):
            plt.plot(
                lmbda_c_star__lmbda_c_crit_steps,
                overline_epsilon_cnu_diss_hat_lmbda_c_star__lmbda_c_crit___nu_chunk_list[cp.nu_slice_indx_list[nu_slice_indx]],
                linestyle='-', color=cp.nu_slice_color_list[nu_slice_indx],
                alpha=1, linewidth=2.5,
                label=cp.nu_slice_label_list[nu_slice_indx])
        plt.legend(loc='best', fontsize=10)
        plt.xlim([0.8, 1.])
        # plt.ylim([1e0, 1e11])
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\lambda_c^{\star}/\lambda_c^{crit}$', 20,
            r'$\hat{\varepsilon}_{c\nu}^{diss}$', 20,
            data_file_prefix+"-rate-independent-nondimensional-scaled-dissipated-chain-scission-energy-per-segment-vs-lmbda_c_star__lmbda_c_crit")
        

        CN_fail_xi_c_star__xi_c_crit___nu_chunk_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-CN_fail_xi_c_star__xi_c_crit___nu_chunk_list")
        p_c_sci_hat_xi_c_star__xi_c_crit___nu_chunk_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-p_c_sci_hat_xi_c_star__xi_c_crit___nu_chunk_list")
        epsilon_cnu_diss_hat_xi_c_star__xi_c_crit___nu_chunk_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-epsilon_cnu_diss_hat_xi_c_star__xi_c_crit___nu_chunk_list")
        # g_c_xi_c_star__xi_c_crit___nu_chunk_list = load_pickle_object(
        #     self.savedir, data_file_prefix+"-g_c_xi_c_star__xi_c_crit___nu_chunk_list")
        # g_c__nu_squared_xi_c_star__xi_c_crit___nu_chunk_list = load_pickle_object(
        #     self.savedir,
        #     data_file_prefix+"-g_c__nu_squared_xi_c_star__xi_c_crit___nu_chunk_list")
        overline_epsilon_cnu_diss_hat_xi_c_star__xi_c_crit___nu_chunk_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-overline_epsilon_cnu_diss_hat_xi_c_star__xi_c_crit___nu_chunk_list")
        # overline_g_c_xi_c_star__xi_c_crit___nu_chunk_list = load_pickle_object(
        #     self.savedir,
        #     data_file_prefix+"-overline_g_c_xi_c_star__xi_c_crit___nu_chunk_list")
        # overline_g_c__nu_squared_xi_c_star__xi_c_crit___nu_chunk_list = load_pickle_object(
        #     self.savedir,
        #     data_file_prefix+"-overline_g_c__nu_squared_xi_c_star__xi_c_crit___nu_chunk_list")
        
        xi_c_star__xi_c_crit_steps = np.linspace(0., 1., cp.xi_c_star_num)

        xi_c_star__xi_c_crit_steps_meshgrid, nu_list_meshgrid = np.meshgrid(
            xi_c_star__xi_c_crit_steps, cp.nu_list)

        CN_fail_xi_c_star__xi_c_crit_list = np.asarray(
            CN_fail_xi_c_star__xi_c_crit___nu_chunk_list)
        
        p_c_sci_hat_xi_c_star__xi_c_crit_list = np.asarray(
            p_c_sci_hat_xi_c_star__xi_c_crit___nu_chunk_list)
        
        epsilon_cnu_diss_hat_xi_c_star__xi_c_crit_list = np.asarray(
            epsilon_cnu_diss_hat_xi_c_star__xi_c_crit___nu_chunk_list)
        
        overline_epsilon_cnu_diss_hat_xi_c_star__xi_c_crit_list = np.asarray(
            overline_epsilon_cnu_diss_hat_xi_c_star__xi_c_crit___nu_chunk_list)
        

        fig, ax1 = plt.subplots()

        ax1.set_facecolor("gray")

        filled_contour_plot = ax1.contourf(
            xi_c_star__xi_c_crit_steps_meshgrid, nu_list_meshgrid,
            CN_fail_xi_c_star__xi_c_crit_list, locator=ticker.LogLocator(),
            cmap=plt.cm.cividis)
        
        for fcp in filled_contour_plot.collections:
            fcp.set_edgecolor('face')
        
        ax1.set_xlim(0.5, 1.)
        ax1.set_xlabel(r'$\xi_c^{\star}/\xi_c^{crit}$', fontsize=20)
        ax1.set_ylabel(r'$\nu$', fontsize=20)
        # ax1.set_xscale('log')
        ax1.set_yscale('log')
        # ax1.tick_params(axis='both', labelsize=16)

        cbar = fig.colorbar(filled_contour_plot)
        cbar.ax.set_ylabel(r'$\textrm{CN}^{\textrm{fail}}$', fontsize=20)
        cbar.ax.tick_params(axis='y', labelsize=14)
        
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)

        plt.tight_layout()
        plt.savefig(self.savedir+data_file_prefix+"-rate-independent-CN-fail-filled-contour-nu-vs-xi_c_star__xi_c_crit"+".pdf")
        plt.close()


        fig = plt.figure()
        for nu_slice_indx in range(len(cp.nu_slice_list)):
            plt.semilogy(
                xi_c_star__xi_c_crit_steps,
                CN_fail_xi_c_star__xi_c_crit___nu_chunk_list[cp.nu_slice_indx_list[nu_slice_indx]],
                linestyle='-', color=cp.nu_slice_color_list[nu_slice_indx],
                alpha=1, linewidth=2.5,
                label=cp.nu_slice_label_list[nu_slice_indx])
        plt.legend(loc='best', fontsize=10)
        plt.xlim([0.5, 1.])
        plt.ylim([1e0, 1e11])
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\xi_c^{\star}/\xi_c^{crit}$', 20,
            r'$\textrm{CN}^{\textrm{fail}}$', 20,
            data_file_prefix+"-rate-independent-CN-fail-vs-xi_c_star__xi_c_crit")
        

        fig, ax1 = plt.subplots()

        ax1.set_facecolor("gray")

        filled_contour_plot = ax1.contourf(
            xi_c_star__xi_c_crit_steps_meshgrid, nu_list_meshgrid,
            p_c_sci_hat_xi_c_star__xi_c_crit_list, locator=ticker.LogLocator(),
            cmap=plt.cm.cividis)
        
        for fcp in filled_contour_plot.collections:
            fcp.set_edgecolor('face')
        
        ax1.set_xlim(0.5, 1.)
        ax1.set_xlabel(r'$\xi_c^{\star}/\xi_c^{crit}$', fontsize=20)
        ax1.set_ylabel(r'$\nu$', fontsize=20)
        # ax1.set_xscale('log')
        ax1.set_yscale('log')
        # ax1.tick_params(axis='both', labelsize=16)

        cbar = fig.colorbar(filled_contour_plot)
        cbar.ax.set_ylabel(r'$d\overline{\mathcal{N}}_{\Gamma}/dN$', fontsize=20)
        cbar.ax.tick_params(axis='y', labelsize=14)
        
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)

        plt.tight_layout()
        plt.savefig(self.savedir+data_file_prefix+"-rate-independent-d_mathcal_N_Gamma__dN-filled-contour-nu-vs-xi_c_star__xi_c_crit"+".pdf")
        plt.close()


        fig = plt.figure()
        for nu_slice_indx in range(len(cp.nu_slice_list)):
            plt.plot(
                xi_c_star__xi_c_crit_steps,
                p_c_sci_hat_xi_c_star__xi_c_crit___nu_chunk_list[cp.nu_slice_indx_list[nu_slice_indx]],
                linestyle='-', color=cp.nu_slice_color_list[nu_slice_indx],
                alpha=1, linewidth=2.5,
                label=cp.nu_slice_label_list[nu_slice_indx])
        plt.legend(loc='best', fontsize=10)
        plt.xlim([0.5, 1.])
        plt.ylim([-0.05, 1.05])
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\xi_c^{\star}/\xi_c^{crit}$', 20,
            r'$d\overline{\mathcal{N}}_{\Gamma}/dN$', 20,
            data_file_prefix+"-rate-independent-d_mathcal_N_Gamma__dN-vs-xi_c_star__xi_c_crit")
        
        fig = plt.figure()
        for nu_slice_indx in range(len(cp.nu_slice_list)):
            plt.semilogy(
                xi_c_star__xi_c_crit_steps,
                p_c_sci_hat_xi_c_star__xi_c_crit___nu_chunk_list[cp.nu_slice_indx_list[nu_slice_indx]],
                linestyle='-', color=cp.nu_slice_color_list[nu_slice_indx],
                alpha=1, linewidth=2.5,
                label=cp.nu_slice_label_list[nu_slice_indx])
        plt.legend(loc='best', fontsize=10)
        plt.xlim([0.5, 1.])
        # plt.ylim([-0.05, 1.05])
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\xi_c^{\star}/\xi_c^{crit}$', 20,
            r'$d\overline{\mathcal{N}}_{\Gamma}/dN$', 20,
            data_file_prefix+"-rate-independent-d_mathcal_N_Gamma__dN-semilogy-vs-xi_c_star__xi_c_crit")
        
        fig = plt.figure()
        for nu_slice_indx in range(len(cp.nu_slice_list)):
            plt.loglog(
                xi_c_star__xi_c_crit_steps,
                p_c_sci_hat_xi_c_star__xi_c_crit___nu_chunk_list[cp.nu_slice_indx_list[nu_slice_indx]],
                linestyle='-', color=cp.nu_slice_color_list[nu_slice_indx],
                alpha=1, linewidth=2.5,
                label=cp.nu_slice_label_list[nu_slice_indx])
        plt.legend(loc='best', fontsize=10)
        plt.xlim([0.5, 1.])
        # plt.ylim([-0.05, 1.05])
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\xi_c^{\star}/\xi_c^{crit}$', 20,
            r'$d\overline{\mathcal{N}}_{\Gamma}/dN$', 20,
            data_file_prefix+"-rate-independent-d_mathcal_N_Gamma__dN-vs-xi_c_star__xi_c_crit-loglog")

        
        fig, ax1 = plt.subplots()

        filled_contour_plot = ax1.contourf(
            xi_c_star__xi_c_crit_steps_meshgrid, nu_list_meshgrid,
            epsilon_cnu_diss_hat_xi_c_star__xi_c_crit_list, 100,
            cmap=plt.cm.cividis)
        
        for fcp in filled_contour_plot.collections:
            fcp.set_edgecolor('face')
        
        ax1.set_xlim(0.5, 1.)
        ax1.set_xlabel(r'$\xi_c^{\star}/\xi_c^{crit}$', fontsize=20)
        ax1.set_ylabel(r'$\nu$', fontsize=20)
        # ax1.set_xscale('log')
        ax1.set_yscale('log')
        # ax1.tick_params(axis='both', labelsize=16)

        cbar = fig.colorbar(filled_contour_plot)
        cbar.ax.set_ylabel(r'$\hat{\varepsilon}_{c\nu}^{diss}$', fontsize=20)
        cbar.ax.tick_params(axis='y', labelsize=14)
        
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        
        save_current_figure_no_labels(
            self.savedir,
            data_file_prefix+"-rate-independent-nondimensional-dissipated-chain-scission-energy-per-segment-filled-contour-nu-vs-xi_c_star__xi_c_crit")
        

        fig = plt.figure()
        for nu_slice_indx in range(len(cp.nu_slice_list)):
            plt.plot(
                xi_c_star__xi_c_crit_steps,
                epsilon_cnu_diss_hat_xi_c_star__xi_c_crit___nu_chunk_list[cp.nu_slice_indx_list[nu_slice_indx]],
                linestyle='-', color=cp.nu_slice_color_list[nu_slice_indx],
                alpha=1, linewidth=2.5,
                label=cp.nu_slice_label_list[nu_slice_indx])
        plt.legend(loc='best', fontsize=10)
        plt.xlim([0.5, 1.])
        # plt.ylim([1e0, 1e11])
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\xi_c^{\star}/\xi_c^{crit}$', 20,
            r'$\hat{\varepsilon}_{c\nu}^{diss}$', 20,
            data_file_prefix+"-rate-independent-nondimensional-dissipated-chain-scission-energy-per-segment-vs-xi_c_star__xi_c_crit")
        
        
        fig, ax1 = plt.subplots()

        filled_contour_plot = ax1.contourf(
            xi_c_star__xi_c_crit_steps_meshgrid, nu_list_meshgrid,
            overline_epsilon_cnu_diss_hat_xi_c_star__xi_c_crit_list, 100,
            cmap=plt.cm.cividis)
        
        for fcp in filled_contour_plot.collections:
            fcp.set_edgecolor('face')
        
        ax1.set_xlim(0.5, 1.)
        ax1.set_xlabel(r'$\xi_c^{\star}/\xi_c^{crit}$', fontsize=20)
        ax1.set_ylabel(r'$\nu$', fontsize=20)
        # ax1.set_xscale('log')
        ax1.set_yscale('log')
        # ax1.tick_params(axis='both', labelsize=16)

        cbar = fig.colorbar(filled_contour_plot)
        cbar.ax.set_ylabel(r'$\overline{\hat{\varepsilon}_{c\nu}^{diss}}$', fontsize=20)
        cbar.ax.tick_params(axis='y', labelsize=14)
        
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        
        save_current_figure_no_labels(
            self.savedir,
            data_file_prefix+"-rate-independent-nondimensional-scaled-dissipated-chain-scission-energy-per-segment-filled-contour-nu-vs-xi_c_star__xi_c_crit")
        

        fig = plt.figure()
        for nu_slice_indx in range(len(cp.nu_slice_list)):
            plt.plot(
                xi_c_star__xi_c_crit_steps,
                overline_epsilon_cnu_diss_hat_xi_c_star__xi_c_crit___nu_chunk_list[cp.nu_slice_indx_list[nu_slice_indx]],
                linestyle='-', color=cp.nu_slice_color_list[nu_slice_indx],
                alpha=1, linewidth=2.5,
                label=cp.nu_slice_label_list[nu_slice_indx])
        plt.legend(loc='best', fontsize=10)
        plt.xlim([0.5, 1.])
        # plt.ylim([1e0, 1e11])
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\xi_c^{\star}/\xi_c^{crit}$', 20,
            r'$\hat{\varepsilon}_{c\nu}^{diss}$', 20,
            data_file_prefix+"-rate-independent-nondimensional-scaled-dissipated-chain-scission-energy-per-segment-vs-xi_c_star__xi_c_crit")
        

        CN_fail_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-CN_fail_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list")
        p_c_sci_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-p_c_sci_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list")
        epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list")
        # g_c_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list = load_pickle_object(
        #     self.savedir, data_file_prefix+"-g_c_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list")
        # g_c__nu_squared_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list = load_pickle_object(
        #     self.savedir,
        #     data_file_prefix+"-g_c__nu_squared_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list")
        overline_epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list = load_pickle_object(
            self.savedir,
            data_file_prefix+"-overline_epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list")
        # overline_g_c_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list = load_pickle_object(
        #     self.savedir,
        #     data_file_prefix+"-overline_g_c_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list")
        # overline_g_c__nu_squared_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list = load_pickle_object(
        #     self.savedir,
        #     data_file_prefix+"-overline_g_c__nu_squared_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list")
        
        lmbda_nu_hat_star__lmbda_nu_hat_crit_steps = np.linspace(0., 1., cp.lmbda_nu_hat_star_num)

        lmbda_nu_hat_star__lmbda_nu_hat_crit_steps_meshgrid, nu_list_meshgrid = np.meshgrid(
            lmbda_nu_hat_star__lmbda_nu_hat_crit_steps, cp.nu_list)

        CN_fail_lmbda_nu_hat_star__lmbda_nu_hat_crit_list = np.asarray(
            CN_fail_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list)
        
        p_c_sci_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit_list = np.asarray(
            p_c_sci_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list)
        
        epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit_list = np.asarray(
            epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list)
        
        overline_epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit_list = np.asarray(
            overline_epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list)
        

        fig, ax1 = plt.subplots()

        ax1.set_facecolor("gray")

        filled_contour_plot = ax1.contourf(
            lmbda_nu_hat_star__lmbda_nu_hat_crit_steps_meshgrid, nu_list_meshgrid,
            CN_fail_lmbda_nu_hat_star__lmbda_nu_hat_crit_list, locator=ticker.LogLocator(),
            cmap=plt.cm.cividis)
        
        for fcp in filled_contour_plot.collections:
            fcp.set_edgecolor('face')
        
        ax1.set_xlim(0.5, 1.)
        ax1.set_xlabel(r'$[\hat{\lambda}_{\nu}^{\star}-1]/\hat{\lambda}_{\nu}^{crit}$', fontsize=20)
        ax1.set_ylabel(r'$\nu$', fontsize=20)
        # ax1.set_xscale('log')
        ax1.set_yscale('log')
        # ax1.tick_params(axis='both', labelsize=16)

        cbar = fig.colorbar(filled_contour_plot)
        cbar.ax.set_ylabel(r'$\textrm{CN}^{\textrm{fail}}$', fontsize=20)
        cbar.ax.tick_params(axis='y', labelsize=14)
        
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)

        plt.tight_layout()
        plt.savefig(self.savedir+data_file_prefix+"-rate-independent-CN-fail-filled-contour-nu-vs-lmbda_nu_hat_star_minus_1__lmbda_nu_hat_crit"+".pdf")
        plt.close()


        fig = plt.figure()
        for nu_slice_indx in range(len(cp.nu_slice_list)):
            plt.semilogy(
                lmbda_nu_hat_star__lmbda_nu_hat_crit_steps,
                CN_fail_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list[cp.nu_slice_indx_list[nu_slice_indx]],
                linestyle='-', color=cp.nu_slice_color_list[nu_slice_indx],
                alpha=1, linewidth=2.5,
                label=cp.nu_slice_label_list[nu_slice_indx])
        plt.legend(loc='best', fontsize=10)
        plt.xlim([0.5, 1.])
        plt.ylim([1e0, 1e11])
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$[\hat{\lambda}_{\nu}^{\star}-1]/\hat{\lambda}_{\nu}^{crit}$', 20,
            r'$\textrm{CN}^{\textrm{fail}}$', 20,
            data_file_prefix+"-rate-independent-CN-fail-vs-lmbda_nu_hat_star_minus_1__lmbda_nu_hat_crit")
        

        fig, ax1 = plt.subplots()

        ax1.set_facecolor("gray")

        filled_contour_plot = ax1.contourf(
            lmbda_nu_hat_star__lmbda_nu_hat_crit_steps_meshgrid, nu_list_meshgrid,
            p_c_sci_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit_list, locator=ticker.LogLocator(),
            cmap=plt.cm.cividis)
        
        for fcp in filled_contour_plot.collections:
            fcp.set_edgecolor('face')
        
        ax1.set_xlim(0.5, 1.)
        ax1.set_xlabel(r'$[\hat{\lambda}_{\nu}^{\star}-1]/\hat{\lambda}_{\nu}^{crit}$', fontsize=20)
        ax1.set_ylabel(r'$\nu$', fontsize=20)
        # ax1.set_xscale('log')
        ax1.set_yscale('log')
        # ax1.tick_params(axis='both', labelsize=16)

        cbar = fig.colorbar(filled_contour_plot)
        cbar.ax.set_ylabel(r'$d\overline{\mathcal{N}}_{\Gamma}/dN$', fontsize=20)
        cbar.ax.tick_params(axis='y', labelsize=14)
        
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)

        plt.tight_layout()
        plt.savefig(self.savedir+data_file_prefix+"-rate-independent-d_mathcal_N_Gamma__dN-filled-contour-nu-vs-lmbda_nu_hat_star_minus_1__lmbda_nu_hat_crit"+".pdf")
        plt.close()


        fig = plt.figure()
        for nu_slice_indx in range(len(cp.nu_slice_list)):
            plt.plot(
                lmbda_nu_hat_star__lmbda_nu_hat_crit_steps,
                p_c_sci_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list[cp.nu_slice_indx_list[nu_slice_indx]],
                linestyle='-', color=cp.nu_slice_color_list[nu_slice_indx],
                alpha=1, linewidth=2.5,
                label=cp.nu_slice_label_list[nu_slice_indx])
        plt.legend(loc='best', fontsize=10)
        plt.xlim([0.5, 1.])
        plt.ylim([-0.05, 1.05])
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$[\hat{\lambda}_{\nu}^{\star}-1]/\hat{\lambda}_{\nu}^{crit}$', 20,
            r'$d\overline{\mathcal{N}}_{\Gamma}/dN$', 20,
            data_file_prefix+"-rate-independent-d_mathcal_N_Gamma__dN-vs-lmbda_nu_hat_star_minus_1__lmbda_nu_hat_crit")
        
        fig = plt.figure()
        for nu_slice_indx in range(len(cp.nu_slice_list)):
            plt.semilogy(
                lmbda_nu_hat_star__lmbda_nu_hat_crit_steps,
                p_c_sci_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list[cp.nu_slice_indx_list[nu_slice_indx]],
                linestyle='-', color=cp.nu_slice_color_list[nu_slice_indx],
                alpha=1, linewidth=2.5,
                label=cp.nu_slice_label_list[nu_slice_indx])
        plt.legend(loc='best', fontsize=10)
        plt.xlim([0.5, 1.])
        # plt.ylim([-0.05, 1.05])
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$[\hat{\lambda}_{\nu}^{\star}-1]/\hat{\lambda}_{\nu}^{crit}$', 20,
            r'$d\overline{\mathcal{N}}_{\Gamma}/dN$', 20,
            data_file_prefix+"-rate-independent-d_mathcal_N_Gamma__dN-semilogy-vs-lmbda_nu_hat_star_minus_1__lmbda_nu_hat_crit")
        
        fig = plt.figure()
        for nu_slice_indx in range(len(cp.nu_slice_list)):
            plt.loglog(
                lmbda_nu_hat_star__lmbda_nu_hat_crit_steps,
                p_c_sci_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list[cp.nu_slice_indx_list[nu_slice_indx]],
                linestyle='-', color=cp.nu_slice_color_list[nu_slice_indx],
                alpha=1, linewidth=2.5,
                label=cp.nu_slice_label_list[nu_slice_indx])
        plt.legend(loc='best', fontsize=10)
        plt.xlim([0.5, 1.])
        # plt.ylim([-0.05, 1.05])
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$[\hat{\lambda}_{\nu}^{\star}-1]/\hat{\lambda}_{\nu}^{crit}$', 20,
            r'$d\overline{\mathcal{N}}_{\Gamma}/dN$', 20,
            data_file_prefix+"-rate-independent-d_mathcal_N_Gamma__dN-vs-lmbda_nu_hat_star_minus_1__lmbda_nu_hat_crit-loglog")

        
        fig, ax1 = plt.subplots()

        filled_contour_plot = ax1.contourf(
            lmbda_nu_hat_star__lmbda_nu_hat_crit_steps_meshgrid, nu_list_meshgrid,
            epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit_list, 100,
            cmap=plt.cm.cividis)
        
        for fcp in filled_contour_plot.collections:
            fcp.set_edgecolor('face')
        
        ax1.set_xlim(0.5, 1.)
        ax1.set_xlabel(r'$[\hat{\lambda}_{\nu}^{\star}-1]/\hat{\lambda}_{\nu}^{crit}$', fontsize=20)
        ax1.set_ylabel(r'$\nu$', fontsize=20)
        # ax1.set_xscale('log')
        ax1.set_yscale('log')
        # ax1.tick_params(axis='both', labelsize=16)

        cbar = fig.colorbar(filled_contour_plot)
        cbar.ax.set_ylabel(r'$\hat{\varepsilon}_{c\nu}^{diss}$', fontsize=20)
        cbar.ax.tick_params(axis='y', labelsize=14)
        
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        
        save_current_figure_no_labels(
            self.savedir,
            data_file_prefix+"-rate-independent-nondimensional-dissipated-chain-scission-energy-per-segment-filled-contour-nu-vs-lmbda_nu_hat_star_minus_1__lmbda_nu_hat_crit")
        

        fig = plt.figure()
        for nu_slice_indx in range(len(cp.nu_slice_list)):
            plt.plot(
                lmbda_nu_hat_star__lmbda_nu_hat_crit_steps,
                epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list[cp.nu_slice_indx_list[nu_slice_indx]],
                linestyle='-', color=cp.nu_slice_color_list[nu_slice_indx],
                alpha=1, linewidth=2.5,
                label=cp.nu_slice_label_list[nu_slice_indx])
        plt.legend(loc='best', fontsize=10)
        plt.xlim([0.5, 1.])
        # plt.ylim([1e0, 1e11])
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$[\hat{\lambda}_{\nu}^{\star}-1]/\hat{\lambda}_{\nu}^{crit}$', 20,
            r'$\hat{\varepsilon}_{c\nu}^{diss}$', 20,
            data_file_prefix+"-rate-independent-nondimensional-dissipated-chain-scission-energy-per-segment-vs-lmbda_nu_hat_star_minus_1__lmbda_nu_hat_crit")
        
        
        fig, ax1 = plt.subplots()

        filled_contour_plot = ax1.contourf(
            lmbda_nu_hat_star__lmbda_nu_hat_crit_steps_meshgrid, nu_list_meshgrid,
            overline_epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit_list, 100,
            cmap=plt.cm.cividis)
        
        for fcp in filled_contour_plot.collections:
            fcp.set_edgecolor('face')
        
        ax1.set_xlim(0.5, 1.)
        ax1.set_xlabel(r'$[\hat{\lambda}_{\nu}^{\star}-1]/\hat{\lambda}_{\nu}^{crit}$', fontsize=20)
        ax1.set_ylabel(r'$\nu$', fontsize=20)
        # ax1.set_xscale('log')
        ax1.set_yscale('log')
        # ax1.tick_params(axis='both', labelsize=16)

        cbar = fig.colorbar(filled_contour_plot)
        cbar.ax.set_ylabel(r'$\overline{\hat{\varepsilon}_{c\nu}^{diss}}$', fontsize=20)
        cbar.ax.tick_params(axis='y', labelsize=14)
        
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        
        save_current_figure_no_labels(
            self.savedir,
            data_file_prefix+"-rate-independent-nondimensional-scaled-dissipated-chain-scission-energy-per-segment-filled-contour-nu-vs-lmbda_nu_hat_star_minus_1__lmbda_nu_hat_crit")
        

        fig = plt.figure()
        for nu_slice_indx in range(len(cp.nu_slice_list)):
            plt.plot(
                lmbda_nu_hat_star__lmbda_nu_hat_crit_steps,
                overline_epsilon_cnu_diss_hat_lmbda_nu_hat_star__lmbda_nu_hat_crit___nu_chunk_list[cp.nu_slice_indx_list[nu_slice_indx]],
                linestyle='-', color=cp.nu_slice_color_list[nu_slice_indx],
                alpha=1, linewidth=2.5,
                label=cp.nu_slice_label_list[nu_slice_indx])
        plt.legend(loc='best', fontsize=10)
        plt.xlim([0.5, 1.])
        # plt.ylim([1e0, 1e11])
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$[\hat{\lambda}_{\nu}^{\star}-1]/\hat{\lambda}_{\nu}^{crit}$', 20,
            r'$\hat{\varepsilon}_{c\nu}^{diss}$', 20,
            data_file_prefix+"-rate-independent-nondimensional-scaled-dissipated-chain-scission-energy-per-segment-vs-lmbda_nu_hat_star_minus_1__lmbda_nu_hat_crit")

if __name__ == '__main__':

    T = 298 # absolute room temperature, K

    AFM_chain_tensile_tests_dict = {
        "al-maawali-et-al": "chain-a", "hugel-et-al": "chain-a"
    }

    al_maawali_et_al_intrinsic_fracture_toughness_characterizer = (
        IntrinsicFractureToughnessCharacterizer(
            paper_authors="al-maawali-et-al", chain="chain-a", T=T)
    )
    # al_maawali_et_al_intrinsic_fracture_toughness_characterizer.characterization()
    al_maawali_et_al_intrinsic_fracture_toughness_characterizer.finalization()

    hugel_et_al_intrinsic_fracture_toughness_characterizer = (
        IntrinsicFractureToughnessCharacterizer(
            paper_authors="hugel-et-al", chain="chain-a", T=T)
    )
    # hugel_et_al_intrinsic_fracture_toughness_characterizer.characterization()
    hugel_et_al_intrinsic_fracture_toughness_characterizer.finalization()