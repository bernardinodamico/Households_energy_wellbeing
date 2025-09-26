import pandas as pd
from DataFusion import gen_training_dataset
from Estimator import ComputeEffects
from Plotter import Plotter
pd.option_context('display.max_rows', None)


import numpy as np
import matplotlib.pyplot as plt


def make_ATEg_plot() -> None:
    #set bin number for real-valued variables
    Y0bn = 35
    Wbn = 13 
    V1bn = 13
    V7bn = 13
    Laplace_sm = 0.001

    discretised_dtset = gen_training_dataset(Y_0_bins_num=Y0bn, W_bins_num=Wbn, V_1_bins_num=V1bn, V_7_bins_num=V7bn)

    ce = ComputeEffects()
    p_Y0_given_doXx_1G, p_Y0_given_doXx_2G, exp_Y0_given_doXx_1G, exp_Y0_given_doXx_2G = ce.compute_ATE(Y_0_bins_num=Y0bn, 
                                                                                                        W_bins_num=Wbn, 
                                                                                                        V_1_bins_num=V1bn, 
                                                                                                        V_7_bins_num=V7bn, 
                                                                                                        Laplace_sm=Laplace_sm, 
                                                                                                        dd=discretised_dtset)

    # Plot ATEs 
    plotter = Plotter()
    plotter.plot_ATE(figure_name=f'ATE', 
                    width_cm=8., 
                    height_cm=10.,
                    doXx_1_distrib=p_Y0_given_doXx_1G, 
                    doXx_2_distrib=p_Y0_given_doXx_2G,
                    exp_Xx_1=exp_Y0_given_doXx_1G,
                    exp_Xx_2=exp_Y0_given_doXx_2G,
                    )
    return

def make_CATEg_plot() -> None:
    #set bin number for real-valued variables
    Y0bn = 10
    Wbn = 12
    V1bn = 13
    V7bn = 13
    Laplace_sm = 0.002

    # Generate training dataset 
    discretised_dtset = gen_training_dataset(Y_0_bins_num=Y0bn, W_bins_num=Wbn, V_1_bins_num=V1bn, V_7_bins_num=V7bn)

    ce = ComputeEffects()
    list_w, list_distribs_doXx_1, list_distribs_doXx_2, list_exp_Y0_given_doXx_1_Ww_1, list_exp_Y0_given_doXx_2_Ww_1 = ce.compute_CATE(Y_0_bins_num=Y0bn, 
                                                                                                                                       W_bins_num=Wbn, 
                                                                                                                                       V_1_bins_num=V1bn, 
                                                                                                                                       V_7_bins_num=V7bn, 
                                                                                                                                       Laplace_sm=Laplace_sm, 
                                                                                                                                       dd=discretised_dtset)

    # Plot CATEs
    plotter = Plotter()
    plotter.plot_CATE(figure_name=f'CATE',
                    width_cm=12.,
                    height_cm=15.5,
                    w_values=list_w,
                    list_distribs_doXx_1=list_distribs_doXx_1,
                    list_exp_Y0_given_doXx_1_Ww_1=list_exp_Y0_given_doXx_1_Ww_1,
                    list_distribs_doXx_2=list_distribs_doXx_2,
                    list_exp_Y0_given_doXx_2_Ww_1=list_exp_Y0_given_doXx_2_Ww_1
                    )
    return




def make_plot_sensitivity_analisys() -> None:
    ATE_G = -2980.1  # kWh/yr

    # Define ranges for pi and gamma
    pi_vals = np.linspace(0, 1, 300)  # pi ranges from 0 (no effect on treatment) to 1 (deterministic)
    gamma_vals = np.linspace(-6000, 0, 300)  # plausible outcome shifts in kWh/yr

    # Create grid
    PI, GAMMA = np.meshgrid(pi_vals, gamma_vals)

    # Compute adjusted ATE
    ATE_adj = ATE_G - PI * GAMMA

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))

    # Shaded heatmap of adjusted effects
    cmap = plt.cm.coolwarm
    im = ax.contourf(PI, GAMMA, ATE_adj, levels=50, cmap=cmap, alpha=0.9)

    # Contour lines
    contours = ax.contour(PI, GAMMA, ATE_adj, levels=[0], colors='black', linewidths=1.6)
    #ax.clabel(contours, inline=True, fontsize=10, fmt={0: 'Tipping line (ATE_adj=0)'})

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$ATE_{adj}$ [kWh/yr]')

    # Axis labels and title
    ax.set_xlabel(r'Confounder-Treatment association ($\pi$)')
    ax.set_ylabel(r'Confounder-Outcome association ($\gamma$) [kWh/yr]')

    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(-6000, 500, 500))

    ax.grid(True, which='both', linestyle='--', color='black', linewidth=0.7, alpha=0.6)
    
    # Reference lines for readability
    ax.axhline(0, color='grey', linestyle='--', linewidth=1)
    ax.axvline(0, color='grey', linestyle='--', linewidth=1)

    plt.tight_layout()
    #plt.show()

    fig.savefig(f"Figures/figure_sensitivity.png", bbox_inches="tight", dpi=600)

    return


if __name__ == "__main__":
    make_ATEg_plot()
    make_CATEg_plot()

    make_plot_sensitivity_analisys()