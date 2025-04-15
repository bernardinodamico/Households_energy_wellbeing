import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
from pandas import DataFrame
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib.ticker import MultipleLocator
import pandas as pd
import matplotlib.gridspec as gridspec
from numpy.polynomial.polynomial import Polynomial


class Plotter():


    def plot_ATE(self, figure_name: str, width_cm: float, height_cm: float, doXx_1_distrib: DataFrame, doXx_2_distrib: DataFrame, exp_Xx_1: float, exp_Xx_2: float) -> None:
        '''
        NOTE: doXx_1_distrib and doXx_2_distrib are two-column dataframes 
        reporting the post-intervention probability distribution P(Y_0 | do(X=1) and P(Y_0 | do(X=2) 
        '''
        doXx_1_distrib.iloc[0, 0] = doXx_1_distrib.iloc[1, 0] - self._bin_width(doXx_1_distrib)
        doXx_1_distrib.iloc[-1, 0] = doXx_1_distrib.iloc[-2, 0] + self._bin_width(doXx_1_distrib)
        doXx_2_distrib.iloc[0, 0] = doXx_2_distrib.iloc[1, 0] - self._bin_width(doXx_2_distrib)
        doXx_2_distrib.iloc[-1, 0] = doXx_2_distrib.iloc[-2, 0] + self._bin_width(doXx_2_distrib)

        fig = plt.figure(figsize=(width_cm/2.54, height_cm/2.54))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2.6, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        

        # Add regression curves
        x, y = self._add_regresssion_curves(df_Xx=doXx_1_distrib, prob_col_name='P(Y_0 | do(X=1))')
        ax1.plot(x, y, linewidth=1.1, color='#42A05C')
        x2, y2 = self._add_regresssion_curves(df_Xx=doXx_2_distrib, prob_col_name='P(Y_0 | do(X=2))')
        ax1.plot(x2, y2, linewidth=1.1, color='#B35933')

        ax1.axvline(exp_Xx_1, color='#42A05C', linestyle='--', linewidth=1.3)
        ax1.axvline(exp_Xx_2, color='#B35933', linestyle='--', linewidth=1.3)

        # Add bars
        ax1.bar(x=doXx_1_distrib['Y_0'], height=doXx_1_distrib['P(Y_0 | do(X=1))'], width=self._bin_width(doXx_1_distrib), edgecolor='#3CB371', alpha=0.8, label="X = false", color='#3CB371')
        ax1.bar(x=doXx_2_distrib['Y_0'], height=doXx_2_distrib['P(Y_0 | do(X=2))'], width=self._bin_width(doXx_2_distrib), edgecolor='#FF6347', alpha=0.65, label="X = true", color='#FF6347')

        plt.rcParams["font.family"] = "Arial"
        ax1.set_xlabel('')
        ax1.set_ylabel(r'$P(Y_0 \mid do(X))_{G}$', fontsize=8)

        ax1.minorticks_on()
        ax1.yaxis.set_major_locator(plt.MultipleLocator(0.05))
        ax1.yaxis.set_minor_locator(plt.MultipleLocator(0.01))

        ax1.tick_params(axis='x', which='major', direction='out', length=4, labelsize=7)
        ax1.tick_params(axis='x', which='minor', direction='out', length=2)
        ax1.tick_params(axis='y', which='major', direction='in', length=4, labelsize=7)
        ax1.tick_params(axis='y', which='minor', direction='in', length=2)

        ax1.axvline(40000, color='dimgray', linestyle='--', linewidth=1.3, label=r'$E(Y_0 \mid do(X))$')
        order = [1, 2, 0]  # Example: third, first, second
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend([handles[idx] for idx in order], [labels[idx] for idx in order], title=r'Wall insulation $(X):$', frameon=True, fontsize=7, title_fontsize=7)
        #ax1.legend(title="Wall insulation:", frameon=True, fontsize=7, title_fontsize=7)
        
        ax1.set_xticklabels([]) 
        
        y_axis_upper_limit = max(doXx_2_distrib['P(Y_0 | do(X=2))'].to_list()) * 1.15
        ax1.set_ylim(0, y_axis_upper_limit)
        ax1.set_xlim(1000, 35000)
        

 
        ax1.annotate("", xy=(exp_Xx_1, y_axis_upper_limit*0.63), xytext=(exp_Xx_2-300, y_axis_upper_limit*0.63), arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='<->', lw=0.8, shrinkB=0.))
        ax1.text(exp_Xx_1+400, y_axis_upper_limit*0.63, r'$ATE_{G}$' + rf'$ = {int(exp_Xx_2-exp_Xx_1)}$', fontsize=7, ha='left', va='center')
        ax1.text(-0.24, 1.07, 'a', transform=ax1.transAxes, fontsize=11, fontweight='bold', va='top', ha='left')
        #-----------------------------------------------------------------------------------------------

        bar_height = pd.DataFrame()
        bar_height['Y_0'] = doXx_1_distrib['Y_0']
        bar_height['PR(Y_0)'] = (doXx_2_distrib['P(Y_0 | do(X=2))'] / doXx_1_distrib['P(Y_0 | do(X=1))']) - 1.
        
        #p = Polynomial.fit(bar_height['Y_0'], bar_height['PR(Y_0)'], 5)
        # Add regression curve
        #x = np.linspace(bar_height['Y_0'].min(), bar_height['Y_0'].max(), 1000)
        #y = p(x)
        #ax2.plot(x, y, linewidth=0.9, color='darkblue')
        
        ax2.axhline(y=0., color='black', linestyle='-', linewidth=0.8)
        ax2.axvline(x=13000., color='darkblue', linestyle='-.', linewidth=0.8)
        ax2.text(13000, -0.9, '13000', fontsize=7, ha='center', va='top', rotation=90)

        ax2.bar(x=doXx_1_distrib['Y_0'], height=bar_height['PR(Y_0)'], width=self._bin_width(doXx_1_distrib), edgecolor='royalblue', alpha=0.8, color='royalblue')
        
        ax2.set_xlabel(r'Gas consumption $(Y_0)$ [kWh/year]', fontsize=8)
        ax2.set_ylabel(r'$\frac{P(Y_0 \mid do(X=true))_{G}}{P(Y_0 \mid do(X=false))_{G}}$', fontsize=10)

        ax2.minorticks_on()
        ax2.yaxis.set_major_locator(plt.MultipleLocator(1))
        ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.5))

        ax2.tick_params(axis='x', which='major', direction='out', length=4, labelsize=7)
        ax2.tick_params(axis='x', which='minor', direction='out', length=2)
        ax2.tick_params(axis='y', which='major', direction='in', length=4, labelsize=7)
        ax2.tick_params(axis='y', which='minor', direction='in', length=2)

        plt.xticks(rotation=90)

        # Set new y labels shifted by +1
        yticks = ax2.get_yticks()
        ax2.set_yticklabels([str(int(tick + 1)) for tick in yticks])
        
        #ax2.set_ylim(-3, 3.)
        ax2.set_xlim(1000, 35000)

        ax2.text(-0.24, -.07, 'b', transform=ax1.transAxes, fontsize=11, fontweight='bold', va='top', ha='left')


        w = str(width_cm).replace('.', '-')
        h = str(height_cm).replace('.', '-')
        fig.savefig(f"Figures/figure_{w}_cm_by_{h}_cm_{figure_name}.png", bbox_inches="tight", dpi=600)

        return
   

    def plot_CATE(self, figure_name: str, width_cm: float, height_cm: float, w_values: list, list_distribs_doXx_1: DataFrame, list_exp_Y0_given_doXx_1_Ww_1: list[float], list_distribs_doXx_2: DataFrame, list_exp_Y0_given_doXx_2_Ww_1: list[float]):
        
        w_values[0] = float(w_values[1]) - (float(w_values[2]) - float(w_values[1]))
        w_values[-1] = float(w_values[-2]) + (float(w_values[2]) - float(w_values[1]))
        samples_df, percentiles, means = self._generate_points_for_CATE(w_values=w_values, list_distribs_doX_given_W=list_distribs_doXx_1)

        fig = plt.figure(figsize=(width_cm/2.54, height_cm/2.54))
        gs = gridspec.GridSpec(3, 4, height_ratios=[4.8, 1.3, 1.7], hspace=0.4)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, :])
        ax_bottom_1 = fig.add_subplot(gs[2, 0])
        ax_bottom_2 = fig.add_subplot(gs[2, 1])
        ax_bottom_3 = fig.add_subplot(gs[2, 2])
        ax_bottom_4 = fig.add_subplot(gs[2, 3])

        # Shaded percentile bands
        ax1.fill_between(percentiles['W_center'], percentiles['q40'], percentiles['q60'], alpha=0.15, label='_40-60%', color='#3CB371', linewidth=0.7, linestyle='--', edgecolor='black')
        ax1.fill_between(percentiles['W_center'], percentiles['q42'], percentiles['q58'], alpha=0.25, label='_42-58%', color='#3CB371', linewidth=0.2)
        ax1.fill_between(percentiles['W_center'], percentiles['q44'], percentiles['q56'], alpha=0.35, label='_44-56%', color='#3CB371', linewidth=0.2)
        ax1.fill_between(percentiles['W_center'], percentiles['q46'], percentiles['q54'], alpha=0.45, label='_46-54%', color='#3CB371', linewidth=0.2)
        ax1.fill_between(percentiles['W_center'], percentiles['q48'], percentiles['q52'], alpha=0.55, label='X = false', color='#3CB371', linewidth=0.2) #'_48-52%'
        ax1.plot(percentiles['W_center'], percentiles['q50'], color='#217338', label='_Median', linestyle='-', linewidth=1.5, alpha=0.7)
        ax1.plot(np.asarray(w_values, dtype=float), list_exp_Y0_given_doXx_1_Ww_1, color='#217338', label='_no_label', linestyle='--', linewidth=1.4)
        median_doX_1 = percentiles['q50']

        ax1.set_xlabel('')
        ax1.set_ylabel(r'Gas consumption $(Y_0)$ [kWh/year]', fontsize=8)
        plt.rcParams["font.family"] = "Arial"
        ax1.minorticks_on()
        ax1.xaxis.set_major_locator(plt.MultipleLocator(0.01))
        ax1.xaxis.set_minor_locator(plt.MultipleLocator(0.005))
        ax1.tick_params(axis='x', which='major', direction='out', length=4, labelsize=7)
        ax1.tick_params(axis='x', which='minor', direction='out', length=2)
        ax1.tick_params(axis='y', which='major', direction='in', length=4, labelsize=7)
        ax1.tick_params(axis='y', which='minor', direction='in', length=2)

        #ax1.set_xticklabels([]) 

        samples_df, percentiles, means = self._generate_points_for_CATE(w_values=w_values, list_distribs_doX_given_W=list_distribs_doXx_2)

        # Shaded percentile bands
        ax1.fill_between(percentiles['W_center'], percentiles['q40'], percentiles['q60'], alpha=0.15, label='_40-60%', color='#FF6347', linewidth=0.7, linestyle='--', edgecolor='black')
        ax1.fill_between(percentiles['W_center'], percentiles['q42'], percentiles['q58'], alpha=0.25, label='_42-58%', color='#FF6347', linewidth=0.2)
        ax1.fill_between(percentiles['W_center'], percentiles['q44'], percentiles['q56'], alpha=0.35, label='_44-56%', color='#FF6347', linewidth=0.2)
        ax1.fill_between(percentiles['W_center'], percentiles['q46'], percentiles['q54'], alpha=0.45, label='_46-54%', color='#FF6347', linewidth=0.2)
        ax1.fill_between(percentiles['W_center'], percentiles['q48'], percentiles['q52'], alpha=0.55, label='X = true', color='#FF6347', linewidth=0.2) #'_48-52%'
        ax1.plot(percentiles['W_center'], percentiles['q50'], color='#b03c0b', label='_Median', linestyle='-', linewidth=1.5, alpha=0.7)
        ax1.plot(np.asarray(w_values, dtype=float), list_exp_Y0_given_doXx_2_Ww_1, color='#b03c0b', label='_no_label', linestyle='--', linewidth=1.4)
        median_doX_2 = percentiles['q50']

        ax1.axvline(40000, color='dimgray', linestyle='--', linewidth=1.4, label=r'$E(Y_0 \mid do(X), W)$')
        ax1.axvline(40000, color='dimgray', linestyle='-', linewidth=1.5, label='Median (50%)')
        ax1.fill_between(percentiles['W_center'], percentiles['q40'] *1000, percentiles['q60']*1000, alpha=0.20, label='CI (40-60%)', color='dimgray', linewidth=0.8, linestyle='--', edgecolor='black')

        ax1.set_xlim(w_values[0], w_values[-1])
        ax1.set_ylim(7500, 22000)

        order = [0, 1, 4, 3, 2]  
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper left', title=r'Wall insulation $(X):$', frameon=True, fontsize=7, title_fontsize=7)
        #-----------------------------------------------------------------------------

        CATE_vals = np.array(list_exp_Y0_given_doXx_2_Ww_1) - np.array(list_exp_Y0_given_doXx_1_Ww_1)
        #CMTE_vals = np.array(median_doX_2) - np.array(median_doX_1) # Conditional Median Treatment Effect

        ax2.plot(np.asarray(w_values, dtype=float), CATE_vals, color='purple', label=r'$CATE_{G}$', linestyle='--', linewidth=1.3)
        ax2.axhline(y=-2980., color='black', linestyle='-.', linewidth=0.9, label=r'$ATE_{G}$')
        ax2.legend(loc='upper left', frameon=True, fontsize=7, ncol=2)

        #ax2.plot(np.array(percentiles['W_center']), CMTE_vals, color='royalblue', label=r'$CMTE_{G}$', linestyle='-', linewidth=1.1)
  

        ax2.set_ylabel(r'Treatment effect', fontsize=8)
        ax2.set_xlabel(r'Energy burden $(W)$ [£/£]', fontsize=8)
        ax2.minorticks_on()
        ax2.xaxis.set_major_locator(plt.MultipleLocator(0.01))
        ax2.xaxis.set_minor_locator(plt.MultipleLocator(0.005))
        ax2.yaxis.set_major_locator(plt.MultipleLocator(1000))
        ax2.yaxis.set_minor_locator(plt.MultipleLocator(500))
        ax2.tick_params(axis='x', which='major', direction='out', length=4, labelsize=7)
        ax2.tick_params(axis='x', which='minor', direction='out', length=2)
        ax2.tick_params(axis='y', which='major', direction='in', length=4, labelsize=7)
        ax2.tick_params(axis='y', which='minor', direction='in', length=2)
        ax2.set_xticks([0., 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11]) 
        ax2.set_xticklabels(['0.00', '0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.09', '0.10', '0.11']) 

        ax2.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5,  color='black', alpha=0.5)
        ax2.set_xlim(w_values[0], w_values[-1])
        ax2.set_ylim(-4000, 0)

        #---------------------------------------------------
        distribs_doXx_2 = list_distribs_doXx_2[0]
        distribs_doXx_1 = list_distribs_doXx_1[0]
        bar_height = pd.DataFrame()
        bar_height['Y_0'] = distribs_doXx_2['Y_0']
        bar_height.loc[bar_height.index[0], 'Y_0'] = bar_height.loc[bar_height.index[1], 'Y_0'] - 4125
        bar_height.loc[bar_height.index[-1], 'Y_0'] = bar_height.loc[bar_height.index[-2], 'Y_0'] + 4125
        bar_height['PR(Y_0)'] = (distribs_doXx_2[f'P(Y_0 | do(X=2), W={0.02})'] / distribs_doXx_1[f'P(Y_0 | do(X=1), W={0.02})']) - 1.
        ax_bottom_1.barh(y=bar_height['Y_0'], width=bar_height['PR(Y_0)'], height=4125, edgecolor='royalblue', alpha=0.8, color='royalblue')

        distribs_doXx_2 = list_distribs_doXx_2[4]
        distribs_doXx_1 = list_distribs_doXx_1[4]
        bar_height = pd.DataFrame()
        bar_height['Y_0'] = distribs_doXx_2['Y_0']
        bar_height.loc[bar_height.index[0], 'Y_0'] = bar_height.loc[bar_height.index[1], 'Y_0'] - 4125
        bar_height.loc[bar_height.index[-1], 'Y_0'] = bar_height.loc[bar_height.index[-2], 'Y_0'] + 4125
        bar_height['PR(Y_0)'] = (distribs_doXx_2[f'P(Y_0 | do(X=2), W={0.0515})'] / distribs_doXx_1[f'P(Y_0 | do(X=1), W={0.0515})']) - 1.
        ax_bottom_2.barh(y=bar_height['Y_0'], width=bar_height['PR(Y_0)'], height=4125, edgecolor='royalblue', alpha=0.8, color='royalblue')

        distribs_doXx_2 = list_distribs_doXx_2[7]
        distribs_doXx_1 = list_distribs_doXx_1[7]
        bar_height = pd.DataFrame()
        bar_height['Y_0'] = distribs_doXx_2['Y_0']
        bar_height.loc[bar_height.index[0], 'Y_0'] = bar_height.loc[bar_height.index[1], 'Y_0'] - 4125
        bar_height.loc[bar_height.index[-1], 'Y_0'] = bar_height.loc[bar_height.index[-2], 'Y_0'] + 4125
        bar_height['PR(Y_0)'] = (distribs_doXx_2[f'P(Y_0 | do(X=2), W={0.0785})'] / distribs_doXx_1[f'P(Y_0 | do(X=1), W={0.0785})']) - 1.
        ax_bottom_3.barh(y=bar_height['Y_0'], width=bar_height['PR(Y_0)'], height=4125, edgecolor='royalblue', alpha=0.8, color='royalblue')

        distribs_doXx_2 = list_distribs_doXx_2[11]
        distribs_doXx_1 = list_distribs_doXx_1[11]
        bar_height = pd.DataFrame()
        bar_height['Y_0'] = distribs_doXx_2['Y_0']
        bar_height.loc[bar_height.index[0], 'Y_0'] = bar_height.loc[bar_height.index[1], 'Y_0'] - 4125
        bar_height.loc[bar_height.index[-1], 'Y_0'] = bar_height.loc[bar_height.index[-2], 'Y_0'] + 4125
        bar_height['PR(Y_0)'] = (distribs_doXx_2[f'P(Y_0 | do(X=2), W={0.11})'] / distribs_doXx_1[f'P(Y_0 | do(X=1), W={0.11})']) - 1.
        ax_bottom_4.barh(y=bar_height['Y_0'], width=bar_height['PR(Y_0)'], height=4125, edgecolor='royalblue', alpha=0.8, color='royalblue')

        ax_bottom_1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax_bottom_2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax_bottom_3.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax_bottom_4.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

        ax_bottom_1.set_ylim(875, 35000 - 875)
        ax_bottom_2.set_ylim(875, 35000 - 875)
        ax_bottom_3.set_ylim(875, 35000 - 875)
        ax_bottom_4.set_ylim(875, 35000 - 875)

        plots = [ax_bottom_1, ax_bottom_2, ax_bottom_3, ax_bottom_4]
        for pl in plots:
            pl.xaxis.set_major_locator(plt.MultipleLocator(1.))
            pl.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
            pl.yaxis.set_major_locator(plt.MultipleLocator(5000))
            pl.yaxis.set_minor_locator(plt.MultipleLocator(2500))

            pl.tick_params(axis='y', which='major', direction='out', length=4, labelsize=7)
            pl.tick_params(axis='y', which='minor', direction='out', length=2)
            pl.tick_params(axis='x', which='major', direction='out', length=4, labelsize=7)
            pl.tick_params(axis='x', which='minor', direction='out', length=2)

            pl.set_xlim(-1, 2)

            pl.set_xticks([-1., 0., 1., 2., 3.]) 
            #pl.set_xticklabels(['0', '1', '2', '3'])
            xticks = pl.get_xticks()
            pl.set_xticklabels([str(int(tick + 1)) for tick in xticks])

            if pl != ax_bottom_1: 
                pl.set_yticklabels([])

        #ax_bottom_1.set_xticks([1, 0., 1., 2.]) 
        #ax_bottom_1..set_xticklabels([-1, 0., 1., 2.]) 
        #ax2.set_xticks([0., 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11]) 
        #ax2.set_xticklabels(['0.00', '0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.09', '0.10', '0.11']) 
        '''
        # Set new x labels shifted by +1
        xticks = ax_bottom_1.get_yticks()
        ax_bottom_1.set_xticklabels([str(int(tick + 1)) for tick in xticks])
        xticks = ax_bottom_2.get_yticks()
        ax_bottom_2.set_xticklabels([str(int(tick + 1)) for tick in xticks])
        xticks = ax_bottom_3.get_yticks()
        ax_bottom_3.set_xticklabels([str(int(tick + 1)) for tick in xticks])
        xticks = ax_bottom_4.get_yticks()
        ax_bottom_4.set_xticklabels([str(int(tick + 1)) for tick in xticks])
        
        ax_bottom_2.set_yticklabels([]) 
        ax_bottom_3.set_yticklabels([])
        ax_bottom_4.set_yticklabels([])
        '''
        
        ax_bottom_1.set_ylabel(r'Gas consumption $(Y_0)$', fontsize=8)

        ax_bottom_1.text(2.3, -.43, r'$\frac{P(Y_0 \mid do(X=true), W)_{G}}{P(Y_0 \mid do(X=false), W)_{G}}$', transform=ax_bottom_1.transAxes, fontsize=10, va='center', ha='center')
        
        ax_bottom_1.text(0.7, 0.9, r'$W=0.02$', transform=ax_bottom_1.transAxes, fontsize=7, va='center', ha='center')
        ax_bottom_2.text(0.7, 0.9, r'$W=0.05$', transform=ax_bottom_2.transAxes, fontsize=7, va='center', ha='center')
        ax_bottom_3.text(0.7, 0.9, r'$W=0.08$', transform=ax_bottom_3.transAxes, fontsize=7, va='center', ha='center')
        ax_bottom_4.text(0.7, 0.9, r'$W=0.11$', transform=ax_bottom_4.transAxes, fontsize=7, va='center', ha='center')
        
        #plt.tight_layout(rect=[0, 0.05, 1, 1])
        #---------------------------------------------------
        ax1.text(-0.218, 1.1, 'a', transform=ax1.transAxes, fontsize=11, fontweight='bold', va='top', ha='left')
        ax2.text(-0.218, 1.16, 'b', transform=ax2.transAxes, fontsize=11, fontweight='bold', va='top', ha='left')
        ax_bottom_1.text(-1, 1.14, 'c', transform=ax_bottom_1.transAxes, fontsize=11, fontweight='bold', va='top', ha='left')
        
        w = str(width_cm).replace('.', '-')
        h = str(height_cm).replace('.', '-')
        fig.savefig(f"Figures/figure_{w}_cm_by_{h}_cm_{figure_name}.png", bbox_inches="tight", dpi=600)

        return


    def _generate_points_for_CATE(self, w_values: list, list_distribs_doX_given_W: DataFrame) -> DataFrame:
        '''
        The method generates a point cloud of {W, Y_0} value pairs based on the w-specific 
        probability distributions  P(Y_0 | do(x), W=w) that is the input parameter: list_distribs_doX_given_W 
        (i.e. the list containing all the dataframes of prob distributions for different values of W)
        '''

        bins_width_W = float(w_values[2]) - float(w_values[1])
        first_df = list_distribs_doX_given_W[0]
        bins_width_Y_0 = first_df.iloc[2]['Y_0'] - first_df.iloc[1]['Y_0']

        long_df_list = []
        for i in range(0, len(list_distribs_doX_given_W)):
            df: DataFrame = list_distribs_doX_given_W[i]
            w_value = w_values[i]
            p_col = df.columns[1]

            temp_df = df.rename(columns={p_col: 'P'})
            temp_df['W'] = float(w_value)

            long_df_list.append(temp_df)

        long_df = pd.concat(long_df_list, ignore_index=True)

        #print(long_df)

        samples = []
        for w_value, group in long_df.groupby('W'):
            probs = group['P'] / group['P'].sum()
            n_samples = 10000
            sampled_y_bin_centers = np.random.choice(group['Y_0'], size=n_samples, p=probs)
            sampled_y = np.random.uniform(sampled_y_bin_centers - bins_width_Y_0 / 2, sampled_y_bin_centers + bins_width_Y_0 / 2)
            sampled_w = np.random.uniform(w_value - bins_width_W / 2, w_value + bins_width_W / 2, size=n_samples)
            samples.append(pd.DataFrame({'Y_0': sampled_y, 'W': sampled_w}))
            
        samples_df = pd.concat(samples, ignore_index=True)
        #-------------------------------------------------------------------------

        bins = np.arange(samples_df['W'].min()-bins_width_W/2, samples_df['W'].max() + 1.5*bins_width_W, bins_width_W)
        samples_df['W_bin'] = pd.cut(samples_df['W'], bins=bins, labels=False)
        quantiles = samples_df.groupby('W_bin')['Y_0'].quantile([0.40, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.60])
        quantiles = quantiles.reset_index()
        percentiles = quantiles.pivot(index='W_bin', columns='level_1', values='Y_0')
        percentiles.columns = ['q40', 'q42', 'q44', 'q46', 'q48', 'q50', 'q52', 'q54', 'q56', 'q58', 'q60']

        percentiles['W_center'] = (bins[:-1] + bins[1:]) / 2

        means = samples_df.groupby('W_bin')['Y_0'].mean()
        means = means.reset_index()


        return samples_df, percentiles, means




    def _bin_width(self, df_Xx: DataFrame) -> float:
        return float(df_Xx.iloc[2, 0]) - float(df_Xx.iloc[1, 0])


    def _add_regresssion_curves(self, df_Xx: DataFrame, prob_col_name: str):

        df_Xx['Y_0'] = df_Xx['Y_0'].replace({'<': '', '>': '', ',': ''}, regex=True).astype(float)
        df_Xx[prob_col_name] = df_Xx[prob_col_name].astype(float)

        probabilities = df_Xx[prob_col_name].to_list()
        values = np.array(list(range(0, len(probabilities))))
        sampled_data = np.random.choice(values, size=20000, p=probabilities)

        kde = gaussian_kde(sampled_data, bw_method=0.3)
        x = np.linspace(min(df_Xx['Y_0'].to_list()), max(df_Xx['Y_0'].to_list()), 1000)
        y = kde(np.linspace(min(values), max(values), 1000))

        return x[:850], y[:850]
    

     