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
        ax1.set_xlim(0, 35000)
        

 
        ax1.annotate("", xy=(exp_Xx_1, y_axis_upper_limit*0.63), xytext=(exp_Xx_2-300, y_axis_upper_limit*0.63), arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='<->', lw=0.8, shrinkB=0.))
        ax1.text(exp_Xx_1+400, y_axis_upper_limit*0.63, r'$ATE_{G}$' + rf'$ = {int(exp_Xx_2-exp_Xx_1)}$', fontsize=7, ha='left', va='center')
        ax1.text(-0.24, 1.07, 'a', transform=ax1.transAxes, fontsize=11, fontweight='bold', va='top', ha='left')
        #-----------------------------------------------------------------------------------------------

        bar_height = pd.DataFrame()
        bar_height['Y_0'] = doXx_1_distrib['Y_0']
        bar_height['PR(Y_0)'] = (doXx_2_distrib['P(Y_0 | do(X=2))'] / doXx_1_distrib['P(Y_0 | do(X=1))']) - 1.
        p = Polynomial.fit(bar_height['Y_0'], bar_height['PR(Y_0)'], 5)
        
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
        ax2.set_xlim(0, 35000)

        ax2.text(-0.24, -.07, 'b', transform=ax1.transAxes, fontsize=11, fontweight='bold', va='top', ha='left')


        w = str(width_cm).replace('.', '-')
        h = str(height_cm).replace('.', '-')
        fig.savefig(f"Figures/figure_{w}_cm_by_{h}_cm_{figure_name}.png", bbox_inches="tight", dpi=600)

        return
   

    def plot_CATE(self, figure_name: str, width_cm: float, height_cm: float, w_values: list, list_distribs_doXx_1: DataFrame):



        '''
        see the chat
        '''

        plt.xlabel(r'Energy burden $(W)$ [£/£]', fontsize=8)
        plt.ylabel(r'Gas consumption $(Y_0)$ [kWh/year]', fontsize=8)

        w = str(width_cm).replace('.', '-')
        h = str(height_cm).replace('.', '-')
        fig.savefig(f"Figures/figure_{w}_cm_by_{h}_cm_{figure_name}.png", bbox_inches="tight", dpi=600)

        return


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
    

     