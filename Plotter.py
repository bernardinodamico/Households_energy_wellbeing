import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
from pandas import DataFrame
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib.ticker import MultipleLocator
import pandas as pd
from scipy.interpolate import griddata


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

        fig, ax = plt.subplots(figsize=(width_cm/2.54, height_cm/2.54))

        # Add regression curves
        x, y = self._add_regresssion_curves(df_Xx=doXx_1_distrib, prob_col_name='P(Y_0 | do(X=1))')
        plt.plot(x, y, linewidth=0.9, color='#42A05C')
        x2, y2 = self._add_regresssion_curves(df_Xx=doXx_2_distrib, prob_col_name='P(Y_0 | do(X=2))')
        plt.plot(x2, y2, linewidth=0.9, color='#B35933')

        plt.axvline(exp_Xx_1, color='#42A05C', linestyle='--', linewidth=1.5)
        plt.axvline(exp_Xx_2, color='#B35933', linestyle='--', linewidth=1.5)

        # Add bars
        plt.bar(x=doXx_1_distrib['Y_0'], height=doXx_1_distrib['P(Y_0 | do(X=1))'], width=self._bin_width(doXx_1_distrib), edgecolor='#3CB371', alpha=0.65, label="X=false", color='#3CB371')
        plt.bar(x=doXx_2_distrib['Y_0'], height=doXx_2_distrib['P(Y_0 | do(X=2))'], width=self._bin_width(doXx_2_distrib), edgecolor='#FF6347', alpha=0.5, label="X=true", color='#FF6347')

        plt.rcParams["font.family"] = "Arial"
        plt.xlabel(r'Gas consumption $(Y_0)$ [kWh/year]', fontsize=8)
        plt.ylabel(r'$P(Y_0 \mid do(X))_{G}$', fontsize=8)

        plt.minorticks_on()
        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.05))  # Major ticks every 0.05units
        plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.01))
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(5000))
        plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1000))

        plt.tick_params(axis='x', which='major', direction='out', length=6, labelsize=7)
        plt.tick_params(axis='x', which='minor', direction='out', length=3)
        plt.tick_params(axis='y', which='major', direction='in', length=6, labelsize=7)
        plt.tick_params(axis='y', which='minor', direction='in', length=3)

        plt.legend(title="Wall insulation:", frameon=True, fontsize=8, title_fontsize=8)

        plt.xticks(rotation=90)
        plt.tight_layout()
        y_axis_upper_limit = max(doXx_2_distrib['P(Y_0 | do(X=2))'].to_list()) * 1.15
        plt.ylim(0, y_axis_upper_limit)
        plt.xlim(0, 35000)

        w = str(width_cm).replace('.', '-')
        h = str(height_cm).replace('.', '-')
        #------------------------------------------------------


        
        ax.annotate("", xy=(exp_Xx_1, y_axis_upper_limit*0.68), xytext=(exp_Xx_2-300, y_axis_upper_limit*0.68), arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='<->', lw=0.8, shrinkB=0.))
        ax.text(exp_Xx_1+400, y_axis_upper_limit*0.68, r'$ATE_{G}$' + f' = {int(exp_Xx_2-exp_Xx_1)}', fontsize=8, ha='left', va='center')


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
    

     