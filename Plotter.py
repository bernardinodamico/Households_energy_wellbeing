import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
from pandas import DataFrame
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib.ticker import MultipleLocator



class Plotter():


    def plot_ATEs(self, figure_name: str, width_cm: float, height_cm: float, doXx_1_distrib_G: DataFrame, doXx_2_distrib_G: DataFrame, exp_Xx_1_G: float, exp_Xx_2_G: float) -> None:
        '''
        NOTE: doXx_1_distrib and doXx_2_distrib are two-column dataframes 
        reporting the post-intervention probability distribution P(Y_0 | do(X=1) and P(Y_0 | do(X=2) 
        '''
        doXx_1_distrib_G.iloc[0, 0] = doXx_1_distrib_G.iloc[1, 0] - self._bin_width(doXx_1_distrib_G)
        doXx_1_distrib_G.iloc[-1, 0] = doXx_1_distrib_G.iloc[-2, 0] + self._bin_width(doXx_1_distrib_G)
        doXx_2_distrib_G.iloc[0, 0] = doXx_2_distrib_G.iloc[1, 0] - self._bin_width(doXx_2_distrib_G)
        doXx_2_distrib_G.iloc[-1, 0] = doXx_2_distrib_G.iloc[-2, 0] + self._bin_width(doXx_2_distrib_G)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width_cm/2.54, height_cm/2.54), gridspec_kw={'width_ratios': [1.5, 1]})

        #---------------------------------------------------------------------------------------------
        # Add regression curves
        x, y = self._add_regresssion_curves(df_Xx=doXx_1_distrib_G, prob_col_name='P(Y_0 | do(X=1))')
        ax1.plot(x, y, linewidth=0.9, color='#42A05C')
        x2, y2 = self._add_regresssion_curves(df_Xx=doXx_2_distrib_G, prob_col_name='P(Y_0 | do(X=2))')
        ax1.plot(x2, y2, linewidth=0.9, color='#B35933')

        ax1.axvline(exp_Xx_1_G, color='#42A05C', linestyle='--', linewidth=1.5)
        ax1.axvline(exp_Xx_2_G, color='#B35933', linestyle='--', linewidth=1.5)

        # set tick values
        ax1.yaxis.set_major_locator(MultipleLocator(0.05)) # Major ticks every 0.05units
        ax1.yaxis.set_minor_locator(MultipleLocator(0.02))
        ax1.xaxis.set_major_locator(MultipleLocator(5000))
        ax1.xaxis.set_minor_locator(MultipleLocator(1000))

        ax1.set_xticks([0, 5000, 10000, 15000, 20000, 25000, 30000, 35000])
        ax1.set_xticklabels(['0', '5000', '10000', '15000', '20000', '25000', '30000', '35000'])

        # Add bars
        ax1.bar(x=doXx_1_distrib_G['Y_0'], height=doXx_1_distrib_G['P(Y_0 | do(X=1))'], width=self._bin_width(doXx_1_distrib_G), edgecolor='#3CB371', alpha=0.65, label="X=false", color='#3CB371')
        ax1.bar(x=doXx_2_distrib_G['Y_0'], height=doXx_2_distrib_G['P(Y_0 | do(X=2))'], width=self._bin_width(doXx_2_distrib_G), edgecolor='#FF6347', alpha=0.5, label="X=true", color='#FF6347')

        plt.rcParams["font.family"] = "Arial"
        ax1.set_xlabel(r'Gas consumption $(Y_0)$ [kWh/year]', fontsize=8)
        #ax1.set_xlabel('')
        ax1.set_ylabel(r'$P(Y_0 \mid do(X))_{G}$', fontsize=8)

        ax1.minorticks_on()

        ax1.tick_params(axis='x', which='major', direction='out', length=6, labelsize=7)
        ax1.tick_params(axis='x', which='minor', direction='out', length=3)
        ax1.tick_params(axis='y', which='major', direction='in', length=6, labelsize=7)
        ax1.tick_params(axis='y', which='minor', direction='in', length=3)
        ax1.tick_params(labelbottom=False)

        ax1.legend(title="Wall insulation:", frameon=True, fontsize=7, title_fontsize=7)

        for label in ax1.get_xticklabels():
            label.set_rotation(90)
        #plt.tight_layout()
        y_axis_upper_limit = 0.1 # max(doXx_2_distrib_G['P(Y_0 | do(X=2))'].to_list()) * 1.15
        ax1.set_ylim([0, y_axis_upper_limit])
        ax1.set_xlim([0, 35000])

        ax1.annotate("", xy=(exp_Xx_1_G, y_axis_upper_limit*0.55), xytext=(exp_Xx_2_G-300, y_axis_upper_limit*0.55), arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='<->', lw=0.8, shrinkB=0.))
        ax1.text(exp_Xx_1_G+400, y_axis_upper_limit*0.55, r'$ATE_{G}$' + f' = {int(exp_Xx_2_G-exp_Xx_1_G)}', fontsize=8, ha='left', va='center')
        #-------------------------------------------------------------------------------------------------

        #-------------------------------------------------------------------------------------------------



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
    

     