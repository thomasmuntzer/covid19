# tick on mondays every week
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd
from typing import Union, List
import matplotlib

matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['legend.fontsize'] = 12
standard_colors = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])

def createXYPlot(dfplot: pd.DataFrame,
                 x: str,
                 y: Union[str, List[str]],
                 plots_folder: str = None,
                 labels: Union[str, List[str]] = None,
                 linestyles: Union[str, List[str]] = None,
                 error = False,
                 asymmetric_error: bool = False,
                 remove_negative: bool = False,
                 figsize_x: int = 10,
                 figsize_y: int = 5,
                 xtitle: str = None,
                 ytitle: str = None,
                 yticks: List[float] = None,
                 dpis: int = 100,
                 bar: bool = False,
                 bar_width: float = 0.75,
                 bar_start: List[float] = None,
                 alphas: List[float] = None,
                 linewidth: float = 1.0,
                 yscale: str = "linear",
                 savename: str = None,
                 title: str = None,
                 colors: List[str] = None,
                 start_date: str = None,
                 alpha_err: float = 0.1,
                 xlim: float = None,
                 days_interval: int = 4):
    
       
    if start_date is not None: 
        dfplot = dfplot[pd.to_datetime(dfplot[x]) > datetime.strptime(start_date,'%Y-%m-%d')]
    
    if remove_negative:
        for yname in y:
            dfplot = dfplot[dfplot[yname] > 0]
            
    dfplot = dfplot.sort_values(x).reset_index() 
    first_day = str(dfplot[x].tolist()[0])
    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y),dpi=dpis)
    
    xindexes = [x for x in sorted(dfplot.index) if x % days_interval == 0]
    xlabels =  [str(dfplot[x].tolist()[i]) for i in xindexes]
    
    # Counter for more objects
    i=0 
    if type(y) != list:
        y = [y]
        
    if type(alphas) != list:
        alphas = [None for i in y]
        
    if type(linestyles) != list:
        linestyles = ["-" for i in y]
        
    if type(colors) != list:
        colors = [standard_colors[i] for i in range(len(y))]
        
    if labels is None:
        labels = y
        
    for y_name in y:
        x_data = dfplot.index
        y_data = np.array(dfplot[y_name])
        y_err = None
        if bar:
            if error:
                if asymmetric_error:
                    y_err = [np.array(dfplot["err_" + y_name + "_lo"]), 
                             np.array(dfplot["err_" + y_name + "_hi"])]
                else:
                    y_err = np.array(dfplot["err_" + y_name])
            plt.bar(x_data + bar_start[i] * bar_width, 
                         y_data, 
                         yerr=y_err,
                         xerr=None,
                         width=bar_width, 
                         align="center", 
                         alpha=alphas[i], 
                         color=colors[i],
                         edgecolor=colors[i],
                         label=labels[i],
                         error_kw={"capsize":1.5,"elinewidth":1}
                   )
        else:
            plt.plot(x_data, 
                     y_data, 
                     alpha=alphas[i], 
                     color=colors[i], 
                     linewidth=linewidth, 
                     linestyle=linestyles[i],
                     label=labels[i])
            if error:
                plt.fill_between(x_data, 
                                 y_data - dfplot["err_" + y_name],
                                 y_data + dfplot["err_" + y_name],
                                 alpha=alpha_err,
                                 color=colors[i]
                                )
                
            
        i+=1
        
    if len(y) > 1: 
        plt.legend()
    # Set Scale
    plt.yscale(yscale)
    
    if xtitle: 
        plt.xlabel(xtitle,fontsize=14)
    if ytitle:
        plt.ylabel(ytitle,fontsize=14)
    if title: 
        plt.title(title,fontsize=16)
    # format the coords message box
    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    plt.xticks(xindexes, 
               labels=xlabels, 
               fontsize=12,
               rotation=50, 
               rotation_mode="anchor", 
               verticalalignment = "top",
               horizontalalignment = "right")
    if yticks is not None:
        plt.yticks(yticks, fontsize=12)
    else:
        plt.yticks(fontsize=12)
    
    if yscale != "linear":
        savename += f"_{yscale}"
    if savename:
        plt.savefig(f"{plots_folder}/{savename}", bbox_inches="tight", facecolor="w")
    plt.show()
    plt.close()
    
    del(fig)
    

def getVariables(df):

    df["data"] = pd.to_datetime(df["data"])
    df["dow"] = df["data"].dt.dayofweek
    df = df.fillna({"tamponi_test_antigenico_rapido":0})
    
    df["tamponi_test_molecolare"] = df["tamponi_test_molecolare"].combine_first(df["tamponi"])
    
    df["nuovi_positivi_lag_7"] = df["nuovi_positivi"].shift(7)
    df["ingressi_terapia_intensiva_lag_7"] = df["ingressi_terapia_intensiva"].shift(7)

    df['giorno'] = df['data'].dt.date
    
    df["nuovi_positivi_test_molecolare"] = df["totale_positivi_test_molecolare"] - df["totale_positivi_test_molecolare"].shift(1)
    df["nuovi_positivi_test_rapido"] = df["totale_positivi_test_antigenico_rapido"] -  df["totale_positivi_test_antigenico_rapido"].shift(1)
    
    df["variazione_deceduti"] = df["deceduti"] - df["deceduti"].shift(1)
    df["variazione_positivi"] = df["totale_positivi"] - df["totale_positivi"].shift(1)
    df["variazione_relativa_positivi"] = df["nuovi_positivi"]/df["totale_positivi"].shift(1)
    df["variazione_deceduti_media_7"] = df.sort_values("data")["variazione_deceduti"].rolling(7).mean()
    
    df["ingressi_terapia_intensiva_30"] = df.sort_values("data")["ingressi_terapia_intensiva"].rolling(30).sum()
    
    df["variazione_guariti"] = df["dimessi_guariti"] - df["dimessi_guariti"].shift(1)
    df["variazione_ospedalizzati"] = df["totale_ospedalizzati"] - df["totale_ospedalizzati"].shift(1)
    df["variazione_ricoverati_con_sintomi"] = df["ricoverati_con_sintomi"] - df["ricoverati_con_sintomi"].shift(1)
    df["variazione_isolamento_domiciliare"] = df["isolamento_domiciliare"] - df["isolamento_domiciliare"].shift(1)
    df["variazione_terapia_intensiva"] = df["terapia_intensiva"] - df["terapia_intensiva"].shift(1)
    df["variazione_tamponi"] = df["tamponi"] - df["tamponi"].shift(1)
    df["variazione_tamponi_molecolari"] = df["tamponi_test_molecolare"] - df["tamponi_test_molecolare"].shift(1)
    df["variazione_tamponi_rapidi"] = df["tamponi_test_antigenico_rapido"] - df["tamponi_test_antigenico_rapido"].shift(1)
    
    df["ti_ratio"] = 100 * df["ingressi_terapia_intensiva"]/df["nuovi_positivi_lag_7"]
    df["death_ti_ratio"] = 100 * df["variazione_deceduti"]/df["ingressi_terapia_intensiva_lag_7"]
    
    df["tasso_positivi"] = 100*(df["nuovi_positivi"]/df["variazione_tamponi"])
    df["tasso_positivi_test_rapido"] = 100*(df["nuovi_positivi_test_rapido"]/df["variazione_tamponi_rapidi"])
    df["tasso_positivi_test_molecolare"] = 100*(df["nuovi_positivi_test_molecolare"]/df["variazione_tamponi_molecolari"])
    df["tasso_positivi_test_molecolare"] =  df["tasso_positivi_test_molecolare"].combine_first(df["tasso_positivi"])

    df["mortalità"] = df["deceduti"]/df["totale_casi"]
    df["variazione_mortalità"] = df["mortalità"]-df["mortalità"].shift(1)
    df["sd_mortalità"] = np.sqrt((df["mortalità"]*(1-df["mortalità"]))/df["totale_casi"])
    df["sd_tasso_positivi"] = np.sqrt((df["tasso_positivi"]*(1-df["tasso_positivi"]))/df["variazione_tamponi"])
    
    
    df["frazione_ospedalizzati"] = df["totale_ospedalizzati"]/df["totale_positivi"]
    df["frazione_terapia_intensiva"] = df["terapia_intensiva"]/df["totale_positivi"]
    df["frazione_isolamento_domiciliare"] = df["isolamento_domiciliare"]/df["totale_positivi"]
    df["frazione_tamponi_rapidi"] = df["variazione_tamponi_rapidi"]/df["variazione_tamponi"]
    
    
        
    roll_mean_vars = ["nuovi_positivi",
                      "ti_ratio",
                      "variazione_tamponi",
                      "variazione_positivi",
                      "variazione_terapia_intensiva",
                      "variazione_relativa_positivi",
                      "tasso_positivi",
                      "tasso_positivi_test_rapido",
                      "tasso_positivi_test_molecolare",
                      "ingressi_terapia_intensiva",
                      "variazione_ospedalizzati",
                      "variazione_guariti",
                      "frazione_ospedalizzati",
                      "frazione_terapia_intensiva",
                      "frazione_isolamento_domiciliare",
                      "mortalità",
                      "frazione_tamponi_rapidi",
                      "variazione_tamponi_rapidi",
                      "variazione_tamponi_molecolari"
                     ]
    
    for var in roll_mean_vars:
        df[var+"_media_7"] = df.sort_values("data")[var].rolling(7).mean()

    df = df.sort_values(by="giorno",ascending = False).reset_index().drop(columns=["index"])
    
    return df
    
    
    
def get_efficacy(df: pd.DataFrame, z_value=1.645):
    
    for vax_status in ["1st_dose", "2nd_dose", "no_vax"]:
        
        df[f"ev_{vax_status}_per_100k"] =  np.round((1e5) * df[f"event_{vax_status}"]/df[f"pop_{vax_status}"],3)
            
        p = df[f"event_{vax_status}"]/df[f"pop_{vax_status}"]
        q = 1 - p
        N = df[f"pop_{vax_status}"]
        df[f"err_ev_{vax_status}_per_100k"] = z_value * (1e5) * np.sqrt(p*q) / np.sqrt(N)
        
        
    for vax_status in ["1st_dose", "2nd_dose"]:
        df[f"efficacy_{vax_status}"] =  np.round(1 - df[f"ev_{vax_status}_per_100k"]/df["ev_no_vax_per_100k"],3)
        
        df.replace([-np.inf], 0, inplace=True)
        
        A = 1 - df[f"ev_{vax_status}_per_100k"]
        B = df["ev_no_vax_per_100k"]
        sA = df[f"err_ev_{vax_status}_per_100k"]
        sB = df[f"err_ev_no_vax_per_100k"]
        f = df[f"efficacy_{vax_status}"]
        df[f"err_efficacy_{vax_status}"] = z_value * np.round(np.abs(f) * np.sqrt((sA/A)**2 + (sB/B)**2),3)
            
        df[f"arr_{vax_status}"] = (df[f"ev_no_vax_per_100k"] - df[f"ev_{vax_status}_per_100k"])/(1e5)
        df[f"nntv_{vax_status}"] = 1/df[f"arr_{vax_status}"]
    
    return df



def clean_israel_data(df, cols_to_clean):
    for colname in cols_to_clean:
        df[colname] = df[colname].apply(lambda x: float(str(x).replace("<","")) -1 if "<" in str(x) else float(str(x)))
    
    for colname in df.columns:
        df = df.rename(columns={colname: colname.lower()})
    
    if "week" in df.columns:
        df["first_day"] = df["week"].apply(lambda x: x.split(" - ")[0])
        df["last_day"] = df["week"].apply(lambda x: x.split(" - ")[1])
    df["over_60"]=df["age_group"].apply(lambda x: "over_60" if x in ('60-69','70-79','80-89','90+') else "under_60")

    df = df.fillna(0)
    return df  


def stratify_population(df_pop, age_groups):
    pop_dict = {}
    for age_bucket in sorted(age_groups):
        if "+" not in age_bucket:
            age_lo = int(age_bucket.split("-")[0])
            age_hi = int(age_bucket.split("-")[1])
        else:
            age_lo = int(age_bucket.replace("+",""))
            age_hi = 99999
        print(f"{age_bucket} -> age lo: {age_lo}, age hi: {age_hi}")

        pop_dict[age_bucket] = np.sum(df_pop[(df_pop.age >= age_lo) & (df_pop.age <= age_hi)]["pop"])
    
    return pd.DataFrame.from_dict(pop_dict,orient="index",columns=["population"]).sort_index()