# tick on mondays every week
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd
from typing import Union, List

def createXYPlot(dfplot: pd.DataFrame,
                 x: str,
                 y: Union[str, List[str]],
                 today: str,
                 plots_folder: str,
                 labels: Union[str, List[str]] = None,
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

    i=0 
    if type(y) != list:
        y = [y]
        
    if type(alphas) != list:
        alphas = [None for x in y]
        
    if type(colors) != list:
        colors = [None for x in y]
        
    if labels is None:
        labels = y
        
    for y_name in y:
        x_data = dfplot.index
        y_data = np.array(dfplot[y_name])
        if error: 
            if asymmetric_error:
                y_err = [np.array(dfplot["err_" + y_name + "_lo"]), np.array(dfplot["err_" + y_name + "_hi"])]
            else:
                y_err = np.array(dfplot["err_" + y_name])
        else:
            y_err=None
        if bar:
            plt.bar(x_data + bar_start[i] * bar_width, 
                         y_data, 
                         yerr=y_err,
                         xerr=None,
                         width=bar_width, 
                         align="center", 
                         alpha=alpha, 
                         color=color, 
                         label=labels[i],
                         error_kw={"capsize":1.5,"elinewidth":1}
                   )
        else:
            plt.plot(x_data, y_data, alpha=alphas[i], color=colors[i], linewidth=linewidth, label=labels[i])
        i+=1
        
    plt.grid(which="both")
    if len(y) > 1: 
        plt.legend(fontsize=12)
    
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
    
    if savename:
        plt.savefig(f"{plots_folder}/{savename}",bbox_inches="tight")
    plt.show()
    plt.close()
    
    del(fig)
    
    
    
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