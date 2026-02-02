import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os


data_file = "/home/z/zoghafoo/www/pT_Cuts/events_dataPostVFP.csv"
pred_file = "/home/z/zoghafoo/www/pT_Cuts/events_ZmumuPostVFP.csv"
out_dir = "/home/z/zoghafoo/www/pT_Cuts"
os.makedirs(out_dir, exist_ok=True)


s = 13000.0
sns.set(style="white", context="talk")

def xmin_xmax(mll, yll):
    xplus = np.exp(yll) * mll / s
    xminus = np.exp(-yll) * mll / s
    return np.minimum(xplus, xminus), np.maximum(xplus, xminus)


data = pd.read_csv(data_file)
pred = pd.read_csv(pred_file)

datasets = {"data": data, "pred": pred}


for l in ["log", "lin"]:

    for label, df in datasets.items():

        xmin, xmax = xmin_xmax(df["mll"].values, df["yll"].values)

        if l == "log":
            xbins = np.logspace(-2.5, -1, 50)
            ybins = np.logspace(-3.5, -2, 50)
        else:
            xbins = np.linspace(0, 0.1, 50)
            ybins = np.linspace(0, 0.01, 50)


        fig = plt.figure(figsize=(15, 12))

        gs = gridspec.GridSpec(
            2, 2,
            width_ratios=[4, 1.2],
            height_ratios=[1.2, 4],
            hspace=0.15,
            wspace=0.15
        )

        ax_top   = fig.add_subplot(gs[0, 0])
        ax_main  = fig.add_subplot(gs[1, 0], sharex=ax_top)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

        h = sns.histplot(
            x=xmax,
            y=xmin,
            bins=[xbins, ybins],
            cmap="cubehelix",
            cbar=False,
            ax=ax_main
        )
        
        cbar_ax = inset_axes(ax_right, 
                             width="100%", height="100%",
                             bbox_to_anchor=(0.05, 1.1, 0.15, 0.3),
                             bbox_transform=ax_right.transAxes,
                             loc='lower left')
        mappable = ax_main.collections[0]
        cbar = fig.colorbar(mappable, cax=cbar_ax)


        ax_top.hist(xmax, bins=xbins, histtype="bar", color="darkmagenta", edgecolor="black")
        ax_right.hist(xmin, bins=ybins, orientation="horizontal", histtype="bar", color="darkmagenta", edgecolor="black")

        ax_top.grid(axis='y', alpha=0.8, linestyle='-')
        ax_right.grid(axis='x', alpha=0.8, linestyle='-')

        if l == "log":
            for ax in [ax_main, ax_top]:
                ax.set_xscale("log")
            for ax in [ax_main, ax_right]:
                ax.set_yscale("log")

            ax_main.xaxis.set_major_locator(LogLocator(base=10))
            ax_main.yaxis.set_major_locator(LogLocator(base=10))
            ax_main.grid(True, which="both", alpha=0.3)


        plt.setp(ax_top.get_xticklabels(), visible=False)
        plt.setp(ax_right.get_yticklabels(), visible=False)

        ax_top.set_ylabel("Events")
        ax_right.set_xlabel("Events")

        ax_main.set_xlabel(r"max($x^\mathrm{obs}_{\mu\mu}$)")
        ax_main.set_ylabel(r"min($x^\mathrm{obs}_{\mu\mu}$)")
        # ax_main.set_title(f"xminll vs xmaxll ({label})")

    
        out_name = f"{out_dir}/xmaxll_xminll_{label}_{l}2D.pdf"
        plt.savefig(out_name)
        plt.close()

        print(f"[OK] Saved {out_name}")
