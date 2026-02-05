import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os


if sys.argv[1] == "oldmax100Files":
    data_file = "/home/z/zoghafoo/WRemnants/csv_files/max100FilesOld/events_dataPostVFP.csv"
    pred_file = "/home/z/zoghafoo/WRemnants/csv_files/max100FilesOld/events_ZmumuPostVFP.csv"
elif sys.argv[1] == "newallFiles":
    data_file = "/home/z/zoghafoo/WRemnants/csv_files/allFiles/events_SingleMuon_2016PostVFP.csv"
    pred_file = "/home/z/zoghafoo/WRemnants/csv_files/allFiles/events_Zmumu_2016PostVFP.csv"
else:
    print("Please provide a valid argument: 'oldmax100Files' or 'newallFiles'")
    sys.exit(1)

max_events = int(sys.argv[2]) if len(sys.argv) > 2 else None

data = pd.read_csv(data_file, nrows=max_events)
pred = pd.read_csv(pred_file, nrows=max_events)

out_dir = "/eos/user/z/zoghafoo/www/pT_Cuts/xmaxll_xminll_2D"
os.makedirs(out_dir, exist_ok=True)


s = 13000.0
sns.set(style="white", context="talk")

def xmin_xmax(mll, yll):
    xplus = np.exp(yll) * mll / s
    xminus = np.exp(-yll) * mll / s
    return np.minimum(xplus, xminus), np.maximum(xplus, xminus)


datasets = {"data": data,
            "pred": pred
            }


for l in [
"log",
"lin"
]:

    for label, df in datasets.items():

        xmin, xmax = xmin_xmax(df["mll"].values, df["yll"].values)

        if l == "log":
            if sys.argv[1] == "oldmax100Files":
                xbins = np.logspace(-2.4, -1, 50)
                ybins = np.logspace(-3.5, -1.9, 50)
            elif sys.argv[1] == "newallFiles":
                xbins = np.logspace(-2.45, -0.9, 200)
                ybins = np.logspace(-3.5, -1.6, 100)
        else:
            if sys.argv[1] == "oldmax100Files":
                xbins = np.linspace(0, 0.1, 50)
                ybins = np.linspace(0, 0.01, 50)
            elif sys.argv[1] == "newallFiles":
                xbins = np.linspace(0, 0.1, 100)
                ybins = np.linspace(0, 0.03, 500)


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
                             loc='lower left',
                             )
        mappable = ax_main.collections[0]
        cbar = fig.colorbar(mappable, cax=cbar_ax)
        cbar.ax.set_title("Events/Bin", pad=6)


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

        # turn the tick labels back on for the shared axes: they make the x-axis labels on the top subplot and the y-axis labels on the right subplot visible (with 'visible=False')
        plt.setp(ax_top.get_xticklabels(), visible=False)
        plt.setp(ax_right.get_yticklabels(), visible=False)

        for ax in [ax_top, ax_right, ax_main]:
            ax.tick_params(which="both", direction="in", top=True, right=True, left=True, bottom=True, pad=8)
            ax.grid(True, which="both", alpha=0.4)


        ax_top.yaxis.tick_left()
        ax_top.set_ylabel("Events/Bin", loc="top")
        ax_right.xaxis.tick_top()
        ax_right.xaxis.set_label_position('top') 
        ax_right.set_xlabel("Events/Bin", loc="right")

        ax_main.set_xlabel(r"max($x^\mathrm{obs}_{\mu\mu}$)")
        ax_main.set_ylabel(r"min($x^\mathrm{obs}_{\mu\mu}$)")
        # ax_main.set_title(f"xminll vs xmaxll ({label})")

        # add diagonal line where xmax = xmin, i.e. y_{mu, mu} = 0
        xmin_lim, xmax_lim = ax_main.get_xlim()
        ymin_lim, ymax_lim = ax_main.get_ylim()

        diag_start = max(xmin_lim, ymin_lim)
        diag_end   = min(xmax_lim, ymax_lim)

        # With zorder=5, the diagonal line and curve appear above the histogram.
        ax_main.plot([diag_start, diag_end], [diag_start, diag_end], color="dodgerblue", linestyle="-", linewidth=2, zorder=5)
        # ax_main.legend(loc="upper right", frameon=False)
        if l == "log":
            if sys.argv[1] == "oldmax100Files":
                ax_main.text(0.012, 0.01, r"$\mathbf{y_{\boldsymbol{\mu\mu}}=0}$", color="dodgerblue")
            elif sys.argv[1] == "newallFiles":
                ax_main.text(0.005, 0.008, r"$\mathbf{y_{\boldsymbol{\mu\mu}}=0}$", color="dodgerblue")
        else:
            if sys.argv[1] == "oldmax100Files":
                ax_main.text(0.012, 0.01, r"$\mathbf{y_{\boldsymbol{\mu\mu}}=0}$", color="dodgerblue")
            elif sys.argv[1] == "newallFiles":
                ax_main.text(-0.001, 0.0125, r"$\mathbf{y_{\boldsymbol{\mu\mu}}=0}$", color="dodgerblue")
        
        # Set xlim based on xbins
        ax_main.set_xlim(xbins[0], xbins[-1])
        
        mll_range = np.sort(df["mll"].values)
        # Add curves where yll = ±2.5
        
        for yll_val in [2.5, -2.5]:
            xmin_curve, xmax_curve = xmin_xmax(mll_range, np.full_like(mll_range, yll_val))
            # Only plot up to x = 0.3
            mask_xlim = xmax_curve <= 0.3
            xmin_curve_filtered = xmin_curve[mask_xlim]
            xmax_curve_filtered = xmax_curve[mask_xlim]
            # Add starting point at xmax=0
            xmax_curve_filtered = np.concatenate([[0], xmax_curve_filtered])
            xmin_curve_filtered = np.concatenate([[0], xmin_curve_filtered])
            order = np.argsort(xmax_curve_filtered)
            ax_main.plot(xmax_curve_filtered[order], xmin_curve_filtered[order], color="green", linestyle="-", linewidth=2, zorder=6, alpha=0.7)
            if l == "log":
                ax_main.text(0.058, 0.00032, r"$\mathbf{|y_{\boldsymbol{\mu\mu}}|=2.5}$", color="green")
            else:
                ax_main.text(0.08, 0.002, r"$\mathbf{|y_{\boldsymbol{\mu\mu}}|=2.5}$", color="green")

        # add curve for 90 < m_ll < 92 GeV
        df_mZ = df[(df["mll"] > 90) & (df["mll"] < 92)]

        xmin_mZ, xmax_mZ = xmin_xmax(df_mZ["mll"].values, df_mZ["yll"].values)
        order = np.argsort(xmax_mZ)
        if l == "log":
            # For log plot, use a straight line between endpoints
            ax_main.plot([xmax_mZ[order[0]], xmax_mZ[order[-1]]], [xmin_mZ[order[0]], xmin_mZ[order[-1]]], color="red", linestyle="-", linewidth=2, zorder=4)
            if sys.argv[1] == "oldmax100Files":
                ax_main.text(0.06, 0.0005, r"$\mathbf{m_{\boldsymbol{\mu\mu}}=m_{Z}}$", color="red")
                ax_main.text(0.056, 0.00044, r"$90\,\mathrm{GeV} < m_{Z} < 92\,\mathrm{GeV}$", color="red", fontsize=10)
            elif sys.argv[1] == "newallFiles":
                ax_main.text(0.021, 0.001, r"$\mathbf{m_{\boldsymbol{\mu\mu}}=m_{Z}}$", color="red")
                ax_main.text(0.0198, 0.00085, r"$90\,\mathrm{GeV} < m_{Z} < 92\,\mathrm{GeV}$", color="red", fontsize=10)
        else:
            # For linear plot, plot the full curve
            ax_main.plot(xmax_mZ[order], xmin_mZ[order], color="red", linestyle="-", linewidth=2, zorder=4)
            if sys.argv[1] == "oldmax100Files":
                ax_main.text(0.07, 0.0002, r"$\mathbf{m_{\boldsymbol{\mu\mu}}=m_{Z}}$", color="red")
                ax_main.text(0.068, -0.0002, r"$90\,\mathrm{GeV} < m_{Z} < 92\,\mathrm{GeV}$", color="red", fontsize=10)
            elif sys.argv[1] == "newallFiles":
                ax_main.text(0.06, -0.0006, r"$\mathbf{m_{\boldsymbol{\mu\mu}}=m_{Z}}$", color="red")
                ax_main.text(0.058, -0.0018, r"$90\,\mathrm{GeV} < m_{Z} < 92\,\mathrm{GeV}$", color="red", fontsize=10)

        # add other infos
        if l == "log":
            ax_main.text(0.05, 0.89, r"$0\,\mathrm{GeV} \leq p^{Z}_{T} < 20\,\mathrm{GeV}$", color="black", transform=ax_main.transAxes)
            if sys.argv[1] == "oldmax100Files":
                ax_main.text(0.05, 0.94, r"$60\,\mathrm{GeV} \leq m_{Z} \leq 120\,\mathrm{GeV}$", color="black", transform=ax_main.transAxes)
            elif sys.argv[1] == "newallFiles":
                ax_main.text(0.05, 0.94, r"$20\,\mathrm{GeV} \leq m_{Z} \leq 30000\,\mathrm{GeV}$", color="black", transform=ax_main.transAxes)    
        else:
            ax_main.text(0.6, 0.89, r"$0\,\mathrm{GeV} \leq p^{Z}_{T} < 20\,\mathrm{GeV}$", color="black", transform=ax_main.transAxes)
            if sys.argv[1] == "oldmax100Files":
                ax_main.text(0.6, 0.94, r"$60\,\mathrm{GeV} \leq m_{Z} \leq 120\,\mathrm{GeV}$", color="black", transform=ax_main.transAxes)
            elif sys.argv[1] == "newallFiles":
                ax_main.text(0.6, 0.94, r"$20\,\mathrm{GeV} \leq m_{Z} \leq 30000\,\mathrm{GeV}$", color="black", transform=ax_main.transAxes)

        if sys.argv[1] == "oldmax100Files":
            out_name = f"{out_dir}/oldmax100Files/xmaxll_xminll_{label}_{l}2D_oldmax100Files.pdf"
        elif sys.argv[1] == "newallFiles":
            out_name = f"{out_dir}/allFiles/xmaxll_xminll_{label}_{l}2D_allFiles.pdf"
        plt.savefig(out_name)
        plt.close()

        print(f"[OK] Saved {out_name}")
