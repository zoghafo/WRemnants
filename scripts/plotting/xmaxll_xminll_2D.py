import ROOT
import uproot
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse
import logging
import pandas as pd
import seaborn as sns


data_file = '/home/z/zoghafoo/www/pT_Cuts/events_dataPostVFP.csv'
pred_file = '/home/z/zoghafoo/www/pT_Cuts/events_ZmumuPostVFP.csv'

data = pd.read_csv(data_file, delimiter=',')
pred = pd.read_csv(pred_file, delimiter=',')

data_mll = data['mll'].to_numpy()
data_yll = data['yll'].to_numpy()

pred_mll = pred['mll'].to_numpy()
pred_yll = pred['yll'].to_numpy()


for d in ["data", "pred"]:
	if d == "data":
		mll = data_mll
		yll = data_yll
	else:
		mll = pred_mll
		yll = pred_yll

	print(f"\n\n\nlen({d}) =  {len(mll)} \t len(yll) = {len(yll)}\n\n\n")

	canvas = ROOT.TCanvas(f"cd_{d}")
	canvas.SetRightMargin(0.15)
	canvas.SetLeftMargin(0.15)
	canvas.SetBottomMargin(0.15)
	canvas.SetTopMargin(0.1)

	canvas.Print(f"/home/z/zoghafoo/www/pT_Cuts/xmaxll_xminll_{d}_2D.pdf[")
	canvas.cd()

	hist_2D = ROOT.TH2F(f"hist_{d}_2D", f"xminll vs xmaxll ({d}); max(x^{{obs}}_{{#mu#mu}}); min(x^{{obs}}_{{#mu#mu}})", 50, 0, 0.1, 10, 0, 0.01)



	if len(mll) == len(yll):
		for i in range(len(mll)):
			xplusll = np.exp( yll[i] ) * mll[i] / np.sqrt( 13000 * 13000 )
			xminusll = np.exp( - yll[i] ) * mll[i] / np.sqrt( 13000 * 13000 )

			xmaxll = max( xplusll, xminusll )
			xminll = min( xplusll, xminusll )

			bin_xmaxll = hist_2D.GetXaxis().FindBin( xmaxll )
			bin_xminll = hist_2D.GetYaxis().FindBin( xminll )

			hist_2D.SetBinContent(bin_xmaxll, bin_xminll, hist_2D.GetBinContent(bin_xmaxll, bin_xminll) + 1)

	else:
		print("Data lengths do not match!")

	hist_2D.SetStats(0)
	# canvas.SetLogx()
	# canvas.SetLogy()
	
	hist_2D.Draw("COLZ")
	
	canvas.Print(f"/home/z/zoghafoo/www/pT_Cuts/xmaxll_xminll_{d}_2D.pdf")
	canvas.Print(f"/home/z/zoghafoo/www/pT_Cuts/xmaxll_xminll_{d}_2D.pdf]")