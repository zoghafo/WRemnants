import ROOT
import uproot
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse
import logging


variables = {
	"mll": "m_{\\mu\\mu} ~ \\text{[GeV]}", 
	"ptll": "p^{\\mu\\mu}_{T} ~ \\text{[GeV]}",
	"yll": "y^{\\mu\\mu}",
	"xmaxll": "x_{\\text{max}}^{\\mu\\mu}",
	"xminll": "x_{\\text{min}}^{\\mu\\mu}",
	}

data_pred = [
	"data",
	"pred"
	]

colours = ["steelblue", "maroon", "darkgreen", "orange", "purple", "brown", "pink", "olive", "cyan", "magenta"]


# Command-line parsing for input files and optional logfile
parser = argparse.ArgumentParser(description="Create stacked hist PDF from ROOT files")
parser.add_argument("files", nargs='+', help="ROOT files (2 or more)")
parser.add_argument("--log", "-l", help="Optional log file path", default=None)
parser.add_argument("--labels", nargs='+', help="Labels for each file (optional)")
args = parser.parse_args()

# Define pt_cuts based on number of files
pt_cuts = {}

default_labels = [
	"p^{Z}_{T} < 2 ~ \\text{GeV}",
	"2 ~ \\text{GeV} \\leq p^{Z}_{T} < 4 ~ \\text{GeV}",
	"4 ~ \\text{GeV} \\leq p^{Z}_{T} < 6 ~ \\text{GeV}",
	"6 ~ \\text{GeV} \\leq p^{Z}_{T} < 8 ~ \\text{GeV}",
	"8 ~ \\text{GeV} \\leq p^{Z}_{T} < 10 ~ \\text{GeV}",
	"10 ~ \\text{GeV} \\leq p^{Z}_{T} < 12 ~ \\text{GeV}",
	"12 ~ \\text{GeV} \\leq p^{Z}_{T} < 14 ~ \\text{GeV}",
	"14 ~ \\text{GeV} \\leq p^{Z}_{T} < 16 ~ \\text{GeV}",
	"16 ~ \\text{GeV} \\leq p^{Z}_{T} < 18 ~ \\text{GeV}",
	"18 ~ \\text{GeV} \\leq p^{Z}_{T} < 20 ~ \\text{GeV}",
]

if args.labels:
	for i, file in enumerate(args.files):
		pt_cuts[file] = args.labels[i] if i < len(args.labels) else f"File {i+1}"
else:
	for i, file in enumerate(args.files):
		pt_cuts[file] = default_labels[i] if i < len(default_labels) else f"File {i+1}"

for v in variables.keys():
	for dp in data_pred:

		# logging will be configured after the output PDF path is set so the
		# default logfile can mirror the PDF filename when --log is not given
		logger = None

		out_pdf = "/home/z/zoghafoo/www/pT_Cuts/" + v + "/" + v + "_stack_" + dp + ".pdf"
		figure = plt.figure()

		# If no logfile was provided, default to same basename as the PDF but with .log
		log_path = args.log if args.log else out_pdf.replace(".pdf", ".log")

		# Clear existing handlers
		for handler in logging.root.handlers[:]:
			logging.root.removeHandler(handler)

		# Configure logging now that out_pdf/log_path are known
		handlers = [logging.StreamHandler(sys.stdout)]
		if log_path:
			# ensure directory exists for logfile
			logdir = os.path.dirname(os.path.abspath(log_path))
			if logdir and not os.path.exists(logdir):
				os.makedirs(logdir, exist_ok=True)
			handlers.append(logging.FileHandler(log_path, mode="w"))
		logging.basicConfig(level=logging.INFO, handlers=handlers, format="%(asctime)s %(levelname)s: %(message)s", force=True)
		logger = logging.getLogger(__name__)

		logger.info("Starting stackplot_rootfiles.py")
		logger.info("Using logfile: %s", log_path)
		logger.info("Command: %s", " ".join(sys.argv))

		logger.info("Opening ROOT files: %s", ", ".join(args.files))
		opened_files = [uproot.open(f) for f in args.files]

		# Extract histograms from all files
		histograms = []
		bins = None
		for i, file in enumerate(opened_files):
			hist, file_bins = file[f"{v}_{dp}"].to_numpy()
			histograms.append(hist)
			if bins is None:
				bins = file_bins
			logger.info("Got histogram: %s (entries=%d)",
						f"{v}_{dp} ({args.files[i]})", np.sum(hist) if hist is not None else 0)

		# Create stacked plot with all histograms
		width = np.diff(bins)
		bottom = np.zeros(len(bins) - 1)
		
		for i, hist in enumerate(histograms):
			label = pt_cuts[args.files[i]]
			plt.bar(bins[:-1], hist, width=width, bottom=bottom, align='edge', color=colours[i], label=fr"${label}$; {np.sum(hist):.0f} entries", alpha=0.8, linewidth=0)
			bottom += hist
		plt.xlabel(fr"${variables[v]}$", loc="right")
		plt.ylabel("Events / GeV", loc="top")

		plt.draw()


		if v == "yll":
			plt.legend(fontsize = 7, loc = 'upper center', ncol = 2 )
			plt.ylim(0, 1.5 * np.max(bottom))
		else:
			plt.legend(fontsize = 7, loc='upper right', frameon = False)
			plt.ylim(0, 1.1 * np.max(bottom))


		figure.savefig(out_pdf, bbox_inches='tight')

		logger.info(f"Plot {out_pdf} Finished")
		plt.close()