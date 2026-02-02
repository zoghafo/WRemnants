import hist
from matplotlib import colormaps
import os

from utilities import parsing
from utilities.styles import styles
from wremnants import syst_tools
from wremnants.datasets.datagroups import Datagroups
from wremnants.histselections import FakeSelectorSimpleABCD
from wremnants.regression import Regressor
from wums import boostHistHelpers as hh
from wums import logging, output_tools, plot_tools

# -----------------------------------------------------------------------------
# ARGUMENTS
# -----------------------------------------------------------------------------

parser = parsing.plot_parser()

parser.add_argument(
    "infile",
    nargs=2,
    help="Two analysis output files to be summed bin-by-bin"
)

parser.add_argument("--ratioToData", action="store_true")
parser.add_argument("-n", "--baseName", type=str, default="nominal")
parser.add_argument("--nominalRef", type=str)
parser.add_argument("--hists", type=str, nargs="+", required=True)

parser.add_argument(
    "-c", "--channel",
    choices=["plus", "minus", "all"],
    default="all"
)

parser.add_argument("--rebin", type=int, nargs="*", default=[])
parser.add_argument("--absval", type=int, nargs="*", default=[])
parser.add_argument("--axlim", type=parsing.str_to_complex_or_int, nargs="*", default=[])
parser.add_argument("--rebinBeforeSelection", action="store_true")
parser.add_argument("--logy", action="store_true")

parser.add_argument("--procFilters", type=str, nargs="*")
parser.add_argument(
    "--excludeProcs",
    type=str,
    nargs="*",
    default=["QCD", "WtoNMu_5", "WtoNMu_10", "WtoNMu_50"],
)

parser.add_argument("--noData", action="store_true")
parser.add_argument("--noFill", action="store_true")
parser.add_argument("--noStack", action="store_true")
parser.add_argument("--noRatio", action="store_true")
parser.add_argument("--density", action="store_true")

parser.add_argument(
    "--flow",
    choices=["show", "sum", "hint", "none"],
    default="none"
)

parser.add_argument("--noRatioErr", action="store_false", dest="ratioError")
parser.add_argument("--rlabel", type=str, default=None)
parser.add_argument("--selection", type=str)
parser.add_argument("--presel", type=str, nargs="*", default=[])
parser.add_argument("--normToData", action="store_true")

parser.add_argument(
    "--fakeEstimation",
    choices=["mc", "simple", "extrapolate", "extended1D", "extended2D"],
    default="extended1D"
)

parser.add_argument(
    "--fakeMCCorr",
    type=str,
    nargs="*",
    default=[None],
    choices=["none", "pt", "eta", "mt"],
)

parser.add_argument("--forceGlobalScaleFakes", type=float, default=None)

parser.add_argument(
    "--fakeSmoothingMode",
    choices=FakeSelectorSimpleABCD.smoothing_modes,
    default="full"
)

parser.add_argument("--fakeSmoothingOrder", type=int, default=3)

parser.add_argument(
    "--fakeSmoothingPolynomial",
    choices=Regressor.polynomials,
    default="chebyshev"
)

parser.add_argument(
    "--fakerateAxes",
    nargs="+",
    default=["eta", "pt", "charge"]
)

parser.add_argument("--fineGroups", action="store_true")

parser.add_argument(
    "--subplotSizes",
    nargs=2,
    type=int,
    default=[4, 2]
)

args = parser.parse_args()
logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

# -----------------------------------------------------------------------------
# CREATE DATAGROUPS (TWO FILES)
# -----------------------------------------------------------------------------

groups_A = Datagroups(
    args.infile[0],
    filterGroups=args.procFilters,
    excludeGroups=args.excludeProcs,
)

groups_B = Datagroups(
    args.infile[1],
    filterGroups=args.procFilters,
    excludeGroups=args.excludeProcs,
)

groups_list = (groups_A, groups_B)

# -----------------------------------------------------------------------------
# PRESELECTION
# -----------------------------------------------------------------------------

select = {}
if args.channel != "all":
    select["charge"] = -1.0j if args.channel == "minus" else 1.0j

if args.presel:
    s = hist.tag.Slicer()
    presel = {}
    for ps in args.presel:
        if "=" in ps:
            ax, rng = ps.split("=")
            lo, hi = map(float, rng.split(","))
            presel[ax] = s[complex(0, lo):complex(0, hi):hist.sum]
        else:
            presel[ps] = s[::hist.sum]

    for g in groups_list:
        g.setGlobalAction(lambda h: h[presel])

# -----------------------------------------------------------------------------
# REBIN
# -----------------------------------------------------------------------------

if args.axlim or args.rebin or args.absval:
    for g in groups_list:
        g.set_rebin_action(
            args.hists[0].split("-"),
            args.axlim,
            args.rebin,
            args.absval,
            args.rebinBeforeSelection,
        )

# -----------------------------------------------------------------------------
# FAKE SETUP
# -----------------------------------------------------------------------------

datasets = groups_A.getNames()

for g in groups_list:
    g.fakerate_axes = args.fakerateAxes
    g.set_histselectors(
        datasets,
        args.baseName,
        smoothing_mode=args.fakeSmoothingMode,
        smoothingOrderSpectrum=args.fakeSmoothingOrder,
        smoothingPolynomialSpectrum=args.fakeSmoothingPolynomial,
        integrate_x=all("mt" not in x.split("-") for x in args.hists),
        mode=args.fakeEstimation,
        forceGlobalScaleFakes=args.forceGlobalScaleFakes,
        mcCorr=args.fakeMCCorr,
    )

# -----------------------------------------------------------------------------
# LOAD HISTOGRAMS
# -----------------------------------------------------------------------------

for g in groups_list:
    if not args.nominalRef:
        nominalName = args.baseName.rsplit("_", 1)[0]
        g.setNominalName(nominalName)
        g.loadHistsForDatagroups(
            args.baseName,
            syst="",
            procsToRead=datasets,
            applySelection=True,
        )
    else:
        nominalName = args.nominalRef
        g.setNominalName(nominalName)
        g.loadHistsForDatagroups(
            nominalName,
            syst=args.baseName,
            procsToRead=datasets,
            applySelection=True,
        )

# -----------------------------------------------------------------------------
# BIN-BY-BIN SUMMATION
# -----------------------------------------------------------------------------

def sum_datagroups(target, source):
    for proc, info in target.groups.items():
        if proc not in source.groups:
            continue

        hA = info["hist"]
        hB = source.groups[proc]["hist"]

        if hA.axes != hB.axes:
            raise RuntimeError(f"Axis mismatch for process {proc}")

        info["hist"] = hA + hB

sum_datagroups(groups_A, groups_B)

# 🔑 Continue with summed object only
groups = groups_A

# -----------------------------------------------------------------------------
# PLOTTING (UNCHANGED FROM HERE)
# -----------------------------------------------------------------------------

groups.sortByYields(args.baseName, nominalName=nominalName)
histInfo = groups.groups

exclude = ["Data"]
unstack = exclude[:]
if args.noData:
    unstack.remove("Data")

prednames = list(
    reversed(
        groups.getNames(
            [d for d in datasets if d not in exclude],
            exclude=False,
            match_exact=True,
        )
    )
)

outdir = output_tools.make_plot_dir(args.outpath, args.outfolder, eoscp=args.eoscp)

for h in args.hists:
    xlabel = plot_tools.get_axis_label(styles, h.split("-"))
    ylabel = r"$Events\,/\,bin$"

    fig = plot_tools.makeStackPlotWithRatio(
        histInfo,
        prednames,
        histName=args.baseName,
        logy=args.logy,
        unstacked=unstack,
        xlabel=xlabel,
        ylabel=ylabel,
        lumi=groups.lumi,
        ratio_to_data=args.ratioToData,
        no_fill=args.noFill,
        no_stack=args.noStack,
        no_ratio=args.noRatio,
        flow=args.flow,
        subplotsizes=args.subplotSizes,
    )

    plot_tools.save_pdf_and_png(outdir, h)

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
