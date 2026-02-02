# Paths
SCRIPT = /home/z/zoghafoo/WRemnants/scripts/plotting/stackplot_rootfiles_python.py
PTCUT_DIR = /home/z/zoghafoo/www/pT_Cuts
FILES = ZpT0to2 ZpT2to4 ZpT4to6 ZpT6to8 ZpT8to10 ZpT10to12 ZpT12to14 ZpT14to16 ZpT16to18 ZpT18to20

# Convert FILES to full paths with .root extension
ROOTS = $(addprefix $(PTCUT_DIR)/,$(addsuffix .root,$(FILES)))

rootfiles:
	./rootfile_creation.sh

ptcutstackplot:
	python3 /home/z/zoghafoo/WRemnants/scripts/plotting/stackplot_rootfiles_python.py $(ROOTS)

xminllxmaxll2Dplot:
	python3 /home/z/zoghafoo/WRemnants/scripts/plotting/xmaxll_xminll_2D.py $(ROOTS)