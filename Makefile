PYTHON ?= python3
RUN = PYTHONPATH=. $(PYTHON) -m

.PHONY: help data test cohort calibrate analysis all clean-cache

help:
	@echo "make data       decompress data/Final_Lebanon_Data.csv.xz to ./Final_Lebanon_Data.csv"
	@echo "make test       determinism and model checks (analysis/test_reproducibility.py)"
	@echo "make analysis   all post-calibration statistics, figures, tables, and macros from the stored calibration results"
	@echo "make all        data + test + analysis"
	@echo "make cohort     redraw corridors and the 100-trajectory cohort (only if outputs/cohorts/paired_cohort.csv is removed)"
	@echo "make calibrate  rerun the paired GA calibration (about 22 CPU-hours; overwrites outputs/results/*_paired_results.csv)"

data: Final_Lebanon_Data.csv

Final_Lebanon_Data.csv: data/Final_Lebanon_Data.csv.xz
	$(PYTHON) -c "import lzma, shutil; shutil.copyfileobj(lzma.open('$<'), open('$@', 'wb'), 1 << 20)"

test: data
	$(RUN) analysis.test_reproducibility

cohort: data
	$(RUN) analysis.paired_pipeline prepare
	$(RUN) analysis.draw_main_cohort

calibrate: data
	PYTHONPATH=. $(PYTHON) -c "from analysis.paired_pipeline import pilot, extended; pilot(); extended()" 2>&1 | grep -v "GA gen=" | tee outputs/rerun_calibration.log

analysis: data
	$(RUN) analysis.dataset_integrity
	$(RUN) analysis.dataset_and_behavior
	$(RUN) analysis.threshold_sensitivity
	$(RUN) analysis.plot_corridors
	$(RUN) analysis.dataset_figures
	$(RUN) analysis.paired_statistics
	$(RUN) analysis.final_diagnostics
	$(RUN) analysis.transfer_and_sensitivity
	$(RUN) analysis.interaction_regime
	$(RUN) analysis.generate_paper_figures
	$(RUN) analysis.generate_tables
	$(RUN) analysis.width_audit

all: data test analysis

clean-cache:
	rm -rf outputs/cache
