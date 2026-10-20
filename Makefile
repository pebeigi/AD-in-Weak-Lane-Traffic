PYTHON ?= python3
RUN = PYTHONPATH=. $(PYTHON) -m

.PHONY: help verify-data data test cohort calibrate analysis all clean-cache

help:
	@echo "make verify-data verify the compressed and uncompressed dataset checksums"
	@echo "make data       verify and decompress data/Final_Lebanon_Data.xz to ./Final_Lebanon_Data.csv"
	@echo "make test       determinism and model checks (analysis/test_reproducibility.py)"
	@echo "make analysis   all post-calibration statistics, figures, tables, and macros from the stored calibration results"
	@echo "make all        data + test + analysis"
	@echo "make cohort     redraw corridors and the 100-trajectory cohort (only if outputs/cohorts/paired_cohort.csv is removed)"
	@echo "make calibrate  rerun the paired GA calibration (about 22 CPU-hours; overwrites outputs/results/*_paired_results.csv)"

verify-data:
	$(RUN) analysis.verify_data

data: Final_Lebanon_Data.csv

Final_Lebanon_Data.csv: data/Final_Lebanon_Data.xz
	$(RUN) analysis.verify_data --extract

test: verify-data data
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
	$(RUN) analysis.model_diagnostics
	$(RUN) analysis.transfer_and_sensitivity
	$(RUN) analysis.interaction_regime
	$(RUN) analysis.generate_paper_figures
	$(RUN) analysis.generate_tables
	$(RUN) analysis.recording_dimension_audit

all: data test analysis

clean-cache:
	rm -rf outputs/cache
