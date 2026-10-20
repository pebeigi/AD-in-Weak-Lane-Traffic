[CmdletBinding()]
param(
    [ValidateSet("Data", "Test", "Analysis", "All", "Cohort", "Calibrate")]
    [string]$Target = "All",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot

function Invoke-PythonModule {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $Python -m @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: -m $($Arguments -join ' ')"
    }
}

function Initialize-Data {
    Invoke-PythonModule analysis.verify_data --extract
}

try {
    switch ($Target) {
        "Data" { Initialize-Data }
        "Test" {
            Initialize-Data
            Invoke-PythonModule unittest analysis.test_reproducibility
        }
        "Analysis" {
            Initialize-Data
            foreach ($module in @(
                "analysis.dataset_integrity",
                "analysis.dataset_and_behavior",
                "analysis.threshold_sensitivity",
                "analysis.plot_corridors",
                "analysis.dataset_figures",
                "analysis.paired_statistics",
                "analysis.final_diagnostics",
                "analysis.transfer_and_sensitivity",
                "analysis.interaction_regime",
                "analysis.generate_paper_figures",
                "analysis.generate_tables",
                "analysis.width_audit"
            )) {
                Invoke-PythonModule $module
            }
        }
        "All" {
            Initialize-Data
            Invoke-PythonModule unittest analysis.test_reproducibility
            & $PSCommandPath -Target Analysis -Python $Python
            if ($LASTEXITCODE -ne 0) { throw "Analysis failed with exit code $LASTEXITCODE" }
        }
        "Cohort" {
            Initialize-Data
            Invoke-PythonModule analysis.paired_pipeline prepare
            Invoke-PythonModule analysis.draw_main_cohort
        }
        "Calibrate" {
            Initialize-Data
            & $Python -c "from analysis.paired_pipeline import pilot, extended; pilot(); extended()"
            if ($LASTEXITCODE -ne 0) { throw "Calibration failed with exit code $LASTEXITCODE" }
        }
    }
}
finally {
    Pop-Location
}
