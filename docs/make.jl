using Documenter
using RNAForecaster

makedocs(
    sitename = "RNAForecaster",
    format = Documenter.HTML(),
    pages = ["Home" => "index.md",
             "Training and Forecasting" => "training.md",
             "Predictions from Splicing Data" => "splicedData.md",
             "Metabolic Labeling for Forecasts" => "forecastMLData.md",
             "Functions" => "functions.md"],
    modules = [RNAForecaster]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/rossinerbe/RNAForecaster.jl"
)
