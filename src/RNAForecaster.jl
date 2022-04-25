module RNAForecaster

using DiffEqFlux, DifferentialEquations
using Flux.Data: DataLoader
using Flux: mse
using Random, Statistics, StatsBase, LinearRegression
using Suppressor
using DataFrames, CategoricalArrays
using Distances


export trainRNAForecaster, predictCellFutures, mostTimeVariableGenes,
 KOeffectPredictions, totalKOImpact, geneKOExpressionChanges, geneResponseToKOs,
 estimateT1LabelingData, filterByZeroProp, filterByGeneVar

include("trainRNAForecaster.jl")
include("makeRecursivePredictions.jl")
include("splicedDataPerturbationEffectPredictions.jl")
include("estimateT1.jl")
include("utils.jl")


end # module
