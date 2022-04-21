module RNAForecaster

using DiffEqFlux, DifferentialEquations
using Flux.Data: DataLoader
using Flux: mse
using Random, Statistics, StatsBase
using Suppressor
using DataFrames
using Distances

export trainRNAForecaster, predictCellFutures, mostTimeVariableGenes,
 KOeffectPredictions, totalKOImpact, geneKOExpressionChanges, geneResponseToKOs

include("trainRNAForecaster.jl")
include("makeRecursivePredictions.jl")
include("splicedDataPerturbationEffectPredictions.jl")


end # module
