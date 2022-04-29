module RNAForecaster

using DiffEqFlux, DifferentialEquations
using Flux.Data: DataLoader
using Flux: mse
using Random, Statistics, StatsBase, LinearRegression
using DataFrames, CategoricalArrays
using Distances
using Distributed
using Base.Iterators: partition
using CUDA
using JLD2
using Flux: loadmodel!


export trainRNAForecaster, predictCellFutures, mostTimeVariableGenes,
 KOeffectPredictions, totalKOImpact, geneKOExpressionChanges, geneResponseToKOs,
 estimateT1LabelingData, filterByZeroProp, filterByGeneVar, createEnsembleForecaster,
 saveForecaster, loadForecaster

include("trainRNAForecaster.jl")
include("makeRecursivePredictions.jl")
include("splicedDataPerturbationEffectPredictions.jl")
include("estimateT1.jl")
include("utils.jl")


end # module
