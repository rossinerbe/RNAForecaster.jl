# Metabolic Labeling for Forecasts

One way to get the two time points needed to train RNAForecaster is to use metabolic
labeling transcriptomic profiling techniques such as scEU-seq. In these techniques,
cells are incubated with modified uridine for a set period of time, after which they
are harvested for scRNA sequencing. The transcripts with incorporated uridine are
labeled, and we therefore know they were transcribed in the labeling time period.
We also know the other transcripts were transcribed earlier.

We can use these labeled and unlabeled transcripts to generate our two input matrices.
To get a clean time point t=0 matrix of total expression that we can use with the total expression
at time point t=1 (when the cells were harvested) we do need to estimate the number of
degraded transcripts in the labeling period, which we calculate via the slope
between the labeled and total transcripts (see [Qiu et al. 2022](https://doi.org/10.1016/j.cell.2021.12.045)).

Here we use a subset of a scEUseq dataset from hTERT immortalized retinal pigment
epithelium cells, published by [Battich et al. 2020](https://doi.org/10.1126/science.aax3072).

```julia
using JLD2
rpeExampleData = load_object("data/rpeExampleData.jld2");
labeledData = Float32.(rpeExampleData[1])
unlabeledData = Float32.(rpeExampleData[2])
totalData = Float32.(rpeExampleData[3])
geneNames = string.(rpeExampleData[4])
labelingTime = vec(rpeExampleData[5])

t0Estimate = estimateT0LabelingData(labeledData, totalData, unlabeledData, labelingTime)
```

For this tutorial, we will subset to cells labeled for 60 minutes, enabling us to
forecast in one hour increments.
```julia
hourCells = findall(x->x == 60.0, labelingTime)
t0_60min = t0Estimate[:,hourCells]
t1_60min = totalData[:,hourCells]

#log normalize
t0_60min = log1p.(t0_60min)
t1_60min = log1p.(t1_60min)

#train RNAForecaster
trainedNetwork = trainRNAForecaster(t0_60min, t1_60min, trainingProp = 1.0,
 learningRate = 0.0001, batchsize = 25);
```

If we wanted to train the network using the GPU,
we would run
`trainedNetwork = trainRNAForecaster(t0_60min, t1_60min, trainingProp = 1.0,
 learningRate = 0.0001, batchsize = 25, useGPU = true)`

Note on GPU errors: on some systems an error can arise related to the way julia and CUDA
handle memory. If you encounter unexpected Out of GPU Memory errors, try setting the
environment variable `JULIA_CUDA_MEMORY_POOL = none`


Now, we can forecast expression predictions 24 hours into the future, and get
predictions at each hour.
```julia
futureExpressionPreds = predictCellFutures(trainedNetwork[1], t1_60min, 24)
```
Predictions can also be performed on the GPU
`futureExpressionPreds = predictCellFutures(trainedNetwork[1], t1_60min, 24,
   useGPU = true)`

The predictions can also be conditioned on arbitrary perturbations in gene expression.
```julia
futureExpressionPredsP = predictCellFutures(trainedNetwork[1], t1_60min, 24,
 perturbGenes = geneNames[1:2], geneNames = geneNames,
 perturbationLevels = [2.0f0, 0.0f0])
```

Once we have forecast expression levels for each gene, we may want to know which
genes expression levels change the most over time, as these are likely to be important
in ongoing biological process we are attempting to model.
To assay this we simply run `mostTimeVariableGenes` which outputs a table of genes
ordered by the most variable over predicted time points.

```julia
geneOutputTable = mostTimeVariableGenes(futureExpressionPreds, geneNames)
```

We can save an RNAForecaster network for later use with:
```julia
saveForecaster(trainedNetwork, "exampleForecaster.jld2")
```

which we can load back into memory using
```julia
loadedNetwork = loadForecaster("exampleForecaster.jld2")
```
WARNING: If you update Flux.jl or DiffEqFlux.jl, saved networks are not guaranteed
to be forward compatible.
