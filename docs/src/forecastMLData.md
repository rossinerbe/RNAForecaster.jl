# Metabolic Labeling for Forecasts

One way to get the two time points needed to train RNAForecaster is to use metabolic
labeling transcriptomic profiling techniques such as scEU-seq. In these techniques,
cells are incubated with modified uridine for a set period of time, after which they
are harvested for scRNA sequencing. The transcripts with incorporated uridine are
labeled, and we therefore know they were transcribed in the labeling time period.
We also know the other transcripts were transcribed earlier.

We can use these labeled and unlabeled transcripts to generate our two input matrices.
To get a clean time point 1 matrix of total expression that we can use with the total expression
at time point 2 (when the cells were harvested) we do need to estimate the number of
degraded transcripts in the labeling period, which we calculate via the slope
between the labeled and total transcripts.
```julia
using JLD2
rpeExampleData = load_object("data/rpeExampleData.jld2");
labeledData = Float32.(rpeExampleData[1])
unlabeledData = Float32.(rpeExampleData[2])
totalData = Float32.(rpeExampleData[3])
geneNames = string.(rpeExampleData[4])
labelingTime = vec(rpeExampleData[5])

t1Estimate = estimateT1LabelingData(labeledData, totalData, unlabeledData, labelingTime)
```

For this tutorial, we will subset to cells labeled for 60 minutes, enabling us to
forecast in one hour increments.
```julia
hourCells = findall(x->x == 60.0, labelingTime)
t1_60min = t1Estimate[:,hourCells]
t2_60min = totalData[:,hourCells]

#log normalize
t1_60min = log1p.(t1_60min)
t2_60min = log1p.(t2_60min)

#train RNAForecaster
trainedNetwork = trainRNAForecaster(t1_60min, t2_60min, trainingProp = 1.0,
 learningRate = 0.0001, batchsize = 25, iterToCheck = 24,
  checkStability = false)
```

Now, we can forecast expression predictions 24 hours into the future, get predictions
at each hour.
```julia
futureExpressionPreds = predictCellFutures(trainedNetwork[1], t2_60min, 24)
```

The predictions can also be conditioned on arbitrary perturbations in gene expression.
```julia
futureExpressionPredsP = predictCellFutures(trainedNetwork[1], t2_60min, 24,
 perturbGenes = geneNames[1:2],
geneNames = geneNames, perturbationLevels = [2.0f0, 0.0f0])
```

Once we have forecast expression levels for each gene, we may want to know which
genes expression levels change the most over time, as these are likely to be important
in ongoing biological process we are attempting to model.
To assay this we simply run `mostTimeVariableGenes` which outputs a table of genes
ordered by the most variable over predicted time points.

```julia
geneOutputTable = mostTimeVariableGenes(futureExpressionPreds, geneNames)
```
