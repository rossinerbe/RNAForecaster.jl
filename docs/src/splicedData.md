# Predictions from Splicing Data

RNAForecaster requires transcriptomics data from the same cell at two different points
in time. The best way to acquire this sort of information from standard single cell
RNA-seq protocols is to leverage incidental capture of introns to determine whether
a transcript is spliced or unspliced. In general, it is a reasonable assumption that
spliced transcripts were produced more recently than unspliced, thus giving some
temporal information on the sequenced transcripts.

Unfortunately, the spliced and unspliced count matrices thus generated cannot be
considered two equivalent time points, which prevents us from being able to
straightforwardly predict future expression states from one time point to the
next. We could attempt to handle this problem using RNA velocity style assumptions
about the splicing and degradation rates to update our predicted expression levels
at each time point. However, this process will propagate too much error for the
resulting predictions to be of value.

Instead, we make a single prediction about the direction of regulation, which
can be compared to predictions under perturbations, allowing us to uncover
key genes relating to the biological process of interest.

Here we will use a small subset of murine pancreatic development data published
 by [Bastidas-Ponce et al. (2019)](https://journals.biologists.com/dev/article/146/12/dev173849/19483/Comprehensive-single-cell-mRNA-profiling-reveals-a)
 to demonstrate how RNAForecaster is used with splicing data.

```julia
#read in data
using DelimitedFiles
splicedSub = readdlm("data/pancSplicedExampleData.csv", ',')
unsplicedSub = readdlm("data/pancUnsplicedExampleData.csv", ',')
geneNamesSub = readdlm("data/pancGeneNamesExampleData.csv", ',')
geneNamesSUb = string.(geneNamesSub)[:,1]

#data must be log normalized and converted to Float32
splicedSub = Float32.(log1p.(splicedSub))
unsplicedSub = Float32.(log1p.(unsplicedSub))

#train RNAForecaster
trainedModel = trainRNAForecaster(splicedSub, unsplicedSub, checkStability = false);
```

So we now have a neural network that can predicted unspliced counts based on
spliced counts. We can now perturb each gene to see how the predictions of
unspliced counts change. This can help uncover regulatory relationships and important
genes underlying the system's biology.

Here we will simulate a KO of each gene of the 10 genes in the data from the
initial conditions of 25 random cells.
```julia
KOEffects = perturbEffectPredictions(trainedModel[1], splicedSub, 25)
```

We can also specify specific genes and specific perturbations.
```julia
perturbEffects = perturbEffectPredictions(trainedModel[1], splicedSub, 25,
  perturbGenes = geneNamesSub[[2,5]], geneNames = geneNamesSub,
  perturbLevels = [1.0f0, 1.5f0])
```

```@docs
perturbEffectPredictions
```

Convenience functions are provided to search the results of the perturbations

```@docs
totalPerturbImpact
genePerturbExpressionChanges
geneResponseToPerturb
```

For example, we can get a list of the genes, and the size of the impact of the KO
on the rest of the genes' expression, measured in euclidean distance, using

```julia
totalPerturbImpact(KOEffects, geneNamesSub)
```
