"""
`perturbEffectPredictions(trainedNetwork, splicedData::Matrix{Float32}, nCells::Int;
     perturbGenes::Vector{String} = Vector{String}(undef, 0), geneNames::Vector{String} = Vector{String}(undef,0),
     seed::Int=123)`

Based on spliced/unspliced counts, predict the immediate transcriptomic effect
of any or all single gene perturbations.
Outputs a tuple containing the cells used for prediction, the expression predictions,
the gene wise differences, and the cell-wise euclidean distances for each perturbation.

This function is capable of running on multiple parallel processes using Distributed.jl.
Call addprocs(n) before running the function to add parallel workers, where n is the
number of additional processes desired.

# Required Arguments
* trainedNetwork - trained neuralODE from trainRNAForecaster
* splicedData - log normalized spliced counts matrix. Must be in Float32 format
* nCells - how many cells from the data should be used for prediction of perturb effect.
 Higher values will increase computational time required.
# Optional Arguments
* perturbGenes - list of genes to simulate a perturbation of. By default all genes are used
* geneNames - if providing a subset of the genes to perturb, a vector of gene names to
 match against, in the order of splicedData
* perturbLevels - list of perturbation levels to use for each perturbed gene. By default
 all genes are set to zero, simulating a KO.
* seed - Random seed for reproducibility on the cells chosen for prediction
"""
function perturbEffectPredictions(trainedNetwork, splicedData::Matrix{Float32}, nCells::Int;
     perturbGenes::Vector{String} = Vector{String}(undef, 0), geneNames::Vector{String} = Vector{String}(undef,0),
     perturbLevels::Vector{Float32} = Vector{Float32}(undef, 0), seed::Int=123)

    Random.seed!(seed)
    cellsToUse = sample(1:size(splicedData)[2], nCells, replace = false)
    splicedSub = splicedData[:, cellsToUse]

    if nprocs() > 1
        println("Predicting perturbation effects using " * string(nprocs()) * " parallel processes.")
    end

    if length(perturbGenes) == 0
        #make prediction for initial conditions
        #matrix to store data in
        initialPredictions = Matrix{Float32}(undef, size(splicedData)[1], nCells)
        for n=1:nCells
             tmpPred = @spawn trainedNetwork(splicedSub[:,n])[1]
             initialPredictions[:,n] = fetch(tmpPred)
        end

        #set negative predictions to zero
        initialPredictions[findall(x->x < 0, initialPredictions)] .= 0

        #create tensor to store perturbation output in: nGenes X perturbations(nGenes) x nCells
        perturbPredictions = Array{Float32}(undef, size(splicedData)[1], size(splicedData)[1], nCells)
        #make predictions
            for n=1:nCells
                for i=1:size(splicedData)[1]
                    tmpCell = splicedSub[:,n]
                    tmpCell[i] = 0
                    tmpPred = @spawn trainedNetwork(tmpCell)[1]
                    perturbPredictions[:,i,n] = fetch(tmpPred)
                end
            end
    else
        if length(geneNames) != size(splicedSub)[1]
            error("Length of gene names is not equal to the number of rows (genes)
            in the input data.")
        end
        perturbGeneInds = findall(in(perturbGenes), geneNames)
        #make prediction for initial conditions
        #matrix to store data in
        initialPredictions = Matrix{Float32}(undef, size(splicedData)[1], nCells)
            for n=1:nCells
                initialPredictions[:,n] = @spawn trainedNetwork(splicedSub[:,n])[1]
            end
        initialPredictions = fetch.(initialPredictions)

        #set negative predictions to zero
        initialPredictions[findall(x->x < 0, initialPredictions)] .= 0

        #create tensor to store perturbation output in: nGenes X perturbations x nCells
        perturbPredictions = Array{Float32}(undef, size(splicedData)[1], length(perturbGeneInds), nCells)
        #make predictions
        for n=1:nCells
            for i= 1:length(perturbGeneInds)
                tmpCell = splicedSub[:,n]
                if length(perturbLevels) == 0
                    tmpCell[perturbGeneInds[i]] = 0
                elseif length(perturbLevels) != length(perturbGenes)
                    error("perturbLevels and perturbGenes must be the same length.")
                else
                    tmpCell[perturbGeneInds[i]] = perturbLevels[i]
                end
                tmpPred = @spawn trainedNetwork(tmpCell)[1]
                perturbPredictions[:,i,n] = fetch(tmpPred)
            end
        end
    end

    #set negative predictions to zero
    perturbPredictions[findall(x->x < 0, perturbPredictions)] .= 0

    #find gene wise difference between initial and perturbations
    perturbGeneChange = Array{Float32}(undef, size(perturbPredictions)[1], size(perturbPredictions)[2], nCells)
    for n=1:nCells
        for i=1:size(perturbPredictions)[2]
            for j=1:size(perturbPredictions)[1]
                    perturbGeneChange[j,i,n] = perturbPredictions[j,i,n] - initialPredictions[j,n]
            end
        end
    end

    #find total cell wise differences via euclidean distance
    perturbDistances = Array{Float32}(undef, size(perturbPredictions)[2], nCells)
    for n=1:nCells
        for i=1:size(perturbPredictions)[2]
            perturbDistances[i,n] = euclidean(perturbPredictions[:,i,n], initialPredictions[:,n])
        end
    end

    return (cellsToUse, perturbPredictions, perturbGeneChange, perturbDistances)
end


"""
`totalPerturbImpact(perturbData, geneNames::Vector{String})`

Function to yield a sorted data frame of the size of the predicted effect of a perturbation
on the cellular transcriptome. Intended to serve as a measure of more or less impactful
gene perturbations.

# Required Arguments
* perturbData - results from perturbEffectPredictions function
* geneNames - vector of gene names in the order of the input expression data.
Should only include perturbed genes
"""
function totalPerturbImpact(perturbData, geneNames::Vector{String})
    #find genes of largest effect on cell transcriptomic state
    #calculate mean and median distnaces for each gene across simulated cells
    meanDists = vec(mean(perturbData[4], dims = 2))
    medianDists = vec(median(perturbData[4], dims = 2))

    distData = DataFrame(Genes = geneNames, MeanDistances = meanDists,
     MedianDistances = medianDists)
    sort!(distData, [:MeanDistances], rev= true)

    return distData
end

"""
`genePerturbExpressionChanges(perturbData, geneNames::Vector{String}, perturbGene::String;
    genesperturbd::Vector{String} = geneNames)`

Function to get a sorted data frame of the predicted effect of a gene perturb on all
other genes.

# Required Arguments
* perturbData - results from perturbEffectPredictions function
* geneNames - vector of gene names in the order of the input expression data
* perturbGene - a gene name to query the predicted perturb effect on expression
# Optional Arguments
* genesPerturbed - If less than all the gene perturbs were performed, the ordered names of the perturb genes must be supplied
"""
function genePerturbExpressionChanges(perturbData, geneNames::Vector{String}, perturbGene::String;
    genesPerturbed::Vector{String} = geneNames)
    #throw an error if the user does not input a gene in the geneList
    if length(findall(x->x==perturbGene, geneNames)) == 0
        error("perturbGene is not found in the list of geneNames")
    elseif length(findall(x->x==perturbGene, genesPerturbed)) == 0
        error("Perturbation of " * perturbGenes * " was not performed with perturbEffectPredictions.")
    end

    if genesPerturbed == geneNames
        geneInd = findall(x->x==perturbGene, geneNames)
    else
        geneInd = findall(x->x==perturbGene, genesPerturbed)
    end

    #get the perturbation data for the gene
    geneData = Matrix{Float32}(undef, length(geneNames), size(perturbData[3])[3])
    for i=1:size(perturbData[3])[3]
        geneData[:,i] = perturbData[3][:,geneInd,i]
    end

    #get mean and median values
    meanChanges = vec(mean(geneData, dims = 2))
    medianChanges = vec(median(geneData, dims = 2))

    geneData = DataFrame(Genes = geneNames, MeanExpressionChange = meanChanges,
     MedianExpressionChange = medianChanges)
    sort!(geneData, [:MeanExpressionChange], rev= true)

    return geneData
end

"""
`geneResponseToPerturb(perturbData, geneNames::Vector{String}, geneOfInterest::String;
    genesPerturbed::Vector{String} = geneNames)`

Function to get a sorted data frame of the predicted effect of all other gene
perturbations on a particular gene of interest.
# Required Arguments
* perturbData - results from perturbEffectPredictions function
* geneNames - vector of gene names in the order of the input expression data
* geneOfInterest - a gene name to query
"""
function geneResponseToPerturb(perturbData, geneNames::Vector{String}, geneOfInterest::String;
    genesPerturbed::Vector{String} = geneNames)
    #throw an error if the user does not input a gene in the geneList
    if length(findall(x->x==geneOfInterest, geneNames)) == 0
        error("geneOfInterest is not found in the list of geneNames")
    end

    geneInd = findall(x->x==geneOfInterest, geneNames)

    #get the perturbation data for the gene
    geneData = Matrix{Float32}(undef, size(perturbData[3])[2], size(perturbData[3])[3])
    for i=1:size(perturbData[3])[3]
        geneData[:,i] = perturbData[3][geneInd,:,i]
    end

    #get mean and median values
    meanChanges = vec(mean(geneData, dims = 2))
    medianChanges = vec(median(geneData, dims = 2))

    geneData = DataFrame(Genes = genesPerturbed, MeanExpressionChange = meanChanges,
     MedianExpressionChange = medianChanges)
    sort!(geneData, [:MeanExpressionChange], rev= true)

    return geneData
end
