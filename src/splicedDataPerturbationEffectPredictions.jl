#based on spliced/unspliced counts predict the immediate transcriptomic effect
#of each single gene KO
#output tuple containing the cells used for prediction, the expression predictions,
# the gene wise differences, and the cell-wise euclidean distances for each "KO"
##Required Arguments
# trainedNetwork - trained neuralODE from trainRNAForecaster
# splicedData - log normalized spliced counts matrix. Must be in Float32 format
# nCells - how many cells from the data should be used for prediction of KO effect. Higher values will increase computational time required.
##Optional Arguments
# KOGenes - list of genes to simulate a KO of. By default all genes are used
# geneNames - if providing a subset of the genes to KO, a vector of gene names to match against, in the order of splicedData
# seed - Random seed for reproducibility on the cells chosen for prediction

function KOeffectPredictions(trainedNetwork, splicedData::Matrix{Float32}, nCells::Int;
     KOGenes::Vector{String} = Vector{String}(undef, 0), geneNames::Vector{String} = Vector{String}(undef,0),
     seed::Int=123)

    Random.seed!(seed)
    cellsToUse = sample(1:size(splicedData)[2], nCells, replace = false)
    splicedSub = splicedData[:, cellsToUse]

    if length(KOGenes) == 0
        #make prediction for initial conditions
        #matrix to store data in
        initialPredictions = Matrix{Float32}(undef, size(splicedData)[1], nCells)
        @suppress begin
            for n=1:nCells
                initialPredictions[:,n] = trainedNetwork(splicedSub[:,n])[1]
            end
        end

        #set negative predictions to zero
        initialPredictions[findall(x->x < 0, initialPredictions)] .= 0

        #create tensor to store perturbation output in: nGenes X perturbations(nGenes) x nCells
        perturbPredictions = Array{Float32}(undef, size(splicedData)[1], size(splicedData)[1], nCells)
        #make predictions
        @suppress begin
            for n=1:nCells
                for i=1:size(splicedData)[1]
                    tmpCell = splicedSub[:,n]
                    tmpCell[i] = 0
                    perturbPredictions[:,i,n] = trainedNetwork(tmpCell)[1]
                end
            end
        end
    else
        if length(geneNames) != size(splicedSub)[1]
            error("Length of gene names is not equal to the number of rows (genes)
            in the input data.")
        end
        KOGeneInds = findall(in(KOGenes), geneNames)
        #make prediction for initial conditions
        #matrix to store data in
        initialPredictions = Matrix{Float32}(undef, size(splicedData)[1], nCells)
        @suppress begin
            for n=1:nCells
                initialPredictions[:,n] = trainedNetwork(splicedSub[:,n])[1]
            end
        end

        #set negative predictions to zero
        initialPredictions[findall(x->x < 0, initialPredictions)] .= 0

        #create tensor to store perturbation output in: nGenes X perturbations x nCells
        perturbPredictions = Array{Float32}(undef, size(splicedData)[1], length(KOGeneInds), nCells)
        #make predictions
        @suppress begin
            for n=1:nCells
                for i= 1:length(KOGeneInds)
                    tmpCell = splicedSub[:,n]
                    tmpCell[KOGeneInds[i]] = 0
                    perturbPredictions[:,i,n] = trainedNetwork(tmpCell)[1]
                end
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



#function to yield a sorted data frame of the size of the predicted effect of a perturbation
#on the cellular transcriptome. Intended to serve as a measure of more or less impactful
#gene KOs
##Required Arguments
# KOData - results from KOeffectPredictions function
# geneNames - vector of gene names in the order of the input expression data. IMPORTANT: should only include KO genes
function totalKOImpact(KOData, geneNames::Vector{String})
    #find genes of largest effect on cell transcriptomic state
    #calculate mean and median distnaces for each gene across simulated cells
    meanDists = vec(mean(KOData[4], dims = 2))
    medianDists = vec(median(KOData[4], dims = 2))

    distData = DataFrame(Genes = geneNames, MeanDistances = meanDists,
     MedianDistances = medianDists)
    sort!(distData, [:MeanDistances], rev= true)

    return distData
end

#function to get a sorted data frame of the predicted effect of a gene KO on all
#other genes
##Required Arguments
# KOData - results from KOeffectPredictions function
# geneNames - vector of gene names in the order of the input expression data
# KOGene - a gene name to query the predicted KO effect on expression
##Optional Arguments
# genesKOd - If less than all the gene KOs were performed, the ordered names of the KO genes must be supplied
function geneKOExpressionChanges(KOData, geneNames::Vector{String}, KOGene::String;
    genesKOd::Vector{String} = geneNames)
    #throw an error if the user does not input a gene in the geneList
    if length(findall(x->x==KOGene, geneNames)) == 0
        error("KOGene is not found in the list of geneNames")
    elseif length(findall(x->x==KOGene, genesKOd)) == 0
        error("KO of " * KOGenes * " was not performed with KOeffectPredictions.")
    end

    if genesKOd == geneNames
        geneInd = findall(x->x==KOGene, geneNames)
    else
        geneInd = findall(x->x==KOGene, genesKOd)
    end

    #get the perturbation data for the gene
    geneData = Matrix{Float32}(undef, length(geneNames), size(KOData[3])[3])
    for i=1:size(KOData[3])[3]
        geneData[:,i] = KOData[3][:,geneInd,i]
    end

    #get mean and median values
    meanChanges = vec(mean(geneData, dims = 2))
    medianChanges = vec(median(geneData, dims = 2))

    geneData = DataFrame(Genes = geneNames, MeanExpressionChange = meanChanges,
     MedianExpressionChange = medianChanges)
    sort!(geneData, [:MeanExpressionChange], rev= true)

    return geneData
end

##function to get a sorted data frame of the predicted effect of all other gene
#KOs on a particular gene of interest
##Required Arguments
# KOData - results from KOeffectPredictions function
# geneNames - vector of gene names in the order of the input expression data
# geneOfInterest - a gene name to query
function geneResponseToKOs(KOData, geneNames::Vector{String}, geneOfInterest::String;
    genesKOd::Vector{String} = geneNames)
    #throw an error if the user does not input a gene in the geneList
    if length(findall(x->x==geneOfInterest, geneNames)) == 0
        error("geneOfInterest is not found in the list of geneNames")
    end

    geneInd = findall(x->x==geneOfInterest, geneNames)

    #get the perturbation data for the gene
    geneData = Matrix{Float32}(undef, size(KOData[3])[2], size(KOData[3])[3])
    for i=1:size(KOData[3])[3]
        geneData[:,i] = KOData[3][geneInd,:,i]
    end

    #get mean and median values
    meanChanges = vec(mean(geneData, dims = 2))
    medianChanges = vec(median(geneData, dims = 2))

    geneData = DataFrame(Genes = genesKOd, MeanExpressionChange = meanChanges,
     MedianExpressionChange = medianChanges)
    sort!(geneData, [:MeanExpressionChange], rev= true)

    return geneData
end
