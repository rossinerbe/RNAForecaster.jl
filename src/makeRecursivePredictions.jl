#function to make future expression predictions using a trained neural ODE
#outputs a 3d tensor containing a predicted expression counts matrix for the
#cell at each time step
##Required Arguments
# trainedNetwork - the trained neural ODE, from the trainRNAForecaster function
# expressionData - the initial expression states that should be used to make predictions from
# tSteps - how many future time steps should be predicted. (Error will propagate with each prediction so predictions will eventually become highly innaccurate at high numbers of time steps)
##Optional Arguments
# KOGenes - a vector of gene names that will be set to zero in the predictions, simulating a knock out.
# geneNames - a vector of gene names in the order of the rows of the expressionData. Used only when simulating KOs.
# enforceMaxPred - should a maximum allowed prediction be enforced? This is used to represent prior knowledge about what sort of expression values are remotely reasonable predictions.
# maxPrediction - if enforcing a maximum prediction, what should the value be? 2 times the maximum of the input expression data by default.
function predictCellFutures(trainedNetwork, expressionData::Matrix{Float32}, tSteps::Int;
     KOGenes::Vector{String} = Vector{String}(undef,0), geneNames::Vector{String} = Vector{String}(undef,0),
     enforceMaxPred::Bool = true, maxPrediction::Float32 = 2*maximum(expressionData))
    if length(KOGenes) == 0
        inputData = copy(expressionData)
        predictions = Array{Float32}(undef, size(expressionData)[1], size(expressionData)[2], tSteps)
        @suppress begin
            for i=1:tSteps
                for j=1:size(expressionData)[2]
                    predictions[:,j,i] = trainedNetwork(inputData[:,j])[1]
                end
                #set negative predictions to zero
                predictions[findall(x->x < 0, predictions)] .= 0

                if enforceMaxPred
                    predictions[findall(x->x > maxPrediction, predictions)] .= maxPrediction
                end

                inputData = predictions[:,:,i]

            end
        end
    else
        if length(geneNames) != size(expressionData)[1]
            error("Length of gene names is not equal to the number of rows (genes)
            in the input data.")
        end

        inputData = copy(expressionData)
        #set virtual KO genes to zero
        KOGeneInds = findall(in(KOGenes), geneNames)
        inputData[KOGeneInds,:] .= 0
        predictions = Array{Float32}(undef, size(expressionData)[1], size(expressionData)[2], tSteps)
        @suppress begin
            for i=1:tSteps
                for j=1:size(expressionData)[2]
                    predictions[:,j,i] = trainedNetwork(inputData[:,j])[1]
                end
                #set negative predictions to zero
                predictions[findall(x->x < 0, predictions)] .= 0
                #set KO genes to zero
                predictions[KOGeneInds,:,i] .= 0

                if enforceMaxPred
                    predictions[findall(x->x > maxPrediction, predictions)] .= maxPrediction
                end

                inputData = predictions[:,:,i]
            end
        end
    end

    return predictions
end

#functions as follows:
#for each cell, take the predicted expression levels of each gene over time
#find the variance
#get the mean/median for each gene's variance across cells for each gene
#idea is that most time variable genes are those most heavily regulated and thus
#likely to be important in the biological system being studied
#outputs a sorted DataFrame containing gene names and the variances over predicted time
##Required Arguments
# cellFutures - a 3d tensor of gene expression over time; the output from predictCellFutures
# geneNames - a vector of gene names corresponding to the order of the genes in cellFutures
##Optional Arguments
# statType - How to summarize the gene variances. Valid options are "mean" or "median"
function mostTimeVariableGenes(cellFutures::AbstractArray{Float32}, geneNames::Vector{String};
     statType = "mean")
    vars = Array{Float32}(undef, size(cellFutures)[1], size(cellFutures)[2])
    for i=1:size(cellFutures)[1]
        for j=1:size(cellFutures)[2]
            vars[i,j] = var(cellFutures[i,j,:])
        end
    end
     if statType == "mean"
         stats = mean(vars, dims=2)
         #put into data frame with gene names
         geneData = DataFrame(GeneNames = geneNames, MeanVariance = vec(stats))
         #sort
         sort!(geneData, [:MeanVariance], rev= true)
     elseif statType == "median"
         stats = median(vars, dims=2)
         #put into data frame with gene names
         geneData = DataFrame(GeneNames = geneNames, MedianVariance = vec(stats))
         #sort
         sort!(geneData, [:MedianVariance], rev= true)
     else
         error("Not a valid statType. Use 'mean' or 'median'")
     end

     return geneData

end

#test on two cells
#testPreds = predictCellFutures(trainedNetwork, t2Data[:,1:2], 5)
#testVGenes = mostTimeVariableGenes(testPreds, geneNames)
