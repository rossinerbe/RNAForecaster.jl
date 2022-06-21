#utility functions for RNAForecaster

#filtering expression count matrices
"""
`filterByZeroProp(t0Counts::Matrix{Float32}, t1Counts::Matrix{Float32},
    zeroProp::Float32)`

Filter by zero proportion for both genes and cells. Very high sparsity prevents
the neural network from achieving a stable solution.

# Required Arguments
* t0Counts - Counts matrix for time 1. Should be genes x cells and Float32.
* t1Counts - Counts matrix for time 2. Should be genes x cells and Float32.

# Keyword Arguments
* zeroProp - proportion of zeroes allowed for a gene or a cell. 0.98f0 by default
"""
function filterByZeroProp(t0Counts::Matrix{Float32}, t1Counts::Matrix{Float32};
    zeroProp::Float32 = 0.98f0)

    zeroPropt0Genes = Array{Float32}(undef, size(t0Counts)[1])
    for i=1:size(t0Counts)[1]
        zeroPropt0Genes[i] = length(findall(x->x == 0, t0Counts[i,:]))/size(t0Counts)[2]
    end
    zeroPropt1Genes = Array{Float32}(undef, size(t1Counts)[1])
    for i=1:size(t1Counts)[1]
        zeroPropt1Genes[i] = length(findall(x->x == 0, t1Counts[i,:]))/size(t1Counts)[2]
    end

    t0Sub = t0Counts[intersect(findall(x->x < zeroProp, zeroPropt0Genes),
     findall(x->x < zeroProp, zeroPropt1Genes)),:]
    t1Sub = t1Counts[intersect(findall(x->x < zeroProp, zeroPropt0Genes),
     findall(x->x < zeroProp, zeroPropt1Genes)),:]

     zeroPropt0Cells = Array{Float32}(undef, size(t0Counts)[2])
     for i=1:size(t0Counts)[2]
         zeroPropt0Cells[i] = length(findall(x->x == 0, t0Counts[:,i]))/size(t0Counts)[1]
     end
     zeroPropt1Cells = Array{Float32}(undef, size(t1Counts)[2])
     for i=1:size(t1Counts)[2]
         zeroPropt1Cells[i] = length(findall(x->x == 0, t1Counts[:,i]))/size(t1Counts)[1]
     end

     t0Sub = t0Sub[:,intersect(findall(x->x < zeroProp, zeroPropt0Cells),
      findall(x->x < zeroProp, zeroPropt1Cells))]
     t1Sub = t1Sub[:,intersect(findall(x->x < zeroProp, zeroPropt0Cells),
      findall(x->x < zeroProp, zeroPropt1Cells))]

     return (t0Sub, t1Sub)
end


"""
`filterByGeneVar(t0Counts::Matrix{Float32}, t1Counts::Matrix{Float32},
    topGenes::Int)`

Filter by gene variance, measured across both count matrices.

# Required Arguments
* t0Counts - Counts matrix for time 1. Should be genes x cells and Float32.
* t1Counts - Counts matrix for time 2. Should be genes x cells and Float32.
* topGenes - the number of top most variable genes to use

# Examples
`mat0 = Float32.(randn(50,50))
mat1 = Float32.(randn(50,50))
hvgFilteredCounts = filterByGeneVar(mat0, mat1, 20)`
"""
function filterByGeneVar(t0Counts::Matrix{Float32}, t1Counts::Matrix{Float32},
    topGenes::Int)

    geneVars = vec(var(t0Counts + t1Counts, dims = 2))
    ordered = sortperm(geneVars)

    t0Counts = t0Counts[ordered[1:topGenes],:]
    t1Counts = t1Counts[ordered[1:topGenes],:]

    return (t0Counts, t1Counts)
end

"""
`saveForecaster(trainedModel, fileName::String)`

Saves the parameters from the neural network after training.

# Required Arguments
* trainedModel - the trained neural network from trainRNAForecaster - just the network
don't include the loss results
* fileName - fileName to save the parameters to. Must end in .jld2
"""
function saveForecaster(trainedModel, fileName::String)
    save_object(fileName, cpu(trainedModel))
end

"""
`loadForecaster(fileName::String, inputNodes::Int, hiddenLayerNodes::Int)`

Recreates a previously saved neural network.

Note: if for some reason you are loading the network and have not first loaded the
DiffEqFlux and DifferentialEquations packages (normally should be loaded when loading
RNAForecaster.jl) then the network will not work, even if you load the required
packages afterwards.

# Required Arguments
* fileName - file name where the parameters are saved
* inputNodes - number of input nodes in the network. Should be the same as the number
of genes in the data the network was trained on
* hiddenLayerNodes - number of hidden layer nodes in the network
"""
function loadForecaster(fileName::String)
    model = load_object(fileName)
    return model
end



function saveEnsembleForecaster(trainedForecaster, fileName::String; gpu::Bool = false)

    if gpu
        trainedNetworksCpu = Vector{Any}(undef, length(trainedForecaster))
        for i=1:length(trainedForecaster)
            trainedNetworksCpu[i] = cpu.(trainedForecaster[i])
        end

        save_object(fileName, trainedNetworksCpu)
    else
        save_object(fileName, trainedForecaster)
    end
end



function loadEnsembleForecaster(fileName::String)

    rnaForecaster = load_object(fileName)
    return rnaForecaster
end
