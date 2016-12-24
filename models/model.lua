local nn = require 'nn'
local pl = require('pl.import_into')()
local nnutils = require 'utils.nnutils'
require 'models.ParallelCriterion2'

modelGen = {}

--local optParser = require '../opts'
--local opt = optParser.parse(arg)
--local Tanh = nn.Tanh

function getModel(opt)
    local Convolution, BatchNorm, ReLU, View
    if opt.cuda == true then
        require 'cudnn'
        Convolution = cudnn.SpatialConvolution
        BatchNorm = cudnn.SpatialBatchNormalization
        ReLU = cudnn.ReLU
    else
        Convolution = nn.SpatialConvolutionMM
        BatchNorm = nn.SpatialBatchNormalization
        ReLU = nn.ReLU
    end
    
    View = nn.View
    
    local model = nn.Sequential()
    
    --input: nx19x19, use padding to keep 19x19
    --Conv layer 92 channels 5x5 kernel
    
    model:add(Convolution(12, 92, 5, 5, 1, 1, 2))
    model:add(ReLU(true))
    model:add(BatchNorm(92))
    
    --Conv layer 384 channels 3x3 kernel  x5
    model:add(Convolution(92, 384, 3, 3, 1, 1, 1))
    model:add(ReLU(true))
    model:add(BatchNorm(384))
    
    model:add(Convolution(384, 384, 3, 3, 1, 1, 1))
    model:add(ReLU(true))
    model:add(BatchNorm(384))
    
    model:add(Convolution(384, 384, 3, 3, 1, 1, 1))
    model:add(ReLU(true))
    model:add(BatchNorm(384))
    
    model:add(Convolution(384, 384, 3, 3, 1, 1, 1))
    model:add(ReLU(true))
    model:add(BatchNorm(384))
    
    model:add(Convolution(384, 384, 3, 3, 1, 1, 1))
    model:add(ReLU(true))
    model:add(BatchNorm(384))

    model:add(Convolution(384, 384, 3, 3, 1, 1, 1))
    model:add(ReLU(true))
    model:add(BatchNorm(384))
    
    model:add(Convolution(384, 384, 3, 3, 1, 1, 1))
    model:add(ReLU(true))
    model:add(BatchNorm(384))
    
    model:add(Convolution(384, 384, 3, 3, 1, 1, 1))
    model:add(ReLU(true))
    model:add(BatchNorm(384))
    
    model:add(Convolution(384, 384, 3, 3, 1, 1, 1))
    model:add(ReLU(true))
    model:add(BatchNorm(384))

    model:add(Convolution(384, 384, 3, 3, 1, 1, 1))
    model:add(ReLU(true))
    model:add(BatchNorm(384))


    model:add(Convolution(384, 1, 3, 3, 1, 1, 1))
    
    --output: 1x19x19
    return model
end

function modelGen.GetModel(inputdim, config)
    
    assert(inputdim[3] == 19)
    assert(inputdim[4] == 19)

    local net = getModel(config)

    if config.nGPU>1 then
        require 'nn'
        require 'cudnn'
        require 'cutorch'
        assert(config.nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than config.nGPU specified')
        local net_single = net
        net = nn.DataParallel(1)
        for i=1, config.nGPU do
            cutorch.withDevice(i, function()
                net:add(net_single:clone())
            end)
        end
    end

    local model = nn.Sequential()
    model:add(net):add(nn.View(config.nstep, 19*19):setNumInputDims(3)):add(nn.SplitTable(1, 2))
    local softmax = nn.Sequential()
    -- softmax:add(nn.Reshape(19*19, true))
    softmax:add(nn.LogSoftMax())
    -- )View(-1):setNumInputDims(2))

    local softmaxs = nn.ParallelTable()
    -- Use self-defined parallel criterion 2, which can handle targets of the format nbatch * #target
    local criterions = nn.ParallelCriterion2()
    for k = 1, config.nstep do
        softmaxs:add(softmax:clone()) --wmd
        local w = 1.0 / k
        criterions:add(nn.ClassNLLCriterion(), w)
    end
    model:add(softmaxs) --wmd
    
    --model:add(nn.LogSoftMax())
    --criterions = nn.ClassNLLCriterion()
    return model, criterions

end

return modelGen


