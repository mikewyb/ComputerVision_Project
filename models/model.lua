
local nn = require 'nn'

local optParser = require '../opts'
local opt = optParser.parse(arg)
local Convolution, BatchNorm, ReLU, View
--local Tanh = nn.Tanh
--[[
if opt.cuda == true then
    require 'cudnn'
    Convolution = cudnn.SpatialConvolution
    BatchNorm = cudnn.SpatialBatchNormalization
    ReLU = cudnn.ReLU
else
--]]
    Convolution = nn.SpatialConvolution
    BatchNorm = nn.SpatialBatchNormalization
    ReLU = nn.ReLU
--end

View = nn.View

local model  = nn.Sequential()

--input: nx19x19, use padding to keep 19x19
--Conv layer 92 channels 5x5 kernel
model:add(Convolution(92, 12, 5, 5, 1, 1, 2))
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

model:add(Convolution(384, 1, 3, 3, 1, 1, 1))

--output: 1x19x19

model:add(View(19*19))

return model
