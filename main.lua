require 'torch'

local optParser = require 'opts'
local opt = optParser.parse(arg)
local tnt = require 'torchnet'

--[[
--from train.lua
local load_closure = function(thread_idx, partition, epoch_size, fm_init, fm_generator, fm_postprocess, bundle, opt)
    local tnt = require 'torchnet'
    local rl = require 'train.rl_framework.infra.env'
    -- It is by default a batchdataset.
    return rl.Dataset{
        forward_model_init = fm_init,
        forward_model_generator = fm_generator,
        forward_model_batch_postprocess = fm_postprocess,
        batchsize = opt.batchsize,
        thread_idx = thread_idx,
        partition = partition,
        bundle = bundle,
        epoch_size = epoch_size,
        opt = opt
    }
end

--from train.lua
local function build_dataset(thread_init, fm_init, fm_gen, fm_postprocess, bundle, partition, epoch_size, opt)
    local dataset
    if opt.nthread > 0 then
        dataset = tnt.ParallelDatasetIterator{
            nthread = opt.nthread,
            init = function()
                require 'cutorch'
                require 'torchnet'
                require 'cudnn'
                require 'train.rl_framework.infra.env'
                require 'train.rl_framework.infra.dataset'
                require 'train.rl_framework.infra.bundle'
                if opt.gpu and opt.nGPU == 1 then
                    cutorch.setDevice(opt.gpu)
                end
                if thread_init ~= nil then thread_init() end
            end,
            closure = function(thread_idx)
                return load_closure(thread_idx, partition, epoch_size, fm_init, fm_gen, fm_postprocess, bundle, opt)
            end
        }
    else
        dataset = tnt.DatasetIterator{
            dataset = load_closure(1, partition, epoch_size, fm_init, fm_gen, fm_postprocess, bundle, opt)
        }
    end
    return dataset
end
--]]

--from infra/framewor.lua
local function load_dataset(partition)
    return tnt.IndexedDataset{
    	fields = { opt.datasource .. "_" .. partition },
    	--path = './dataset'
    	path = opt.path
    }
end

-- training/test set, already a iterator
local train_dataset = load_dataset("train")
local test_dataset = load_dataset("test")

print(train_dataset:get(1))


