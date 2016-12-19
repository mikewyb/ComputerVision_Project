require 'torch'

local optParser = require 'opts'
local opt = optParser.parse(arg)
local tnt = require 'torchnet'
local sgfloader = require 'utils.sgf'
local board = require 'board.board'
local common = require 'common.common'
local goutils = require 'utils.goutils'
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

local function protected_play(b, game)
    local x, y, player = sgfloader.parse_move(game:play_current(), false)
	
    if player ~= nil and board.play(b, x, y, player) then
        game:play_next()
		print("---------------------------------------------------------")
		print(player)
		print(board.show(b, "all"))				
		local opt = {}
		opt.userank = false
		opt.feature_type = "complete"
		local feature = goutils.extract_feature(b, player, opt, rank, 'test');
		print(feature:size())
        return true
    else
        return false
    end
end

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
local sample = train_dataset:get(1)
print(sample.table)

sample_idx = math.random(train_dataset:size())
local sample = train_dataset:get(sample_idx)
for k, v in pairs(sample) do
    sample = v
    break
end

local content = sample.table.content
local filename = sample.table.filename

b = board.new()

game = sgfloader.parse(content:storage():string(), filename)
print(content:storage():string())
print(sgfloader.show_move(game))
print(game:num_round())

--[[
local nstep = 3
for i = 1, nstep do
	local x1, y1, player = sgfloader.parse_move(game:play_current(i - 1), false)
	print(x1)
	print(y1)
	print(player)
end
--]]

if game ~= nil and game:has_moves() and game:get_boardsize() == common.board_size and game:play_start() then
    board.clear(b)
    goutils.apply_handicaps(b, game)

    local game_play_through = true
    local round = math.random(game:num_round()) - 1
    for j = 1, round do
		--print(game)
		if not protected_play(b, game) then
       		break
      	end
  	end
 end

