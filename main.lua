require 'torch'
require 'optim'
require 'os'
require 'xlua'
require 'torch'
require 'nn'
require 'nngraph'
require 'xlua'

-- require 'cunn'
-- require 'cudnn' -- faster convolutions

local tnt = require 'torchnet'
local image = require 'image'
local optParser = require 'opts'
local sgfloader = require 'utils.sgf'
local board = require 'board.board'
local common = require 'common.common'
local goutils = require 'utils.goutils'

local pl = require 'pl.import_into'()

local opt = pl.lapp[[
    --alpha          (default 0.1)
    --nthread        (default 0)
    --batchsize      (default 128)
    --progress                                 Whether to print the progress
    --nEpochs             (default 20)      Epoch size
    --nGPU                (default 1)          Number of GPUs to use.
    --nstep               (default 1)          Number of steps.
    --model_name          (default 'model-12-parallel-384-n-output-bn')
    --datasource          (default 'kgs')
    --feature_type        (default 'extended')
    --cuda                (default 'true')
    --path                (default './dataset')
    --min_move            (default 30)
    --max_move            (default 50)
    --momentum            (default 0.9)
    --verbose             (default 'true')
    --output              (default 'submission.csv')
    --logDir              (default 'logs')
]]

print(pl.pretty.write(opt))

local apply_random_moves = true
local min_random_moves = opt.min_move
local max_random_moves = opt.max_move

local function protected_play(b, game)
    local x, y, player = sgfloader.parse_move(game:play_current(), false)
    if player ~= nil and board.play(b, x, y, player) then
        game:play_next()
        return true
    else
        return false
    end
end

-- info: 'train' or 'test'
local function get_sa(b, game, sample_idx, info, nstep)
    local x, y, player = sgfloader.parse_move(game:play_current())
    local rank
    
    local opt = {}
    opt.userank = false
	opt.feature_type = "complete"
		
    local feature = goutils.extract_feature(b, player, opt, rank, info)
    local style = 0
    
    if feature:ne(feature):sum() > 0 or move ~= move then
        print(feature)
        print(move)
    end

    local sample
    --local nstep = 1
    local moves = torch.LongTensor(nstep)
    local xys = torch.LongTensor(nstep, 2)
    for i = 1, nstep do
        local x, y, player = sgfloader.parse_move(game:play_current(i - 1))
        local x_rot, y_rot = goutils.rotateMove(x, y, style)
        moves[i] = goutils.xy2moveIdx(x_rot, y_rot)
        if moves[i] < 1 or moves[i] > 361 then
            board.show(b, 'last_move')
            print("Original loc")
            print(x)
            print(y)

            print("rotated")
            print(x_rot)
            print(y_rot)
            print(player)
            print(moves[i])
            error("Move invalid!")
        end

        xys[i][1] = x_rot
        xys[i][2] = y_rot
    end

    return feature, moves, xys, game.ply
end

local function load_random_game(sample_idx, dataset, game, b)
	--print("load game")
    local moveforward = 2
    while true do
        local sample = dataset:get(sample_idx)
        for k, v in pairs(sample) do
            sample = v
            break
        end
        
        local content = sample.table.content
        local filename = sample.table.filename
        game = sgfloader.parse(content:storage():string(), filename)
	
        local max_moves = max_random_moves
        local max_can_move = game:num_round()-moveforward
        if max_can_move < 0 then
		    max_can_move = 0
	    end
	    if max_moves > max_can_move and max_can_move >= 0 then
            max_moves = max_can_move
        end
        if min_random_moves > max_can_move and max_can_move >= 0 then
            min_random_moves = max_can_move
        end
        moveforward = moveforward + 1
        
        if max_can_move == 0 and moveforward == 3 then
            return game, b, false

	end

        if game ~= nil and game:has_moves() and game:get_boardsize() == common.board_size and game:play_start() then
            board.clear(b)
            goutils.apply_handicaps(b, game)
			--print("------------play-------------")
            local game_play_through = true
            if apply_random_moves then
                local round = math.random(game:num_round()) - 1
				if round < min_random_moves then
					round = min_random_moves
				end
				if round > max_moves then
					round = max_moves
				end								
				--print(round)
                
                for j = 1, round do
                    if not protected_play(b, game) then
                        game_play_through = false
                        break
                    end
                end
            end
            if game_play_through then 
                break 
            end
        end
    end
	--print(board.show(b, "all"))				
	--print(game)
    return game, b, true
end

local function randomPlayAndGetFeature(sample_idx, dataset, info)
    local move_counter = 1
    -- set range for move
    local max_move_counter = 100
    local nstep = 1
    local game_restarted = false
    local b = board.new()
    local game

    game, b, ok = load_random_game(sample_idx, dataset, game, b)
	if ok == false then
        local retmove = torch.LongTensor(1)
        retmove[1] = 0
        -- TODO change layer here
        return torch.DoubleTensor(12,19,19), retmove, torch.LongTensor(1, 2), 0
    end

    repeat
    if game_restarted or game:play_get_ply() >= game:play_get_maxply() - nstep + 1 then
        game, b, ok = load_random_game(sample_idx, dataset, game, b)

        game_restarted = false
    else
        if not protected_play(b, game) then
            game_restarted = true
        else
            move_counter = move_counter + 1
            if move_counter >= max_move_counter then
                game_restarted = true
            end
        end
    end
    -- Check whether it is a valid situation.
    if not game_restarted then
        for i = 1, nstep do
            local x1, y1, player = sgfloader.parse_move(game:play_current(i - 1), false)
            if player == nil or y1 == 0 or y1 == common.board_size + 1 then
                game_restarted = true
                break
            end
        end
    end
    until not game_restarted

	--print(board.show(b, "all"))				
	--print(game:show_info())

    return get_sa(b, game, sample_idx, info, nstep)
end

-- Build simple models.
function build_policy_model(opt)
    --local network_maker = require('models.' .. opt.model_name)
    --local network, crit, outputdim, monitor_list = network_maker({1, 12, 19, 19}, opt) -- change from 25
    
    local network_maker = require('models.model')
    local network, crit = network_maker.GetModel({1, 12, 19, 19}, opt) 

    print(network)
    --crit = nn.ClassNLLCriterion()
    
    --TODO replace network with another model
    --local network = require("./models/model") 
    --print(network)
    
    if opt.nGPU > 1 then
        require 'cutorch'
        require 'cunn'
        require 'cudnn'
        return network:cuda(), crit:cuda()
    end
    return network, crit
end


function getTrainSample(train_dataset, idx)
    --print("in getTrainSample")
 
    feature, move, xys, ply = randomPlayAndGetFeature(idx, train_dataset, 'train')
    --print("----------- feature ----------")
    --print(feature)
    --print("----------- move ----------")
    --print(move)
    --print("----------- xys ----------")
    --print(xys)
    --print("----------- ply ----------")
    --print(ply)
	--print("----------- idx ----------")
	--print(idx)

    --TODO fix bugs here
    --return torch.FloatTensor(12,19,19):double(), move--torch.LongTensor(nstep)
    return feature:double(), move
end

function getTrainTraget(dataset, idx)
    --print("in getTrainTraget")
end

function getTestSample(test_dataset, idx)
    ---print("in getTestSample")
    feature, move, xys, ply = randomPlayAndGetFeature(idx, test_dataset, 'test')
    --print("----------- feature ----------")
    --print(feature)
    --print("----------- move ----------")
    --print(move)
    --print("----------- xys ----------")
    --print(xys)
    --print("----------- ply ----------")
    --print(ply)
	--print("----------- idx ----------")
	--print(idx)
    --TODO fix bugs here
    --return torch.FloatTensor(12, 19, 19):double(), move-- torch.LongTensor(nstep)
	return feature:double(), move, ply, idx
end

function getTestLabel(dataset, idx)
    --print("in getTestLabel")
end


function getIterator(dataset)
    return tnt.DatasetIterator{
        dataset = tnt.BatchDataset{
            batchsize = opt.batchsize,
            dataset = dataset
        }
    }
end

local function load_dataset(partition)
    return tnt.IndexedDataset{
    	fields = { opt.datasource .. "_" .. partition },
    	--path = './dataset'
    	path = opt.path
    }
end

local trainData = load_dataset("train")
local testData = load_dataset("test")

--TODO change size
local trainLength = 40000 ---144748
local testLength = 1000--26814

trainDataset = tnt.SplitDataset{
    partitions = {train=0.7, val=0.3},
    initialpartition = 'train',
    dataset = tnt.ListDataset{
        list = torch.range(1, trainLength):long(),
        load = function(idx)
            local i, t = getTrainSample(trainData, idx)
            return {
                input = i,
                target = t,
            }
        end
    }
}


testDataset = tnt.ListDataset{
    list = torch.range(1, testLength):long(),
    load = function(idx)
        local i, s, sid, mid = getTestSample(testData, idx)
        return {
            input = i, 
            sfgid = sid,
            moveid = mid,
            index = idx
        }
    end
}

local model, criterion = build_policy_model(opt)
print("------------------ model -----------------")
print(model)

local function compute_aver_loss(state)
    local aver_train_errs = { }
    local err_str = ""
    for k, e in pairs(state.errs) do
        aver_train_errs[k] = e:sum(1) / state.errs_count / e:size(1)
        err_str = err_str .. string.format("[%s]: %5.6f ", k, aver_train_errs[k][1])
    end
    return aver_train_errs, err_str
end

local engine = tnt.OptimEngine()
local meter = tnt.AverageValueMeter()
--local criterion = nn.CrossEntropyCriterion()
local clerr = tnt.ClassErrorMeter{topk = {1}}
local timer = tnt.TimeMeter()
local batch = 1
local logName = opt.output or "submission"
local convergeLog = assert(io.open("outputs/".. logName .. "_cvgLog.logs", "w"))
local lr_acc = 0

engine.hooks.onStart = function(state)
    --print("In onStart\n")
    meter:reset()
    clerr:reset()
    timer:reset()
    batch = 1
    if state.training then
        mode = 'Train'
    else
        mode = 'Val'
    end
end

if opt.cuda == 'true' then
    require 'cunn'
    require 'cudnn'

    model       = model:cuda()
    criterion   = criterion:cuda()
    local igpu  = torch.CudaTensor()
    local tgpu = torch.CudaTensor()
    engine.hooks.onSample = function(state)
        igpu:resize(state.sample.input:size()):copy(state.sample.input)
        tgpu:resize(state.sample.target:size()):copy(state.sample.target)
        state.sample.input  = igpu
        state.sample.target = tgpu
    end
end

engine.hooks.onForwardCriterion = function(state)
    --print("In onForwardCriterion")
    meter:add(state.criterion.output)
	--local train_aver_loss, train_err_str = compute_aver_loss(state)
    --clerr:add(state.network.output, state.sample.target)
    if opt.verbose == true then
        --local t_str = os.date("%c", os.time())
        --print(string.format("| %s | %s Batch: %d/%d; avg. loss: %2.4f; avg_loss: %2.4f, tain_err: %s",
        --t_str, mode, batch, state.iterator.dataset:size(), meter:value(), train_aver_loss, train_err_str))
        --io.flush()
    else
        xlua.progress(batch, state.iterator.dataset:size())
    end
    batch = batch + 1 -- batch increment has to happen here to work for train, val and test.
    timer:incUnit()
end

engine.hooks.onUpdate = function(state)
    if state.epoch >= 10 then
        lr_acc = lr_acc + 1
        if lr_acc > 20 then
            opt.alpha = opt.alpha / 1.2
            lr_acc = 0
        end
    end
	state.network:updateParameters(opt.alpha)
end

engine.hooks.onEnd = function(state)
    print("In engin onEnd\n")
    --print(string.format("%s: avg. loss: %2.4f; avg. error: %2.4f, time: %2.4f",
    --mode, meter:value(), clerr:value{k = 1}, timer:value()))
    --convergeLog:write(string.format("%s: avg. loss: %2.4f; avg. error: %2.4f, time: %2.4f\n",
    --mode, meter:value(), clerr:value{k = 1}, timer:value()))
    print(string.format("%s: avg. loss: %2.4f; time: %2.4f",mode, meter:value(), timer:value()))
    convergeLog:write(string.format("%s: avg. loss: %2.4f; time: %2.4f\n",mode, meter:value(), timer:value()))
end

local epoch = 1

while epoch <= opt.nEpochs do
    --print("epoch:")
    --print(epoch)

    --print("train")
    trainDataset:select('train')
    engine:train{
        network = model,
        criterion = criterion,
        --TODO
        iterator = getIterator(trainDataset),
        optimMethod = optim.sgd,
        maxepoch = 1,
        config = {
            learningRate = opt.alpha,
            momentum = opt.momentum
        }
    }

    --print("validate")
    -- trainDataset = validdataset
    --TODO
    trainDataset:select('val')
    engine:test{
        network = model,
        criterion = criterion,
        iterator = getIterator(trainDataset)
    }
    print('Done with Epoch '..tostring(epoch))
    epoch = epoch + 1
end

local subName = opt.output or "submission"
local submission = assert(io.open(opt.logDir .. "/".. subName .. ".csv", "w"))
submission:write("Index, SGFId, MoveId, Move\n")
batch = 1

--[[
--  This piece of code creates the submission
--  file that has to be uploaded in kaggle.
--]]
engine.hooks.onForward = function(state)
    --print("In engin onforward")
    local index = state.sample.index
    local sfg  = state.sample.sfgid
    local move = state.sample.mid
    local _, pred = state.network.output:max(2)
    pred = pred - 1
    for i = 1, pred:size(1) do
        -- index, sfgid, moveid, move ((x-1)*19+y)
        submission:write(string.format("%d, %d, %d, %d\n", index, sfg, move, pred[i][1]))
    end
    xlua.progress(batch, state.iterator.dataset:size())
    batch = batch + 1
end

if opt.cuda == 'true' then
    require 'cunn'
    require 'cudnn'

    model       = model:cuda()
    criterion   = criterion:cuda()
    local igpu  = torch.CudaTensor()
    -- TO CHeck
    engine.hooks.onSample = function(state)
        igpu:resize(state.sample.input:size()):copy(state.sample.input)
        state.sample.input  = igpu
    end
end

engine.hooks.onEnd = function(state)
    --print("In engin onEnd")
    submission:close()
    convergeLog:close()
end

engine:test{
    network = model,
    iterator = getIterator(testDataset)
}

print("The End!")
