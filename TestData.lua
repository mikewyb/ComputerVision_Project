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


-- TODO: Batchsize 256
--local
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
    --LR                  (default 0.1)
    --output              (default 'submission.csv')
    --logDir              (default 'logs')
    --idx                 (default 467)
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
    -- Now we have a valid situation, Extract feature for the current game.
    local x, y, player = sgfloader.parse_move(game:play_current())
    local rank

    -- require 'fb.debugger'.enter()
    local opt = {}
    opt.userank = false
	opt.feature_type = "complete"
		
    local feature = goutils.extract_feature(b, player, opt, rank, info)
    local style = 0

    -- Check if we see any NaN.
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
	print("load game")
    while true do
        local sample = dataset:get(sample_idx)
        for k, v in pairs(sample) do
            sample = v
            break
        end
            -- require 'fb.debugger'.enter()
        local content = sample.table.content
        local filename = sample.table.filename
        print(content:storage():string())
        game = sgfloader.parse(content:storage():string(), filename)
        local max_moves = max_random_moves
        local max_can_move = game:num_round()-1
        if max_moves > max_can_move then
            max_moves = max_can_move
        end
        if game ~= nil and game:has_moves() and game:get_boardsize() == common.board_size and game:play_start() then
            board.clear(b)
            goutils.apply_handicaps(b, game)
			print("------------play-------------")
            local game_play_through = true
            if apply_random_moves then
                local round = math.random(game:num_round()) - 1
				if round < min_random_moves then
					round = min_random_moves
				end
				if round > max_moves then
					round = max_moves
				end								
				print(round)
                
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
    return game, b
end

local function randomPlayAndGetFeature(sample_idx, dataset, info)
    local move_counter = 1
    -- set range for move
    local max_move_counter = 100
    local nstep = 1
    local game_restarted = false
    local b = board.new()
    local game

    game, b = load_random_game(sample_idx, dataset, game, b)
	
    repeat
    if game_restarted or game:play_get_ply() >= game:play_get_maxply() - nstep + 1 then
        game, b = load_random_game(sample_idx, dataset, game, b)
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
            -- player ~= nil: It is a valid move
            -- y1 > 0: it is not a pass/(0, 0) or resign/(1, 0)
            -- Sometime pass = (20, 20) so we need to address this as well.
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


function getTrainSample(train_dataset, idx)
    
    feature, move, xys, ply = randomPlayAndGetFeature(idx, train_dataset, 'train')
    print("----------- feature ----------")
    print(feature)
    print("----------- move ----------")
    print(move)
    print("----------- xys ----------")
    print(xys)
    print("----------- ply ----------")
    print(ply)
	print("----------- idx ----------")
	print(idx)

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
local trainLength = 40000
local testLength = 10000

local i, t = getTrainSample(trainData, opt.idx)


print("The End!")
