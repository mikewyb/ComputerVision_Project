require 'torch'

local optParser = require 'opts'
local opt = optParser.parse(arg)
local tnt = require 'torchnet'
local sgfloader = require 'utils.sgf'
local board = require 'board.board'
local common = require 'common.common'
local goutils = require 'utils.goutils'

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
		print(feature[1])
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

--Get the distribution of the number of round for each game
local function print_numOfRound(dataset)
    local countRound = torch.Tensor(8):zero()
    for i = 1, dataset:size() do
        sample = dataset:get(i)
        for k, v in pairs(sample) do
            sample = v
            break
        end
        content = sample.table.content
    	filename = sample.table.filename
    	game = sgfloader.parse(content:storage():string(), filename)
    	round = game:num_round()
    	if round == 0 then
            countRound[1] = countRound[1] + 1
    	elseif round <= 50 then
	    countRound[2] = countRound[2] + 1
        elseif round <= 100 then
	    countRound[3] = countRound[3]+ 1
    	elseif round <= 150 then
	    countRound[4] = countRound[4] + 1
    	elseif round <= 200 then
	    countRound[5] = countRound[5] + 1
    	elseif round <= 250 then
	    countRound[6] = countRound[6] + 1
    	elseif round <= 300 then
	    countRound[7] = countRound[7] + 1
    	else
	    countRound[8] = countRound[8] + 1
        end
    end
    for i = 1, 8 do
        print(countRound[i])
    end
end


--Get the distribution of the number of round for each game
local function print_numOfRank(dataset)
    local countRank = torch.Tensor(4):zero()
    for i = 1, dataset:size() do
    	sample = dataset:get(i)
    	for k, v in pairs(sample) do
            sample = v
            break
    	end
    	content = sample.table.content
    	filename = sample.table.filename
    	game = sgfloader.parse(content:storage():string(), filename)
    	br, wr = game:get_ranks('kgs')
    	if wr == '1d' or wr == '2d' or wr == '3d' or wr == '4d' then
	    countRank[1] = countRank[1] + 1
    	elseif wr == '5d' or wr == '6d' or wr == '7d' then
	    countRank[2] = countRank[2] + 1
    	elseif wr == '8d' or wr == '9d' then
	    countRank[3] = countRank[3] + 1
    	else
	    countRank[4] = countRank[4] + 1
        end
    end
    for i = 1, 4 do
    	print(countRank[i])
    end
end

local function print_feature(dataset)
    sample_idx = math.random(dataset:size())
    local sample = dataset:get(sample_idx)
    for k, v in pairs(sample) do
    	sample = v
    	break
    end

    local content = sample.table.content
    local filename = sample.table.filename

    b = board.new()

    game = sgfloader.parse(content:storage():string(), filename)
    --print(content:storage():string())
    --print(sgfloader.show_move(game))
    --local br, wr = game:get_ranks('kgs')
    --print(br)
    --print(wr)

    --print(game:num_round())

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
end

-- training/test set
local train_dataset = load_dataset("train")
local test_dataset = load_dataset("test")

print("Training data round distribution:")
print_numOfRound(train_dataset)
print("Testing data round distribution:")
print_numOfRound(test_dataset)
print("Training data rank distribution:")
print_numOfRound(train_dataset)

print("Print feature:")
print_feature(train_dataset)

--print(train_dataset:size())
--print(test_dataset:size())

--print(train_dataset:get(1))
--local sample = train_dataset:get(2)
--print(sample.table)


