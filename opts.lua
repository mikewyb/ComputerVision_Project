local M = {}

function M.parse(arg)
    local cmd = torch.CmdLine();
    cmd:text()
    cmd:text('Predicting the next move of playing Go: A multi-class classification ')
    cmd:text()
    cmd:text('Options:')
    cmd:option('-path',             './dataset',    'Path to dataset')
    cmd:option('-datasource',       'kgs',          'Data source')
    --cmd:option('-val',              10,             'Percentage to use for validation set')
    --cmd:option('-nEpochs',          256,            'Maximum epochs')
    cmd:option('-batchsize',        128,            'Batch size for epochs')
    cmd:option('-nThreads',         1,              'Number of dataloading threads')
    --cmd:option('-manualSeed',       '0',            'Manual seed for RNG')
    cmd:option('-LR',               0.1,            'initial learning rate')
    --cmd:option('-momentum',         0.9,            'momentum')
    --cmd:option('-weightDecay',      1e-4,           'weight decay')
    --cmd:option('-logDir',           'logs',         'log directory')
    cmd:option('-model',            '',             'Model to use for training')
    --cmd:option('-verbose',          'false',        'Print stats for every batch')
    --[[
    -- Hint: Use this option to convert your code to use GPUs
    --]]
    cmd:option('-cuda',             'false',             'Use cuda tensor')

    local opt = cmd:parse(arg or {})

    --if opt.model == '' or not paths.filep('models/'..opt.model..'.lua') then
    --    cmd:error('Invalid model ' .. opt.model)
    --end

    if opt.nThreads == nil then
	cmd:error("opt.nthread cannot be nil")
    end

    if not opt.batchsize then
	cmd:error("opt.batchsize cannot be nil")
    end

    return opt
end

return M
