--
-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.
--

local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'
local doc = require 'argcheck.doc'

local pl = require 'pl.import_into'()

local rl = require 'train.rl_framework.infra.env'
require 'train.rl_framework.infra.bundle'

doc[[

### rl.Agent

The Agent module implements the agent behavior.

For opt we have:
   optim:
      supervised:
      q_learning:
      double_q_learning:
      policy_gradient:
      actor_critic:
]]

require 'nn'

local Agent = torch.class('rl.Agent', rl)

-- Methods
-- Pure supervised approach.
-- input.s is the input state and input.a is the next action the agent is supposed to take.
Agent.optim_supervised = function(self, input, test_mode)
    -- minimize ||Q(s, a) - (r + \gamma max_a' Q_fix(s', a'))||^2
    local bundle = self.bundle
    local alpha = self.alpha

    bundle:forward(input.s, { policy=true } )
    local gradOutput, errs = bundle:backward_prepare(input.s, { policy=input.a })

    if not test_mode then
        -- require 'fb.debugger'.enter()
        bundle:backward(input.s, gradOutput)
        -- require 'fb.debugger'.enter()
        bundle:update(alpha)
        -- require 'fb.debugger'.enter()
    end
    return errs
end

--------------- Make them a table ----------------
Agent.optims = {
    supervised = Agent.optim_supervised,
}

Agent.__init = argcheck{
    noordered = true,
   {name="self", type="rl.Agent"},
   {name="bundle", type="rl.Bundle"},
   {name="opt", type='table'},
   call =
      function(self, bundle, opt)
          self.bundle = bundle
          self.opt = pl.tablex.deepcopy(opt)
          -- Learning rate.
          self.alpha = self.opt.alpha
          -- From opt, we could specify the behavior of agent.
          self.optim = rl.func_lookup(self.opt.optim, Agent.optims)
      end
}

Agent.training = argcheck{
   {name="self", type="rl.Agent"},
   call = function(self)
       self.bundle:training()
       self.test_mode = false
   end
}

Agent.evaluate = argcheck{
   {name="self", type="rl.Agent"},
   call = function(self)
       self.bundle:evaluate()
       self.test_mode = true
   end
}

Agent.reduce_lr = argcheck{
   {name="self", type="rl.Agent"},
   {name="ratio", type="number"},
   call = function(self, ratio)
       self.alpha = self.alpha / ratio
   end
}

Agent.get_lr = argcheck{
   {name="self", type="rl.Agent"},
   call = function(self)
       return self.alpha
   end
}

-- Agent behavior.
Agent.optimize = argcheck{
   {name="self", type="rl.Agent"},
   {name="input", type="table"},
   call =
      function(self, input)
          -- we run the optimization method.
          self.bundle:clear_forwarded()
          return self:optim(input, self.test_mode)
      end
}

local function copy_model(dst, dst_str, src, src_str)
    -- Update the parameters.
    -- require 'fb.debugger'.enter()
    local params_dst = dst:parameters()
    local params_src = src:parameters()
    assert(#params_dst == #params_src,
           string.format("#%s [%d] is not equal to #%s [%d]!", dst_str, #params_dst, src_str, #params_src))
    for i = 1, #params_dst do
        local dst_sizes = params_dst[i]:size()
        local src_sizes = params_src[i]:size()
        assert(#dst_sizes == #src_sizes,
               string.format("Parameter tensor order at layer %d: #%s [%d] is not equal to #%s [%d]!", i, dst_str, #dst_sizes, src_str, #src_sizes))
        for j = 1, #dst_sizes do
            assert(dst_sizes[j] == src_sizes[j],
                   string.format("Parameter tensor size at layer %d: %s [%d] is not equal to %s [%d]!", i, dst_str, dst_sizes[j], src_str, src_sizes[j]))
        end

        params_dst[i]:copy(params_src[i])
    end
end

-- Update the model bundle used for dataset sampling.
Agent.update_sampling_model = argcheck{
    {name="self", type="rl.Agent"},
    {name="cb_before", type="function", opt=true, default=nil},
    {name="cb_after", type="function", opt=true, default=nil},
    call = function(self, cb_before, cb_after)
        if cb_before then cb_before() end

        if cb_after then cb_after() end
    end
}

Agent.get_bundle = argcheck{
   {name="self", type="rl.Agent"},
   call =
      function(self)
          return self.bundle
      end
}

