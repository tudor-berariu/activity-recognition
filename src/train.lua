--------------------------------------------------------------------------------
---  1. Load needed modules
--------------------------------------------------------------------------------

require("torch")                                         -- the almighty tensors
require("nn")                                                   -- neural models
require("optim")                                      -- optimization algorithms

--------------------------------------------------------------------------------
--- 2. the training procedure for the posture
--------------------------------------------------------------------------------

local train = function(dataset, model, criterion, opt)

   -----------------------------------------------------------------------------
   --- A. Get model's parameters and gradients
   -----------------------------------------------------------------------------

   w, dldw = model:getParameters()

   -----------------------------------------------------------------------------
   --- B. Define closure to compute loss and gradients for new batches
   -----------------------------------------------------------------------------

   local feval = function(wNew)         -- closure to compute loss and gradients
      if w ~= wNew then                                     -- update parameters
         w:copy(wNew)
      end

      x, t = dataset:updateBatch("train")

      dldw:zero()                                             -- erase gradients
      local loss = 0

      for i = 1,dataset.batchSize do
         loss = loss + criterion:forward(model:forward(x[i]), t[i])
         model:backward(x[i], criterion:backward(model.output, t[i]))
      end

      loss = loss / dataset.batchSize
      dldw = dldw:div(dataset.batchSize)

      return loss, dldw                            -- return error and gradients
   end

   for i = 1, opt.maxEpochs do
      local loss = 0
      dataset:resetBatch("train")
      repeat
         _, fs = optim.sgd(feval, w, opt)
         loss = loss + fs[1]
      until dataset.epochFinished
      print("loss = " .. loss)
   end

end

return train
