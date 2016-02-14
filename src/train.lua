--------------------------------------------------------------------------------
---  1. Load needed modules
--------------------------------------------------------------------------------

require("torch")                                         -- the almighty tensors
require("nn")                                                   -- neural models
require("optim")                                      -- optimization algorithms
require("gnuplot")                                                 -- plot error

--------------------------------------------------------------------------------
--- 2. the training procedure for the posture
--------------------------------------------------------------------------------

local train = function(dataset, model, criterion, opt)

   -----------------------------------------------------------------------------
   --- A. Get model's parameters and gradients
   -----------------------------------------------------------------------------

   w, dldw = model:getParameters()

   -----------------------------------------------------------------------------
   --- B. Others: confussion matrix, etc.

   confusionMatrix = optim.ConfusionMatrix(dataset.classes)
   trainLosses = {}
   validationLosses = {}


   -----------------------------------------------------------------------------
   --- C. Define closure to compute loss and gradients for new batches
   -----------------------------------------------------------------------------

   local feval = function(wNew)         -- closure to compute loss and gradients
      if w ~= wNew then                                     -- update parameters
         w:copy(wNew)
      end

      x, t = dataset:updateBatch("train")

      dldw:zero()                                             -- erase gradients
      local loss = 0

      loss = criterion:forward(model:forward(x), t)
      model:backward(x, criterion:backward(model.output, t))

      --[[
      for j = 1, dataset.batchSize do
         loss = loss + criterion:forward(model:forward(x[j]), t[j])
         model:backward(x[j], criterion:backward(model.output, t[j]))
      end

      loss = loss / dataset.batchSize
      dldw = dldw:div(dataset.batchSize)

     --]]


      return loss, dldw                            -- return error and gradients
   end

   for epoch = 1, opt.maxEpochs do
      --------------------------------------------------------------------------
      --- Train on all training examples
      --------------------------------------------------------------------------
      local trainLoss = 0
      local time = sys.clock()
      dataset:resetBatch("train")
      repeat
         _, fs = optim.sgd(feval, w, opt)
         trainLoss = trainLoss + fs[1]
      until dataset.epochFinished
      time = (sys.clock() - time) * 1000
      trainLoss = trainLoss * (dataset.batchSize / dataset.trainNo)
      table.insert(trainLosses, trainLoss)

      --------------------------------------------------------------------------
      --- Compute loss on validation set
      --------------------------------------------------------------------------
      local validationLoss = 0
      dataset:resetBatch("valid")
      confusionMatrix:zero()
      repeat
         x, t = dataset:updateBatch("valid")
         prediction = model:forward(x)
         validationLoss = validationLoss +
            criterion:forward(model:forward(x), t)

         for i = 1,dataset.batchSize do
            confusionMatrix:add(prediction[i], t[i])
         end
      until dataset.epochFinished
      validationLoss = validationLoss * (dataset.batchSize / dataset.validNo)
      table.insert(validationLosses, validationLoss)

      --------------------------------------------------------------------------
      --- Report progress
      --------------------------------------------------------------------------
      output = string.format("%2d/%2d: ", epoch, opt.maxEpochs)
      output = output .. string.format("training loss = %.5f", trainLoss)
      output = output .. string.format(", valid. loss = %.5f", validationLoss)
      output = output .. string.format(", epoch time = %.5f ms", time)

      print(output)
      print(confusionMatrix)

      print(torch.Tensor(trainLosses))
      print(torch.Tensor(validationLosses))

      gnuplot.plot(
         {'Training', torch.linspace(1, epoch, epoch) ,
          torch.Tensor(trainLosses),  '-'},
         {'Validation', torch.linspace(1, epoch, epoch),
          torch.Tensor(validationLosses), '-'})
      gnuplot.xlabel('Epoch')
      gnuplot.ylabel('Loss')
      gnuplot.plotflush()

   end

end

return train
