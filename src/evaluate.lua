--------------------------------------------------------------------------------
--- 1. Require needed packages
--------------------------------------------------------------------------------


--------------------------------------------------------------------------
--- Compute loss on test set
--------------------------------------------------------------------------

local evaluate = function(dataset, model, criterion, opt)
   local testLoss = 0
   local confusionMatrix = optim.ConfusionMatrix(dataset.classes)

   dataset:resetBatch("test")
   confusionMatrix:zero()

   repeat
      x, t = dataset:updateBatch("test")
      prediction = model:forward(x)
      testLoss = testLoss + criterion:forward(model:forward(x), t)

      for i = 1,dataset.batchSize do
         confusionMatrix:add(prediction[i], t[i])
      end
   until dataset.epochFinished

   testLoss = testLoss / dataset.testNo

   print("----------------------------------------")
   print("Loss on test dataset: " .. testLoss)
   print(confusionMatrix)
   print("----------------------------------------")
end

return evaluate
