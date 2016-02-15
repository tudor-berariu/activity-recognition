--------------------------------------------------------------------------------
--- 1. Load needed modules
--------------------------------------------------------------------------------

require('nn')

--------------------------------------------------------------------------------
--- 2. The function that creates the model
--------------------------------------------------------------------------------

local function getModel(dataset, opt)

   if opt.gpuid > 0 then
      require("cutorch")
      require("cunn")
   end

   -----------------------------------------------------------------------------
   --- A. Check size
   -----------------------------------------------------------------------------

   assert(dataset.batchSize > 0)                             -- check batch size
   assert(dataset.inHeight > 0)                            -- check image height
   assert(dataset.inWidth > 0)                              -- check image width
   assert(dataset.classesNo > 0)                      -- check number of classes

   -----------------------------------------------------------------------------
   --- B. Build model
   -----------------------------------------------------------------------------

   local model = nn.Sequential()
   model:add(nn.Reshape(dataset.inHeight * dataset.inWidth))
   model:add(nn.Linear(dataset.inHeight * dataset.inWidth, dataset.classesNo))
   model:add(nn.LogSoftMax())

   if opt.gpuid > 0 then
      model = model:cuda()
   end

   return model
end

return getModel
