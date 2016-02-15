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
   assert(dataset.fmaps > 0)
   assert(dataset.inHeight > 0)                            -- check image height
   assert(dataset.inWidth > 0)                              -- check image width
   assert(dataset.classesNo > 0)                      -- check number of classes

   -----------------------------------------------------------------------------
   --- B. Build model
   -----------------------------------------------------------------------------

   local model = nn.Sequential()
   local mapsNo = dataset.fmaps
   local height = dataset.inHeight
   local width = dataset.inWidth

   while height * width > 256 do
      nextMapsNo = math.max(math.min(256, mapsNo * 2), 16)
      model:add(nn.SpatialConvolutionMM(mapsNo, nextMapsNo, 3, 3, 1, 1, 1, 1))
      model:add(nn.ReLU())
      model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

      width  = (width  - 1)- 2 + 3
      height = (height - 1) - 2 + 3

      width  = math.floor((width - 2) / 2 + 1)
      height = math.floor((height - 2) / 2 + 1)

      mapsNo = nextMapsNo
   end

   model:add(nn.Reshape(mapsNo * height * width))
   model:add(nn.Linear(mapsNo * height * width, mapsNo))
   model:add(nn.Tanh())
   model:add(nn.Linear(mapsNo, dataset.classesNo))
   model:add(nn.LogSoftMax())

   if opt.gpuid > 0 then
      model = model:cuda()
      if opt.verbose then print("[convnet1] Convnet model using CUDA"); end
   end

   return model
end

return getModel
