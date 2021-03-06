--------------------------------------------------------------------------------
--- 1. Load required libraries
--------------------------------------------------------------------------------

require("torch")

require("activity_dataset")
train = require("train")
evaluate = require("evaluate")

require("image")
locales = {'en_US.UTF-8'}
os.setlocale(locales[1])


--------------------------------------------------------------------------------
--- 2. Parse command line options
--------------------------------------------------------------------------------

cmd = torch.CmdLine()

cmd:text()
cmd:text("Activity recognition!")
cmd:text()
cmd:text("Options:")

--- General : the most important

cmd:option("-dataset", "posture", "the dataset to solve (posture / activity)")
cmd:option("-model", "logistic", "the model to be used for classification")
cmd:option("-seeData", false, "just see the data")
cmd:option("-sleep", 0, "just see the data")
cmd:option("-justTest", false, "just test a previously trained model")

--- Dataset

cmd:option("-noCache", false, "use tensors cached on disk (if they exist)")
cmd:option("-limit", 0, "include only the first n images in the dataset")
cmd:option("-oneOutput", false, "encode targets as single values; def: 1-of-k")
cmd:option("-justNames", false, "do not load images, but only their names")
cmd:option("-testRatio", 0.2, "test set")
cmd:option("-validRatio", 0.1, "validation set")
cmd:option("-shuffle", false, "shuffle all before train / valid / test split")
cmd:option("-allTags", false, "learn other tags, not just the the first one")
cmd:option("-scale", 0.1, "what scale?")

--- Dataset augmentation

cmd:option("-noFlip", false, "horizontal flip")
cmd:option("-vertCropRatio", 0.1, "vertical crop ratio")
cmd:option("-horizCropRatio", 0.2, "horizontal crop ratio")
cmd:option("-vertCrop", 0, "vertical crop ratio")
cmd:option("-horizCrop", 0, "horizontal crop ratio")

--- Info during training

cmd:option("-verbose", false, "display information")
cmd:option("-visualize", false, "display images during training")
cmd:option("-plot", false, "display plot during training")

--- Hardware

cmd:option("-gpuid", -1, "NVIDIA GPU id; -1 for CPU")
cmd:option("-allOnGPU", false, "Put scaled dataset on GPU")

--- Optimization paramters

cmd:option("-batchSize", 10, "batch size")
cmd:option("-notContiguous", false, "do not take contiguous batches")

cmd:option("-algorithm", "sgd", "The optimization algorithm to be used")
cmd:option("-learningRate", 1e-3, "learning rate")
cmd:option("-learningRateDecay", 1e-5, "learning rate decay")
cmd:option("-momentum", .0, "momentum")
cmd:option("-maxEpochs", 100, "number of maximum epochs")

--- State

cmd:option("-seed", 666, "seed for the random number generator")
cmd:option("-saveEvery", 0, "Save paramters every...")
cmd:option("-params", "", "File to load paramters from")

--- Parse

opt = cmd:parse(arg)

opt.useCache = not opt.noCache
opt.flip = not opt.noFlip
opt.contiguous = not opt.notContiguous
opt.oneHot = not opt.oneOutput

if opt.justTest then
   if opt.params == "" then
      print("What parameters do you want to test?")
      os.exit()
   end
end

torch.manualSeed(opt.seed)

--------------------------------------------------------------------------------
--- 3A. Inspect dataset
--------------------------------------------------------------------------------

dataset = ActivityDataset(opt)

if opt.seeData then
   -- require("image")
   dataset:resetBatch("train")
   repeat
      local X, T = dataset:updateBatch("train")
      local band = X[1]:clone()
      for i=2,dataset.batchSize do
         band = torch.cat(band, X[i])
      end
      win = image.display{image=band, win=win, zoom=5, legend='batch'}
      sys.sleep(tonumber(opt.sleep))
   until dataset.epochFinished
   os.exit()
end -- if opt.seeData

--------------------------------------------------------------------------------
--- 3B. Train the model
--------------------------------------------------------------------------------

local getModel, model, criterion

getModel = require("models/" .. opt.model)

model = getModel(dataset, opt)
if opt.oneHot then
   criterion = nn.MSECriterion()           -- TODO
else
   criterion = nn.ClassNLLCriterion()
end

if opt.gpuid > 0 then
   criterion = criterion:cuda()
   model = model:cuda()
end

print(model)

local ms = sys.clock()

train(dataset, model, criterion, opt)
evaluate(dataset, model, criterion, opt)

ms = (sys.clock() - ms) * 1000

--------------------------------------------------------------------------------
--- 4. Done!
--------------------------------------------------------------------------------

print("Done in " .. ms .. " ms!")

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
