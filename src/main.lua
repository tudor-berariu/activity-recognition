--------------------------------------------------------------------------------
--- 1. Load required libraries

require("torch")

require("activity_dataset")
--train = require("train")

--------------------------------------------------------------------------------
--- 2. Parse command line options

cmd = torch.CmdLine()
cmd:text()
cmd:text("Activity recognition!")
cmd:text()
cmd:text("Options:")

--- General : the most important

cmd:option("-dataset", "posture", "the dataset to solve (posture / activity)")
cmd:option("-model", "logistic", "the model to be used for classification")
cmd:option("-see_data", false, "just see the data")

--- Dataset

cmd:option("-no_cache", false, "use tensors cached on disk (if they exist)")
cmd:option("-limit", 0, "include only the first n images in the dataset")
cmd:option("-one_output", false, "encode targets as single values; def: 1-of-k")
cmd:option("-just_names", false, "do not load images, but only their names")
cmd:option("-test_ratio", 0.2, "test set")
cmd:option("-valid_ratio", 0.1, "validation set")
cmd:option("-shuffle", false, "shuffle all before train / valid / test split")
cmd:option("-all_tags", false, "learn other tags, not just the the first one")
cmd:option("-scale", 0.1, "what scale?")

--- Dataset augmentation

cmd:option("-no_flip", false, "horizontal flip")
cmd:option("-vert_crop_ratio", 0.1, "vertical crop ratio")
cmd:option("-horiz_crop_ratio", 0.2, "horizontal crop ratio")
cmd:option("-vert_crop", 0, "vertical crop ratio")
cmd:option("-horiz_crop", 0, "horizontal crop ratio")

--- Info during training

cmd:option("-verbose", false, "display information")
cmd:option("-visualize", false, "display images during training")
cmd:option("-plot", false, "display plot during training")

--- Hardware

cmd:option("-gpuid", -1, "GPU id; -1 for CPU")

--- Optimization paramters

cmd:option("-batch_size", 10, "batch size")
cmd:option("-not_contiguous", false, "do not take contiguous batches")

cmd:option("-learning_rate", 1e-3, "learning rate")
cmd:option("-momentum", .0, "momentum")
cmd:option("-max_epochs", 100, "number of maximum epochs")

--- Miscellaneous

cmd:option("-seed", 666, "seed for the random number generator")

--- Parse

opt = cmd:parse(arg)

opt.use_cache = not opt.no_cache
opt.flip = not opt.no_flip
opt.contiguous = not opt.not_contiguous
opt.one_of_k = not opt.one_output

torch.manualSeed(opt.seed)

--------------------------------------------------------------------------------
--- 3. Train the model

ds = ActivityDataset(opt)

if opt.see_data then
   require("image")
   ds:reset_batch("train")
   repeat
      local X, T = ds:update_batch("train")
      local band = X[1]:clone()
      for i=2,ds.batch_size do
         band = torch.cat(band, X[i])
      end
      win = image.display{image=band, win=win, zoom=0.5, legend='batch'}
      print("$")
   until ds.epoch_finished
end -- if opt.see_data


--------------------------------------------------------------------------------
--- 4. Done!

print("Done!")
