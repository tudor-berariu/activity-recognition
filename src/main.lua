--------------------------------------------------------------------------------
--- 1. Load required libraries

require("torch")
-- require("train_conv_net")

--------------------------------------------------------------------------------
--- 2. Parse command line options

cmd = torch.CmdLine()
cmd:text()
cmd:text("Activity recognition!")
cmd:text()
cmd:text("Options:")

-- General : the most important
cmd:option("-dataset", "posture", "the dataset to solve (posture / activity)")
cmd:option("-model", "logistic", "the model to be used for classification")

-- Info during training
cmd:option("-verbose", true, "display information")
cmd:option("-visualize", true, "display images during training")
cmd:option("-plot", true, "display plot during training")

-- Hardware
cmd:option("-gpuid", -1, "GPU id; -1 for CPU")

-- Optimization paramters
cmd:option("-learning_rate", 1e-3, "learning rate")
cmd:option("-momentum", .0, "momentum")
cmd:option("-batch_size", 10, "batch size")
cmd:option("-max_epochs", 100, "number of maximum epochs")

-- Miscellaneous
cmd:option("-seed", 666, "seed for the random number generator")

-- Parse
opt = cmd:parse(arg)

--------------------------------------------------------------------------------
--- 3. Train the model

torch.manualSeed(opt.seed)

--------------------------------------------------------------------------------
--- 4. Done!

print("Done!")
