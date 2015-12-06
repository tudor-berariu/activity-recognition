require("torch")
require("train_conv_net")

do
   args = {
      -- general arguments
      ["verbose"] = true,
      ["plot"] = true,
      ["visualize"] = true,
      -- dataset
      ["scale"] = 0.2,
      ["limit"] = 1000,
      ["one_of_k"] = true,
      ["filter"] = {["sitting"] = true, ["standing"] = true},
      -- preprocess
      ["crop_height"] = 91,
      ["crop_width"] = 91,
      -- training
      ["momentum"] = 0.9,
      ["learningRate"] = 1.0
   }
   train_model(args)
end
