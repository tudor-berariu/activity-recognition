local module conv_net_models = {}
require("nn")

function conv_net_models.get_network(args)
   local nn = require("nn")
   local args = args or {}
   local width = args.width or 110
   local height = args.height or 110
   local classes_no = args.classes_no or 4
   local model_name = args.model or "5x5conv"
   local model
   if model_name == "5x5conv" then
      model = nn.Sequential()
      --- first convolution
      model:add(nn.SpatialZeroPadding(2, 2, 2, 2))
      model:add(nn.SpatialConvolutionMap(nn.tables.full(1, 32), 5, 5, 1, 1))
      model:add(nn.ReLU())
      model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      width = math.ceil(width / 2)
      height = math.ceil(height / 2)
      print("anticipated (1): " .. height .. "x" .. width)
      --- second convolution
      model:add(nn.SpatialZeroPadding(2, 2, 2, 2))
      model:add(nn.SpatialConvolutionMap(nn.tables.full(32, 64), 5, 5, 1, 1))
      model:add(nn.ReLU())
      model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      width = math.ceil(width / 2)
      height = math.ceil(height / 2)
      print("anticipated (2): " .. height .. "x" .. width)
      --- third convolution
      model:add(nn.SpatialZeroPadding(1, 1, 1, 1))
      model:add(nn.SpatialConvolutionMap(nn.tables.full(64, 128), 3, 3, 1, 1))
      model:add(nn.ReLU())
      model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      width = math.ceil(width / 2)
      height = math.ceil(height / 2)
      print("anticipated (3): " .. height .. "x" .. width)
      --- fourth convolution
      model:add(nn.SpatialZeroPadding(1, 1, 1, 1))
      model:add(nn.SpatialConvolutionMap(nn.tables.full(128, 128), 3, 3, 1, 1))
      model:add(nn.ReLU())
      model:add(nn.SpatialMaxPooling(2, 2, 2, 2):ceil())
      width = math.ceil(width / 2)
      height = math.ceil(height / 2)
      print("anticipated (4): " .. height .. "x" .. width)
      --- a last fully connected layer
      model:add(nn.Reshape(128 * height * width))
      model:add(nn.Linear(128 * height * width, classes_no))
      -- model:add(nn.Tanh())
   end

   model:add(nn.SoftMax())
   return model
end

return conv_net_models
