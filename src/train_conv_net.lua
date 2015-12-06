local ds = require("activity_dataset")
local conv_models = require("conv_net_models")

function sgd_epoch(model, images, labels, classes, args)
   local criterion = nn.ClassNLLCriterion()
   local parameters, gradParameters = model:getParameters()
   local confusion = optim.ConfusionMatrix(classes)
   args = args or {}
   local batch_size = args.batch_size or 20
   local idx = torch.randperm(images:size(1))
   -- go through the dataset
   local iter = 0
   for i = 1, images:size(1), batch_size do
      xlua.progress(t, images:size(1) + 1)
      j = math.min(images:size(1), i + batch_size -1)

      local feval = function(x)
         if x ~= parameters then
            parameters:copy(x)
         end
         gradParameters:zero()
         local f = 0
         local input = images:index(1, idx[{{i, j}}])
         local targets = labels:index(1, idx[{{i, j}}]
         local output = model:forward()
         local df_do = torch.Tensor(output:size(1), labels:size(2))
         for k = 1, output:size(1) do
            local err = criterion:forward(output[k], targets[k])
            f = f + err
            df_do[k]:copy(criterion:backward(output[k], targets[k]))
            confusion:add(output[k], targets[k])
         end
         model:backward(inputs, df_do)
         gradParameters:div(j - i + 1)
         f = f / (j - i + 1)
         return f, gradParameters
      end
      optim.sgd(feval, parameters, args)
      iter = iter + 1
      if iter % 1000 == 0 then
         collectgarbage("collect")
      end
   end
   xlua.progress(images:size(1), images:size(1)+1)
   return confusion
end
