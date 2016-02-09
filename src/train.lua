require("preprocessor")

function sgd_epoch(model, images, labels, classes, preprocessor, args)
   local nn = require("nn")
   local optim = require("optim")

   args = args or {}

   local criterion = args.criterion or nn.MSECriterion()
   local parameters, gradParameters = model:getParameters()
   local confusion = optim.ConfusionMatrix(classes)

   local batch_size = args.batch_size or 20
   local idx = torch.randperm(images:size(1)):long()
   -- go through the dataset
   local train_err = 0
   local iter = 0
   for i = 1, images:size(1), batch_size do
      xlua.progress(i, images:size(1) + 1)
      j = math.min(images:size(1), i + batch_size -1)

      local feval = function(x)
         if x ~= parameters then
            parameters:copy(x)
         end
         gradParameters:zero()
         local f = 0
         local input = preprocessor:process_batch(images:index(1, idx[{{i,j}}]))
         local targets = labels:index(1, idx[{{i, j}}])
         local output = model:forward(input)
         local df_do = torch.Tensor(output:size(1), labels:size(2))
         for k = 1, output:size(1) do
            local err = criterion:forward(output[k], targets[k])
            f = f + err
            df_do[k]:copy(criterion:backward(output[k], targets[k]))
            confusion:add(output[k], targets[k])
         end
         model:backward(input, df_do)
         gradParameters:div(j - i + 1)
         f = f / (j - i + 1)
         train_err = train_err + f
         return f, gradParameters
      end
      optim.sgd(feval, parameters, args)
      iter = iter + 1
      if iter % 1000 == 0 then
         collectgarbage("collect")
      end
   end
   xlua.progress(images:size(1), images:size(1)+1)
   return confusion, train_err
end

function train_model(args)
   local cv_models = require("conv_net_models")
   local ds = require("activity_dataset")
   

   args = args or {}
   local epochs_no = args.epochs or 10

   -- get data
   images, classes, labels,stats = ds.get_posture_dataset(args)
   local util = require("util")
   classes = util.reverse_table(classes)
   images:reshape(images, images:size(1), 1, images:size(2), images:size(3))
   images:add(-stats.mean)
   images:div(stats.stddev)

   args["height"] = images:size(3)
   args["width"] = images:size(4)
   args["classes_no"] = #classes

   -- preprocess
   preprocessor = Preprocessor(args)
   args["height"] = preprocessor.height
   args["width"] = preprocessor.width
   -- get model
   net = cv_models.get_network(args)
   print(net)
   for epoch = 1, epochs_no do
      local train_conf, train_err, test_conf, test_err
      local time = sys.clock()
      train_conf, train_err =
         sgd_epoch(net, images, labels, classes, preprocessor, args)
      time = sys.clock() - time
      if args.verbose then
         print("epoch " .. epoch .. " : " .. (time * 1000) .. " ms")
      end
      local train_acc = train_conf.totalValid
      print("train accuracy: " .. train_acc)
      print("train error: " .. train_err)
   end
end

return true
