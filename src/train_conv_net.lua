ds = require('activity_dataset')
require('torch')

function get_mini_batch(img_names, labels, B)
   local N = #img_names
   local idx = torch.rand(B):mul(N):ceil():long()
   local batch_names = {}
   for i = 1, B do batch_names[i] = img_names[idx[i]] end
   local X = ds.get_posture_batch(batch_names)
   local T = labels:index(1, idx)
   return X, T
end

do
   local img_names, class_names, labels = ds.get_posture_dataset()
   local N = #img_names
   local X, T
   X, T = get_mini_batch(img_names, labels, 20)
   print(X:size())
   print(T:size())
end -- do
