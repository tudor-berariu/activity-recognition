local module activity_dataset = {}

--[[

   This function prepares a new batch of posture images. It returns
   just the pre-processed images in a tensor.

--]]

function activity_dataset.get_posture_batch(names, path, args)
   local image = require("image");
   path = path or "../data/datasets/posture/" -- default folder
   args = args or {}

   -- take the first image to extract size information
   local first_img = image.load(path .. names[1])
   -- image should be 3 x height x width
   assert(first_img:nDimension() == 3 and first_img:size(1) == 3)

   local original_height = first_img:size(2)
   local original_width = first_img:size(3)

   -- scale images
   local scale = args.scale or 1
   local scaled_height = torch.round(original_height * scale)
   local scaled_width = torch.round(original_width * scale)

   -- crop to given size
   local vert_crop = scaled_height - (args.crop_height or scaled_height)
   local horiz_crop = scaled_width - (args.crop_width or scaled_width)
   local height = scaled_height - vert_crop
   local width = scaled_width - horiz_crop

   -- function to extract (scale and crop) images
   local function extract_image(img, dst)
      if args.scale then img = image.scale(img, scaled_width, scaled_height) end
      local up = torch.random(1 + vert_crop)
      local left = torch.random(1 + horiz_crop)
      image.crop(dst, img, left, up, left + width - 1, up + height - 1)
   end

   -- allocate tensor
   local X = torch.Tensor(#names, height, width)
   -- extract images
   extract_image(first_img, X[i])
   for i = 2,X:size(1) do
      local img = image.load(path .. names[i])
      extract_image(img[1], X[i])
   end

   return X
end -- function


--[[

   This function returns a table with all the names and a tensor for
   the labels.

--]]

function activity_dataset.get_posture_dataset(args)
   args = args or {}

   -- check if we're in debug mode
   verbose = args.verbose or false

   local path = args.path or "../data/datasets/posture" -- default relative path
   if path:sub(path:len()) ~= "/" then path = path .. "/" end

   local f = io.open(path .. "info") -- info file must exist
   if not f then error() end

   -- go through all examples once
   local classes = {} -- class correspondence
   local images = {}  -- an array of images
   local K = 0        -- the number of classes
   while true do
      -- read a new line from info
      local line = f:read()
      if not line then break end -- skip empty lines (there shouldn't be any)

      local is_first = true
      for word in line:gmatch("([^,]+)") do -- parse line
         if is_first then -- the first word represents the file name
            images[#images+1] = word
            is_first = false
         elseif not classes[word] then -- other words are labels
            classes[word] = K + 1
            K = K + 1
         end
      end -- for s
   end -- while true
   f:close()

   if verbose then
      print(#images .. " lines read; " .. K .. " different classes")
   end

   -- now go again through all examples to build target tensor
   local T = torch.zeros(#images, K)
   local f = io.open(path .. "info")

   for n = 1, #images do
      local line
      repeat
         line = f:read()  -- skip empty lines (there shouldn't be any)
      until line

      local is_first = true
      for tag in line:gmatch("([^,]+)") do
         if not is_first then
            T[{n, classes[tag]}] = 1 -- one hot codification
         else
            assert(images[n] == tag) -- just to be sure we are on the right line
            is_first = false         -- skip the file name
         end -- if c_idx > 0
      end -- for tag
   end -- for n

   f:close()

   return images, classes, T
end -- function get_posture_dataset


return activity_dataset
