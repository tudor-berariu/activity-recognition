--------------------------------------------------------------------------------
--- 1. Load needed modules
--------------------------------------------------------------------------------

require("torch")                                            -- tensor operations
require("preprocessor")                      -- data set augmentation on the fly
require("util")                                             -- various functions

--------------------------------------------------------------------------------
--- 2. Dataset loader implementation - highly customizable
--------------------------------------------------------------------------------

local ActivityDataset = torch.class('ActivityDataset')

--------------------------------------------------------------------------------
--- 2.1. Method to initialize object (it reads the dataset from disk)
--------------------------------------------------------------------------------

--- Relevant options: verbose, batch_size, contiguous


function ActivityDataset:__init(opt)
   -----------------------------------------------------------------------------
   --- A. Configuration
   -----------------------------------------------------------------------------

   self.verbose = opt.verbose                      -- write sutff as they happen
   self.batch_size = opt.batch_size                                -- batch size
   self.contiguous = opt.contiguous               -- batches are served in order
   self.fmaps = 1                                       -- images are grey-scale

   local use_cache = opt.use_cache   -- the local variable might get overwritten

   local test_ratio = util.round(opt.test_ratio, 2)  -- ratio to use for testing
   local valid_ratio = util.round(opt.valid_ratio, 2)          -- ... validation
   local train_ratio = 1 - test_ratio - valid_ratio              -- ... training

   assert(test_ratio >= 0.0 and valid_ratio >= 0.0 and train_ratio > 0.0)

   self.batch_idxs = torch.LongTensor(self.batch_size)   -- allocate batch tesor

   if opt.all_tags and not opt.one_of_k then              -- sorry, not possible
      print("Incompatible configuration: scalar outputs and multiple classes")
      os.exit()
   end

   -----------------------------------------------------------------------------
   --- B. Configure paths on disk
   -----------------------------------------------------------------------------

   if opt.dataset == "posture" then
      self.path = opt.dataset_path or "../data/datasets/posture/"
   else
      print(opt.dataset .. " : unknown dataset")              -- unknown dataset
      os.exit()
   end

   if self.path:sub(self.path:len()) ~= "/" then
      self.path = self.path .. "/"               -- make sure path ends with "/"
   end

   self:print("loading the " .. opt.dataset .. " dataset from " .. self.path)

   local cache_path = self.path .. ".torch_cache/"     -- folder to save tensors

   -----------------------------------------------------------------------------
   --- C. Image preprocessing configuration
   -----------------------------------------------------------------------------

   local image = require("image")             -- module needed for image scaling

   local info = assert(io.open(self.path .. "info"), "info file not found")
   local line = info:read()                               -- read the first line
   info:close()                                       -- close info file for now

   local first_image_name = line:sub(1, line:find(",")-1) -- ignore tags for now
   local first_image = image.load(self.path .. first_image_name)[{{1}}]

   self:set_resolution(first_image:size(), opt) -- assume all have the same size
   assert(self.height > 0 and self.width > 0)           -- check the two members

   self:print("scaling to " .. self.height .. "x" .. self.width)

   self.preprocessor = Preprocessor(self.height, self.width, opt)
   assert(self.preprocessor.height > 0 and self.preprocessor.width > 0)

   self:print("size after preprocess:")
   self:print(self.preprocessor.height .. "x" .. self.preprocessor.width)

   self.X = torch.Tensor(                    -- allocate memory for batch tensor
      self.batch_size, self.fmaps,
      self.preprocessor.height, self.preprocessor.width
   )

   -----------------------------------------------------------------------------
   --- D. Split info
   -----------------------------------------------------------------------------

   local wc_output = io.popen("wc -l " .. self.path .. "info | cut -d\" \" -f1")
   local str = assert(wc_output:read("*a"))
   wc_output:close()
   self.all_no = tonumber(str)

   if opt.limit > 0 then self.all_no = math.min(opt.limit, self.all_no) end

   self:print("data set size set to " .. self.all_no)

   do                                        -- compute the size for each subset
      local rest

      self.valid_no = torch.round(valid_ratio * self.all_no)   -- validation set
      rest = self.valid_no % self.batch_size
      self.valid_no = self.valid_no - rest

      self.test_no = torch.round(test_ratio * self.all_no)      -- test set size
      rest = self.test_no % self.batch_size
      self.test_no = self.test_no - rest

      self.train_no = self.all_no - self.valid_no - self.test_no    -- train set
      rest = self.train_no % self.batch_size
      self.train_no = self.train_no - rest

      self.all_no = self.train_no + self.valid_no + self.test_no   -- final size
   end


   -----------------------------------------------------------------------------
   --- D. prefix used for file names
   -----------------------------------------------------------------------------

   local names_prefix = ""                                  -- names file prefix
   local images_prefix = ""                                -- images file prefix
   local labels_prefix = ""                                -- labels file prefix
   local classes_prefix = ""                              -- classes file prefix
   local misc_prefix = ""                            -- various info file prefix

   if opt.one_of_k then                    -- attach output encoding information
      labels_prefix = "k_" .. labels_prefix
      misc_prefix = "k_" .. misc_prefix
   end

   if opt.shuffle then                                      -- append order info
      names_prefix = "s_" .. opt.seed .. "_" .. names_prefix
      images_prefix = "s_" .. opt.seed .. "_" .. images_prefix
      labels_prefix = "s_" .. opt.seed .. "_" .. labels_prefix
      misc_prefix = "s_" .. opt.seed .. "_" .. misc_prefix
   end

   if opt.all_tags then                                  -- append classes info
      labels_prefix = "all_" .. labels_prefix
      classes_prefix = "all_" .. classes_prefix
      misc_prefix = "all_" .. misc_prefix
   end

   do                                                       -- apend split info
      local fmt = "%.2f_%.2f_%.2f_"
      local split_info = fmt:format(train_ratio, valid_ratio, test_ratio)
      names_prefix = split_info .. names_prefix
      images_prefix = split_info .. images_prefix
      labels_prefix = split_info .. labels_prefix
      misc_prefix = split_info .. misc_prefix
   end

   --[[
      do                                             -- attach scale information
      local scale_prefix = self.height .. "x" .. self.width .. "_"
      images_prefix = scale_prefix .. images_prefix
      misc_prefix = scale_prefix .. misc_prefix
      end
   --]]

   do                                              -- atach info about data size
      names_prefix = self.all_no .. "_" .. names_prefix
      images_prefix = self.all_no .. "_" .. images_prefix
      labels_prefix = self.all_no .. "_" .. labels_prefix
      classes_prefix = self.all_no .. "_" .. classes_prefix
      misc_prefix = self.all_no .. "_" .. misc_prefix
   end

   local names_path = cache_path .. names_prefix .. "names" .. ".dat"
   local images_path = cache_path .. images_prefix .. "images" .. ".dat"
   local labels_path = cache_path .. labels_prefix .. "labels" .. ".dat"
   local classes_path = cache_path .. classes_prefix .. "classes" .. ".dat"
   local misc_path = cache_path .. misc_prefix .. "misc" .. ".dat"

   -----------------------------------------------------------------------------
   --- E. use cache or get images from files
   -----------------------------------------------------------------------------

   if use_cache then
      self:print("searching for saved data on disk")
   else
      self:print("not using saved data")
   end

   if use_cache then

      local files_to_check = {}             -- make a list with all needed files

      if opt.just_names then
         table.insert(files_to_check, names_path)                  -- names file
      else
         table.insert(files_to_check, images_path)                -- images file
      end
      table.insert(files_to_check, labels_path)                   -- labels file
      table.insert(files_to_check, classes_path)                 -- classes file
      table.insert(files_to_check, misc_path)              -- miscellaneous file

      for _,v in pairs(files_to_check) do
         local f = io.open(v)                            -- try opening the file
         if not f then use_cache = false else f.close() end          -- fallback
      end -- for _,v

   end -- if use_cache

   if use_cache then
      self:print("using cached data")
   else
      self:print("cannot use cached data")
   end

   -----------------------------------------------------------------------------
   --- F. if saved data is to be used, load tensors from disk
   -----------------------------------------------------------------------------

   if use_cache then
      self.misc = assert(torch.load(misc_path))                          -- misc
      assert(self.misc.height == self.height)
      assert(self.misc.width == self.width)
      assert(self.misc.all_no == self.all_no)
      assert(self.misc.train_no == self.train_no)
      assert(self.misc.valid_no == self.valid_no)
      assert(self.misc.test_no == self.test_no)
      self.classes_no = self.misc.classes_no

      self.classes = assert(torch.load(classes_path))                 -- classes
      for c = 1,self.classes_no do assert(self.classes[c]); end

      self.labels = assert(torch.load(labels_path))                    -- labels
      assert(self.misc.all_no == self.labels:size(1))
      if opt.one_of_k then
         assert(self.labels:nDimension() == 2)
         assert(self.classes_no == self.labels:size(2))
      else
         assert(self.labels_train:nDimension() == 1)
      end

      if self.just_names then
         self.names = assert(torch.load(names_path))                     -- names
         assert(self.all_no == #(self.names))
      else
         self.images = assert(torch.load(images_path))                  -- images
         assert(self.all_no == self.images:size(1))
         assert(self.fmaps == self.images:size(2))
         assert(self.height == self.images:size(3))
         assert(self.width == self.images:size(4))
      end

      if opt.one_of_k then
         self.T = torch.Tensor(self.batch_size, self.misc.classes_no)
      else
         self.T = torch.Tensor(self.batch_size)
      end

      self.train_labels = self.labels:narrow(1, 1, self.train_no)
      self.valid_labels = self.labels:narrow(1, self.train_no + 1, self.valid_no)
      self.test_labels =
         self.labels:narrow(1, self.train_no + self.valid_no + 1, self.test_no)

      if self.images then
         self.train_images = self.images:narrow(1, 1, self.train_no)
         self.valid_images =
            self.images:narrow(1, self.train_no+1, self.valid_no)
         self.test_images =
            self.images:narrow(1, self.train_no+self.valid_no+1, self.test_no)
      end

      return                                          -- data loaded successfully
   end

   -----------------------------------------------------------------------------
   --- G. read data from disk
   -----------------------------------------------------------------------------


   if opt.just_names then
      self.names = {}                                                   -- names
   else
      self.images =                                                    -- images
         torch.Tensor(self.all_no, self.fmaps, self.height, self.width)
   end

   local inv_classes = {}                                  -- classes to numbers
   self.classes = {}                                       -- numbers to classes

   local alloc_classes = 2

   if opt.one_of_k then
      self.labels = torch.zeros(self.all_no, alloc_classes)    -- one hot labels
   else
      self.labels = torch.zeros(self.all_no)                    -- scalar labels
   end

   local sidx                          -- map line number to position in dataset
   if opt.shuffle then
      sidx = torch.randperm(self.all_no)                              -- shuffle
   else
      sidx = torch.linspace(1, self.all_no, self.all_no):long()      -- original
   end

   info = assert(io.open(self.path .. "info"))               -- reopen info file

   local mean = 0                                                        -- mean
   local stddev = 0                                        -- standard deviation

   self.classes_no = 0

   for line_no = 1, self.all_no do
      local idx = sidx[line_no]                       -- position in the dataset
      local line = info:read()                                 -- read next line

      if not line or not line:find(",") then                 -- exit if not good
         print("Something went wrong on line " .. line_no)
         os.exit()
      end

      local cnt = 1

      for word in line:gmatch("([^,]+)") do                        -- parse line
         if cnt == 1 then            -- the first word represents the image name
            local oimg = image.load(self.path .. word)[{{1}}]  -- keep channel 1
            local simg = image.scale(oimg[1], self.width, self.height)

            mean = mean + simg:mean()
            stddev = stddev + torch.cmul(simg, simg):mean()

            if opt.just_names then
               self.names[idx] = word
            else
               self.images[idx]:copy(simg)
            end

         elseif opt.all_tags or cnt == 2 then
            if not inv_classes[word] then                 -- is this a new class
               self.classes_no = self.classes_no + 1
               inv_classes[word] = self.classes_no
               self.classes[self.classes_no] = word
               if opt.one_of_k and self.classes_no > alloc_classes then
                  self.labels =
                     torch.cat(self.labels, torch.zeros(alloc_classes), 2)
                  alloc_classes = alloc_classes + alloc_classes
               end
            end

            local class = inv_classes[word]            -- the current class
            if opt.one_of_k then
               self.labels[idx][class] = 1
            else
               self.labels[idx] = class
            end
         end -- if cnt
         cnt = cnt + 1
      end -- for word
      xlua.progress(line_no, self.all_no)                     -- report progress
   end

   info:close()
   collectgarbage()

   mean = mean / self.all_no
   stddev = math.sqrt(stddev / self.all_no - mean * mean)

   self.mean = mean
   self.stddev = stddev

   if opt.one_of_k then
      self.labels = self.labels[{{1, self.all_no}, {1, self.classes_no}}]
      self.T = torch.Tensor(self.batch_size, self.classes_no)
   else
      self.T = torch.Tensor(self.batch_size)
   end

   -----------------------------------------------------------------------------
   --- H. Link subtensors
   -----------------------------------------------------------------------------

   self.train_labels = self.labels:narrow(1, 1, self.train_no)
   self.valid_labels = self.labels:narrow(1, self.train_no + 1, self.valid_no)
   self.test_labels =
      self.labels:narrow(1, self.train_no + self.valid_no + 1, self.test_no)

   if self.images then
      self.train_images = self.images:narrow(1, 1, self.train_no)
      self.valid_images =
         self.images:narrow(1, self.train_no+1, self.valid_no)
      self.test_images =
         self.images:narrow(1, self.train_no+self.valid_no+1, self.test_no)
   end


   self:display_info()

   -----------------------------------------------------------------------------
   --- I. Save to disk and return
   -----------------------------------------------------------------------------

   assert(os.execute("mkdir -p " .. cache_path))

   if opt.just_names then
      torch.save(names_path, self.names)                           -- save names
   else                                                                    -- or
      torch.save(images_path, self.images)                        -- save images
   end

   torch.save(labels_path, self.labels)                           -- save labels
   torch.save(classes_path, self.classes)                        -- save classes

   do
      local misc = {
         ["mean"] = self.mean, ["stddev"] = self.stddev,
         ["all_no"] = self.all_no, ["train_no"] = self.train_no,
         ["valid_no"] = self.valid_no, ["test_no"] = self.test_no,
         ["classes_no"] = self.classes_no,
         ["height"] = self.height, ["width"] = self.width
      }

      torch.save(misc_path, misc)                     -- save miscellaneous
   end

   self:print("Saved on disk")
end

function ActivityDataset:print(message)
   if self.verbose then
      print("[activity_dataset] " .. message)
   end
end

--------------------------------------------------------------------------------

--[[

   set_resolution finds out the size images are scaled to

--]]

function ActivityDataset:set_resolution(original_size, opt)
   local n = original_size:size()

   assert((n == 2 or n == 3), "Strange image size")

   if n == 2 then
      self.fmaps = 1
   elseif n == 3 then
      self.fmaps = original_size[1]
   end

   self.height = original_size[n-1]
   self.width  = original_size[n]

   if opt.scale then
      -- if scale is specified
      self.height = torch.round(self.height * opt.scale)
      self.width = torch.round(self.width * opt.scale)
   elseif opt.height and opt.width then
      -- if exact height and widht are specified
      self.height = opt.height
      self.width = opt.width
   elseif opt.height then
      -- if exact height is specified
      self.width = torch.round(self.width * opt.height / self.height)
      self.height = opt.height
   elseif opt.width then
      -- if exact width is specified
      self.height = torch.round(self.height * opt.width / self.width)
      self.width = opt.width
   end

end

--[[

   This function returns a random mini batch of size B.
   It's supposed to be used when it's impossible to load the full dataset in
   memory.
--]]

function ActivityDataset:update_batch(subset)
   local image = require("image")

   local first_idx, last_idx
   if subset == "train" then
      first_idx = 1
      last_idx = self.train_no
   elseif subset == "valid" then
      first_idx = self.train_no + 1
      last_idx = self.train_no + self.valid_no
   else
      first_idx = self.train_no + self.valid_no + 1
      last_idx = self.all_no
   end

   --- Compute the indexes for the next batch of examples
   if self.contiguous then
      self.idx = self.idx or first_idx
      self._idxs = torch.linspace(
         self.idx,
         self.idx + self.batch_size - 1,
         self.batch_size
      ):long()

      self.idx = self.idx + self.batch_size
      if self.idx > last_idx then
         self.idx = 1
         self.epoch_finished = true
      else
         self.epoch_finished = false
      end

   else
      self._idxs = torch.rand(self.batch_size):mul(last_idx - first_idx)
         :ceil():add(first_idx - 1):long()
      self.epoch_finished = true
   end

   --- Prepare targets
   self.T:copy(self.labels:index(1, self._idxs))

   --- Prepare images
   if self.images then
      for i = 1,self.batch_size do
         self.preprocessor:process_image(self.X[i], self.images[self._idxs[i]])
      end
   else
      for i = 1,self.batch_size do
         self.preprocessor:process_image(
            self.X[i],
            image.load(self.path .. self.names[self._idxs[i]])[{{1}}]
         )
      end
   end

   collectgarbage()
   return self.X, self.T
end

function ActivityDataset:reset_batch(subset)
   if subset == "train" then
      self.idx = 1
   elseif subset == "valid" then
      self.idx = train_no + 1
   else
      self.idx = train_no + valid_no + 1
   end
end

function ActivityDataset:display_info()
   for n,c in pairs(self.classes) do
      print(n .. "=>" .. c)
   end
end
