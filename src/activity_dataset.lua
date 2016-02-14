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

--- Relevant options: verbose, batchSize, contiguous


function ActivityDataset:__init(opt)
   -----------------------------------------------------------------------------
   --- A. Configuration
   -----------------------------------------------------------------------------

   self.verbose = opt.verbose                      -- write sutff as they happen
   self.batchSize = opt.batchSize                                  -- batch size
   self.contiguous = opt.contiguous               -- batches are served in order
   self.fmaps = 1                                       -- images are grey-scale

   local useCache = opt.useCache     -- the local variable might get overwritten

   local testRatio = util.round(opt.testRatio, 2)    -- ratio to use for testing
   local validRatio = util.round(opt.validRatio, 2)            -- ... validation
   local trainRatio = 1 - testRatio - validRatio                 -- ... training

   assert(testRatio >= 0.0 and validRatio >= 0.0 and trainRatio > 0.0)

   self.batchIdxs = torch.LongTensor(self.batchSize)     -- allocate batch tesor

   if opt.allTags and not opt.oneHot then                 -- sorry, not possible
      print("Incompatible configuration: scalar outputs and multiple classes")
      os.exit()
   end

   -----------------------------------------------------------------------------
   --- B. Configure paths on disk
   -----------------------------------------------------------------------------

   if opt.dataset == "posture" then
      self.path = opt.datasetPath or "../data/datasets/posture/"
   else
      print(opt.dataset .. " : unknown dataset")              -- unknown dataset
      os.exit()
   end

   if self.path:sub(self.path:len()) ~= "/" then
      self.path = self.path .. "/"               -- make sure path ends with "/"
   end

   self:print("loading the " .. opt.dataset .. " dataset from " .. self.path)

   local cachePath = self.path .. ".torch_cache/"      -- folder to save tensors

   -----------------------------------------------------------------------------
   --- C. Image preprocessing configuration
   -----------------------------------------------------------------------------

   local image = require("image")             -- module needed for image scaling

   local info = assert(io.open(self.path .. "info"), "info file not found")
   local line = info:read()                               -- read the first line
   info:close()                                       -- close info file for now

   local firstImageName = line:sub(1, line:find(",")-1)           -- ignore tags
   local firstImage = image.load(self.path .. firstImageName)[{{1}}]

   self:setResolution(firstImage:size(), opt)   -- assume all have the same size
   assert(self.height > 0 and self.width > 0)           -- check the two members

   self:print("scaling to " .. self.height .. "x" .. self.width)

   self.preprocessor = Preprocessor(self.height, self.width, opt)
   assert(self.preprocessor.height > 0 and self.preprocessor.width > 0)
   self.inHeight = self.preprocessor.height
   self.inWidth = self.preprocessor.width

   self:print("size after preprocess:" .. self.inHeight .. "x" .. self.inWidth)

   self.X = torch.Tensor(                    -- allocate memory for batch tensor
      self.batchSize, self.fmaps, self.inHeight, self.inWidth
   )

   -----------------------------------------------------------------------------
   --- D. Split info
   -----------------------------------------------------------------------------

   local wcOutput = io.popen("wc -l " .. self.path .. "info | cut -d\" \" -f1")
   local wcResult = assert(wcOutput:read("*a"))
   wcOutput:close()
   self.allNo = tonumber(wcResult)

   if opt.limit > 0 then self.allNo = math.min(opt.limit, self.allNo) end

   self:print("data set size set to " .. self.allNo)

   do                                        -- compute the size for each subset
      local rest

      self.validNo = torch.round(validRatio * self.allNo)      -- validation set
      rest = self.validNo % self.batchSize
      self.validNo = self.validNo - rest

      self.testNo = torch.round(testRatio * self.allNo)         -- test set size
      rest = self.testNo % self.batchSize
      self.testNo = self.testNo - rest

      self.trainNo = self.allNo - self.validNo - self.testNo        -- train set
      rest = self.trainNo % self.batchSize
      self.trainNo = self.trainNo - rest

      self.allNo = self.trainNo + self.validNo + self.testNo       -- final size
   end


   -----------------------------------------------------------------------------
   --- D. prefix used for file names
   -----------------------------------------------------------------------------

   local namesPrefix = ""                                   -- names file prefix
   local imagesPrefix = ""                                 -- images file prefix
   local labelsPrefix = ""                                 -- labels file prefix
   local classesPrefix = ""                               -- classes file prefix
   local miscPrefix = ""                             -- various info file prefix

   if opt.oneHot then                      -- attach output encoding information
      labelsPrefix = "k_" .. labelsPrefix
      miscPrefix = "k_" .. miscPrefix
   end

   if opt.shuffle then                                      -- append order info
      namesPrefix = "s_" .. opt.seed .. "_" .. namesPrefix
      imagesPrefix = "s_" .. opt.seed .. "_" .. imagesPrefix
      labelsPrefix = "s_" .. opt.seed .. "_" .. labelsPrefix
      miscPrefix = "s_" .. opt.seed .. "_" .. miscPrefix
   end

   if opt.allTags then                                    -- append classes info
      labelsPrefix = "all_" .. labelsPrefix
      classesPrefix = "all_" .. classesPrefix
      miscPrefix = "all_" .. miscPrefix
   end

   do                                                        -- apend split info
      local fmt = "%.2f_%.2f_%.2f_"
      local splitInfo = fmt:format(trainRatio, validRatio, testRatio)
      namesPrefix = splitInfo .. namesPrefix
      imagesPrefix = splitInfo .. imagesPrefix
      labelsPrefix = splitInfo .. labelsPrefix
      miscPrefix = splitInfo .. miscPrefix
   end

   --[[
      do                                             -- attach scale information
      local scale_prefix = self.height .. "x" .. self.width .. "_"
      imagesPrefix = scale_prefix .. imagesPrefix
      miscPrefix = scale_prefix .. miscPrefix
      end
   --]]

   do                                              -- atach info about data size
      namesPrefix = self.allNo .. "_" .. namesPrefix
      imagesPrefix = self.allNo .. "_" .. imagesPrefix
      labelsPrefix = self.allNo .. "_" .. labelsPrefix
      classesPrefix = self.allNo .. "_" .. classesPrefix
      miscPrefix = self.allNo .. "_" .. miscPrefix
   end

   local namesFilePath = cachePath .. namesPrefix .. "names" .. ".dat"
   local imagesFilePath = cachePath .. imagesPrefix .. "images" .. ".dat"
   local labelsFilePath = cachePath .. labelsPrefix .. "labels" .. ".dat"
   local classesFilePath = cachePath .. classesPrefix .. "classes" .. ".dat"
   local miscFilePath = cachePath .. miscPrefix .. "misc" .. ".dat"

   -----------------------------------------------------------------------------
   --- E. use cache or get images from files
   -----------------------------------------------------------------------------

   if useCache then
      self:print("searching for saved data on disk")
   else
      self:print("not using saved data")
   end

   if useCache then

      local filesToCheck = {}             -- make a list with all needed files

      if opt.justNames then
         table.insert(filesToCheck, namesFilePath)                 -- names file
      else
         table.insert(filesToCheck, imagesFilePath)               -- images file
      end
      table.insert(filesToCheck, labelsFilePath)                  -- labels file
      table.insert(filesToCheck, classesFilePath)                -- classes file
      table.insert(filesToCheck, miscFilePath)             -- miscellaneous file

      for _,fname in pairs(filesToCheck) do
         local f = io.open(fname)                        -- try opening the file
         if not f then useCache = false else f.close() end           -- fallback
      end -- for _,fname

   end -- if useCache

   if useCache then
      self:print("using cached data")
   else
      self:print("cannot use cached data")
   end

   -----------------------------------------------------------------------------
   --- F. if saved data is to be used, load tensors from disk
   -----------------------------------------------------------------------------

   if useCache then
      self.misc = assert(torch.load(miscFilePath))                       -- misc
      assert(self.misc.height == self.height)
      assert(self.misc.width == self.width)
      assert(self.misc.allNo == self.allNo)
      assert(self.misc.trainNo == self.trainNo)
      assert(self.misc.validNo == self.validNo)
      assert(self.misc.testNo == self.testNo)
      self.classesNo = self.misc.classesNo

      self.classes = assert(torch.load(classesFilePath))              -- classes
      for c = 1,self.classesNo do assert(self.classes[c]); end

      self.labels = assert(torch.load(labelsFilePath))                 -- labels
      assert(self.misc.allNo == self.labels:size(1))
      if opt.oneHot then
         assert(self.labels:nDimension() == 2)
         assert(self.classesNo == self.labels:size(2))
      else
         assert(self.labels:nDimension() == 1)
      end

      if self.justNames then
         self.names = assert(torch.load(namesFilePath))                 -- names
         assert(self.allNo == #(self.names))
      else
         self.images = assert(torch.load(imagesFilePath))              -- images
         assert(self.allNo == self.images:size(1))
         assert(self.fmaps == self.images:size(2))
         assert(self.height == self.images:size(3))
         assert(self.width == self.images:size(4))
      end

      if opt.oneHot then
         self.T = torch.Tensor(self.batchSize, self.misc.classesNo)
      else
         self.T = torch.Tensor(self.batchSize)
      end

      self.trainLabels = self.labels:narrow(1, 1, self.trainNo)
      self.validLabels = self.labels:narrow(1, self.trainNo + 1, self.validNo)
      self.testLabels =
         self.labels:narrow(1, self.allNo - self.testNo + 1, self.testNo)

      if self.images then
         self.trainImages = self.images:narrow(1, 1, self.trainNo)
         self.validImages = self.images:narrow(1, self.trainNo+1, self.validNo)
         self.testImages =
            self.images:narrow(1, self.trainNo+self.validNo+1, self.testNo)
      end

      return                                         -- data loaded successfully
   end

   -----------------------------------------------------------------------------
   --- G. read data from disk
   -----------------------------------------------------------------------------


   if opt.justNames then
      self.names = {}                                                   -- names
   else
      self.images =                                                    -- images
         torch.Tensor(self.allNo, self.fmaps, self.height, self.width)
   end

   local invClasses = {}                                   -- classes to numbers
   self.classes = {}                                       -- numbers to classes

   local allocClasses = 2

   if opt.oneHot then
      self.labels = torch.zeros(self.allNo, allocClasses)      -- one hot labels
   else
      self.labels = torch.zeros(self.allNo)                     -- scalar labels
   end

   local sidx                          -- map line number to position in dataset
   if opt.shuffle then
      sidx = torch.randperm(self.allNo)                               -- shuffle
   else
      sidx = torch.linspace(1, self.allNo, self.allNo):long()        -- original
   end

   info = assert(io.open(self.path .. "info"))               -- reopen info file

   local mean = 0                                                        -- mean
   local stddev = 0                                        -- standard deviation

   self.classesNo = 0

   for lineNo = 1, self.allNo do
      local idx = sidx[lineNo]                        -- position in the dataset
      local line = info:read()                                 -- read next line

      if not line or not line:find(",") then                 -- exit if not good
         print("Something went wrong on line " .. lineNo)
         os.exit()
      end

      local cnt = 1

      for word in line:gmatch("([^,]+)") do                        -- parse line
         if cnt == 1 then            -- the first word represents the image name
            local oImg = image.load(self.path .. word)[{{1}}]  -- keep channel 1
            local sImg = image.scale(oImg[1], self.width, self.height)

            mean = mean + sImg:mean()
            stddev = stddev + torch.cmul(sImg, sImg):mean()

            if opt.justNames then
               self.names[idx] = word
            else
               self.images[idx]:copy(sImg)
            end

         elseif opt.allTags or cnt == 2 then
            if not invClasses[word] then                  -- is this a new class
               self.classesNo = self.classesNo + 1
               invClasses[word] = self.classesNo
               self.classes[self.classesNo] = word
               if opt.oneHot and self.classesNo > allocClasses then
                  self.labels =
                     torch.cat(self.labels, torch.zeros(allocClasses), 2)
                  allocClasses = allocClasses + allocClasses
               end
            end

            local class = invClasses[word]                  -- the current class
            if opt.oneHot then
               self.labels[idx][class] = 1
            else
               self.labels[idx] = class
            end
         end -- if cnt
         cnt = cnt + 1
      end -- for word
      xlua.progress(lineNo, self.allNo)                       -- report progress
   end

   info:close()
   collectgarbage()

   mean = mean / self.allNo
   stddev = math.sqrt(stddev / self.allNo - mean * mean)

   self.mean = mean
   self.stddev = stddev

   if opt.oneHot then
      self.labels = self.labels[{{1, self.allNo}, {1, self.classesNo}}]
      self.T = torch.Tensor(self.batchSize, self.classesNo)
   else
      self.T = torch.Tensor(self.batchSize)
   end

   -----------------------------------------------------------------------------
   --- H. Link subtensors
   -----------------------------------------------------------------------------

   self.trainLabels = self.labels:narrow(1, 1, self.trainNo)
   self.validLabels = self.labels:narrow(1, self.trainNo + 1, self.validNo)
   self.testLabels =
      self.labels:narrow(1, self.trainNo + self.validNo + 1, self.testNo)

   if self.images then
      self.trainImages = self.images:narrow(1, 1, self.trainNo)
      self.validImages =
         self.images:narrow(1, self.trainNo+1, self.validNo)
      self.testImages =
         self.images:narrow(1, self.trainNo+self.validNo+1, self.testNo)
   end


   self:displayInfo()

   -----------------------------------------------------------------------------
   --- I. Save to disk and return
   -----------------------------------------------------------------------------

   assert(os.execute("mkdir -p " .. cachePath))

   if opt.justNames then
      torch.save(namesFilePath, self.names)                        -- save names
   else                                                                    -- or
      torch.save(imagesFilePath, self.images)                     -- save images
   end

   torch.save(labelsFilePath, self.labels)                        -- save labels
   torch.save(classesFilePath, self.classes)                     -- save classes

   do
      local misc = {
         ["mean"] = self.mean, ["stddev"] = self.stddev,
         ["allNo"] = self.allNo, ["trainNo"] = self.trainNo,
         ["validNo"] = self.validNo, ["testNo"] = self.testNo,
         ["classesNo"] = self.classesNo,
         ["height"] = self.height, ["width"] = self.width
      }

      torch.save(miscFilePath, misc)                       -- save miscellaneous
   end

   self:print("Saved on disk")
end

function ActivityDataset:print(message)
   if self.verbose then
      print("[activity dataset] " .. message)
   end
end

--------------------------------------------------------------------------------

--[[

   set_resolution finds out the size images are scaled to

--]]

function ActivityDataset:setResolution(originalSize, opt)
   local n = originalSize:size()

   assert((n == 2 or n == 3), "Strange image size")

   if n == 2 then
      self.fmaps = 1
   elseif n == 3 then
      self.fmaps = originalSize[1]
   end

   self.height = originalSize[n-1]
   self.width  = originalSize[n]

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

function ActivityDataset:updateBatch(subset)
   local image = require("image")

   local firstIdx, lastIdx
   if subset == "train" then
      firstIdx = 1
      lastIdx = self.trainNo
   elseif subset == "valid" then
      firstIdx = self.trainNo + 1
      lastIdx = self.trainNo + self.validNo
   else
      firstIdx = self.trainNo + self.validNo + 1
      lastIdx = self.allNo
   end

   --- Compute the indexes for the next batch of examples
   if self.contiguous then
      self.idx = self.idx or firstIdx
      self.batchIdxs = torch.linspace(
         self.idx,
         self.idx + self.batchSize - 1,
         self.batchSize
      ):long()

      self.idx = self.idx + self.batchSize
      if self.idx > lastIdx then
         self.idx = 1
         self.epochFinished = true
      else
         self.epochFinished = false
      end

   else
      self.batchIdxs = torch.rand(self.batchSize):mul(lastIdx - firstIdx)
         :ceil():add(firstIdx - 1):long()
      self.epochFinished = true
   end

   --- Prepare targets
   self.T:copy(self.labels:index(1, self.batchIdxs))

   --- Prepare images
   if self.images then
      for i = 1,self.batchSize do
         self.preprocessor:processImage(
            self.X[i], self.images[self.batchIdxs[i]]
         )
      end
   else
      for i = 1,self.batchSize do
         self.preprocessor:processImage(
            self.X[i],
            image.load(self.path .. self.names[self.batchIdxs[i]])[{{1}}]
         )
      end
   end

   collectgarbage()
   return self.X, self.T
end

function ActivityDataset:resetBatch(subset)
   if subset == "train" then
      self.idx = 1
   elseif subset == "valid" then
      self.idx = self.trainNo + 1
   else
      self.idx = self.trainNo + self.validNo + 1
   end
end

function ActivityDataset:displayInfo()
   for n, c in pairs(self.classes) do
      print(n .. "=>" .. c)
   end
end
