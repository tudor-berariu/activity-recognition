--------------------------------------------------------------------------------
--- 1. Load needed modules

require("torch")
require("preprocessor")

--------------------------------------------------------------------------------
--- 2. Dataset loader implementation

local ActivityDataset = torch.class('ActivityDataset')

function ActivityDataset:__init(opt)
   -----------------------------------------------------------------------------
   --- A. other parameters

   if opt.use_cache ~= nil then
      self.use_cache = opt.use_cache
   else
      self.use_cache = true
   end

   self.just_names = opt.just_names or false
   self.verbose = opt.verbose or false

   self.batch_size = opt.batch_size or 10
   if opt.contiguous ~= nil then
      self.contiguous = opt.contiguous
   else
      self.contiguous = true
   end
   self._idxs = torch.LongTensor(self.batch_size)

   -----------------------------------------------------------------------------
   --- B. set the paths

   opt.dataset = opt.dataset or "posture"

   if opt.dataset == "posture" then
      self.path = opt.dataset_path or "../data/datasets/posture/"
   else
      print(opt.dataset .. " dataset not supported.")
      os.exit()
   end

   if self.path:sub(self.path:len()) ~= "/" then
      self.path = self.path .. "/"
   end
   local cache_path = self.path .. ".torch_cache/"

   local suffix, full_suffix, l_suffix

   if opt.limit and opt.limit > 0 then
      self:print("limit to " .. opt.limit .. " examples")
      suffix = "_" .. opt.limit .. ".dat"
   else
      suffix = ".dat"
   end

   -----------------------------------------------------------------------------
   --- C. get original image size

   local image = require("image")

   local info = assert(io.open(self.path .. "info"), "info file not found")
   local line = info:read()
   info:close()

   local first_image_name = line:sub(1, line:find(",")-1)
   local first_image = image.load(self.path .. first_image_name)[{{1}}]

   self:set_resolution(first_image:size(), opt)

   self.preprocessor = Preprocessor(self.height, self.width, opt)
   self.X = torch.Tensor(self.batch_size, self.fmaps,
                         self.preprocessor.height, self.preprocessor.width)

   self:print("going for " .. self.height .. "x" .. self.width)

   full_suffix = self.height .. "x" .. self.width .. suffix

   if opt.one_of_k then
      l_suffix = "_k" .. suffix
   else
      l_suffix = suffix
   end

  -----------------------------------------------------------------------------
   --- D. use cache or get images from files

   if self.use_cache then
      self:print("searching for cached info")
   else
      self:print("not using cache")
   end

   if self.use_cache then
      --- look for classes file
      local classes_file = io.open(cache_path .. "classes" .. suffix)
      if not classes_file then
         self.use_cache = false
      else
         classes_file.close()
      end

      --- look for labels file
      local labels_file = io.open(cache_path .. "labels" .. l_suffix)
      if not labels_file then
         self.use_cache = false
      else
         labels_file.close()
      end

      --- look for stats file
      local stats_file = io.open(cache_path .. "stats" .. full_suffix)
      if not stats_file then self.use_cache = false else stats_file.close() end

      --- look for names or images file
      if self.just_names then
         local names_file = io.open(cache_path .. "names" .. suffix)
         if not names_file then
            self.use_cache = false
         else
            names_file.close()
         end
      else
         local images_file = io.open(cache_path .. "images" .. full_suffix)
         if not images_file then
            self.use_cache = false
         else
            images_file.close()
         end
      end
   end

   if self.use_cache then
      self:print("using cached data")
   else
      self:print("cannot use cached data")
   end

  -----------------------------------------------------------------------------
   --- E. if cache is to be used, load and return tensors

   if self.use_cache then
      self.classes = assert(torch.load(cache_path .. "classes" .. suffix))
      self.labels = assert(torch.load(cache_path .. "labels" .. l_suffix))
      self.stats = assert(torch.load(cache_path .. "stats" .. full_suffix))

      assert(self.stats.N == self.labels:size(1))
      if opt.one_of_k then
         assert(self.labels:nDimension() == 2)
         assert(self.stats.K == self.labels:size(2))
         self.T = torch.Tensor(self.batch_size, self.stats.K)
      else
         assert(self.labels:nDimension() == 1)
         self.T = torch.Tensor(self.batch_size)
      end

      assert(self.stats.height == self.height)
      assert(self.stats.width == self.width)

      if self.just_names then
         --- just names
         self.names = assert(torch.load(cache_path .. "names" .. suffix))
         assert(self.stats.N == #(self.names))
         return
      else
         --- images
         self.images =
            assert(torch.load(cache_path .. "images" .. full_suffix))
         assert(self.stats.N == self.images:size(1))
         assert(self.height == self.images:size(3))
         assert(self.width == self.images:size(4))
         return
      end
   end

   -----------------------------------------------------------------------------
   --- F. no cached data

   local wc_output = io.popen("wc -l " .. self.path .. "info | cut -d\" \" -f1")
   local str = assert(wc_output:read("*a"))
   wc_output:close()
   local lines_no = tonumber(str)

   if opt.limit > 0 then lines_no = math.min(opt.limit, lines_no) end

   if self.just_names then
      self.names = {}
   else
      self.images = torch.Tensor(lines_no, 1, self.height, self.width)
   end

   self.classes = {}

   if opt.one_of_k then
      self.labels = torch.zeros(lines_no, 1)
   else
      self.labels = torch.zeros(lines_no)
   end

   self.stats = {
      ["mean"] = 0, ["stddev"] = 0,
      ["K"] = 0, ["N"] = 0,
      ["height"] = self.height, ["width"] = self.width
   }

   info = assert(io.open(self.path .. "info"))

   while true do
      --- read a new line from info
      local line = info:read()
      if not line or not line:find(",") then break end

      --- progress
      self.stats.N = self.stats.N + 1
      xlua.progress(self.stats.N, lines_no + 1)

      local is_first = true
      for word in line:gmatch("([^,]+)") do -- parse line
         if is_first then -- the first word represents the file name

            local oimg = image.load(self.path .. word)[{{1}}]
            local simg = image.scale(oimg[1], self.width, self.height)

            self.stats.mean = self.stats.mean + simg:mean()
            self.stats.stddev = self.stats.stddev + torch.cmul(simg,simg):mean()

            if self.just_names then
               self.names[#(self.names)+1] = word
            else
               self.images[self.stats.N]:copy(simg)
            end

            is_first = false

         else  -- other words are labels
            if not opt.filter or opt.filter[word] then

               if not self.classes[word] then
                  self.stats.K = self.stats.K + 1
                  self.classes[word] = self.stats.K
                  if opt.one_of_k and self.stats.K > self.labels:size(2) then
                     self.labels = torch.cat(self.labels,
                                             torch.zeros(self.labels:size()),
                                             2)
                  end
               end

               if opt.one_of_k then
                  self.labels[self.stats.N][self.classes[word]] = 1
               else
                  self.labels[self.stats.N] = self.classes[word]
               end

            end -- if filter

         end -- if is_first
      end -- for word

      if opt.limit > 0 and self.stats.N >= opt.limit then break end
   end -- while true
   info:close()

   collectgarbage()

   self.stats.mean = self.stats.mean / self.stats.N
   self.stats.stddev = math.sqrt(self.stats.stddev / self.stats.N -
                                    self.stats.mean * self.stats.mean)

   if opt.one_of_k then
      self.labels = self.labels[{{1, self.stats.N}, {1, self.stats.K}}]
      self.T = torch.Tensor(self.batch_size, self.stats.K)
   else
      self.labels = self.labels[{{1, self.stats.N}}]
      self.T = torch.Tensor(self.batch_size)
   end

   if self.just_names then
      assert(#(self.names) == self.stats.N)
   else
      self.images = self.images[{{1, self.stats.N}, {}, {}}]
   end

   xlua.progress(lines_no + 1, lines_no + 1)

   self:print(self.stats.N .. " images found")
   self:print(self.stats.K .. " different classes")
   self:print("mean: " .. self.stats.mean)
   self:print("std. dev.: " .. self.stats.stddev)

   -----------------------------------------------------------------------------
   --- G. Save to disk and return

   assert(os.execute("mkdir -p " .. cache_path))

   torch.save(cache_path .. "classes" .. suffix, self.classes)
   torch.save(cache_path .. "labels" .. l_suffix, self.labels)
   torch.save(cache_path .. "stats" .. full_suffix, self.stats)

   if self.just_names then
      torch.save(cache_path .. "names" .. suffix, self.names)
      self:print("saved to disk for future use")
   else
      torch.save(cache_path .. "images" .. full_suffix, self.images)
      self:print("saved to disk for future use")
   end

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

function ActivityDataset:update_batch()
   local image = require("image")

   --- Compute the indexes for the next batch of examples
   if self.contiguous then
      self.idx = self.idx or 1      self._idxs = torch.linspace(
         self.idx,
         self.idx + self.batch_size - 1,
         self.batch_size
      ):long()

      self.idx = self.idx + self.batch_size
      if self.idx > (self.stats.N - self.batch_size + 1) then
         self.idx = 1
         self.epoch_finished = true
      else
         self.epoch_finished = false
      end

   else
      self._idxs = torch.rand(self.batch_size):mul(self.stats.N):ceil():long()
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
