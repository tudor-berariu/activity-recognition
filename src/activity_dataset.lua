local module activity_dataset = {}
require("torch")

-- TODO: generalize for activities

--[[
   --------------------
   ---- Usage examples:

   -- (1). Getting a tensor with the images

   local ds = require("activity_dataset")
   local args = {
      ["verbose"] = true,
      ["scale"] = 0.25,
      ["just_names"] = false,
      ["limit"] = 2000
   }
   images, classes, labels, stats = ds.get_posture_dataset()

   -- (2). Getting just the file names and small batches of images on demand

   local ds = require("activity_dataset")
   local args = {
      ["verbose"] = true,
      ["scale"] = 0.25,
      ["just_names"] = true,
      ["limit"] = 2000
   }
   names, classes, labels, stats = ds.get_posture_dataset()
   args["size"] = 100
   X, T = ds.generate_batch(names, labels, args)

--]]

activity_dataset.default_path = "../data/datasets/posture/"

--------------------------------------------------------------------------------

--[[

   This function returns the desired height and width for the images.
   It modifies the original height and width using information supplied in args.

--]]

function activity_dataset.get_resolution(height, width, args)
   -- compute the target size
   if args.scale then
      height = torch.round(height * args.scale)
      width = torch.round(width * args.scale)
   elseif args.height and args.width then
      height = args.height
      width = args.width
   elseif args.height then
      width = torch.round(width * args.height / height)
      height = args.height
   elseif args.width then
      height = torch.round(height * args.width / width)
      width = args.width
   end
   return height, width
end

--[[

   This function returns a random mini batch of size B.
   It's supposed to be used when it's impossible to load the full dataset in
   memory.

   args:
    - size       : long    , default: 100
    - contiguous : boolean , default: false
    - start      : long   (works only for contiguous mini-batches)
--]]

function activity_dataset.generate_batch(names, labels, args)
   local image = require("image")
   args = args or {}
   local path = args.path or activity_dataset.default_path
   if path:sub(path:len()) ~= "/" then path = path .. "/" end

   local N = #names                               -- total number of examples
   local B = args.size or torch.min(100, N)       -- batch size (default: 100)
   local idx
   if args.contiguous then
      local start = args.start or torch.random(N-B+1)
      idx = torch.linspace(start, start+B-1, B):long()
   else
      idx = torch.rand(B):mul(N):ceil():long() -- generate random indices
   end
   local batch_names = {}
   for i = 1, B do batch_names[i] = names[idx[i]] end -- get image names

   -- allocate tensor
   local first_img = image.load(path .. names[1])
   local height = first_img:size(2)
   local width = first_img:size(3)
   height, width = activity_dataset.get_resolution(height, width, args)

   local X = torch.Tensor(#names, height, width)
   for i = 1,X:size(1) do
      local o_img = image.load(path .. names[i])
      X[i] = image.scale(o_img[1], width, height)
   end -- for i

   local T = labels:index(1, idx)
   return X, T
end


--[[
   get_posture_dataset(args)
    - path       : string,  default: "../data/datasets/posture/"]
    - scale      : number,  default: 1
    - height     : number
    - width      : number
    - use_cache  : boolean, default: true
    - just_names : boolean, default: false
    - verbose    : boolean, default: false
    - one_of_k   : boolean, default: false
    - filter     : table,   default: nil
--]]

function activity_dataset.get_posture_dataset(args)
   ---------------------------------------------------------
   local names, images, labels, targets, stats
   args = args or {}

   ---------------------------------------------------------
   -- check if we're in debug mode
   verbose = args.verbose or false

   ---------------------------------------------------------
   -- the path to the posture dataset
   local path = args.path or "../data/datasets/posture/"
   if path:sub(path:len()) ~= "/" then path = path .. "/" end
   local cache_path = path .. ".torch_cache/"

   ---------------------------------------------------------
   -- set the scale for the dataset

   -- fetch one image to see the original size
   local image = require("image")

   local info = io.open(path .. "info")
   local line = info:read()
   local first_image = line:sub(1, line:find(",")-1)
   info:close()
   local img = image.load(path .. first_image)
   local height = img:size(2)
   local width = img:size(3)
   height, width = activity_dataset.get_resolution(height, width, args)

   if verbose then
      print("<dataset> going for " .. height .. "x" .. width)
   end
   ---------------------------------------------------------
   -- use cache or get images from files
   local use_cache = args.use_cache or true
   local suffix, full_suffix, lsuffix
   if args.limit then
      suffix = "_" .. args.limit .. ".dat"
      full_suffix = height .. "x" .. width .. "_" .. args.limit .. ".dat"
   else
      suffix = ".dat"
      full_suffix = height .. "x" .. width .. ".dat"
   end
   if args.one_of_k then
      lsuffix = "_k" .. suffix
   else
      lsuffix = suffix
   end

   if verbose then
      if use_cache then
         print("<dataset> searching for cached info")
      else
         print("<dataset> not using cache")
      end
   end

   if use_cache then
      -- check if serialized information exists
      local classes_file = io.open(cache_path .. "classes" .. suffix)
      if not classes_file then use_cache = false else classes_file.close() end

      local labels_file = io.open(cache_path .. "labels" .. lsuffix)
      if not labels_file then use_cache = false else labels_file.close() end

      local stats_file = io.open(cache_path .. "stats" .. full_suffix)
      if not stats_file then use_cache = false else stats_file.close() end

      if args.just_names then
         local names_file = io.open(cache_path .. "names" .. suffix)
         if not names_file then use_cache = false else names_file.close() end
      else
         local images_file = io.open(cache_path .. "images" .. full_suffix)
         if not images_file then use_cache = false else images_file.close() end
      end
   end

   if verbose then
      if use_cache then
         print("<dataset> using cache")
      else
         print("<dataset> not using cache")
      end
   end

   if use_cache then
      classes = assert(torch.load(cache_path .. "classes" .. suffix))
      labels = assert(torch.load(cache_path .. "labels" .. lsuffix))
      stats = assert(torch.load(cache_path .. "stats" .. full_suffix))

      assert(stats.N == labels:size(1))
      if args.one_of_k then
         assert(labels:nDimension() == 1)
         assert(stats.K == labels:size(2))
      else
         assert(labels:nDimension() == 1)
      end

      if args.just_names then
         names = assert(torch.load(cache_path .. "names" .. suffix))
         assert(stats.N == #names)
         return names, classes, labels, stats
      else
         images = assert(torch.load(cache_path .. "images" .. full_suffix))
         assert(stats.N == images:size(1))
         assert(height == images:size(2))
         assert(width == images:size(3))
         return images, classes, labels, stats
      end
   end

   ---------------------------------------------------------
   -- try to estimate the size of the dataset
   local wc_output = io.popen("wc -l " .. path .. "info | cut -d\" \" -f1")
   local str = assert(wc_output:read("*a"))
   wc_output:close()
   local lines_no = tonumber(str)
   if args.limit then lines_no = math.min(args.limit, lines_no) end

   -- prepare data structures
   if args.just_names then
      names = {}                                     -- file names
   else
      images = torch.Tensor(lines_no, height, width) -- tensor with all images
   end
   classes = {}                          -- class correspondence
   if args.one_of_k then
      labels = torch.Tensor(lines_no, 1)
   else
      labels = torch.Tensor(lines_no)
   end
   stats = {["mean"] = 0, ["stddev"] = 0, ["K"] = 0, ["N"] = 0} -- stats

   info = assert(io.open(path .. "info"))

   while true do
      -- read a new line from info
      local line = info:read()
      if not line or not line:find(",") then break end

      -- progress
      stats.N = stats.N + 1
      xlua.progress(stats.N, lines_no + 1)

      local is_first = true
      for word in line:gmatch("([^,]+)") do -- parse line
         if is_first then -- the first word represents the file name
            local o_img = image.load(path .. word)
            local s_img = image.scale(o_img[1], width, height)
            stats.mean = stats.mean + s_img:mean()
            stats.stddev = stats.stddev + torch.cmul(s_img, s_img):mean()

            if args.just_names then
               names[#names+1] = word
            else
               images[stats.N] = s_img
            end

            is_first = false
         else  -- other words are labels
            if not classes[word] and (not args.filter or args.filter[word]) then
               stats.K = stats.K + 1
               classes[word] = stats.K
               if args.one_of_k and stats.K > labels:size(2) then
                  labels = torch.cat(labels, torch.zeros(labels:size()), 2)
               end
            end
            if not args.filter or args.filter[word] then
               if args.one_of_k then
                  labels[stats.N][classes[word]] = 1
               else
                  labels[stats.N] = classes[word]
               end
            end
         end -- if is_first
      end -- for word

      if args.limit and stats.N >= args.limit then break end
   end -- while true
   info:close()

   stats.mean = stats.mean / stats.N
   stats.stddev = math.sqrt(stats.stddev / stats.N - stats.mean * stats.mean)

   if args.one_of_k then
      labels = labels[{{1, stats.N}, {1, stats.K}}]
   else
      labels = labels[{{1, stats.N}}]
   end
   if args.just_names then
      assert(#names == stats.N)
   else
      images = images[{{1, stats.N}, {}, {}}]
   end

   xlua.progress(lines_no + 1, lines_no + 1)

   if verbose then
      print(stats.N .. " images found; " .. stats.K .. " different classes")
      print("mean: ", stats.mean .. "; std. dev.: " .. stats.stddev)
   end

   -- Save in torch cache and return

   assert(os.execute("mkdir -p " .. cache_path))
   torch.save(cache_path .. "classes" .. suffix, classes)
   torch.save(cache_path .. "labels" .. lsuffix, labels)
   torch.save(cache_path .. "stats" .. full_suffix, stats)
   if args.just_names then
      torch.save(cache_path .. "names" .. suffix, names)
      if verbose then print("<dataset> saved to torch cache; done") end
      return names, classes, labels, stats
   else
      torch.save(cache_path .. "images" .. full_suffix, images)
      if verbose then print("<dataset> saved to torch cache; done") end
      return images, classes, labels, stats
   end
end -- function get_posture_dataset

return activity_dataset
