local module activity_dataset = {}

function activity_dataset.get_posture_batch(names, path, params)
  local image = require("image");
  path = path or "../data/datasets/posture/" -- default folder
  params = params or {}

  -- take the first image to extract size information
  local first_img = image.load(path .. names[1])
  -- image should be 3 x height x width
  assert(first_img:nDimension() == 3 and first_img:size(1) == 3)

  local original_height = first_img:size(2)
  local original_width = first_img:size(3)

  -- scale images
  local scale = params.scale or 1
  local scaled_height = torch.round(original_height * scale)
  local scaled_width = torch.round(original_width * scale)

  -- crop to given size
  local vert_crop = scaled_height - (params.crop_height or scaled_height)
  local horiz_crop = scaled_width - (params.crop_width or scaled_width)
  local height = scaled_height - vert_crop
  local width = scaled_width - horiz_crop

  -- function to extract (scale and crop) images
  local function extract_image(img, dst)
    if params.scale then img = image.scale(img, scaled_width, scaled_height) end
    local up = torch.random(1 + vert_crop)
    local left = torch.random(1 + horiz_crop)
    image.crop(dst, img, left, up, left + width - 1, up + height - 1)
  end

  -- allocate tensor
  local X = torch.Tensor(#names, height, width)
  -- extract images
  extract_image(first_img, X[i])
  for i = 2,#x do
    local img = image.load(path .. names[i])
    extract_image(img[1], X[i])
  end

  return X
end -- function


function activity_dataset.get_posture_dataset(argv)
  argv = argv or {}
  local path = argv.path or "../data/datasets/posture/"
  local f = io.open(path .. "info")
  if not f then error() end

  -- go through all examples once
  local N = 0
  local classes = {}
  local K = 0
  all_files = {}
  while true do
    -- read a new line from info
    local line = f:read()
    if line == nil then break end
    local is_first = true
    -- parse line
    for word in line:gmatch("([^,]+)") do
      if is_first then
        all_files[N] = word
        is_first = false
      else
        if classes[word] == nil then
          K = K + 1
          classes[word] = K
        end -- if classes_s
      end -- if s_idx
    end -- for s
    N = N + 1
  end -- while true
  print(N .. " lines read; " .. K .. " different classes")
  f:close()
  -- now go again to build target tensor
  local T = torch.zeros(N, K)
  local f = io.open(path .. "info")
  for n = 1, N do
    local line = f:read()
    local is_first = true
    for tag in line:gmatch("([^,]+)") do
      if not is_first then
        T[{n, classes[tag]}] = 1
      else
        is_first = false
      end -- if c_idx > 0
    end -- for tag
  end -- for n
  f:close()
  return all_files, T, classes
end -- function get_posture_dataset


return activity_dataset
