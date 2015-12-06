local class = require("class")
Preprocessor = class("Preprocessor")

function Preprocessor:__init(args)
   args = args or {}

   local in_height = args.heigth or 96
   local in_width = args.widht or 128

   -- crop to given size (given crop size should be less or equal than original)
   self.vert_crop = in_height - (args.crop_height or 5)
   self.horiz_crop = in_width - (args.crop_width or 37)
   self.height = in_height - self.vert_crop
   self.width = in_width - self.horiz_crop

   -- flip?
   self.flip = args.flip or false
end

   -- function to extract (scale and crop) images

function Preprocessor:process_batch(imgs)
   local image = require("image")
   local up, left, img2, oimgs

   oimgs = torch.Tensor(imgs:size(1), imgs:size(2), self.height, self.width)
   for i = 1, imgs:size(1) do
      -- crop
      up = torch.random(1 + self.vert_crop) - 1
      left = torch.random(1 + self.horiz_crop) - 1
      img2 = image.crop(imgs[i], left, up, left+self.width, up+self.height)
      -- flip
      if self.flip and math.random() > 0.5 then
         oimgs[i] = image.flip(img2, 2)
      else
         oimgs[i] = img2
      end -- flip
   end -- for i
   return oimgs
end

return true
