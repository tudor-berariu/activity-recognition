--------------------------------------------------------------------------------
--- 1. Load required libraries

local class = require("class")

--------------------------------------------------------------------------------
--- 2. Define the preprocessor class
---
--- The preprocessor applies individual transformations to each image.
--- Other preprocessing like scaling or standardization should be applied to all
--- images somewhere else.

Preprocessor = class("Preprocessor")

function Preprocessor:__init(in_height, in_width, opt)
   self.verbose = opt.verbose
   -----------------------------------------------------------------------------
   --- A. Crop

   if opt.vert_crop then                                  -- fixed vertical crop
      self.vert_crop = opt.vert_crop
   else                                                   -- vertical crop ratio
      self.vert_crop = torch.floor(in_height * opt.vert_crop_ratio)
   end

   self.height = in_height - self.vert_crop

   if opt.horiz_crop then
      self.horiz_crop = opt.horiz_crop
   else
      self.horiz_crop = torch.floor(in_width * opt.horiz_crop_ratio)
   end

   self.width = in_width - self.horiz_crop

   self.flip = opt.flip or false                              -- horizontal flip

   self:print("Final size: " .. self.height .. "x" .. self.width)
   self:print("Flip: " .. tostring(self.flip))
 end

function Preprocessor:process_image(dest_img, src_img)
   -- load needed modules
   local image = require("image")

   if self.vert_crop or self.horiz_crop then
      local up = torch.random(1 + self.vert_crop) - 1
      local left = torch.random(1 + self.horiz_crop) - 1

      if self.flip and (math.random() > 0.5) then
         image.hflip(
            dest_img,
            image.crop(src_img, left, up, left + self.width, up + self.height)
         )
      else
         image.crop(
            dest_img, src_img,
            left, up,
            left + self.width, up + self.height
         )
      end -- if flip
   else
      if self.flip and math.random() > 0.5 then
         image.hflip(dest_img, src_img)
      else
         dest_img:copy(src_img)
      end -- if flip
   end -- if crop
end

function Preprocessor:print(message)
   if self.verbose then
      print("[preprocessor] " .. message)
   end
end



return Preprocessor
