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

function Preprocessor:__init(height, width, opt)
   self.verbose = opt.verbose
   -----------------------------------------------------------------------------
   --- A. Crop
   -----------------------------------------------------------------------------

   if opt.vertCrop > 0 then                               -- fixed vertical crop
      self.vertCrop = opt.vertCrop
   else                                                   -- vertical crop ratio
      self.vertCrop = torch.floor(height * opt.vertCropRatio)
   end

   self.height = height - self.vertCrop

   if opt.horizCrop > 0 then
      self.horizCrop = opt.horizCrop
   else
      self.horizCrop = torch.floor(width * opt.horizCropRatio)
   end

   self.width = width - self.horizCrop

   -----------------------------------------------------------------------------
   --- B. Flip
   -----------------------------------------------------------------------------
   self.flip = opt.flip or false                              -- horizontal flip

   self:print("Final size: " .. self.height .. "x" .. self.width)
   self:print("Flip: " .. tostring(self.flip))
 end

function Preprocessor:processImage(destImg, srcImg)
   -- load needed modules
   local image = require("image")

   if self.vertCrop or self.horizCrop then
      local up = torch.random(1 + self.vertCrop) - 1
      local left = torch.random(1 + self.horizCrop) - 1

      if self.flip and (math.random() > 0.5) then
         image.hflip(
            destImg,
            image.crop(srcImg, left, up, left + self.width, up + self.height)
         )
      else
         image.crop(
            destImg, srcImg,
            left, up,
            left + self.width, up + self.height
         )
      end -- if flip
   else
      if self.flip and math.random() > 0.5 then
         image.hflip(destImg, srcImg)
      else
         destImg:copy(srcImg)
      end -- if flip
   end -- if crop
end

function Preprocessor:print(message)
   if self.verbose then
      print("[preprocessor] " .. message)
   end
end

return Preprocessor
