--[[
   This script verifies that reading the dataset works.
   Run with qt support or comment the last display command.
--]]

do
   local function my_print(s) print("<test-activity> " .. s) end
   local ds = require("activity_dataset")
   -- setup args
   local args = {
      ["verbose"] = true,
      ["scale"] = 0.25,
      ["just_names"] = true,
      ["limit"] = 200,
      ["size"] = 1,
      ["filter"] = {["standing"] = true, ["sitting"] = true}
   }
   local idx = torch.random(args.limit)
   -- get by name
   local names1, classes1, labels1, stats1
   names1, classes1, labels1, stats1 = ds.get_posture_dataset(args)
   local X1, T1
   X1, T1 = ds.generate_batch({names1[idx]}, labels1[{{idx,idx}}], args)
   my_print("Done (1/2)")
   -- get images
   local images2, classes2, labels2, stats2
   args["just_names"] = false
   images2, classes2, labels2, stats2 = ds.get_posture_dataset(args)
   my_print("Done (2/2)")
   -- compare results
   local diff = X1 - images2[idx]
   if diff:sum() < 0.00001 then
      my_print("Ok!")
   else
      my_print("Error!")
   end
   -- show one image
   local util = require("util")
   captions = util.reverse_table(classes2)

   image.display{image=images2[idx], win=win_input, zoom=4,
                 legend=captions[labels2[idx]]}
end
