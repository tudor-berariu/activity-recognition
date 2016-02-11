local module util = {}

--[[

   Various functions that do not deserve to go in a separate module yet.

--]]

--[[

   Reverse a table: keys become values and values become keys.

--]]

function util.reverse_table(t)
   local u = {}
   for k, v in pairs(t) do u[v] = k end
   return u
end

--[[

   Shallow comparison of tables

--]]

function util.shallow_eq(t1, t2)
   for k1, v1 in pairs(t1) do
      if not t2[k1] or t2[k1] ~= v1 then return false end
   end
   for k2, v2 in pairs(t2) do
      if not t1[k2] or t1[k2] ~= v2 then return false end
   end
   return true
end

--[[

   Deep comparison of tables.

--]]

function util.eq(t1, t2, depth)
   if type(t1) ~= type(t2) then return false end
   if type(t1) ~= "table" then return t1 == t2 end
   for k1, v1 in pairs(t1) do
      if not t2[k1] or not util.eq(t2[k1], v1) then return false end
   end
   for k2, v2 in pairs(t2) do
      if not t1[k2] or not util.eq(t1[k2], v2) then return false end
   end
   return true
end

assert(util.shallow_eq(
          {["a"] = 2, ["b"] = 3},
          util.reverse_table(util.reverse_table({["b"] = 3, ["a"] = 2}))
))

--[[

   Round numbers to a specific number of decimals

--]]

function util.round(n, d)
   return torch.round(n * math.pow(10, d)) / math.pow(10, d)
end

return util
