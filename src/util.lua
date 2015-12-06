local module util = {}

function util.reverse_table(t)
   u = {}
   for k, v in pairs(t) do u[v] = k end
   return u
end

return util
