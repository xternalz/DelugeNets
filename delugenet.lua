--
--  DelugeNet
--

local nn = require 'nn'
require 'cunn'
require 'models/CrossLayerDepthwiseConvolution'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = cudnn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
   local depth = opt.depth
   local iChannels, nLayers = nil, 1

   -- Bottleneck composite layer
   local function bottleneck(n, stride, type)
      local nInputPlane = iChannels
      iChannels = n * 2

      local block = nn.Sequential()

      local s = nn.Sequential()
      if type == 'both_preact' or type ~= 'no_preact' then
         if nLayers == 1 then
            s:add(SBatchNorm(nInputPlane))
            s:add(ReLU(true))
         elseif nLayers > 1 then
            s:add(nn.CrossLayerDepthwiseConvolution(nInputPlane, nLayers))
            s:add(SBatchNorm(nInputPlane))
            -- block transition
            if type ~= 'no_preact' and type ~= nil then
               nLayers = 1
               s:add(Convolution(nInputPlane,n*2,3,3,stride,stride,1,1))
               return block:add(s)
            end
            s:add(ReLU(true))
         end
      end
      s:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n*2,1,1,1,1,0,0))

      if type == nil then
         nLayers = nLayers + 1
         return block
            :add(nn.ConcatTable()
               :add(s)
               :add(nn.Identity()))
            :add(nn.FlattenTable())
      else
         nLayers = 1
         return block:add(s)
      end
   end

   -- Network block
   local function block(features, count, stride, type)
      local s = nn.Sequential()
      if count < 1 then
        return s
      end
      s:add(bottleneck(features, stride,
                  type == 'first' and 'no_preact' or 'both_preact'))
      for i=2,count do
         s:add(bottleneck(features, 1, nil))
      end
      return s
   end

   local model = nn.Sequential()
   if opt.dataset == 'cifar10' or opt.dataset == 'cifar100' then
      -- Configurations:
      --  # compositeLayer, # features, compositeLayer type
      local cfg = {
         [146.1]  = {8,1},
         [146.2] = {8,1.75},
         [218] = {12,1}
      }

      assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
      local n, width = table.unpack(cfg[depth])
      iChannels = 32*width
      print(' | DelugeNet-' .. depth .. ' CIFAR-10')

      -- The CIFAR model
      model:add(Convolution(3,32*width,3,3,1,1,1,1))
      model:add(block(32*width, n, 1, nil))
      model:add(block(64*width, n*2+1, 2, nil))
      model:add(block(128*width, n*3+1, 2, nil))
      model:add(nn.CrossLayerDepthwiseConvolution(iChannels, nLayers))
      model:add(SBatchNorm(iChannels))
      model:add(ReLU(true))
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(iChannels):setNumInputDims(3))
      model:add(nn.Linear(iChannels, opt.dataset == 'cifar10' and 10 or 100))
   else
      error('invalid dataset: ' .. opt.dataset)
   end

   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         if v.weight then
            v.weight:fill(1)
            v.bias:zero()
         end
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:cuda()

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   return model
end

return createModel
