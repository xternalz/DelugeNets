require 'nn'

-- Cross-layer Depthwise Convolutional Layer

local CrossLayerDepthwiseConvolution, parent = torch.class('nn.CrossLayerDepthwiseConvolution', 'nn.Container')

function CrossLayerDepthwiseConvolution:__init(nInputPlane, nLayers)
  parent.__init(self)
  self.nLayers = nLayers
  self.nInputPlane = nInputPlane
  self.running_mean = torch.zeros(nInputPlane,nLayers)
  self.running_var = torch.ones(nInputPlane,nLayers)
  self.weight = torch.Tensor(nInputPlane,nLayers)
  self.bias = torch.Tensor(nInputPlane,nLayers)
  self.gradWeight = torch.Tensor(nInputPlane,nLayers)
  self.gradBias = torch.Tensor(nInputPlane,nLayers)
  self:reset()
  self.SBatchNorm = nn.SpatialBatchNormalization(nInputPlane)
  self.modules = {self.SBatchNorm}
  self.train = true
  self.gradInput = {}
  for i = 1, nLayers do
    self.gradInput[i] = torch.Tensor()
  end
end

function CrossLayerDepthwiseConvolution:reset()
  self.weight:fill(1/self.nLayers)
  self.bias:zero()
  self.running_mean:zero()
  self.running_var:fill(1)
end

function CrossLayerDepthwiseConvolution:training()
  self.SBatchNorm:training()
  self.train = true
end

function CrossLayerDepthwiseConvolution:evaluate()
  self.SBatchNorm:evaluate()
  self.train = false
end

function CrossLayerDepthwiseConvolution:updateOutput(input)
  assert(torch.type(input) == 'table', 'Input must be a table')
  self.output:resizeAs(input[1]):zero()
  self.save_mean = self.save_mean or self.running_mean.new()
  self.save_mean:resize(self.nInputPlane,self.nLayers)
  self.save_std = self.save_std or self.running_var.new()
  self.save_std:resize(self.nInputPlane,self.nLayers)
  for i = 1, self.nLayers do
    self.SBatchNorm.running_mean:copy(self.running_mean:select(2,i))
    self.SBatchNorm.running_var:copy(self.running_var:select(2,i))
    self.SBatchNorm.weight:copy(self.weight:select(2,i))
    self.SBatchNorm.bias:copy(self.bias:select(2,i))
    self.output:add(self.SBatchNorm:forward(input[i]))
    if self.train then
      self.running_mean:select(2,i):copy(self.SBatchNorm.running_mean)
      self.running_var:select(2,i):copy(self.SBatchNorm.running_var)
      self.save_mean:select(2,i):copy(self.SBatchNorm.save_mean)
      self.save_std:select(2,i):copy(self.SBatchNorm.save_std)
    end
  end
  return self.output
end

function CrossLayerDepthwiseConvolution:backward(input, gradOutput, scale)
  scale = scale or 1
  assert(torch.type(input) == 'table', 'Input must be a table')
  for i = 1, self.nLayers do
    self.gradInput[i] = self.gradInput[i] or input[i].new()
    self.gradInput[i]:resizeAs(input[i]):fill(0)
    self.SBatchNorm.weight:copy(self.weight:select(2,i))
    self.SBatchNorm.bias:copy(self.bias:select(2,i))
    self.SBatchNorm.save_mean:copy(self.save_mean:select(2,i))
    self.SBatchNorm.save_std:copy(self.save_std:select(2,i))
    self.gradInput[i]:copy(self.SBatchNorm:backward(input[i],gradOutput,scale))
    self.gradWeight:select(2,i):add(self.SBatchNorm.gradWeight)
    self.gradBias:select(2,i):add(self.SBatchNorm.gradBias)
    self.SBatchNorm:zeroGradParameters()
  end
  return self.gradInput
end

function CrossLayerDepthwiseConvolution:parameters()
  return {self.weight, self.bias}, {self.gradWeight, self.gradBias}
end

function CrossLayerDepthwiseConvolution:zeroGradParameters()
  self.gradWeight:zero()
  self.gradBias:zero()
end
