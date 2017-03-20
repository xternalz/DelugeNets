# DelugeNets
Deluge Networks (DelugeNets) are a novel class of neural networks facilitating massive and flexible cross-layer information inflows from preceding layers to succeeding layers. For more technical details of DelugeNets, please refer to the paper: [https://arxiv.org/abs/1611.05552](https://arxiv.org/abs/1611.05552).
<br><br>
# Prerequisite
fb.resnet.torch: [https://github.com/facebook/fb.resnet.torch](https://github.com/facebook/fb.resnet.torch)<br>
optnet: [https://github.com/fmassa/optimize-net](https://github.com/fmassa/optimize-net)
<br><br>
# Further Memory Optimization
Insert these lines to the `M.setup` function in `models/init.lua`:
```
local cache = {}
model:apply(function(m)
  local moduleType = torch.type(m)
  if moduleType == 'nn.CrossLayerDepthwiseConvolution' then
     if cache['cldc-o'] == nil then
        cache['cldc-o'] = torch.CudaStorage(1)
     end
     m.SBatchNorm.output = torch.CudaTensor(cache['cldc-o'], 1, 0)
  end
end)
```
<br>


## Training Divergence
There are occassional divergence issues when training the networks. For stability, please set cuDNN to deterministic mode:
```
-cudnn deterministic
```
<br><br>
## CIFAR-10 & CIFAR-100
| Model              | #Params | CIFAR-10 error | CIFAR-100 error |
|--------------------|--------:|---------------:|----------------:|
| DelugeNet-146      | 6.7M    | 3.98           | 19.72           |
| DelugeNet-218      | 10.0M   | 3.88           | 19.31           |
| Wide-DelugeNet-146 | 20.2M   | 3.76           | 19.02           |

#### How to run
1. Make sure that [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch) runs well on your machine.<br><br>
2. Copy `delugenet.lua` and `CrossLayerDepthwiseConvolution.lua` to `fb.resnet.torch/models`.<br><br>
3. Modify the learning rate schedule codes in `train.lua` for CIFAR-10 and CIFAR-100 to:<br>
`decay = epoch >= 225 and 2 or epoch >= 150 and 1 or 0`<br><br>
4. Train DelugeNets:


DelugeNet-146 on CIFAR-10<br>
`th main.lua -batchSize 64 -nEpochs 300 -optnet true -netType delugenet -dataset cifar10 -depth 146.1`<br><br>
DelugeNet-146 on CIFAR-100<br>
`th main.lua -batchSize 64 -nEpochs 300 -optnet true -netType delugenet -dataset cifar100 -depth 146.1`<br><br>
DelugeNet-218 on CIFAR-10<br>
`th main.lua -batchSize 64 -nEpochs 300 -optnet true -netType delugenet -dataset cifar10 -depth 218`<br><br>
DelugeNet-218 on CIFAR-100<br>
`th main.lua -batchSize 64 -nEpochs 300 -optnet true -netType delugenet -dataset cifar100 -depth 218`<br><br>
Wide-DelugeNet-146 on CIFAR-10<br>
`th main.lua -batchSize 64 -nEpochs 300 -optnet true -netType delugenet -dataset cifar10 -depth 146.2`<br><br>
Wide-DelugeNet-146 on CIFAR-100<br>
`th main.lua -batchSize 64 -nEpochs 300 -optnet true -netType delugenet -dataset cifar100 -depth 146.2`<br>
<br><br>
## ImageNet
Validation errors (single-crop 224x224)

| Model              | #Params |  GigaFLOPs  |   top-1 error  |   top-5 error   |
|--------------------|--------:|------------:|---------------:|----------------:|
| DelugeNet-92       | 43.4M   | 11.8        | 22.05           | 6.03            |
| DelugeNet-104      | 51.4M   | 13.2        | 21.86           | 5.98            |
| DelugeNet-122      | 63.6M   | 15.2        | 21.53           | 5.86            |

#### How to run
1. Follow the guide at [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch) repository on how to set up ImageNet dataset.<br><br>
2. To train ImageNet-based models, you need a minimum of 8 TITAN X GPUs:


DelugeNet-92<br>
`th main.lua -nGPU 8 -batchSize 256 -nEpochs 100 -optnet true -netType delugenet -dataset imagenet -data IMAGENET_PATH -nThreads 13 -depth 92`<br><br>
DelugeNet-104<br>
`th main.lua -nGPU 8 -batchSize 256 -nEpochs 100 -optnet true -netType delugenet -dataset imagenet -data IMAGENET_PATH -nThreads 13 -depth 104`<br><br>
DelugeNet-122<br>
`th main.lua -nGPU 8 -batchSize 256 -nEpochs 100 -optnet true -netType delugenet -dataset imagenet -data IMAGENET_PATH -nThreads 13 -depth 122`<br>

Replace `IMAGENET_PATH` with ImageNet dataset path.
