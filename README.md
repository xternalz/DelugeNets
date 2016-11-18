# DelugeNets
Deluge Networks (DelugeNets) are a novel class of neural networks facilitating massive and flexible cross-layer information inflows from preceding layers to succeeding layers. For more technical details of DelugeNets, please refer to the paper: [https://arxiv.org/abs/1611.05552](https://arxiv.org/abs/1611.05552).
<br><br>
# Prerequisite
fb.resnet.torch: [https://github.com/facebook/fb.resnet.torch](https://github.com/facebook/fb.resnet.torch)<br>
optnet: [https://github.com/fmassa/optimize-net](https://github.com/fmassa/optimize-net)
<br><br>
# CIFAR-10 & CIFAR-100
| Model              | #Params | CIFAR-10 error | CIFAR-100 error |
|--------------------|--------:|---------------:|----------------:|
| DelugeNet-146      | 6.7M    | 3.98           | 19.72           |
| DelugeNet-218      | 10.0M   | 3.88           | 19.31           |
| Wide-DelugeNet-146 | 20.2M   | 3.76           | 19.02           |

####How to run
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
  
<br>
# ImageNet
coming soon
