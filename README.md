# Neural_Network_From_Scratch
A simple neural network using only python + numpy.

- This is a fully-connected network with one hidden layer.  
- The layout is **Input-784 -> FC-2048 -> ReLU -> FC-10 -> Sigmoid**, (where FC-Num stands for fully-connected layer with Num neurons)  
- The network uses L2 loss : $\frac{1}{N}*\sum_i{\\;\(\text{target}_i - \text{predicted}_i\)^2}$

## Results on MNIST numbers dataset:
- Accuracy and loss:
> 0 | Test_set: loss : 0.0225, acc : 88.21% | Train_set: loss : 0.0234, acc : 87.49%  
5 | Test_set: loss : 0.0225, acc : 88.25% | Train_set: loss : 0.0234, acc : 87.477%

- First 400 batches:
![Manual Neural Network](https://i.imgur.com/5dkDZIO.png)


## Mathematical formulation
With $\underline{Z^0}$ as the input vector, the network can be written mathematically as a sequence:  
  
$$\Large\underline{Z^1}\\; =\\; \mathbf{W^1} * \underline{Z^0}\\;+\\;\underline{B^1},$$  
  
$$\Large\underline{A^1}\\; =\\;ReLU\(\\;\underline{Z^1}\\;\)\\;\\;\\;\\;\\;\\;,$$   
  
$$\Large\underline{Z^2}\\; =\\; \mathbf{W^2} * \underline{Z^1}\\;+\\;\underline{B^2},$$  
  
$$\Large\underline{A^2}\\; =\\;Sigmoid\(\\;\underline{Z^2}\\;\)\\;\\;,$$  
  
where $\underline{Z^i}$ is the logits vector in layer $i$, $\underline{A^i}$ is the activation vector, $\mathbf{W^i}$ is the weight matrix and $\underline{B^i}$ is the bias vector  

The gradients are calculated from:  
  
$$\Large\frac{dC_i}{d\underline{Z^2_i}}\\;=\\;\frac{dC_i}{d\underline{A^2_i}}*\frac{d\underline{A^2_i}}{d\underline{Z^2_i}}\\;=\\;\frac{dC_i}{d\underline{B^2_i}}\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;,$$  
   
$$\Large\frac{dC_i}{d\mathbf{W}^2_{ij}}\\;=\\;\frac{dC_i}{d\underline{Z^2_i}}*\frac{d\underline{Z^2_i}}{d\mathbf{W}^2_{ij}}\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;,$$

$$\Large\frac{dC}{d\underline{Z^1_i}} \\;=\\; \sum_k{ \( \frac{dC_i}{d\underline{Z^2_i}} * \mathbf{W}^2_{ki}* \frac{d\underline{A^1_i}}{d\underline{Z^1_i}}\)}\\;=\\;\frac{dC_i}{d\underline{B^1_i}},$$  

$$\Large\frac{dC_i}{d\mathbf{W}^1_{ij}}\\;=\\;\frac{dC_i}{d\underline{Z^1_i}}*\frac{d\underline{Z^1_i}}{d\mathbf{W}^1_{ij}}\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;\\;,$$  

  
