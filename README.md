## <center>VGG</center>
###### <center>2018, OCT 16</center>

[VGG paper ](https://arxiv.org/pdf/1409.1556.pdf)

本人基于tensorflow搭了一个的VGG16跑了cifar-10，epoch=30，Accuracy已经有近0.7，只为通过代码更好的理解VGG，有兴趣的话点击下方查看代码[VGG - CIFAR-10](https://github.com/WeiYangBin/Notes-Deep-Learning/blob/master/cifar-10%20-%20VGG.ipynb)

看过CNN看VGG给人的感觉还是比较容易理解，比较起CNN我认为主要在两个方面有了变化，一个是层数变深，另一个变化就是比起CNN的7 * 7, 11 * 11,VGG全部用的都是3 * 3的filter/kernel

看完paper会发现作者举了从A,B,C一直到E的六种网络架构,越来越deep，因为用的比较多的还是VGG16和VGG19 (这里的16和19代表的是层数)，然后又因为VGG16和VGG19的accuracy差的不会太多，所以这篇文章主要谈的是VGG16

paper中的图片是这样的

![image](https://github.com/WeiYangBin/Notes-Deep-Learning/blob/master/Picture/VGG%E6%9E%B6%E6%9E%84.png)

换一种角度来看可能会更容易理解

![image](https://github.com/WeiYangBin/Notes-Deep-Learning/blob/master/Picture/VGG.jpg)

##### Step1:

首先输入图片
```
Input : Image_size = [224 * 224 * 3], [ height, weight, channel]
```

计算卷积后的图片大小变化的公式如下
```math
[(h + 2p - f )/ s + 1  ,  (w + 2p - f )/ s + 1]

h :height  

w:weight

p:padding

f:filter

s = strides
```

##### Step 2:

对输入的图片进行**第一个卷积块**

其中包含了两个卷积层，利用[3 * 3 *3]的核，对图片进行卷积,之后对卷积后的图片进行activation，normalization,和pooling,
activation,normalization不会改变图像大小故不变
```
Convolution：filter/kernel = [3 * 3 * 3], [ height, weight, channel] 
filter/kernel 个数： 64
strides = 1
padding = 'SAME'  
padding = 1

#利用上面公式计算(224 + 2 * 1 - 3) / 1 + 1  = 224 
conv1_1 = [224 * 224 * 64]
conv1_1 = relu(conv1_1) #size = [224 * 224 * 64]
conv1_2 = [224 * 224 * 64]
conv1_2 = relu(conv1_2) #size = [224 * 224 * 64]

Max_pool : kernel_size = (2 * 2), strides = 2
padding = 'VALID' 
padding = 0
#利用上面公式计算(224 - 2) / 2 + 1  = 112
pool1 = [112 * 112 * 64]
```

##### Step 3:

对输入的图片进行**第二个卷积块**
其中包含了两个卷积层

```
Convolution：filter/kernel = [3 * 3 * 64], [ height, weight, channel] 
filter/kernel 个数： 128
strides = 1
padding = 'SAME'  
padding = 1

#利用上面公式计算(112 + 2 * 1 - 3) / 1 + 1  = 112 
conv2_1 = [112 * 112 * 128]
conv2_1 = relu(conv2_1) #size = [112 * 112 * 128]
conv2_2 = [112 * 112 * 128]
conv2_2 = relu(conv2_2) #size = [112 * 112 * 128]

Max_pool : kernel_size = (2 * 2), strides = 2
padding = 'VALID' 
padding = 0
#利用上面公式计算(112 - 2) / 2 + 1  = 56
pool2 = [56 * 56 * 128]
```

##### Step 4:

对输入的图片进行**第三个卷积块**
其中包含了三个卷积层

```
Convolution：filter/kernel = [3 * 3 * 128], [ height, weight, channel] 
filter/kernel 个数： 256
strides = 1
padding = 'SAME'  
padding = 1

#利用上面公式计算(56 + 2 * 1 - 3) / 1 + 1  = 56
conv3_1 = [56 * 56 * 256]
conv3_1 = relu(conv3_1) #size = [56 * 56 * 256]
conv3_2 = [112 * 112 * 128]
conv3_2 = relu(conv3_2) #size = [56 * 56 * 256]
conv3_3 = [112 * 112 * 128]
conv3_3 = relu(conv3_3) #size = [56 * 56 * 256]

Max_pool : kernel_size = (2 * 2), strides = 2
padding = 'VALID' 
padding = 0
#利用上面公式计算(56 - 2) / 2 + 1  = 28
pool3 = [28 * 28 * 256]
```

##### Step 5:

对输入的图片进行**第四个卷积块**
其中包含了三个卷积层
```
Convolution：filter/kernel = [3 * 3 * 512], [ height, weight, channel] 
filter/kernel 个数： 512
strides = 1
padding = 'SAME'  
padding = 1

#利用上面公式计算(28 + 2 * 1 - 3) / 1 + 1  = 28
conv4_1 = [28 * 28 * 512]
conv4_1 = relu(conv4_1) #size = [28 * 28 * 512]
conv4_2 = [28 * 28 * 512]
conv4_2 = relu(conv4_2) #size = [28 * 28 * 512]
conv4_3 = [28 * 28 * 512]
conv4_3 = relu(conv4_3) #size = [28 * 28 * 512]


Max_pool : kernel_size = (2 * 2), strides = 2
padding = 'VALID' 
padding = 0
#利用上面公式计算(28 - 2) / 2 + 1  = 14
pool4 = [14 * 14 * 512]
```

##### Step 6:

对输入的图片进行**第五个卷积块**
其中包含了三个卷积层
```
Convolution：filter/kernel = [3 * 3 * 512], [ height, weight, channel] 
filter/kernel 个数： 512
strides = 1
padding = 'SAME'  
padding = 1

#利用上面公式计算(14 + 2 * 1 - 3) / 1 + 1  = 28
conv5_1 = [14 * 14 * 512]
conv5_1 = relu(conv5_1) #size = [14 * 14 * 512]
conv5_2 = [14 * 14 * 512]
conv5_2 = relu(conv5_2) #size = [14 * 14 * 512]
conv5_3 = [14 * 14 * 512]
conv5_3 = relu(conv5_3) #size = [14 * 14 * 512]


Max_pool : kernel_size = (2 * 2), strides = 2
padding = 'VALID' 
padding = 0
#利用上面公式计算(14 - 2) / 2 + 1  = 7
pool5 = [7 * 7 * 512]
```

#### Step 7:

对输入的图片进行**flatten，全连接块**
包含三个全连接层，平铺开我们的池化层与4096个神经元进行全连接，然后
```
全连接
fully_connect6 = [7 * 7 * 512, 4096]
fully_connect6 = relu(fully_connect6 )
dropout(fully_connect6)   #dropout随机失活，防止过拟合的一种方式
fully_connect7 = [4096, 4096]
fully_connect7 = relu(fully_connect7 )
dropout(fully_connect7)   #dropout随机失活，防止过拟合的一种方式

输出层
fully_connect8 = [4096, 1000]
fully_connect8 = relu(fully_connect8 )
y_conv = tf.nn.softmax(fc8)
```

因为有1000类所以输出为1000，利用softmax进行评分。
