# caffe_stn
there are eight flies about Spatial Transformer Networks 
## 前言
Spatial Transformer Networks是Google DeepMind在15年提出的，相当于在传统的一层Convolution中间，装了一个“插件”，可以使得传统的卷积带有了**[裁剪]、[平移]、[缩放]、[旋转]**等特性；理论上，可以减少CNN的训练数据量，以及减少做data argument，让CNN自己学会数据的形状变换。
本次自定义神经层就是 以STN为例。

### 1.自定义神经层一般的步骤
>* **1.创建新定义的头文件include/caffe/layers/my_neuron_layer.hpp**
重新Layer名的方法：virtual inline const char*  type() const { return "MyNeuron"; }
如果只是需要cpu方法的话，可以注释掉forward/backward_gpu()这两个方法
*  **2.创建对应src/caffe/src/my_neuron_layer.cpp的源文件**
重写方法LayerSetUp,实现从能从prototxt读取参数
重写方法Reshape，如果对继承类没有修改的话，就不需要重写
重写方法Forward_cpu
重写方法Backward_cpu(非必须)
 如果要GPU支持，则还需要创建src/caffe/src/my_neuron_layer.cu，同理重写方法Forward_gpu/Backward_gpu(非必须)
* **3.proto/caffe.proto注册新的Layer**
* **4.my_neuron_layer.cpp添加注册的宏定义**
* **5. 重新编译和install**

大概的就是以上几个步骤，其中主要是第一和第二步，神经层的具体实现。我们可以继承至Layer， 或者NeuronLayer。


### 2.具体实现STN层的步骤
**stn的实现是比较难得的，但是在github上有人写出来了。可以先在我的github上下载这个stn层的具体实现代码，包括.cpp   .hpp 和 .cu 六个文件 **
####st_layer.cpp ,  st_layer.cu , st_loss_layer.cu 和st_loss_layer.cpp放到caffe/src/caffe/src/layers下。st_layer.hpp和st_losss_layer.hpp放到caffe/include/caffe/layers/下。
然后在caffe.protobuf中注册新的layer，主要有以下几个地方要改：
添加新layer的数据结果
```
message SpatialTransformerParameter {

  // How to use the parameter passed by localisation network
  optional string transform_type = 1 [default = "affine"];
  // What is the sampling technique
  optional string sampler_type = 2 [default = "bilinear"];

  // If not set,stay same with the input dimension H and W
  optional int32 output_H = 3;
  optional int32 output_W = 4;

  // If false, only compute dTheta, DO NOT compute dU
  optional bool to_compute_dU = 5 [default = true];

  // The default value for some parameters
  optional double theta_1_1 = 6;
  optional double theta_1_2 = 7;
  optional double theta_1_3 = 8;
  optional double theta_2_1 = 9;
  optional double theta_2_2 = 10;
  optional double theta_2_3 = 11;
}

// added by Kaichun Mo
message PowerFileParameter {

  optional string shift_file = 1;
}

// added by Kaichun Mo
message STLossParameter {

  // Indicate the resolution of the output images after ST transformation
  required int32 output_H = 1;
  required int32 output_W = 2;
}

// added by Kaichun Mo
message LocLossParameter {

  required double threshold = 1;
}
```
在message LayerParmeter中注册新layer，添加
```
  optional SpatialTransformerParameter st_param = 150;
  optional STLossParameter st_loss_param = 151;
  optional PowerFileParameter power_file_param = 152;
  optional LocLossParameter loc_loss_param = 153;
```
有的blog说要在message V1LayerParameter中添加新的层名称，其实是不需要的，因为这个是
是旧版caffe中的 已经废弃 现在使用LayerParameter。而我们自己定义的网络层是不需要往里面加的。
### 3.重新编译
```
cd /caffe_root/
make all
---------------------------------------------------------
/bin/sh: 1: cannot create .build_release/src/caffe/proto/caffe.pb.o.warnings.txt: Permission denied
make: *** [.build_release/src/caffe/proto/caffe.pb.o] Error 1
//居然出错了
----------------------------------------------------------
sudo make all  //就好了
-----------------------------------------------------------
NVCC src/caffe/layers/reduction_layer.cu
NVCC src/caffe/layers/cudnn_pooling_layer.cu
NVCC src/caffe/layers/batch_reindex_layer.cu
NVCC src/caffe/layers/embed_layer.cu
NVCC src/caffe/layers/softmax_loss_layer.cu
NVCC src/caffe/layers/tanh_layer.cu
AR -o .build_release/lib/libcaffe.a
LD -o .build_release/lib/libcaffe.so.1.0.0-rc3

```
这编译通过了但是在网络中调用SpatialTransformer就会出错：
```
Check failed: registry.count(type) == 1 (0 vs. 1) Unknown layer type: SpatialTransformer 
```
没有STN层，应该是没有编译到st_layer.cpp文件，回去看编译结果输出，果然没有st_layer.cpp的编译。检查了一遍，有些低级错误。然后重新编译。
```sh
--------------------------------------------------------
NVCC src/caffe/layers/st_layer.cu
NVCC src/caffe/layers/st_loss_layer.cu
CXX src/caffe/layers/st_layer.cpp
CXX src/caffe/layers/st_loss_layer.cpp
--------------------------------------------------------
```
如上，这次就编译到这个STN文件了。
### 4.测试STN层
采用mnist网络测试一下STN网络，网络文件可以在我的github中下载，加入STN的网络是先采用卷积和全连接层给STN那六个转换参数，然后STN层的输出当做data数据，后面接卷积和全连接层正常训练
![enter image description here](https://leanote.com/api/file/getImage?fileId=59783cf0ab64410cc4001634)
由上可以看到准确率达到99.57%.说明STN还是有一定的效果的。

###KAM
[TOC]




参考资料：
1.[Christopher Bourez's blog](http://christopher5106.github.io/big/data/2016/04/18/spatial-transformer-layers-caffe-tensorflow.html)
2.https://github.com/christopher5106/last_caffe_with_stn






