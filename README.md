# yolov8obb_rknn_Cplusplus
yolov8obb 旋转目标检测部署rknn的C++代码


## 编译和运行

1）编译

```
cd examples/rknn_yolov8_obb_demo

bash build-linux_RK3588.sh

```

2）运行

```
cd install/rknn_yolov8obb_demo_Linux

./rknn_yolov8obb_demo

```

注意：修改模型、测试图像、保存图像的路径，修改文件为src下的main.cc

```

int main(int argc, char **argv)
{
    char model_path[256] = "/home/firefly/zhangqian/rknn/examples/rknn_yolov8_obb_demo/model/RK3588/yyolov8n-obb.rknn";
    char image_path[256] = "/home/firefly/zhangqian/rknn/examples/rknn_yolov8_obb_demo/test.jpg";
    char save_image_path[256] = "/home/firefly/zhangqian/rknn/examples/rknn_yolov8_obb_demo/test_result.jpg";

    detect(model_path, image_path, save_image_path);
    return 0;
}
```


# 测试效果
## onnx 测试效果
![test_onnx_result](https://github.com/user-attachments/assets/cff5c466-7f16-4b4c-a518-0fce14ca1d1d)


## rk3588上测试效果

冒号“:”前的数子是15类对应的类别，后面的浮点数是目标得分。（类别:得分）

![images](https://github.com/cqu20160901/yolov8obb_rknn_Cplusplus/blob/main/examples/rknn_yolov8_obb_demo/test_result.jpg)

把板端模型推理和后处理时耗也附上，供参考，使用的芯片rk3588，模型输入640x640，检测类别15类。

![image](https://github.com/user-attachments/assets/d5fff943-460a-44c3-8397-db4abcbcf119)



# 导出onnx 参考

[【yolov8-obb 旋转目标检测 瑞芯微RKNN芯片部署、地平线Horizon芯片部署、TensorRT部署】](https://blog.csdn.net/zhangqian_1/article/details/139437315)


