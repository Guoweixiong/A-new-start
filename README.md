# image data augmentation script
功能说明：<br>
* 输入图片后，对图片进行裁剪，旋转，改变颜色，投影变换，得到不同效果的图片并保存，以此进行图像数据增广，获得大量数据<br>
<br>
预先编写函数random_warp和random_light_color,再进行处理<br>
<br>
函数*random_warp*对图片进行投影变换，*random_light_color*改变图片的颜色<br>
程序运行时设定有两次输入<br>
*  第一次输入原始图片的路径，路径名不含中文字符,且用/代替\ <br>
*  第二次输入为生成图片的文件名，默认保存在同一路径文件夹内

