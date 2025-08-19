from PIL import Image

# 加载 RGBA 图像
input_path = "./datasets/before/1_t.JPG"  # 替换为你的 RGBA 图像路径
output_path = "./datasets/JPEGImages/01.JPG"  # 转换后保存的路径

# 打开图像
img = Image.open(input_path)

# 确保图像是 RGBA 模式
if img.mode == 'RGBA':
    # 转换为 RGB 模式
    rgb_img = img.convert('RGB')
    # 保存 RGB 图像
    rgb_img.save(output_path)
    print(f"图像已成功从 RGBA 转换为 RGB 并保存为: {output_path}")
else:
    print("输入图像不是 RGBA 模式，无需转换。")
