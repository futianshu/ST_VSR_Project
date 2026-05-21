import argparse
import os
import glob
import shutil
import tempfile
from mmagic.apis import MMagicInferencer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='basicvsr_pp')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 自己读取目录下所有的图片并排序
    img_paths = sorted(glob.glob(os.path.join(args.input_dir, '*.png')) + 
                       glob.glob(os.path.join(args.input_dir, '*.jpg')))
    
    if not img_paths:
        raise ValueError(f"在 {args.input_dir} 中没有找到任何图片！")

    print(f"正在推理: {args.input_dir}，共发现 {len(img_paths)} 帧")

    # 2. 初始化推理器
    editor = MMagicInferencer(model_name=args.model_name)
    
    # 3. 🌟 终极绝招：使用临时文件夹，构建符合 MMagic 脾气的目录结构
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, img_path in enumerate(img_paths):
            # 严格按照 REDS 格式的 8位数字命名，如 00000000.png
            target_name = f"{i:08d}.png"
            target_path = os.path.join(temp_dir, target_name)
            
            # 优先使用软链接(省空间省时间)，如果系统不支持则使用 copy
            try:
                os.symlink(os.path.abspath(img_path), target_path)
            except OSError:
                shutil.copy(img_path, target_path)
        
        # 4. 将伪装好的临时文件夹路径传给 MMagic
        editor.infer(video=temp_dir, result_out_dir=args.output_dir)

if __name__ == "__main__":
    main()