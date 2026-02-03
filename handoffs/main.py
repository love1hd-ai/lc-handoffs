import argparse
from datetime import datetime
from pathlib import Path


def get_current_time() -> str:
    """
    获取当前时间，格式为 yyyy-mm-dd HH:mm:ss

    Returns:
        str: 格式化的时间字符串，例如 '2026-02-03 14:30:45'
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    """主函数：处理命令行参数"""
    parser = argparse.ArgumentParser(description="处理图片和文本内容")
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        help="图片路径",
        default=None,
    )
    parser.add_argument(
        "-t",
        "--text",
        type=str,
        help="文本内容",
        default=None,
    )

    args = parser.parse_args()

    # 处理图片和文本
    print(f"当前时间: {get_current_time()}")

    if args.image:
        # 验证图片路径是否存在
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"错误: 图片文件不存在: {args.image}")
        else:
            print(f"图片路径: {args.image}")
    else:
        print("图片路径: 未提供")

    if args.text:
        print(f"文本内容: {args.text}")
    else:
        print("文本内容: 未提供")


if __name__ == "__main__":
    main()
