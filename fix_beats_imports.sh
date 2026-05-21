#!/bin/bash
# fix_beats_imports.sh
# 用途：将 beats 包内所有模块间的绝对导入改为相对导入

TARGET_DIR="beats"

if [ ! -d "$TARGET_DIR" ]; then
    echo "错误: 目录 $TARGET_DIR 不存在，请确认路径。"
    exit 1
fi

cd "$TARGET_DIR" || exit 1

# 收集目录下所有 .py 文件的模块名（去掉 .py 后缀）
modules=()
for f in *.py; do
    [[ "$f" == "__init__.py" ]] && continue
    basename="${f%.py}"
    modules+=("$basename")
done

echo "检测到以下模块: ${modules[*]}"

# 遍历所有 .py 文件，将 "from <模块名> " 替换为 "from .<模块名> "
for file in *.py; do
    [[ ! -f "$file" ]] && continue
    echo "处理 $file ..."
    for mod in "${modules[@]}"; do
        # 避免将自身替换成相对导入（比如 backbone.py 里 from backbone import ... 不会出现）
        # 使用正则：行首以 from 开头，后面是模块名，然后跟空格或换行
        # 注意：不替换已经带点的导入
        sed -i -E "s/^from (${mod})([[:space:]])/from .\1\2/g" "$file"
    done
done

echo "完成！所有绝对导入已改为相对导入。"
cd ..
