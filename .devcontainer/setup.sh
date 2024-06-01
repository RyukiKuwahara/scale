#! /bin/bash

# リポジトリのルートディレクトリを見つける関数
find_repo_root() {
    local dir=$(pwd)
    while [ "$dir" != "/" ]; do
        if [ -d "$dir/.git" ]; then
            echo "$dir"
            return
        fi
        dir=$(dirname "$dir")
    done
    echo ""
}

DIR_PATH=$(find_repo_root)
# ルートディレクトリが見つかったか確認
if [ -n "$DIR_PATH" ]; then
    echo "Repository root found at: $DIR_PATH"
    # 安全なディレクトリとしてリポジトリのルートを追加
    git config --global --add safe.directory "$DIR_PATH"
    echo "Added $DIR_PATH as a safe directory."
else
    echo "No Git repository found. Exiting..."
    exit 1
fi

# gitの設定
git config --global user.name 'RyukiKuwahara'
git config --global user.email 'ryukikuwahara@outlook.jp'

