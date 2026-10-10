#!/usr/bin/env bash
# auto_push.sh —— 一键提交并推送 Research-notes
# 用法：
#   ./auto_push.sh                  # 默认 commit message: "notes: update YYYY-MM-DD HH:MM"
#   ./auto_push.sh "add SPROUT"     # 自定义 commit message

set -euo pipefail

# 仓库根目录（脚本所在目录）
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"

# 没有改动就直接退出
if git diff --quiet && git diff --cached --quiet && [ -z "$(git ls-files --others --exclude-standard)" ]; then
    echo "✓ 工作区干净，无需提交"
    exit 0
fi

MSG="${1:-notes: update $(date '+%Y-%m-%d %H:%M')}"

echo "→ 仓库: $REPO_DIR"
echo "→ 分支: $BRANCH"
echo "→ 提交信息: $MSG"

git add -A
git commit -m "$MSG"
git push origin "$BRANCH"

echo "✓ 已推送到 origin/$BRANCH"
echo "  GitHub Actions 会自动构建并部署到 https://hhhjyz.github.io/Research-notes/"
