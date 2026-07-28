#!/usr/bin/env bash
#
# 清理本地远程跟踪引用，只保留 origin/release/therock-7.14。
# 只影响本地 clone，不会改动 GitHub 上的任何分支（可通过 git fetch 恢复）。
#
# 用法：  sudo ./cleanup-branches.sh
#
set -euo pipefail

# 需要保留的分支（remote-tracking ref 全名）
KEEP_REF="refs/remotes/origin/release/therock-7.14"
KEEP_BRANCH="release/therock-7.14"

# 以 root 运行时，绕过 git 对 root 拥有仓库的 "dubious ownership" 检查
GIT=(git -c "safe.directory=*")

# 定位仓库工作树（脚本所在目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> 仓库工作树: $SCRIPT_DIR"
echo "==> git-dir: $("${GIT[@]}" rev-parse --git-dir)"

# 校验目标分支存在
if ! "${GIT[@]}" show-ref --verify --quiet "$KEEP_REF"; then
    echo "错误: 目标分支 $KEEP_REF 不存在，已中止。" >&2
    exit 1
fi

# 1) 限制 fetch 配置，未来 fetch 只拉取这一个分支
echo "==> 设置 fetch refspec 只保留 $KEEP_BRANCH"
"${GIT[@]}" config remote.origin.fetch \
    "+refs/heads/${KEEP_BRANCH}:${KEEP_REF}"
"${GIT[@]}" config --get-all remote.origin.fetch

# 2) 统计删除前数量
before="$("${GIT[@]}" for-each-ref refs/remotes/origin | wc -l)"
echo "==> 删除前远程跟踪引用数: $before"

# 2.1) 先单独删掉符号引用 origin/HEAD（它指向某个分支，不能和目标一起批量删）
if "${GIT[@]}" symbolic-ref -q "refs/remotes/origin/HEAD" >/dev/null 2>&1; then
    echo "==> 删除符号引用 refs/remotes/origin/HEAD"
    "${GIT[@]}" symbolic-ref -d "refs/remotes/origin/HEAD" 2>/dev/null || \
        "${GIT[@]}" update-ref --no-deref -d "refs/remotes/origin/HEAD" 2>/dev/null || true
fi

# 3) 批量删除除目标外的所有远程跟踪引用（单次 update-ref 批处理）
echo "==> 正在删除多余引用..."
"${GIT[@]}" for-each-ref --format='%(refname)' refs/remotes/origin \
    | grep -vx "$KEEP_REF" \
    | grep -vx "refs/remotes/origin/HEAD" \
    | sed 's/^/delete /' \
    | "${GIT[@]}" update-ref --stdin

# 4) 打包清理，回收空间
echo "==> pack-refs 清理"
"${GIT[@]}" pack-refs --all --prune >/dev/null 2>&1 || true

# 5) 统计删除后数量并展示保留结果
after="$("${GIT[@]}" for-each-ref refs/remotes/origin | wc -l)"
echo "==> 删除后远程跟踪引用数: $after"
echo "==> 保留的远程分支:"
"${GIT[@]}" for-each-ref --format='    %(refname)' refs/remotes/origin

echo "完成。已删除 $((before - after)) 个远程跟踪引用。"
