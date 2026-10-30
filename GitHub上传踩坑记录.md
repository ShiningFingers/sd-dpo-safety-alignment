# GitHub 上传踩坑记录（Windows + Clash 代理环境）

> 适用范围：Windows，使用 Clash 系统代理（`127.0.0.1:7897`），无 SSH key、未装 gh CLI，只能走 HTTPS 推送。
>
> 一句话结论：**先配代理、再把 HTTP 版本降到 1.1、用临时内嵌 token 的 remote URL 推送，推完立刻清 token。** 读操作（`ls-remote`）通不代表 push 能通——这是最容易踩的坑。

---

## 一、一图看懂：排查决策顺序

```
连通性测试
  └─ git ls-remote <repo>  超时/Connection reset
        → 没配代理（git 不吃系统代理）→ 配 http.proxy / https.proxy
  └─ ls-remote 秒回，push 卡死（CPU≈0、零字节）
        → HTTP/2 与代理不兼容 → git config http.version HTTP/1.1
  └─ push 报 401 / 弹 GCM 窗口跳不过去
        → 认证走不通 → 临时用 PAT 内嵌 remote URL
  └─ push 成功
        → 立刻把 remote URL 改回无 token 形式 + Select-String 查残留
        → 核验：rev-parse HEAD == ls-remote origin refs/heads/main
```

---

## 二、问题清单表

| # | 症状 | 根因 | 修复（一条命令/一步操作） |
|---|---|---|---|
| 1 | 目录混着 `.venv/`、`__pycache__/`、临时脚本、多个 docx | 没 `.gitignore` + 没分拣 | 写 `.gitignore`；一次性脚本移到 `tools/`；只留最终产物 docx |
| 2 | `git ls-remote` 报 `Failed to connect to github.com port 443` / `Connection was reset` | git **不读系统代理** | `git config --local http.proxy http://127.0.0.1:7897`（及 https.proxy） |
| 3 | `ls-remote` 秒回，但 `git push` 卡死：多个 git 进程 CPU≈0、两分钟零字节上传、远端无 refs | **Git 默认 HTTP/2 与该 HTTP 代理不兼容**（已知问题） | `git config --local http.version HTTP/1.1` |
| 4 | GCM 弹窗「Sign in with your browser」点了不跳转，授权走不通；`git credential fill` 挂住 | 浏览器授权被代理/默认浏览器拦截；GCM 任何探测命令都可能弹交互窗 | 改用 PAT 临时内嵌 remote URL 推送（见第五节） |
| 5 | 贡献图不显示这条 commit | 贡献图有缓存（最多 24h），且只统计默认分支；不是邮箱没绑 | 用 API 查 commit 顶层 `author` 字段确认是否已关联到账号（见第七节） |
| 6 | `commit.verification.verified=false` | 没做 GPG 签名 | **无需处理**，与贡献计数无关 |
| 7 | 仓库首页 description 为空 | GitHub 不会自动从 README 取简介 | 仓库首页 About 齿轮手填 description + topics |

---

## 三、直接可复用：新项目上传 6 步

### 步骤 0：分拣与清理（先做，别等 add）

- 写 `.gitignore`（模板见下）。
- 一次性脚本（本次是 11 个 `_*.py`）**移动**到 `tools/`，不是删除——删之前先 `grep` 主脚本有没有 import 它们（本次核实无依赖）。
- 多个同源 docx 只保留 `final.docx`，其余移走或删除。

`.gitignore` 模板：

```gitignore
# Python
.venv/
venv/
__pycache__/
*.pyc
*.pyo

# IDE
.idea/
.vscode/
.trae/

# OS
.DS_Store
Thumbs.db

# 临时
_doc_dump.txt
*.tmp
*.log
```

### 步骤 1：初始化与首次提交

```powershell
git init
git add .
git commit -m "init: 首次提交"
git branch -M main
```

### 步骤 2：配代理 + HTTP/1.1（本仓库生效，不污染全局）

```powershell
# 本仓库生效（--local）
git config --local http.proxy  http://127.0.0.1:7897
git config --local https.proxy http://127.0.0.1:7897
git config --local http.version HTTP/1.1
```

> 为什么是 `--local`：只影响当前仓库，不影响别的项目；切到无代理环境也不会坑到自己。

### 步骤 3：加远端并测试读连通

```powershell
git remote add origin https://github.com/<user>/<repo>.git
git ls-remote origin          # 秒回 = 网络+凭证都通；但不代表 push 能通
```

### 步骤 4：用临时 PAT 推送（绕开 GCM）

生成 PAT：GitHub → Settings → Developer settings → Personal access tokens → **Tokens (classic)** → 只勾 `repo` → Expiration 选 `7 days`。

```powershell
# 临时把 token 内嵌进 remote URL
git remote set-url origin "https://<user>:<TOKEN>@github.com/<user>/<repo>.git"

# 推送（设环境变量防交互弹窗卡住）
$env:GIT_TERMINAL_PROMPT=0
git push -u origin main

# === 成功后立刻执行下面两行，不要跳过 ===
git remote set-url origin "https://github.com/<user>/<repo>.git"
git remote -v   # 确认 fetch/push 都不含 token
```

### 步骤 5：清 token 残留 + 核验

```powershell
# 确认 .git/config 里没有 token 全文（只打印有无，不打印 token）
Select-String -Path .git/config -Pattern "ghp_" -SimpleMatch
# 无输出 = 干净

# 核验 commit 一致
git rev-parse HEAD
git ls-remote origin refs/heads/main        # 两者 hash 应完全相同

# 核验跟踪关系
git status -sb
# 应显示：## main...origin/main  （无 ahead/behind）
```

### 步骤 6：用完 Revoke token

GitHub → Settings → Developer settings → Personal access tokens → 找到刚用的那枚 → **Revoke**。即使选了 7 天过期，用完就撤最稳。

---

## 四、命令速查（PowerShell 友好）

| 用途 | 命令 | 生效范围 |
|---|---|---|
| 配代理 | `git config --local http.proxy http://127.0.0.1:7897` | 本仓库 |
| 配 HTTP/1.1 | `git config --local http.version HTTP/1.1` | 本仓库 |
| 看所有配置（含来源） | `git config --list --show-origin` | — |
| 测读连通 | `git ls-remote origin` | — |
| 防交互弹窗 | `$env:GIT_TERMINAL_PROMPT=0`（当前会话） | 本会话 |
| 看跟踪状态 | `git status -sb` | — |
| 看远端 refs | `git ls-remote origin refs/heads/main` | — |
| 查 config 里有无 token | `Select-String -Path .git/config -Pattern "ghp_"` | — |

> ⚠️ **PowerShell 坑**：`git status -sb` 输出里的 `@{u}` 在 PowerShell 里会被当哈希表解析，所以引用时必须加单引号，例如 `git rev-parse --abbrev-ref '@{u}'`。直接敲 `git status -sb` 查看是没问题的，但如果要把 `@{u}` 当参数传给别的命令，务必写成 `'@{u}'`。

---

## 五、安全清单

- [ ] PAT scope 只勾 `repo`，不要勾多余权限；Expiration 选 7 天。
- [ ] token **只临时内嵌** remote URL，推完立刻改回无 token 地址。
- [ ] 用 `Select-String` 确认 `.git/config` 无 `ghp_` 残留。
- [ ] 用完去 GitHub **Revoke** token，别等自动过期。
- [ ] `.gitignore` 必须覆盖 `.venv/`、`__pycache__/`、`.idea/`、`.trae/`、临时脚本。
- [ ] **永远不要**把密钥、`.env`、token 写进代码或 commit。
- [ ] 报告/文档含个人信息时：**先建 Private 仓库**，确认脱敏后再改 Public（Settings → Danger Zone）。
- [ ] 大二进制文件（模型、数据集、大量 PDF）用 Git LFS 或 Release 附件，别直接入库。

---

## 六、本次事件时间线

1. 初始化仓库，分拣文件（写 `.gitignore`、11 个 `_*.py` 移到 `tools/`、只留 `final.docx`），首提交 `650759d`（30 文件）。
2. `git ls-remote` 直连超时 → 发现 git 不读系统代理 → 配 `http.proxy` / `https.proxy` 后秒回。
3. `git push` 卡死（4 个 git 进程 CPU≈0，零字节）→ 根因 HTTP/2 与代理不兼容 → `http.version HTTP/1.1`。
4. GCM 浏览器授权跳不过去 → 改用 classic PAT 内嵌 remote URL 推送成功。
5. 立刻改回无 token remote URL，`Select-String` 确认无残留。
6. 核验：`rev-parse HEAD` 与 `ls-remote origin refs/heads/main` 一致；`status -sb` 显示 `main...origin/main`。
7. API 核验：commit 顶层 `author` 已关联到 `ShiningFingers`（贡献图需等缓存刷新）；仓库 description 为空需手填。

---

## 七、附：判断"卡住"还是"在传"

| 信号 | 卡住（需要干预） | 正常在传 |
|---|---|---|
| `Get-Process git` 的 CPU | 恒为 0，长时间不动 | 有波动 |
| 网络上传速率 | 0 bit/s | 有持续流量 |
| `git ls-remote origin refs/heads/main` | 无输出/无对应 refs | 返回 push 后的新 hash |
| 远端仓库网页 | 没出现新 commit | 出现新 commit |

> **核心经验**：`git ls-remote` 通 ≠ push 能通。读操作走通只证明网络和凭证没问题；push 还受 HTTP 版本、代理兼容性影响。卡死时先看 CPU——CPU≈0 不是"在努力传"，是"在等"（等交互窗、等 HTTP/2 握手、等代理响应）。

---

## 八、附：贡献图不显示的排查

贡献图不显示 ≠ commit 没关联到账号。按这个顺序查：

1. **等缓存**：贡献图最多 24 小时刷新，刚推的 commit 可能要等。
2. **确认在默认分支**：只有推到默认分支（本次是 `main`）的 commit 才计入。
3. **用 API 查关联**：

   ```powershell
   $headers = @{ "User-Agent" = "trae" }
   $c = Invoke-RestMethod -Uri "https://api.github.com/repos/<user>/<repo>/commits/<sha>" -Proxy "http://127.0.0.1:7897" -Headers $headers
   $c.author   # 非 null 且 login = 你的账号 = 已关联；null = 邮箱没绑到账号
   ```

   - 本次 `$c.author.login` = `ShiningFingers`，**已关联**，不是邮箱问题。
   - 若 `author` 为 null：去 **Settings → Emails** 把 commit 里的邮箱添加并验证。

4. **别误判 GPG**：`commit.verification.verified=false` 只表示没 GPG 签名，**与贡献计数无关**。
