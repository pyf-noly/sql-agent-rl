# 把项目上传到 GitHub（手把手）

> 目标：把这套 `sql-agent-rl` 发布成一个**别人能看懂、能复现**的公开仓库。
> 本文按「先理解限制 → 本地准备 → 建远程仓库 → 认证 → push → 验证」的顺序写，
> 每一步都给可直接复制的命令。

---

## 0. 先理解三件事（不然一定会踩坑）

1. **GitHub 是代码托管，不是网盘**：单文件硬上限 **100MB**（>50MB 会警告）。
   本项目 `data/raw/bird` 有 **42GB**，`runs/*/adapter` 每个 **127MB**，
   这些**绝对不能**进 git。
2. **密钥一旦 push 就永久留在历史里**：即使下一个提交删掉，别人仍能从历史里翻出来。
   所以 `.env` 必须在第一次 push 前就被忽略。
3. **git 只记录你 `add` 的东西**：没 add 的文件不会上传——这既是保护，也是你要检查的地方。

---

## 1. 这台机器现在的状态（已经处理好了）

| 项 | 状态 |
|---|---|
| Git | 系统没装；已下载**免安装 MinGit** 到 `E:\deepseek工作区\.tools\mingit\cmd\git.exe` |
| 仓库 | `E:\deepseek工作区\sql-agent-rl`，分支 `main`，已有 4 个提交（P0~P3） |
| 历史 | 只有 **1.4MB**，最大文件 220KB，**没提交过 `.env`** |
| 待提交 | P4~P7 的全部代码、文档、结果（约 20MB） |
| 已加 | `LICENSE`（MIT）、`.gitattributes`、`scripts/check_public_repo.py`（上传前自检） |
| 已改 | `.gitignore`：结果/文档进库，原始数据/缓存/权重不进 |

> 如果你以后装了正式版 Git（`winget install Git.Git`），把下文的
> `E:\deepseek工作区\.tools\mingit\cmd\git.exe` 换成 `git` 即可，命令完全一样。

---

## 2. 哪些文件会被上传

| 会进库 | 大小 | 为什么 |
|---|---|---|
| `envs/ rollout/ training/ tests/ scripts/ configs/` | ~2MB | 代码本体（227 个单测） |
| `docs/ guides/` | ~0.2MB | 全部实验记录与手册 |
| `results/*.json` | ~10MB | **所有结论的原始证据**（含逐条 Episode） |
| `runs/*/summary.json` + `metrics.jsonl` | 0.1MB | 训练曲线证据（不含 adapter 权重） |
| `data/bird_p6/sft_train.jsonl` 等 | ~20MB | 多轮 SFT 数据（别人 clone 后可直接训） |
| `README.md LICENSE .env.example ...` | <1MB | 门面与配置模板 |

| 不进库 | 大小 | 原因 |
|---|---|---|
| `data/raw/` | **42GB** | 原始 BIRD 数据，README 里给下载命令 |
| `runs/*/adapter/` | 127MB/个 | 超过 GitHub 单文件限制；复现靠训练脚本 |
| `dist/*.zip` | 61MB | 构建产物；要发布走 GitHub Releases |
| `.cache/` | 0.8MB | LLM 响应缓存，对别人无价值 |
| `.env` | — | **密钥，绝对不能上传** |

---

## 3. 第一步：在 GitHub 建一个空仓库

1. 登录 [github.com](https://github.com) → 右上角 **+** → **New repository**
2. **Repository name**：`sql-agent-rl`
3. 选 **Public**
4. **不要**勾选 "Add a README file" / "Add .gitignore" / "Choose a license"
   （本地已经有了，勾了会冲突）
5. 点 **Create repository**
6. 页面上会显示地址，形如：

```
https://github.com/<你的用户名>/sql-agent-rl.git
```

记下它，第 6 步要用。

---

## 4. 第二步：给 git 配身份（只需一次）

```powershell
& 'E:\deepseek工作区\.tools\mingit\cmd\git.exe' config --global user.name "你的GitHub用户名"
& 'E:\deepseek工作区\.tools\mingit\cmd\git.exe' config --global user.email "你的邮箱"
```

> 这只决定提交记录里显示的作者，**不是登录认证**。

---

## 5. 第三步：认证（推荐 Personal Access Token）

GitHub 早就不允许用账号密码 push 了，要用 **token**。

1. GitHub 右上角头像 → **Settings**
2. 左侧最下 **Developer settings**
3. **Personal access tokens** → **Tokens (classic)**
4. **Generate new token (classic)**
5. 填写：
   - Note：`sql-agent-rl-upload`
   - Expiration：90 天
   - Scopes：勾 **`repo`**（整个大类）
6. 点 **Generate token**，**立刻复制**（只显示一次）

push 时：
- `Username`：你的 GitHub 用户名
- `Password`：**粘贴 token**（不是账号密码）

Windows 会把它存进「凭据管理器」，以后不用再输。

---

## 6. 第四步：本地提交（已经替你做完，这里解释命令）

```powershell
cd E:\deepseek工作区\sql-agent-rl
& 'E:\deepseek工作区\.tools\mingit\cmd\git.exe' status
& 'E:\deepseek工作区\.tools\mingit\cmd\git.exe' add -A
& 'E:\deepseek工作区\.tools\mingit\cmd\git.exe' commit -m "feat(P4-P7): BIRD 训练、轮级优势、全量评测与服务化实验"
```

**提交前务必先跑自检**（检查大文件和密钥）：

```powershell
& 'E:\deepseek工作区\.venv\Scripts\python.exe' scripts\check_public_repo.py
```

看到 `[结论] 可以上传` 再往下走。

---

## 7. 第五步：关联远程仓库并 push

```powershell
cd E:\deepseek工作区\sql-agent-rl
& 'E:\deepseek工作区\.tools\mingit\cmd\git.exe' remote add origin https://github.com/<你的用户名>/sql-agent-rl.git
& 'E:\deepseek工作区\.tools\mingit\cmd\git.exe' push -u origin main
```

第一次 push 会提示输入用户名/token（见第 5 步）。

看到类似输出即成功：

```
To https://github.com/xxx/sql-agent-rl.git
 * [new branch]      main -> main
branch 'main' set up to track 'origin/main'.
```

---

## 8. 第六步：验证

打开 `https://github.com/<你的用户名>/sql-agent-rl`，检查四件事：

1. **README 自动渲染**（表格、链接正常）；
2. 右上 **About** 加一句描述 + topics（`text-to-sql`、`agent`、`reinforcement-learning`、`grpo`）；
3. **Settings → 最下方 Repository size**：应在 30~50MB 量级；
4. 搜索确认没有 `.env`、没有 `data/raw/`、没有 `adapter_model.safetensors`。

---

## 9. 以后怎么更新（三件套）

```powershell
cd E:\deepseek工作区\sql-agent-rl
& 'E:\deepseek工作区\.tools\mingit\cmd\git.exe' status          # 看改了什么
& 'E:\deepseek工作区\.tools\mingit\cmd\git.exe' add -A
& 'E:\deepseek工作区\.tools\mingit\cmd\git.exe' commit -m "docs: 更新 README"
& 'E:\deepseek工作区\.tools\mingit\cmd\git.exe' push
```

> 习惯：一次提交只做一件事；commit message 用中文也完全没问题。

---

## 10. 常见错误对照表

| 报错 | 原因 | 解决 |
|---|---|---|
| `remote origin already exists` | 已经加过 remote | `git remote set-url origin <新地址>` |
| `failed to push some refs ... rejected` | 远程有本地没有的提交（比如建仓时勾了 README） | `git pull --rebase origin main` 再 push |
| `File ... is 105.00 MB; this exceeds GitHub's file size limit` | 有大文件进了暂存区 | `git rm --cached <文件>`，确认 `.gitignore` 覆盖它，再重新提交 |
| `Authentication failed` | 密码栏填了账号密码 | 填 **token**；token 过期就重新生成 |
| `LF will be replaced by CRLF` | Windows 换行符差异 | 可忽略（已加 `.gitattributes`） |
| 中文文件名显示成 `\xxx` | git 默认转义 | `git config --global core.quotepath false` |
| `Failed to connect ... 443` / 超时 | 网络问题 | 换网络或开代理；`git config --global http.proxy http://127.0.0.1:端口` |

---

## 11. 可选加分项

- **Releases**：把 `dist/train_bundle.zip` 作为 Release 附件（不进 git 历史，适合大文件）；
- **Actions**：加 `.github/workflows/tests.yml`，每次 push 自动跑 227 个单测——面试很加分；
- **About**：一句英文描述 + topics，别人搜得到；
- **Pin**：在个人主页把这仓库置顶。

---

## 12. 一句话

**本地已经整理好，你只需要：建空仓库 → 生成 token → 两条命令（remote add + push）。**
