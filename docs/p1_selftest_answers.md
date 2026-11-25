# P1 遗留的 14 个自测问题：答案与代码依据

> 交代背景：P1 阶段收官时（2026-11-03）只验收了 `pytest` 全绿，项目执行计划里
> `guides/P1.md` §3 的 14 个问题一直挂着没答。P7 收口时补上，每题给出
> **结论 → 代码依据（文件:行）→ 面试口径**。
>
> 读代码的版本：`envs/sql_sandbox.py`（332 行）、`envs/error_taxonomy.py`（114 行）、
> `envs/feedback.py`（193 行）、`envs/db_registry.py`（248 行）。
> 行号以本文件写作时为准（2026-11-06）。

---

## 3.1 `sql_sandbox.py`

### 1. `open_readonly` 里的 `?mode=ro` 是什么意思？为什么它是"第一道物理防线"？

**结论**：`mode=ro` 是 SQLite 自己的 URI 打开参数（`envs/sql_sandbox.py:211-223`），
连接一建立就处于只读状态；此后任何 `INSERT/UPDATE/DELETE/DROP` 都会由 SQLite 内核
直接返回 `attempt to write a readonly database`，**不经过我们的任何代码判断**。

**为什么叫"物理防线"**：我们的白名单（`validate_sql`）、单语句检查、关键字黑名单都是
**逻辑层**的——只要解析器有洞（sqlparse 版本差异、畸形 SQL 绕过），逻辑层就可能失效。
`mode=ro` 不依赖解析：即使一条写操作骗过了 `validate_sql`，数据库文件也改不了。
这是"纵深防御"的最内层，也是唯一不依赖我们自己代码正确性的那一层。

**面试口径**：沙箱的第一道防线必须是数据库自己给的，不能是我写的正则。

---

### 2. `validate_sql` 为什么要先 `sqlite3.complete_statement` 再 `sqlparse.split`？只用一个行不行？

**结论**：两个工具解决的是两个不同问题，缺一不可（`envs/sql_sandbox.py:160-208`）。

| 工具 | 能做什么 | 单独用会漏什么 |
|---|---|---|
| `sqlite3.complete_statement` | 用 **SQLite 自己的词法器**判断"这条语句是否完整"：引号/括号是否闭合、分号是否在字符串里 | 它只回答"完不完整"，不告诉你里面有几条语句 |
| `sqlparse.split` | 按语句边界切分，能数出"几条语句" | 它对语法的合法性一无所知；而且 `SELECT ';'` 这类字符串里的分号，靠朴素切分会切错 |

组合起来才两头都堵住：
1. 先 `complete_statement` 挡掉"引号没闭合 / 括号没闭合"这类**残缺**输入（`sql_sandbox.py:167`）；
2. 再 `sqlparse.split` 数条数，`!= 1` 直接拒（`sql_sandbox.py:176-181`）；
3. 还有一处容易漏的细节：**先去尾分号再校验一次**（`sql_sandbox.py:170-174`）。
   `SELECT 1;` 是一条合法语句，但带分号时 `complete_statement` 认为"已经结束了"，
   而 `split` 可能算出 2 段——去掉尾分号后重新确认，才能区分"一条带终止符"和
   "两条语句"。

**被测试覆盖的真实攻击**：`SELECT 1; DROP TABLE employee` 会被第 2 步拦成 `multi_statement`，
同时第 3 步的关键字黑名单（`sql_sandbox.py:201-206`）还会再拦一次。

---

### 3. `set_progress_handler(_guard, 2000)` 里的 `2000` 是什么？调大调小各有什么后果？

**结论**：`2000` 是 **SQLite 虚拟机指令（opcode）的执行步数**——每执行 2000 条 VM 指令，
SQLite 回调一次 `_guard`（`envs/sql_sandbox.py:288`）。它决定"多久检查一次是否超时"。

- **调小**（如 1）：几乎每条指令都回调一次 **Python 函数**，跨 C/Python 边界的开销
  会把查询拖慢一个数量级，评测吞吐直接崩掉；
- **调大**（如 10⁶）：回调次数少、开销低，但检查粒度变粗——一条跑 0.5s 的语句可能
  在 2s 预算外又跑很久才被打断，超时变成"软"的；
- **2000 是折中**：对普通查询来说一次检查在亚毫秒级，既有足够的时间分辨率，
  Python 回调开销也可以忽略。

**必须知道的局限**：这是**协作式**超时，只在 VM 执行指令时触发。如果卡在单个昂贵的
opcode 里（大排序、大聚合）或阻塞 I/O，回调不会进来——所以它是"软超时"，不是硬实时。

---

### 4. 超时的时候 `_guard` 返回 `1`，SQLite 会做什么？为什么不是直接 `sys.exit`？

**结论**：progress handler 返回非 0 时，SQLite 主动中断当前语句（`SQLITE_INTERRUPT`），
`con.execute` 抛 `sqlite3.OperationalError: interrupted`；我们在
`envs/sql_sandbox.py:304-312` 捕获它，归一成 `error_class="timeout"` 的
`QueryResult` 返回。**连接本身仍然可用**，`finally` 里正常清 handler、关连接。

**为什么不能 `sys.exit`**：

1. `sys.exit` 抛的是 `SystemExit`，它不是"这条 SQL 失败"，而是"整个进程退出"；
   评测是 6 线程并发的（`run_baseline.py`），一个超时会把整批评测带走；
2. 从 C 的 VM 回调里抛 Python 异常穿过 SQLite 的 C 栈，属于**未定义行为**，
   轻则状态不一致，重则直接 abort；
3. `sys.exit` 无法被"按样本"处理——而我们需要的是把这一条标成失败、
   继续评测下一条。

**面试口径**：沙箱的失败必须是可以被上层当作数据处理的返回值，不能让进程陪葬。

---

### 5. `fetchmany(max_rows + 1)` 为什么是 `+1` 而不是 `max_rows`？

**结论**：`+1` 是用来**检测截断**的（`envs/sql_sandbox.py:292-295`）。

- 取 `max_rows + 1` 行：如果真拿到了 `max_rows + 1` 行，说明结果**比预算多**
  → 置 `truncated=True`，再砍回 `max_rows`；
- 只取 `max_rows` 行的话，"刚好 `max_rows` 行"和"至少 `max_rows` 行、被截断了"
  **无法区分**——而这两件事在评测里后果完全不同：截断后的结果做等价比较会产生
  **假阳性**（两边都被截到同样的前缀 → 判"相同"）。

这也是 BIRD 评测必须把 `max_rows` 从 1000 放大到 20000 的原因：
BIRD 有 55 条 gold 结果超过 1000 行，截断会把"错得不一样"判成"对"。

---

## 3.2 `error_taxonomy.py`

> 先纠正一个数量：`guides/P1.md` 说"至少 6 类"，实际 `ErrorClass` 是 **9 类**
> （`envs/error_taxonomy.py:19-28`）：`invalid_sql` / `syntax` / `no_such_column` /
> `no_such_table` / `ambiguous_column` / `timeout` / `empty_result` / `db_error` / `unknown`。
> 多出来的三类有明确用途：`invalid_sql` 是"沙箱拒绝"（不是数据库报错）、
> `empty_result` 是"执行成功但零行"（P6 值检查要单独处理）、`unknown` 是"成功且非空"。

### 6. `_PATTERNS` 里为什么 `AMBIGUOUS_COLUMN` 必须排在 `NO_SUCH_COLUMN` 前面？举个会被顺序搞错的例子。

**结论**：`classify` 是"**第一个命中的模式即返回**"（`envs/error_taxonomy.py:70-72`），
所以更具体的类别必须排在更笼统的类别前面（`envs/error_taxonomy.py:33-43`）。

两者共享 `column name` 这个字面：
- 歧义 → `ambiguous column name: dept_id`（SQLite 原话）
- 不存在 → `no such column: T2.emp_name`

如果 `NO_SUCH_COLUMN` 的正则被放宽成包含 `column`/`column name` 的兜底模式，
那么 `ambiguous column name: dept_id` 会先命中它，反馈就会说
**"这列不存在，可用列名是……"**——而真实情况是"这列在 `department` 和 `employee`
里都有，请用表名限定"。**两类错误的修复动作完全相反**：一个要改名，一个要加限定符。
顺序错了不会报错，只会让模型在原地打转——这是最典型的一类静默 bug。

当前的正则（`no such column`）已经足够窄，顺序仍保留为防御：`_PATTERNS` 将来若有人
为了兜底而放宽某条模式，顺序就是唯一防线。

---

### 7. `classify` 在 `error_class` 已经是 `invalid_sql` 时为什么直接透传，不再匹配正则？

**结论**：见 `envs/error_taxonomy.py:61-67`——沙箱已经判定的类别直接返回，
唯一的例外是 `db_error`（这一类太笼统，需要靠报错原文细分）。

三个理由：

1. **文本来源不同**：`invalid_sql` 的 message 是我们自己写的理由码
   （`multi_statement:` / `not_select:` / `forbidden_keyword:`…，见
   `envs/sql_sandbox.py:160-208`），根本不是 SQLite 的报错文本，拿它去匹配
   SQLite 的报错模式属于跨来源匹配；
2. **会误伤**：被拒的 SQL 文本里可能恰好包含 `syntax error` 之类的字样
   （比如注释里写着"fix syntax error"），重匹配会把"沙箱拒绝"翻成"数据库语法错误"，
   反馈模板随之给错建议；
3. **下游要的是精确理由**：`FeedbackFormatter._invalid_sql`
   （`envs/feedback.py:121-128`）会把 `reason` 原样回灌，告诉模型
   "你只能回一条 SELECT"。透传保证这条信息不丢，也让 `classify` 幂等。

---

### 8. `extract_offender` 从 `no such column: t.emp_name` 里抠出 `t.emp_name`，反馈模板拿它干什么？

**结论**：`extract_offender`（`envs/error_taxonomy.py:76-84`）用一条正则抠出"惹事的标识符"，
反馈格式化器拿它做三件事：

1. **点名**：反馈里直接写 `` `t.emp_name` ``，而不是"你有一列写错了"——
   模型的注意力有落点；
2. **按表裁剪列清单**（`envs/feedback.py:146-161`）：如果 offender 带表前缀
   （`t.emp_name` → `t`），**只列 `t` 的列**，不把整库的列倒出来。既省 token
   （BIRD 的库最多 199 列），也避免"给一堆无关列让模型再猜一次"；
3. **歧义时定位候选表**（`envs/feedback.py:163-170`）：用 offender 去
   `columns_by_table` 里反查，告诉模型"`dept_id` 在 `department` 和 `employee`
   里都有，请用别名限定"。

同一个函数也被 `no_such_table` 用（`envs/feedback.py:139-144`）：点名缺失的表，
并附上最多 20 个真实表名。

---

## 3.3 `feedback.py`

### 9. 为什么反馈模板要单独一个文件，而不是写在沙箱里？

**结论**：反馈文本是"环境 → 模型"的**唯一文字接口**，也是本项目的**实验变量**
（`envs/feedback.py:1-13` 的模块说明写了这条）。单独成文件有四个直接好处：

1. **可消融**：P6 要做的对照就是"只回报错原文" vs "报错 + 可用列名"。
   改 `FeedbackConfig`（`envs/feedback.py:29-36`）即可，沙箱一行不用动；
2. **沙箱保持纯粹**：沙箱只负责"执行 + 判定 + 返回结构化结果"，
   不掺任何自然语言表达，单测好写（执行正确性 vs 文案正确性分开测）；
3. **可审阅**：所有进模型上下文的话集中在一个 193 行的文件里，
   review 和 grep 泄漏都只在一处；
4. **唯一出口便于防泄漏**：`test_no_gold_leakage`（`tests/test_error_taxonomy.py:125-137`）
   只需要守住这一个出口。

---

### 10. `FeedbackConfig(include_schema_hint=False)` 这个开关是给谁用的？（提示：P6 消融）

**结论**：给**消融实验**用的：关掉以后，`no_such_table` / `no_such_column` /
`ambiguous_column` 的反馈不再附"可用表名/列名"（`envs/feedback.py:141-169`），
只剩数据库报错原文。

它要回答的问题是：**修复能力到底来自"环境把列名喂到嘴边"，还是来自模型自己看 schema？**
如果关掉提示后修复率明显下降，说明我们的收益有一部分是"提示工程"而不是"模型学会了用反馈"——
这种归因不做，简历上的 +2.93pp 就说不清楚。

配套的测试是 `test_schema_hint_can_be_disabled`
（`tests/test_error_taxonomy.py:118-123`），确保开关真的生效而不是摆设。

---

### 11. 看 `test_no_gold_leakage` 这个测试——它在防什么事故？如果反馈里带了 gold SQL，训练会发生什么？

**结论**：防止**答案泄漏进反馈**（`tests/test_error_taxonomy.py:125-137`）。测试把
`"SELECT COUNT(*) FROM employee"`、`"Alice"`、`"42"` 三种形态的 gold 片段喂给所有反馈分支，
断言它们一个字都不能出现在回灌文本里。

如果反馈里带了 gold SQL 或正确结果，会发生三件坏事：

1. **评测变开卷**：模型只要照抄反馈就能"修好"，EA 会虚高，而且这个数字**没有任何
   现实意义**——线上没有 gold；
2. **训练学到错误策略**：SFT 会学"先随便写一条，等反馈给答案"；RL 会把
   "等答案"这条路径强化出来（reward 一直在涨，真实能力一直不涨），属于典型的
   reward hacking；
3. **不可发现**：这种泄漏不会报错，只会让指标好看。所以必须在测试层用断言钉死。

对应地，`rollout/agent.py:137-139` 也在代码里写了同一条纪律：评测器只用于
"判对才停"的停机决策，**正确性信息绝不写进反馈**。

---

## 3.4 `db_registry.py`

### 12. 为什么用 `PRAGMA table_info` 读列，而不是直接读 `tables.json`？

**结论**：schema 的**唯一可信来源是 sqlite 文件本身**（`envs/db_registry.py:144-176`）。

| 角度 | `PRAGMA table_info`（采用） | `tables.json`（仅用于交叉校验） |
|---|---|---|
| 真实性 | 文件里实际有什么就是什么 | Spider 的人工标注，可能漏列/多列/顺序不同 |
| 数据集覆盖 | Spider / BIRD 通用 | BIRD 根本没有这个文件，要额外分支 |
| 额外信息 | 类型、主键、`PRAGMA foreign_key_list` 的外键 | 只有列名，外键要另读 |
| 成本 | 一次打开 + 每表 1~2 条 PRAGMA，可缓存 | 读一个大 JSON |

`load_spider_tables`（`envs/db_registry.py:193-201`）保留下来只做**交叉校验**，
不参与构造 schema——"我们信数据库文件，不信标注文件"是刻意的。

---

### 13. `_load_schema` 上挂 `lru_cache` 是为了什么？没有它会慢在哪？

**结论**（`envs/db_registry.py:144-145`）：`_load_schema` 每次调用都要
**打开 sqlite 文件 + 1 条 `sqlite_master` 查询 + 每张表 1~2 条 PRAGMA**。
这个函数在热路径上被反复调用：

- 评测：每道题渲染 prompt 要一次 schema，307 题 / 11 个库 → 307 次；
- SFT 数据装配：3918 条样本 / 54 个库 → 3918 次；
- 多轮 rollout：每一轮重建 messages 又要一次。

没有缓存就是"几千次重复打开文件"，在机械盘/大库上会变成肉眼可见的启动延迟；
BIRD 最大库有几十张表，PRAGMA 次数还要乘以表数。

缓存键是 **resolve 后的绝对路径字符串**（不是 `Path` 对象），`maxsize=512`
覆盖本项目所有库；返回的 `DbSchema` / `TableInfo` 都是 `frozen=True` 的不可变对象，
所以共享缓存不会被下游改坏。

---

### 14. `render_ddl` 为什么输出 `CREATE TABLE` 而不是 JSON？

**结论**（`envs/db_registry.py:68-113`）：

1. **贴模型先验**：Text-to-SQL 的预训练语料里，"schema"几乎总是以
   `CREATE TABLE x (...)` 的形态出现；JSON 的 `{"table_names": [...], "column_names": [[...]]}`
   是 Spider 数据集内部的表示，模型没见过这种"考试专用格式"；
2. **省 token**：DDL 只有标识符本身；JSON 还要付键名、引号、括号、嵌套的成本。
   本项目的 prompt 本来就以 schema 为主（P4 实测 **88.7% 的 token 是上下文**），
   这一项直接决定单条样本能不能塞进 2560/3072 的窗口；
3. **可读性/可 diff**：DDL 是人类和模型都能直接核对的形态，出问题时
   人肉一眼能看出"列名写错了"，JSON 要先在脑子里反序列化；
4. **可参数化**：`render_ddl` 支持 `include_types` / `include_fk` / `quote_identifiers`
   / 列预算，Spider 用默认（与 P2~P5 的历史 prompt 逐字一致），BIRD 走
   `BIRD_PROMPT` 打开全部选项（列名带空格必须加引号，否则模型会自己删空格 → 语法错）。

---

## 附：这 14 题的共同点

回头看，P1 的设计都在回答同一个问题：**"环境凭什么可信？"**

- 只读连接 → 不靠自己的解析代码；
- 双层语句校验 → 不靠单一工具；
- 协程式超时 → 不靠杀进程；
- 错误分类顺序 → 不靠"大概对"；
- 反馈单独成文件 + 防泄漏测试 → 不靠人工 review；
- schema 从库里读 + 缓存 → 不靠标注文件。

后面 P2~P6 的所有数字，都建立在这六条上。这也是 P1 阶段"测试不全绿不放行"的原因。
