[English](reproduction.en.md)

# 复现与新实验

## 服务性能

复制 `configs/serving.example.json`，填写自己的端点和模型；密钥仅从 `SPARKBENCH_API_KEY` 环境变量读取。示例默认6条、并发1、128输出上限。题库JSONL每行包含唯一`id`与`messages`；完整历史60条负载在实验`protocols/protocol/workloads.jsonl`。

```bash
sparkbench serve --config configs/serving.example.json --workloads configs/smoke-workloads.jsonl
# 核对预览后执行；使用新的输出目录。
sparkbench serve --config configs/serving.example.json --workloads configs/smoke-workloads.jsonl --output outputs/new-serving --execute
```

`arrival.example.json`使用独立固定到达；忙时记录本地拒绝，不自动改为闭环排队。泊松到达及历史复杂干扰设计保留在参考源码与冻结方案，不能把新固定到达适配器声称为完整重现历史泊松协议。

仅明确Cloudflare HTTP错误可有限重试；超时／连接中断保留未知结果，不盲目重放。记录每次尝试，末次成功不抹除前次失败。全局截止不延长。`outputs/`中的`STOP`停止新请求；在途请求按有界超时排空。请求和响应内容不默认持久化，保存计数、时序、载荷散列与错误类别。

## 原生引擎与缓存

`native.example.json`连接原生SGLang `/generate`。JSONL需提供经同版本tokenizer验证的`input_ids`，不把字符数当词元数。`ignore_eos=true`仅作为强制长度负载；实际词元计数与预填充／解码字段保存在`native_meta`。非流式客户端没有观测到首词元时间，不用总时长伪造TTFT。

可选配置`metrics_url`、`counter_names`，监控密钥来自`SPARKBENCH_METRICS_KEY`。服务计数器须记录重置、标签变化和并行业务污染。缓存复测用相同完整前缀、明确seed/probe身份与等待时间。剖析API与导出轨迹因引擎版本而异，先核对已安装版本；历史自动结束剖析验证代码在`reference/2026-study/completion/`，不默认远程启动profiler。

## GPU与双机

硬件执行需专用Linux GB10、已安装且验证过的CUDA/PyTorch环境；本套件不安装驱动、不停推理服务、不修改频率功耗策略。原实验的库版本与内核路径见完整报告和记录。NVFP4需要匹配的FlashInfer/CUTLASS后端；不支持的路径记录为unsupported，而非0吞吐。

```bash
sparkbench hardware --mode base
sparkbench hardware --mode base --output outputs/base --execute
sparkbench hardware --mode extensions --output outputs/extensions --execute
sparkbench hardware --mode soak --output outputs/soak --duration-limit 36000 --execute
# 两台机器分别运行；把 192.0.2.10 改成节点0的实际测试网地址。
sparkbench hardware --mode collectives --rank 0 --master-addr 192.0.2.10 --output outputs/nccl-r0 --execute
sparkbench hardware --mode collectives --rank 1 --master-addr 192.0.2.10 --output outputs/nccl-r1 --execute
```

`overlap`与collectives相同双机启动方式。通信配置由环境变量`NCCL_*`决定并写入元数据；必须检查NCCL日志才能证明实际传输路径。单机扩展不自动启动双机重叠。FP64与代表性Nsight计数器的历史程序、协议和数据也在参考目录与实验中，其独立适配不纳入当前CLI的自动执行。

每次输出目录必须是新目录。内存保留不少于16GiB或总内存15%；温度85°C退避，低于75°C稳定30秒恢复，最长冷却600秒且不延长总窗口。集合通信遇热限终止，重新执行时新建两rank配对批次。`STOP_GPU`停止后续提交。不得在正式服务上直接复制历史停服协调器。

本次仓库发布验证了CPU导入、配置、SSE、重试、统计、模拟HTTP与离线数据复算；未重跑硬件适配器，不能宣称任意新CUDA环境已通过实机验收。
