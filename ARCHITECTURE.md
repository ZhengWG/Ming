# Ming / BailingMM2 架构文档（代码层）

本文档基于仓库当前代码，描述 Ming（产品名）在代码中的 BailingMM2 实现架构，重点覆盖：

- 模型初始化链路
- 推理执行链路（文本/多模态）
- 模块连接关系（接口与关键张量）
- 可选分支（Image Generation / Talker）

---

## 1. 总览

### 1.1 命名关系

- 对外产品名：**Ming**（如 Ming-flash-omni-2.0）
- 代码实现名：**Bailing**（如 `BailingMM2NativeForConditionalGeneration`）

同一套系统，名称不同。

### 1.2 代码分层

1. 入口层  
   - `README.md`, `cookbook.ipynb`, `test_infer.py`, `examples/*`
2. 编排层（多模态总装）  
   - `modeling_bailingmm2.py`
3. 主干 LLM 层（MoE）  
   - `modeling_bailing_moe_v2.py`
4. 模态处理层（视觉/音频/Processor）  
   - `processing_bailingmm2.py`  
   - `image_processing_bailingmm2.py`  
   - `audio_processing_bailingmm2.py`  
   - `qwen3_moe_vit.py`  
   - `modeling_whisper_encoder.py`
5. 可选扩展层  
   - 图像生成：`diffusion/*`, `bizgen/*`  
   - 语音生成：`modeling_bailing_talker.py`, `talker_module/*`, `AudioVAE/*`

---

## 2. 配置与 AutoClass 映射

### 2.1 模型配置（`config.json`）

- 主架构：`BailingMM2NativeForConditionalGeneration`
- 三个核心子配置：
  - `llm_config` -> `BailingMoeV2ForCausalLM`
  - `vision_config` -> `Qwen3MoeVisionTransformer`
  - `audio_config` -> `WhisperAudioEncoder`（含 whisper encoder 参数）

### 2.2 Processor 映射（`preprocessor_config.json`）

- `AutoProcessor` -> `processing_bailingmm2.BailingMM2Processor`
- `AutoImageProcessor` -> `BailingMM2ImageProcessor`
- `AutoFeatureExtractor` -> `BailingMM2AudioProcessor`

---

## 3. 初始化架构（from_pretrained）

入口：`BailingMM2NativeForConditionalGeneration.from_pretrained(...)`  
文件：`modeling_bailingmm2.py`

### 3.1 默认主干加载（`load_vlm=True`）

`__init__` 主要构造：

1. 视觉编码器  
   `self.vision = Qwen3MoeVisionTransformer(self.config.vision_config)`
2. 音频编码器  
   `self.audio = WhisperAudioEncoder(**self.config.audio_config.whisper_encoder_config)`
3. 语言模型  
   `self.model = BailingMoeV2ForCausalLM(self.config.llm_config)`
4. 对齐层
   - `linear_proj`：视觉特征投影到 LLM hidden size
   - `linear_proj_audio`：音频特征投影到 LLM hidden size

### 3.2 可选加载：图像生成分支

参数：`load_image_gen=True`  
调用：`load_image_gen_modules(...)`

加载内容：

- `query_tokens_dict`（多尺度可学习 token）
- `connector`（额外因果模型用于条件变换）
- `proj_in / proj_out`
- `diffusion_loss`（`SD3Loss` / `SANALoss` / `ZImageLoss`）
- 可选 `byt5` 文本条件增强

### 3.3 可选加载：Talker 分支

参数：`load_talker=True`  
加载：

- `model.talker = BailingTalker2.from_pretrained(...)`
- `model.talker_vae = AudioVAE.from_pretrained(...)`

---

## 4. 主执行链路（多模态文本生成）

以 `test_infer.py` 的 `generate()` 为标准路径。

### 4.1 Step A：消息到模型输入（Processor 侧）

调用序列：

1. `processor.apply_chat_template(messages, ...)`  
   生成统一 prompt 文本（含 system/user/assistant 结构）。
2. `processor.process_vision_info(messages)`  
   从消息内容中提取并读取 image/video/audio。
3. `processor(...)`  
   输出 `BatchFeature`，关键字段：
   - 文本：`input_ids`, `attention_mask`
   - 图像：`pixel_values`, `image_grid_thw`
   - 视频：`pixel_values_videos`, `video_grid_thw`
   - 音频：`audio_feats`, `audio_feats_lengths`, `audio_placeholder_loc_lens`

其中占位符扩展逻辑：

- `_expand_image_tokens`：`<IMAGE>` -> `<image> + N*<imagePatch> + </image>`
- `_expand_video_tokens`：`<VIDEO>` -> `<video> + N*<framePatch> + </video>`
- `_expand_audio_tokens`：`<AUDIO>` -> `<audio> + N*<audioPatch> + </audio>`

作用：保证 token 序列与后续特征 patch 数量一一对应。

### 4.2 Step B：BailingMM2 顶层 generate

入口：`BailingMM2NativeForConditionalGeneration.generate(...)`

1. 视觉特征提取（如有图像）
   - 输入：`pixel_values`, `image_grid_thw`
   - 调用：`extract_image_feature(...)`
   - 输出：`image_embeds`

2. 视频特征提取（如有视频）
   - 输入：`pixel_values_videos`, `video_grid_thw`
   - 调用：`extract_image_feature(...)`
   - 输出：`video_embeds`

3. 音频特征提取（如有音频）
   - 输入：`audio_feats`, `audio_feats_lengths`
   - 调用：`extract_audio_feature(...)`
   - 输出：`audio_embeds`, `audio_embeds_lengths`

4. 进入 LLM 生成
   调用 `self.model.generate(...)`，传入：
   - `query_embeds_image`
   - `query_embeds_video`
   - `query_embeds_audio`
   - `query_embeds_audio_lengths`
   - `placeholder_audio_loc_lens`
   - `image_grid_thw`
   - `image_grid_thw_video`

### 4.3 Step C：LLM 内部融合与解码

入口：`BailingMoeV2ForCausalLM.forward / generate`

#### 4.3.1 多模态注入（`prompt_wrap_navit`）

位置：`BailingMoeV2Model.prompt_wrap_navit(...)`

- `prompt_wrap_vision(...)`
  - 依据 `input_ids == image_patch_token / video_patch_token`
  - 将对应 token 的 embedding 替换为视觉/视频特征
- `prompt_wrap_audio(...)`
  - `patch_continuous_features(...)`
  - 依据 `placeholder_audio_loc_lens` 将连续音频特征写入 embedding
  - 同时构建 `audio_mask`

返回：

- `inputs_embeds`（已注入多模态）
- `vision_mask`
- `audio_mask`

#### 4.3.2 位置编码与 RoPE

`image_grid_thw` / `image_grid_thw_video` 参与 3D/video RoPE 的位置索引计算。  
这保证图像/视频 token 的空间与时序结构在注意力中被正确编码。

#### 4.3.3 MoE Router 分流

`BailingMoeV2SparseMoeBlock.forward(hidden_states, image_mask, audio_mask)`

在 `router_type=MultiRouter` 时：

- 同时计算 `gate / image_gate / audio_gate`
- 对视觉 token 用 `image_gate` 路由
- 对音频 token 用 `audio_gate` 路由
- 对文本 token 用默认 `gate` 路由

通过 `image_mask/audio_mask` 完成 token 级别路由切换。

#### 4.3.4 自回归生成与缓存

`prepare_inputs_for_generation(...)` 管理：

- `past_key_values`
- 裁剪后的 `input_ids`
- 动态 `position_ids`
- 继续携带多模态上下文参数（`query_embeds_*`, `grid_thw`, masks）

最终 `lm_head` 输出 logits，迭代直到 EOS 或达到 `max_new_tokens`。

### 4.4 Step D：输出解码

上层调用：

- `generated_ids_trimmed = out_ids[len(in_ids):]`
- `processor.batch_decode(...)`

输出最终文本。

---

## 5. 模块连接契约（I/O 视角）

### 5.1 Processor -> Model 契约

| 字段 | 生产者 | 消费者 | 说明 |
|---|---|---|---|
| `input_ids` | `BailingMM2Processor` | `BailingMM2.generate -> LLM` | 文本 token 序列（含模态 patch token） |
| `attention_mask` | `BailingMM2Processor` | `LLM.forward` | 有效 token mask |
| `pixel_values` | `BailingMM2ImageProcessor` | `extract_image_feature` | 图像 patch 序列 |
| `image_grid_thw` | `BailingMM2ImageProcessor` | `vision encoder` + RoPE | 图像 token 网格结构 |
| `pixel_values_videos` | `BailingMM2ImageProcessor` | `extract_image_feature` | 视频 patch 序列 |
| `video_grid_thw` | `BailingMM2ImageProcessor` | `vision encoder` + RoPE | 视频时空网格结构 |
| `audio_feats` | `BailingMM2AudioProcessor` | `extract_audio_feature` | 音频特征序列 |
| `audio_feats_lengths` | `BailingMM2AudioProcessor` | `extract_audio_feature` | 音频段长度 |
| `audio_placeholder_loc_lens` | `BailingMM2Processor` | `prompt_wrap_audio` | 音频占位位置与长度 |

### 5.2 编排层 -> LLM 契约

| 字段 | 生产者 | 消费者 | 作用 |
|---|---|---|---|
| `query_embeds_image` | `extract_image_feature` | `prompt_wrap_navit` | 替换图像 patch token embedding |
| `query_embeds_video` | `extract_image_feature` | `prompt_wrap_navit` | 替换视频 patch token embedding |
| `query_embeds_audio` | `extract_audio_feature` | `prompt_wrap_audio` | 覆盖音频占位段 embedding |
| `query_embeds_audio_lengths` | `extract_audio_feature` | `patch_continuous_features` | 控制每段音频写入长度 |
| `image_grid_thw/video_grid_thw` | Processor | LLM RoPE index | 位置编码结构信息 |

### 5.3 LLM 内部契约

| 字段 | 生产者 | 消费者 | 作用 |
|---|---|---|---|
| `vision_mask` | `prompt_wrap_navit` | MoE block | 标识视觉 token 路由 |
| `audio_mask` | `prompt_wrap_audio` | MoE block | 标识音频 token 路由 |
| `past_key_values` | 每步 forward | 下一步 generation | 增量解码缓存 |

---

## 6. 可选分支连接

### 6.1 Image Generation 分支（`image_gen=True`）

路径：

1. `get_condition_embeds_for_image_gen(...)`  
   从 LLM hidden states 组织图像条件向量
2. `connector + proj_out`  
   将条件映射到 diffusion 所需维度
3. `diffusion_loss.sample(...)`  
   采样生成图像

支持项：

- 负向条件（negative embeds）
- CFG / seed / steps
- 参考图条件（`image_gen_pixel_values_reference`）

### 6.2 Talker 分支（语音生成）

`BailingTalker2` 内部结构：

- 文本 backbone：`Qwen2Model`
- 时序生成：`CFM + DiT`
- 聚合器：`Aggregator`
- 声纹提取：`SpkembExtractor`（ONNX）
- token->wav：detokenizer/vae 流式解码

该分支在主 VLM 文本问答路径外独立运行，可通过 `load_talker=True` 挂载到主模型对象。

---

## 7. 端到端时序（简化）

```text
messages
  -> Processor(chat template + multimodal preprocess)
  -> input_ids / attention_mask / pixel_values / audio_feats / grids / placeholder_lens
  -> BailingMM2.generate
      -> vision/audio encode + projector
      -> BailingMoeV2.generate
          -> prompt_wrap_navit (inject embeds + masks)
          -> RoPE by grid_thw
          -> MoE decode with MultiRouter
          -> logits / next token
  -> decode text
```

---

## 8. 关键设计点

1. **统一 token 序列 + 外部特征注入**  
   文本中先放 patch 占位，后在 embedding 层替换/覆盖，避免在 decoder 结构内硬编码多模态分支。

2. **结构信息显式传递**  
   `image_grid_thw/video_grid_thw` 贯穿预处理、编码与 RoPE，保持时空一致性。

3. **模态感知 MoE 路由**  
   `MultiRouter + mask` 让不同模态 token 使用不同门控策略。

4. **可插拔扩展**  
   image_gen/talker 都通过 `from_pretrained` 开关挂载，不侵入基础文本推理路径。

---

## 9. 参考入口

- 文本/多模态推理：`test_infer.py`
- 图像生成示例：`test_infer_imagegen.py`
- 主模型实现：`modeling_bailingmm2.py`
- 主干 LLM 实现：`modeling_bailing_moe_v2.py`
- Processor：`processing_bailingmm2.py`

---

## 10. 函数级连接清单（可用于代码走查）

### 10.1 输入侧连接（Message -> BatchFeature）

| 上游函数 | 下游函数 | 连接对象 |
|---|---|---|
| `BailingMM2Processor.apply_chat_template` | `BailingMM2Processor.__call__` | 格式化后文本 prompt |
| `bailingmm_utils.process_vision_info` | `BailingMM2Processor.__call__` | 图片/视频/音频原始载荷 |
| `BailingMM2Processor._expand_image_tokens` | `tokenizer(...)` | `<IMAGE>` 替换为 patch token 序列 |
| `BailingMM2Processor._expand_video_tokens` | `tokenizer(...)` | `<VIDEO>` 替换为帧 patch token 序列 |
| `BailingMM2Processor._expand_audio_tokens` | `tokenizer(...)` | `<AUDIO>` 替换为音频 patch token 序列 |
| `BailingMM2ImageProcessor.preprocess` | `BailingMM2Processor.__call__` | `pixel_values`, `image_grid_thw`, `video_grid_thw` |
| `BailingMM2AudioProcessor.preprocess` | `BailingMM2Processor.__call__` | `audio_feats`, `audio_feats_lengths`, `encoder_feats_lengths` |

### 10.2 编排层连接（BatchFeature -> Query Embeds）

| 上游函数 | 下游函数 | 连接对象 |
|---|---|---|
| `BailingMM2NativeForConditionalGeneration.generate` | `extract_image_feature` | `pixel_values`, `image_grid_thw` |
| `BailingMM2NativeForConditionalGeneration.generate` | `extract_image_feature` | `pixel_values_videos`, `video_grid_thw` |
| `extract_image_feature` | `BailingMoeV2ForCausalLM.generate` | `query_embeds_image/query_embeds_video` |
| `extract_audio_feature` | `BailingMoeV2ForCausalLM.generate` | `query_embeds_audio/query_embeds_audio_lengths` |
| `BailingMM2Processor.__call__` | `BailingMoeV2ForCausalLM.generate` | `placeholder_audio_loc_lens` |

### 10.3 LLM 内部连接（Query Embeds -> Decoder）

| 上游函数 | 下游函数 | 连接对象 |
|---|---|---|
| `BailingMoeV2Model.prompt_wrap_vision` | `BailingMoeV2Model.forward` | 替换后的 `inputs_embeds` |
| `BailingMoeV2Model.prompt_wrap_audio` | `BailingMoeV2Model.forward` | 替换后的 `inputs_embeds` + `audio_mask` |
| `BailingMoeV2Model.prompt_wrap_navit` | `BailingMoeV2SparseMoeBlock.forward` | `image_mask` / `audio_mask` |
| `BailingMoeV2Model.forward` | `get_t_scale_rope_index/get_rope_index` | `image_grid_thw/video_grid_thw` |
| `BailingMoeV2SparseMoeBlock.forward` | experts 执行 | `topk_idx/topk_weight`（按模态 mask 分流） |
| `BailingMoeV2ForCausalLM.forward` | `lm_head` | `hidden_states -> logits` |
| `BailingMoeV2ForCausalLM.prepare_inputs_for_generation` | 下一步 `forward` | cache + 裁剪输入 + 多模态上下文 |

### 10.4 Image Gen 分支连接（可选）

| 上游函数 | 下游函数 | 连接对象 |
|---|---|---|
| `BailingMM2NativeForConditionalGeneration.generate(image_gen=True)` | `get_condition_embeds_for_image_gen` | LLM hidden states / image embeds |
| `get_condition_embeds_for_image_gen` | `connector -> proj_out` | diffusion 条件向量 |
| `load_image_gen_modules` | `self.diffusion_loss.sample` | diffusion 模块实例与参数 |

---

## 11. Mermaid 时序图

### 11.1 标准多模态文本生成

```mermaid
sequenceDiagram
    participant U as User Messages
    participant P as BailingMM2Processor
    participant M as BailingMM2Native
    participant V as Vision/Audio Encoders
    participant L as BailingMoeV2

    U->>P: messages (text/image/video/audio)
    P->>P: apply_chat_template + process_vision_info
    P->>P: __call__ -> BatchFeature
    P-->>M: input_ids/attention_mask/pixel_values/audio_feats/grids
    M->>V: extract_image_feature / extract_audio_feature
    V-->>M: query_embeds_image/video/audio
    M->>L: generate(query_embeds_*, grids, placeholder_lens)
    L->>L: prompt_wrap_navit (inject embeds + masks)
    L->>L: RoPE by image_grid_thw/video_grid_thw
    L->>L: MoE decode with MultiRouter
    L-->>M: generated_ids
    M-->>P: generated_ids_trimmed
    P-->>U: batch_decode(text)
```

### 11.2 图像生成分支

```mermaid
sequenceDiagram
    participant P as Processor
    participant M as BailingMM2Native
    participant L as BailingMoeV2
    participant D as Diffusion

    P-->>M: input_ids/attention_mask/(optional)pixel_values
    M->>L: forward to get hidden states (or use provided llm_hidden_states)
    L-->>M: condition hidden states
    M->>M: get_condition_embeds_for_image_gen
    M->>D: diffusion_loss.sample(condition, negative_condition, cfg, steps)
    D-->>M: generated image(s)
```

