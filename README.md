## Update Notes
### 2024/12/12
1. Use gguf model to convert to pytorch format. Now only hunyuan-3b model is tested.
2. Hunyuan's pytorch model can now be accelerated using OpenVINO.

## Running Guide
### Installation


```bash
git clone https://github.com/zhaohb/hunyuan-torch.git
pip install transformers==4.46.3
pip install torch
pip install torchvision

```
### Convert GGUF model to PyTorch and testing:
```shell
cd hunyuan-torch
#Weight format conversion
python3 convert_hunyuan_gguf_to_torch.py --input ./hunyuan-3b-Q4_0.gguf --output ./test --just_weights

# Now you can get the pytorch_model.bin file in the test directory, which can be used by transformers

#In the hunyuan_3b_dq_fp16 directory, we have prepared various files for transformers, we just copy pytorch_model.bin.
cp test/pytorch_model.bin hunyuan_3b_dq_fp16  
```
### TransFormers infer
```shell
python3 hunyuan_infer.py

output:
    query:  你是谁?
    answer:  我是混元助手，一个由腾讯开发的大型语言模型。我具备丰富的语义理解和计算能力，可以为用户提供问答式的服务，例如回答问题和提供建议。有什么可以帮助您的吗？<|eos|>
```
### Convert HunYuan PyTorch model to OpenVINO and testing (Intel(R) Xeon(R) Gold 6252N CPU @ 2.30GHz):
#### convert
```shell
python3 test_ov.py -m hunyuan_3b_dq_fp16/ -ov ./hunyuan_ov -llm_int4_com -llm_int8_quant -convert_model_only
cp hunyuan_3b_dq_fp16/merges.txt hunyuan_ov/
cp hunyuan_3b_dq_fp16/tokenizer_config.json hunyuan_ov/
cp hunyuan_3b_dq_fp16/vocab.json hunyuan_ov/
```
#### FP16 testing
```shell
#linux
python3 test_ov.py -m /path/to/hunyuan_3b -ov hunyuan-ov 

#output
INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, onnx, openvino
OpenVINO version 
 2024.5.0-17288-7975fa5da0c-refs/pull/3856/head


query:  你是谁?
answer:  你好，我是混元助手，一个由腾讯开发的大型语言模型。我具备丰富的语义理解和计算能力，可以为用户提供问答式的服务。有什么可以帮助您的吗？<|eos|>

性能测试中...


LLM Model First token latency: 97.87 ms, Output len: 12, Avage token latency: 80.27 ms
```
#### INT4 compress + INT8 dynamic quant + insertslice opt
```shell
python3 test_ov.py -m /path/to/hunyuan_3b -ov hunyuan-ov -llm_int4_com -llm_int8_quant  

#output
INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, onnx, openvino
OpenVINO version 
 2024.5.0-17288-7975fa5da0c-refs/pull/3856/head


query:  你是谁?
answer:  你好，我是混元助手，一个由腾讯开发的大型语言模型。我具备理解人类语言的能力，能够应对各种复杂场景下的对话需求，包括解答问题、提供建议等。欢迎您向我提问任何问题。<|eos|>

性能测试中...

LLM Model First token latency: 44.60 ms, Output len: 13, Avage token latency: 34.10 ms
```
### LNL iGPU INT4 compress + INT8 dynamic quant + insertslice opt (You must use this ov branch: https://github.com/zhaohb/openvino/tree/hunyuan_2024_6)
```shell
python test_ov.py -m /path/to/hunyuan_3b -ov hunyuan-ov -d GPU -llm_int4_com -llm_int8_quant
INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, openvino
OpenVINO version
 2024.6.0-17418-38261652a93-refs/pull/28116/head


query:  你是谁?
answer:  你好，我是腾讯混元大模型，一个由腾讯研发的大语言模型。我具备理解人类语言的能力，能够应对各种不同场景的问答需求 ，例如：回答问题、提供建议等。你可以向我提问任何问题，我会尽我所能为你提供准确、有用的信息。<|eos|>

性能测试中...


LLM Model First token latency: 37.58 ms, Output len: 11, Avage token latency: 25.31 ms
```
### Parsing test_ov.py's arguments :
```shell
python3 test_ov.py  --help
usage: Export HunYuan Model to IR [-h] [-m MODEL_ID] -ov OV_IR_DIR [-d DEVICE] [-p PROMPT] [-max MAX_NEW_TOKENS] [-llm_int4_com]
                                  [-llm_int8_quant] [-convert_model_only]

options:
  -h, --help            show this help message and exit
  -m MODEL_ID, --model_id MODEL_ID
                        model_id or directory for loading
  -ov OV_IR_DIR, --ov_ir_dir OV_IR_DIR
                        output directory for saving model
  -d DEVICE, --device DEVICE
                        inference device
  -p PROMPT, --prompt PROMPT
                        prompt
  -max MAX_NEW_TOKENS, --max_new_tokens MAX_NEW_TOKENS
                        max_new_tokens
  -llm_int4_com, --llm_int4_compress
                        llm int4 weights compress
  -llm_int8_quant, --llm_int8_quant
                        llm int8 weights quantize
  -convert_model_only, --convert_model_only
                        convert model to ov only, do not do inference test
```
