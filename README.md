## Update Notes
### 2024/12/12
1. Use gguf model to convert to pytorch format. Now only hunyuan-3b model is tested.

## Running Guide
### Installation


```bash
git clone https://github.com/zhaohb/hunyuan-torch.git
pip install transformers==4.44.2
pip install torch
pip install torchvision

```
### Convert GGUF model to PyTorch and testing:
```shell
cd hunyuan-torch
#Weight format conversion
python3 convert_hunyuan_gguf_to_torch.py --input ./hunyuan-3b-Q4_0.gguf --output ./test --just_weights

# Now you can get the pytorch_model.bin file in the test directory, which can be used by transformers

cp test/pytorch_model.bin hunyuan_3b_dq_fp16  #In the hunyuan_3b_dq_fp16 directory, we have prepared various files for transformers reasoning.
```
### TransFormers infer
```shell
python3 hunyuan_infer.py

output:
    query:  你是谁?
    answer:  我是混元助手，一个由腾讯开发的大型语言模型。我具备丰富的语义理解和计算能力，可以为用户提供问答式的服务，例如回答问题和提供建议。有什么可以帮助您的吗？<|eos|>
```