from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from hunyuan_3b_dq_fp16.hunyuan_tokenizer import HunYuanTokenizer, encode_tokens
import torch
import json, os

def apply_chat_template(query):
    return '<|startoftext|>' + query + '<|extra_4|>'

def hunyuan_torch_infer(model_path=None):
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to('cpu').eval()

    query = '你是谁?'
    print("query: ", query)
    query = apply_chat_template(query)

    tokenizer_path = ("./hunyuan_3b_dq_fp16/vocab.json", "./hunyuan_3b_dq_fp16/merges.txt", "./hunyuan_3b_dq_fp16/tokenizer_config.json")
    tokenizer = HunYuanTokenizer(*tokenizer_path)
    tokenizer_conf = json.load(open("./hunyuan_3b_dq_fp16/tokenizer_config.json"))
    special_tokens = {v["content"]: k for k, v in tokenizer_conf.get("added_tokens_decoder", {}).items()}

    encoded = encode_tokens(tokenizer, special_tokens, query,
                            bos=tokenizer_conf.get("add_bos_token", False))
    encoded = torch.tensor([encoded.tolist()], dtype=torch.int32)

    # copy from https://hf-mirror.com/tencent/Tencent-Hunyuan-Large/blob/main/Hunyuan-A52B-Instruct/generation_config.json
    # Get the eos_toeken_id/pad_token_id of the hunyuan model
    generation_config = dict(
                            max_new_tokens=64, top_k=20, top_p=0.6, temperature=0.7, 
                            do_sample=True, eos_token_id=[127960, 127967], pad_token_id=127961)
    result = model.generate(input_ids=encoded, **generation_config)

    print("answer: ", tokenizer.decode(result.tolist()[0][len(encoded[0]):]))

if __name__ == '__main__':
    hunyuan_torch_infer('./hunyuan_3b_dq_fp16')
