import argparse
import openvino as ov
from pathlib import Path
from ov_hunyuan import OVHunYuanForCausalLM, HunYuan_OV
from transformers import TextStreamer
import time, json
from hunyuan_3b_dq_fp16.hunyuan_tokenizer import HunYuanTokenizer, encode_tokens
import torch
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser("Export HunYuan Model to IR", add_help=True)
    parser.add_argument("-m", "--model_id", required=False, help="model_id or directory for loading")
    parser.add_argument("-ov", "--ov_ir_dir", required=True, help="output directory for saving model")
    parser.add_argument('-d', '--device', default='CPU', help='inference device')
    parser.add_argument('-p', '--prompt', default="Describe this image.", help='prompt')
    parser.add_argument('-max', '--max_new_tokens', default=512, help='max_new_tokens')
    parser.add_argument('-llm_int4_com', '--llm_int4_compress', action="store_true", help='llm int4 weights compress')
    parser.add_argument('-llm_int8_quant', '--llm_int8_quant', action="store_true", help='llm int8 weights quantize')
    parser.add_argument('-convert_model_only', '--convert_model_only', action="store_true", help='convert model to ov only, do not do inference test')

    args = parser.parse_args()
    model_id = args.model_id
    ov_model_path = args.ov_ir_dir
    device = args.device
    max_new_tokens = int(args.max_new_tokens)
    question = args.prompt
    llm_int4_compress = args.llm_int4_compress
    llm_int8_quant = args.llm_int8_quant
    convert_model_only=args.convert_model_only

    if not Path(ov_model_path).exists():
        minicpm3_ov = HunYuan_OV(pretrained_model_path=model_id, ov_model_path=ov_model_path, device=device, llm_int4_compress=llm_int4_compress)
        minicpm3_ov.export_vision_to_ov()
        del minicpm3_ov.model
        # del minicpm3_ov.tokenizer
        del minicpm3_ov
    elif Path(ov_model_path).exists() and llm_int4_compress is True and not Path(f"{ov_model_path}/llm_stateful_int4.xml").exists():
        minicpm3_ov = HunYuan_OV(pretrained_model_path=model_id, ov_model_path=ov_model_path, device=device, llm_int4_compress=llm_int4_compress)
        minicpm3_ov.export_vision_to_ov()
        del minicpm3_ov.model
        # del minicpm3_ov.tokenizer
        del minicpm3_ov
    
    if not convert_model_only:
        llm_infer_list = []
        core = ov.Core()
        hunyuan_model = OVHunYuanForCausalLM(core=core, ov_model_path=ov_model_path, device=device, llm_int4_compress=llm_int4_compress, llm_int8_quant=llm_int8_quant, llm_infer_list=llm_infer_list)

        version = ov.get_version()
        print("OpenVINO version \n", version)
        print('\n')

        generation_config = dict(
                            max_new_tokens=64, top_k=20, top_p=0.6, temperature=0.7, 
                            do_sample=True, eos_token_id=[127960, 127967], pad_token_id=127961)
        query = '你是谁?'
        print("query: ", query)
        query = hunyuan_model.apply_chat_template(query)

        encoded = encode_tokens(hunyuan_model.tokenizer, hunyuan_model.special_tokens, query,
                                bos=hunyuan_model.tokenizer_conf.get("add_bos_token", False))
        encoded = torch.tensor([encoded.tolist()], dtype=torch.int32)

        inputs_embeds = hunyuan_model.get_input_embeds(input_ids=encoded)

        model_outputs = hunyuan_model.generate(
            inputs_embeds=inputs_embeds,
            **generation_config,
        )
        print("answer: ", hunyuan_model.tokenizer.decode(model_outputs.tolist()[0]))

        print("\n性能测试中...")
        for i in range(2):
            model_outputs = hunyuan_model.generate(
                inputs_embeds=inputs_embeds,
                **generation_config,
            )

        print("\n")
        if len(llm_infer_list) > 1:
            avg_token = sum(llm_infer_list[1:]) / (len(llm_infer_list) - 1)
            print(f"LLM Model First token latency: {llm_infer_list[0]:.2f} ms, Output len: {len(llm_infer_list) - 1}, Avage token latency: {avg_token:.2f} ms")
