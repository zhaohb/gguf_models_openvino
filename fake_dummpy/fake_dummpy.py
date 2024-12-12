from hunyuan_3b_dq_fp16.modeling_hunyuan import HunYuanForCausalLM
from hunyuan_3b_dq_fp16.configuration_hunyuan import HunYuanConfig
import torch

llamaconfig = HunYuanConfig.from_pretrained("config.json")

model = HunYuanForCausalLM(llamaconfig)

output_dir = "hunyuan_dummpy/models"

model.save_pretrained(output_dir,
    safe_serialization=False,  # 关闭安全序列化
    save_config=True  # 保存配置
    )

