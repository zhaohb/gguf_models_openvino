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
        #query = "请总结以下内容： 憨憨”总是伴随、提供着有趣 的梗，会让受众感受到娱乐性，而不容易被厌烦。\n对于 艺人来说，偶尔的憨憨行 为、语言，又可以拉近与大众 的距离而显得接地气许多，同时又能呈现出反差萌的感觉，而更容易引起更多 路 人盘的好感。\n所以，一贯印象中霸气、美艳的张雨绮在节目中拿到X卡并高呼“无限可能”时，“铁憨憨”也是 观众对其完全不同的另一 面的反馈和评价。赵露思在《哦！我的皇帝陛下》《传 闻 中的陈芊芊》等剧中可爱而搞笑的角色的加持下，也将这一“憨憨”人设延续到了日常，为手滑点赞给宋茜道歉等以及片场花絮中的行为、言语，也一度受到不少好评，让不少人对她印象深刻。\n《传闻中的陈芊芊》剧照\n同时，“憨憨”人设也有利于拓展综艺等资源。毕竟“憨憨”能引发的笑果是直观的，如果要有明确的梗和标签更是加分，而综艺节目也需要有梗、笑果的人来增强综艺感又不会让综艺感显得那么刻意。可以看到，不少被形容为“铁憨憨”的艺人，也经常出现在不少综艺节目中，并大多能拥有不错的适配性，为节目提供综艺感。\n或许正是因为“憨论、控评的方 法之 一。如上文所说，除了艺人，“憨憨”人 设成为公 关、洗白方式 已经蔓延到多个领域。\n自嘲、为大众提供梗和笑料，从某种程度上能够降低甚至转变大众或批评、或气愤等的 态度。“‘弱智’ 无罪嘛，  大家一般都会原谅‘傻子’”。一位粉丝对壹娱观察（ID：yiyuguancha）解释道。\n诚然，艺人偶尔表现出憨憨行为大概率是 可以加分的 ，但如果涉及到立人设则情况可能会出现 不同 。因 为人设是一个持续性、需要不断巩固的过程，这对于更广泛的人群来说，首先面临的是持续的相信感的问题，一旦失误，很可能出现反噬。而“憨憨”人设对于艺人本身也存在一些伤害。\n 粉丝 都说好，路人已看清\n对于 艺人和经纪公司来说，粉丝是基础和首先要维护的对象，但更大的路人盘也是不能放 弃的高地。如上文所说 ，憨憨人设算得上吸引路人好感的方式之一，但却并不是万无一失。\n豆瓣上关于吐槽明星走白痴、无出现反噬知、铁憨憨、“白痴美人”等人设，以及粉丝给偶像立铁憨憨等帖子不在少数。\n有网友表示：“不喜欢明明蠢、白痴还凹成单纯人设，也 不喜欢 把无知当 可爱人设。”“还是喜欢聪明 人， 聪明不外露的那种。”“真的搞不懂怎么想的，在娱乐圈这么多年还蠢萌蠢萌，你当别人先不说本身就是娱乐圈 里的偶像，怎么可能不去注意实时的娱乐 动态？那你也可以说他专心自己的工作不去看这些影响心情，但这么大了连包邮是什么都不知 道？这种憨憨男孩的不问世事的 人设偶尔立立还好，但一过头。 就会让人感觉很刻意。”\n网络截图\n 然而，在周震南事件的评论区里 ，还有大批粉丝在表达“周震南只 是个孩子，是个憨憨”的类似话术，这个时候，“憨憨”二字无疑进一步刺痛了万千网友的神经。“讨伐” 周震南的声浪也 随之更高  。\n显然 ，不真实的观感很容易导致大众的不信服，并且容易获得吐槽声，从而 适得其反。而一旦翻车，过往历史就会成为群嘲的素材来源。\n对于艺人来说，尤其是演员，“憨憨” 人设还可 能 会影响其作品的观感。但是你知道没打飒飒的撒 法撒旦的撒大苏打大\n立人设的目的之一就是让艺人拥有一个明显的标签，从而能够具有独特性而被 大众认知。这就意味着，这个人设 会在大 众心中留 下深刻印象。试想， 如果一个演员在综艺、社交媒体总呈现出“憨憨”的一面，那么他再尝试诠释各种不同性格角色，出演不同风格影视剧作品时，大概率会影响受众的代入感 ，而让受众产生出戏的感觉。这对于艺人，尤其是演员，算得上是一种伤害。\n 在这点上，邓超或许算得上一个代表， 无论是“跑 男”系列以 及最新综艺节目《哈哈哈哈哈》带来的直观感受，还是在微博上与网友和 孙俪的互动，频繁出现的“超哥，你真是 个憨憨”一定 程度上也 在降弱大众对于他演员身份的认知。\n真人秀综艺《哈哈哈哈哈》\n另 外，对于“憨憨”集中出现的偶像爱豆行业 ，这两个字被看到很多次后，团队也该有所警惕，毕竟对于演员和歌手来说，他们还能靠演技 和作品进行“弥补”，去获得市场新的认可，而国内的偶像产业，在舞台缺失的情况下，往返于综艺和低成本网剧里的爱豆们，不迅速去寻找自身的定位，而浸入“憨憨”的蜜罐，甚至上升到固定的人设，那么， 在偶像更迭加速的形势下，国内的偶像筛选机制和出道体系也欠 缺严谨，因此， 这群爱豆们的“ 憨憨”好立， 但更容易被毁，那时，团队也只好忍泪买单，而偶像们也只好加速自身的淘汰。\n诚然，艺人是可以展现出自己“憨憨”一面的。如上文所说，这一面的展现可以是真实性 格一面的体 现，也可以显得接地气而更具亲和题。其一在于如上述豆瓣网友评论展现的那样，“憨”总与“无知”“白痴”等联系在一起。这也说明，如果把握不好度的问题，“憨憨”也无法带给受众  真实性。\n另外，立人设需要持续性，但同时 也就存在危险性，很容易引起反噬，尤其是与本身性格、特质相差甚远的，翻车的可能性 就更大。\n网友已经对不久之前频频登上热度的赵露思有了吐槽：“营 销过度真的会反噬，不要立什么傻白甜憨憨人设了，戏太多了适得其反。”“其实一开始知道赵露思的时候觉得这个女孩憨憨傻傻的很可爱亲切，但…这一系列操作真的无语了。”\n而前不久周震南父母老赖事件出现并持续发酵后，“2G少年”等也成为网友吐槽的点之一。“平时 接触到他的营销就是什么2G网、我周震南了，说本身就是娱乐圈里的偶像，怎么可能不去注意实时的娱乐动态？那你也可以说他专心自己的工作不去看这些影响心情，但这么大了连包邮是什么都不知道？这种憨憨男孩的不问世事的人设偶尔立立还好，但一过头。就会让人感觉很刻意。”\n网络截 图\n然而，在 周震南 事件的评论区里，还有大批粉丝在表达“周震南笑果是直观的只是个 孩子，是个憨憨”的类似话 术，这个时候，“憨憨”二字无疑进一步刺痛了万千网友的神经。“讨伐”周震 南的声浪也随之更高。\n显然， 不真实的观感很容易导致大众的不信服，并且容易获得吐槽声，从而适得其反。而一旦翻车，过往历史就会成为群嘲的素材来源。\n对于艺人来说，尤其是演员，“憨 憨”人设 还可能会 影响其作 品的观感。\n立人设的目的之 一就是让艺人拥有一个明显的标签，从而能够具 有独特性而被大众认知。这就意 味着， 这个人设会在大众心中留下深刻印象。试 想，如果一 个演 员在综艺、社交媒体总呈现出“憨憨”的一面，那么他再尝试诠释各种不同性格角色，出演不同风格影视剧作品时，大概率会影响受众的代 入感，而让受众产生出戏的感觉。这对 于艺人，尤其是演员，算得上是一种伤害。\n在这点上，邓超或许算得上一个代表 ，无论是“跑男”系列以及最新综艺节目《哈哈哈哈 哈》带来的直观感受，还是在微博上与网友和孙俪的互动，频繁出现的“超哥， 你真 是个 憨憨”一定程度上也在降弱大众对于他演员身份的认知。\n真人秀综艺《哈哈哈哈哈》\n另外，对于“憨憨”集中出现是让艺人拥有一个明显的标签，从而能够具有独特性而被大众认知。这就意味 着，这个人设会在大众心中留下深刻印象。试想，如果一个演员在综艺、 社交媒体总呈现出“憨憨”的一面，那么他再尝试诠释 各种不同性格角色，出演不同风格影视 剧作品时，大概率 会影响受众的代入感， 而让受众产生出戏的感觉。这对于艺人，尤其是演员，算得上是一种伤害。\n在这点上，邓超或许算得上一个代表 ，无论是“跑男”系列 以及最新综艺节 目《哈哈哈哈哈》带来的直观感受，还 是在微博上与网友和孙俪的互动，频繁出现的“ 超哥，你真是个憨憨”一定程度 上也在降弱大众对于他演员身份的认知。\n真人秀综艺《 哈哈哈哈哈》\n另 外，对于“憨憨”集中 出现的偶 像爱豆行业，这两个字被看到很多次后， 团队也该有所警惕，毕竟对于演员和歌手来 说，他们还能靠演技和作品进行“弥补”，去获得市场新的认?"
        #query = "请总结以下内容： 憨憨”总是伴随、提供着有趣 的梗，会让受众感受到娱乐性，而不容易被厌烦。\n对于 艺人来说，偶尔的憨憨行 为、语言，又可以拉近与大众 的距离而显得接地气许多，同时又能呈现出反差萌的感觉，而更容易引起更多 路 人盘的好感。\n所以，一贯印象中霸气、美艳的张雨绮在节目中拿到X卡并高呼“无限可能”时，“铁憨憨”也是 观众对其完全不同的另一 面的反馈和评价。赵露思在《哦！我的皇帝陛下》《传 闻 中的陈芊芊》等剧中可爱而搞笑的角色的加持下，也将这一“憨憨”人设延续到了日常，为手滑点赞给宋茜道歉等以及片场花絮中的行为、言语，也一度受到不少好评，让不少人对她印象深刻。\n《传闻中的陈芊芊》剧照\n同时，“憨憨”人设也有利于拓展综艺等资源。毕竟“憨憨”能引发的笑果是直观的，如果要有明确的梗和标签更是加分，而综艺节目也需要有梗、笑果的人来增强综艺感又不会让综艺感显得那么刻意。可以看到，不少被形容为“铁憨憨”的艺人，也经常出现在不少综艺节目中，并大多能拥有不错的适配性，为节目提供综艺感。\n或许正是因为“憨论、控评的方 法之 一。如上文所说，除了艺人，“憨憨”人 设成为公 关、洗白方式 已经蔓延到多个领域。\n自嘲、为大众提供梗和笑料，从某种程度上能够降低甚至转变大众或批评、或气愤等的 态度。“‘弱智’ 无罪嘛，  大家一般都会原谅‘傻子’”。一位粉丝对壹娱观察（ID：yiyuguancha）解释道。\n诚然，艺人偶尔表现出憨憨行为大概率是 可以加分的 ，但如果涉及到立人设则情况可能会出现 不同 。因 为人设是一个持续性、需要不断巩固的过程，这对于更广泛的人群来说，首先面临的是持续的相信感的问题，一旦失误，很可能出现反噬。而“憨憨”人设对于艺人本身也存在一些伤害。\n 粉丝 都说好，路人已看清\n对于 艺人和经纪公司来说，粉丝是基础和首先要维护的对象，但更大的路人盘也是不能放 弃的高地。如上文所说 ，憨憨人设算得上吸引路人好感的方式之一，但却并不是万无一失。\n豆瓣上关于吐槽明星走白痴、无出现反噬知、铁憨憨、“白痴美人”等人设，以及粉丝给偶像立铁憨憨等帖子不在少数。\n有网友表示：“不喜欢明明蠢、白痴还凹成单纯人设，也 不喜欢 把无知当 可爱人设。”“还是喜欢聪明 人， 聪明不外露的那种。”“真的搞不懂怎么想的，在娱乐圈这么多年还蠢萌蠢萌，你当别人先不说本身就是娱乐圈 里的偶像，怎么可能不去注意实时的娱乐 动态？那你也可以说他专心自己的工作不去看这些影响心情，但这么大了连包邮是什么都不知 道？这种憨憨男孩的不问世事的 人设偶尔立立还好，但一过头。 就会让人感觉很刻意。”\n网络截图\n 然而，在周震南事件的评论区里 ，还有大批粉丝在表达“周震南只 是个孩子，是个憨憨”的类似话术，这个时候，“憨憨”二字无疑进一步刺痛了万千网友的神经。“讨伐” 周震南的声浪也 随之更高  。\n显然 ，不真实的观感很容易导致大众的不信服，并且容易获得吐槽声，从而 适得其反。而一旦翻车，过往历史就会成为群嘲的素材来源。\n对于艺人来说，尤其是演员，“憨憨” 人设还可 能 会影响其作品的观感。但是你知道没打飒飒的撒 法撒旦的撒大苏打大\n立人设的目的之一就是让艺人拥有一个明显的标签，从而能够具有独特性而被 大众认知。这就意味着，这个人设 会在大 众心中留 下深刻印象。试想， 如果一个演员在综艺、社交媒体总呈现出“憨憨”的一面，那么他再尝试诠释各种不同性格角色，出演不同风格影视剧作品时，大概率会影响受众的代入感 ，而让受众产生出戏的感觉。这对于艺人，尤其是演员，算得上是一种伤害。\n 在这点上，邓超或许算得上一个代表， 无论是“跑 男”系列以 及最新综艺节目《哈哈哈哈哈》带来的直观感受，还是在微博上与网友和 孙俪的互动，频繁出现的“超哥，你真是 个憨憨”一定 程度上也 在降弱大众对于他演员身份的认知。\n真人秀综艺《哈哈哈哈哈》\n另 外，对于“憨憨”集中出现的偶像爱豆行业 ，这两个字被看到很多次后，团队也该有所警惕，毕竟对于演员和歌手来说，他们还能靠演技 和作品进行弥补"
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
