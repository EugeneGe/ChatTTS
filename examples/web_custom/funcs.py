import random
from typing import Optional
from time import sleep

import gradio as gr

import sys

sys.path.append("..")
sys.path.append("../..")
from tools.audio import float_to_int16, has_ffmpeg_installed, load_audio
from tools.logger import get_logger

logger = get_logger(" WebUI ")

from tools.seeder import TorchSeedContext
from tools.normalizer import normalizer_en_nemo_text, normalizer_zh_tn

import ChatTTS

chat = ChatTTS.Chat(get_logger("ChatTTS"))

custom_path: Optional[str] = None

has_interrupted_text = False
is_in_generate_text = False
has_interrupted_audio = False
is_in_generate_audio = False

seed_min = 1
seed_max = 4294967295

use_mp3 = has_ffmpeg_installed()
if not use_mp3:
    logger.warning("没有安装 ffmpeg，使用 wav 文件输出")

# 音色选项：用于预置合适的音色 TODO 如果想要自定义音色，需要录音使用其他方式匹配，获取音色，放入下面选项中
voices = {
    "默认": {"seed": 300},
    "磁性男声": {"seed": 900},
    "知心姐姐": {"seed": 1400},
    "知心哥哥": {"seed": 1500},
    "青春女声": {"seed": 1700},
    "青春女声2": {"seed": 3400},
    "清晰男声": {"seed": 7400},
    "Timbre8": {"seed": 8888}, #10100
    "Timbre9": {"seed": 9999},
}


def generate_seed():
    return gr.update(value=random.randint(seed_min, seed_max))


# 返回选择音色对应的seed
def on_voice_change(vocie_selection):
    return voices.get(vocie_selection)["seed"]


def on_audio_seed_change(audio_seed_input):
    with TorchSeedContext(audio_seed_input):  # 进入种子上下文，设置 torch 的随机种子
        rand_spk = chat.sample_random_speaker()  # 生成随机 speaker
    return rand_spk  # 返回 speaker


def load_chat(cust_path: Optional[str], coef: Optional[str]) -> bool:
    if cust_path == None:
        ret = chat.load(coef=coef)
    else:
        logger.info("local model path: %s", cust_path)
        ret = chat.load("custom", custom_path=cust_path, coef=coef)
        global custom_path
        custom_path = cust_path
    if ret:
        try:
            chat.normalizer.register("en", normalizer_en_nemo_text())
        except ValueError as e:
            logger.error(e)
        except:
            logger.warning("Package nemo_text_processing not found!")
            logger.warning(
                "Run: conda install -c conda-forge pynini=2.1.5 && pip install nemo_text_processing",
            )
        try:
            chat.normalizer.register("zh", normalizer_zh_tn())
        except ValueError as e:
            logger.error(e)
        except:
            logger.warning("Package WeTextProcessing not found!")
            logger.warning(
                "Run: conda install -c conda-forge pynini=2.1.5 && pip install WeTextProcessing",
            )
    return ret


def reload_chat(coef: Optional[str]) -> str:
    """重新加载聊天模型，并校验 DVAE 系数"""
    global is_in_generate, custom_path

    if is_in_generate:
        raise gr.Error("正在生成内容，无法重新加载！")

    try:
        chat.unload()
        gr.Info("模型已卸载。")
    except Exception as e:
        raise gr.Error(f"模型卸载失败：{e}")

    # 校验 coef 参数
    if not isinstance(coef, str) or len(coef) != 230:
        gr.Warning("无效的 DVAE 系数，已忽略。")
        coef = None

    try:
        ret = load_chat(custom_path, coef)
    except Exception as e:
        raise gr.Error(f"模型加载失败：{str(e)}")

    if not ret or not hasattr(chat, "coef"):
        raise gr.Error("模型加载失败，或缺少必要的系数。")

    gr.Info("模型重新加载成功。")
    return chat.coef


def on_upload_sample_audio(sample_audio_input: Optional[str]) -> str:
    if sample_audio_input is None:
        return ""
    sample_audio = load_audio(sample_audio_input, 24000)
    spk_smp = chat.sample_audio_speaker(sample_audio)
    del sample_audio
    return spk_smp


def refine_text(
        text,  # 输入文本
        text_seed_input,  # 设定文本生成的随机种子
        refine_text_flag,  # 是否启用文本优化
        temperature,  # 温度参数，控制生成文本的随机性
        top_P,  # 采样时使用的 top-p (核采样) 参数
        top_K,  # 采样时使用的 top-k 参数
        split_batch,  # 是否拆分文本进行优化（大于 0 表示拆分）
):
    global chat  # 使用全局 chat 对象进行推理
    print(text,text_seed_input,refine_text_flag,temperature,top_P,top_K,split_batch)
    # 如果不启用文本优化，则直接返回原始文本
    if not refine_text_flag:
        sleep(1)  # 休眠 1 秒，防止 UI 加载标记过快消失
        return text

    # 调用 chat 模型进行文本优化
    text = chat.infer(
        text,  # 输入文本
        skip_refine_text=False,  # 不跳过文本优化
        refine_text_only=True,  # 仅执行文本优化
        params_refine_text=ChatTTS.Chat.RefineTextParams(  # 传入优化参数
            temperature=temperature,  # 影响文本生成的多样性
            top_P=top_P,  # 核采样阈值
            top_K=top_K,  # 仅从前 K 个最高概率的 token 中采样
            manual_seed=text_seed_input,  # 设定随机种子，保证可复现性
        ),
        split_text=split_batch > 0,  # 是否拆分文本进行优化
    )

    # 如果返回结果是列表，则取第一个元素，否则直接返回文本
    return text[0] if isinstance(text, list) else text


def generate_audio(
        text,  # 需要转换为语音的文本
        temperature,  # 影响语音生成的随机性
        top_P,  # 采样时的 top-P (核采样) 参数
        top_K,  # 采样时的 top-K 参数
        spk_emb_text: str,  # 说话人嵌入（用于控制音色）
        stream,  # 是否以流式方式返回音频
        audio_seed_input,  # 设定随机种子，保证音频一致性
        split_batch,  # 是否拆分文本进行生成（适用于长文本）
):
    global chat  # 使用全局 chat 实例进行语音推理
    print(text,temperature,top_P,top_K,spk_emb_text,stream,audio_seed_input,split_batch)
    # 1. 检查输入是否合法
    # - 如果文本为空，或者说话人嵌入不以 "蘁淰" 开头（说明无效），直接返回 None
    if not text or not spk_emb_text.startswith("蘁淰"):
        return None

    # 2. 构造音频生成参数
    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb=spk_emb_text,  # 设定说话人嵌入
        temperature=temperature,  # 影响生成的音频多样性
        top_P=top_P,  # 采样时的核采样参数
        top_K=top_K,  # 采样时的前 K 采样参数
        manual_seed=audio_seed_input,  # 设定随机种子，保证一致性
    )

    # 3. 如果提供了示例文本和示例音频编码，则使用它们进行风格匹配
    # 4. 调用 chat 进行语音生成
    wav = chat.infer(
        text,  # 输入文本
        skip_refine_text=True,  # 跳过文本优化，直接生成音频
        params_infer_code=params_infer_code,  # 传入语音生成参数
        stream=stream,  # 是否流式返回
        split_text=split_batch > 0,  # 是否拆分文本进行处理
        max_split_batch=split_batch,  # 设置拆分的最大批次
    )

    # 5. 处理返回的音频数据
    if stream:
        # 流式生成：逐步返回生成的音频数据
        for gen in wav:
            audio = gen[0]  # 获取生成的音频
            if audio is not None and len(audio) > 0:
                yield 24000, float_to_int16(audio).T  # 转换格式并返回音频
            del audio  # 释放变量，减少内存占用
    else:
        # 非流式生成：直接返回完整音频
        yield 24000, float_to_int16(wav[0]).T  # 转换格式并返回音频


def _set_generate_text_buttons(is_generating):
    """更新按钮状态"""
    return gr.update(visible=not is_generating, interactive=not is_generating), \
        gr.update(visible=is_generating, interactive=is_generating)


def interrupt_generate_text():
    """终止文本生成"""
    global chat, has_interrupted_text
    has_interrupted_text = True
    chat.interrupt()


def set_buttons_before_generate_text():
    """开始生成文本前，更新按钮状态"""
    global has_interrupted_text, is_in_generate_text
    has_interrupted_text = False
    is_in_generate_text = True
    return _set_generate_text_buttons(is_generating=True)


def set_buttons_after_generate_text():
    """文本生成完成或被终止后，更新按钮状态"""
    global has_interrupted_text, is_in_generate_text
    is_in_generate_text = False
    return _set_generate_text_buttons(is_generating=False)


def _set_generate_audio_buttons(generate_audio_button, interrupt_audio_button, is_reset=False):
    return gr.update(
        value=generate_audio_button, visible=is_reset, interactive=is_reset
    ), gr.update(value=interrupt_audio_button, visible=not is_reset, interactive=not is_reset)


def interrupt_generate_audio():
    global chat, has_interrupted_audio

    has_interrupted_audio = True
    chat.interrupt()


def set_buttons_before_generate_audio(generate_audio_button, interrupt_audio_button):
    global has_interrupted_audio, is_in_generate_audio

    has_interrupted_audio = False
    is_in_generate_audio = True

    return _set_generate_audio_buttons(
        generate_audio_button,
        interrupt_audio_button,
    )


def set_buttons_after_generate_audio(generate_audio_button, interrupt_audio_button, audio_output):
    global has_interrupted_audio, is_in_generate_audio

    is_in_generate_audio = False

    return _set_generate_audio_buttons(
        generate_audio_button,
        interrupt_audio_button,
        audio_output is not None or has_interrupted_audio,
    )
