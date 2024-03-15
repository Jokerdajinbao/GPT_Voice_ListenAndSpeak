import gradio as gr
import openai
import datetime
import wave #转录声音为wav格式
from openaifun.api_answer import api_answer #调用openai的api回答问题
from voice_baidu.save_as_wav import save_as_wav #保存录音为wav文件
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

openai.api_base = 'https://key.langchain.com.cn/v1'
openai.api_key = 'sk-kcfJcDXKztSEuMxaSqVjvuniMFIlz8HSr2xApuxivkNINiEc'

API_KEY = "***"#填百度TTS的api
SECRET_KEY = "***"

def textask(input_text):
    print("\n用户提问：" + input_text)
    text = api_answer(input_text)
    #audio = text2audio(text,API_KEY,SECRET_KEY)
    output = sambert_hifigan_tts(input=text, voice='zhiyan_emo')
    wav = output[OutputKeys.OUTPUT_WAV]
    with open('temp2.wav', 'wb') as f:
        f.write(wav)
    return "temp2.wav"  # 返回临时文件的路径

def outputvoice(invoice):
    save_as_wav(invoice,"temp.wav")
    #text = voice2text("temp.wav",API_KEY,SECRET_KEY)
    text = inference_16k_pipline(audio_in='temp.wav')
    print("\n用户提问：" + text['text'])
    text = api_answer(text['text'])
    #audio = text2audio(text,API_KEY,SECRET_KEY)
    output = sambert_hifigan_tts(input=text, voice='zhiyan_emo')
    wav = output[OutputKeys.OUTPUT_WAV]
    with open('temp2.wav', 'wb') as f:
        f.write(wav)
    return "temp2.wav"  # 返回临时文件的路径

with gr.Blocks(title="语音对话AI",css="#chatbot{height:500px} .overflow-y-auto{height:500px}") as rxbot:
    gr.HTML("""<h1 align="center"><font size=5>语音对话AI</h1>""")
    state = gr.State([])
    with gr.Column():
        txt = gr.Textbox(show_label=False, placeholder="文字提问").style(container=False)
        submitBtn = gr.Button("提问", variant="primary")
    with gr.Column():
        in_voice = gr.Audio(type="filepath", source="microphone", label="语音提问")
        btn_voice = gr.Button("提问")
        out_voice = gr.Audio(type="filepath", label="输出音频")
    txt.submit(textask,txt,out_voice)
    submitBtn.click(textask,txt,out_voice)
    btn_voice.click(outputvoice, in_voice, out_voice)

if __name__ == "__main__":
    inference_16k_pipline = pipeline(task=Tasks.auto_speech_recognition,model='speech_UniASR_asr_2pass-zh-cn-16k-common-vocab8358-tensorflow1-online')
    sambert_hifigan_tts = pipeline(task=Tasks.text_to_speech, model='speech_sambert-hifigan_tts_zh-cn_16k')
    rxbot.queue().launch(
                server_name='0.0.0.0',
                #server_name='[2409:8a55:3c45:4d01:79b7:77a:ca07:850d]',
                #server_name='[::1]', # 相当于127.0.0.1
                server_port=6006,
                show_api=False,
                #share=True,
                share=False,
                inbrowser=True,
                ssl_verify=False,#用签名部署https网页以避免gradio录音被不信任
                ssl_certfile="cert.pem",
                ssl_keyfile="key.pem"
                )