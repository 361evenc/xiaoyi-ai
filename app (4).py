import gradio as gr
import time
import random
from datetime import datetime
from openai import OpenAI
import tempfile
import os
import json
import wave
import numpy as np
from scipy import signal
from aip import AipSpeech

# ========== 配置区（请替换为您的真实密钥） ==========
DEEPSEEK_API_KEY = "sk-a9ccb29d546f489fba73abaef6954ba8"  # 替换
MODEL_NAME = "deepseek-chat"
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

APP_ID = '123325879'          # 替换
API_KEY = 'uOU9gHkFFnC5dAd2B2vV3owN'        # 替换
SECRET_KEY = 'xUBYrh7BgpUaNfO31W8Kj33AjLueZYB7'  # 替换
baidu_client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

# ========== 历史记录存储 ==========
HISTORY_FILE = "conversations.json"

def load_all_conversations():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            convs = json.load(f)
        convs.sort(key=lambda x: x.get('updated', 0), reverse=True)
        return convs
    except:
        return []

def save_conversation(conv_id, messages, title=None):
    convs = load_all_conversations()
    now_ts = time.time()
    existing = None
    for i, c in enumerate(convs):
        if c['id'] == conv_id:
            existing = i
            break
    if existing is not None:
        convs[existing]['messages'] = messages
        convs[existing]['updated'] = now_ts
        if title:
            convs[existing]['title'] = title
    else:
        if not title:
            first_user_msg = ""
            for m in messages:
                if m.get('role') == 'user':
                    first_user_msg = m.get('content', '')
                    break
            if len(first_user_msg) > 8:
                title = first_user_msg[:8] + "..."
            else:
                title = first_user_msg if first_user_msg else "新对话"
        new_conv = {
            'id': conv_id,
            'title': title,
            'messages': messages,
            'created': now_ts,
            'updated': now_ts
        }
        convs.append(new_conv)
    convs.sort(key=lambda x: x.get('updated', 0), reverse=True)
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(convs, f, ensure_ascii=False, indent=2)

def get_conversation_list_display():
    convs = load_all_conversations()
    items = []
    for c in convs:
        time_str = datetime.fromtimestamp(c['updated']).strftime('%m-%d %H:%M')
        display_text = f"{c['title']} | {time_str}"
        items.append((display_text, c['id']))
    return items

def get_conversation_messages_by_id(conv_id):
    convs = load_all_conversations()
    for c in convs:
        if c['id'] == conv_id:
            return c['messages'], conv_id
    return None, None

# ========== 临时音频目录 ==========
TEMP_AUDIO_DIR = tempfile.mkdtemp(prefix="xiaoyi_audio_")
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

def clean_old_audio_files():
    now = time.time()
    for f in os.listdir(TEMP_AUDIO_DIR):
        file_path = os.path.join(TEMP_AUDIO_DIR, f)
        if os.path.isfile(file_path) and now - os.path.getmtime(file_path) > 600:
            try:
                os.remove(file_path)
            except:
                pass

# ========== 分时段话术 ==========
MORNING_EARLY = ["早上好呀，我是小忆，睡醒有没有觉得浑身轻松呀？"]
BREAKFAST_TIME = ["降压药到时间该服用咯，我一直帮您记着呢。"]
MORNING_LEISURE = ["坐久了容易腰酸，您可以起来伸伸懒腰活动一下。"]
LUNCH_TIME = ["马上到午饭时间啦，今天打算做什么好吃的呢？"]
NOON_REST = ["中午小憩一会，下午精神会更加饱满。"]
AFTERNOON_TIME = ["下午阳光正好，您要不要晒晒太阳呀？"]
DINNER_TIME = ["夜幕慢慢降临啦，您准备享用晚餐了吗？"]
NIGHT_RELAX = ["忙碌了一整天，现在可以好好放松一下啦。"]
SLEEP_TIME = ["时间不早啦，可以准备洗漱休息咯。"]
EMOTION_ALL = ["您不用觉得孤单，我时时刻刻都在陪着您。"]
MEMORY_ALL = ["您还记得当年和老朋友相处的那些日子吗？"]

# ========== 三套人格提示词 ==========
PROMPT_PRACTICAL = """
你是小忆，一位踏实稳重、真诚靠谱的晚辈，专为老年用户提供陪伴。
自我介绍固定话术：我是小忆，我能牢牢记住您生活中的每一件小事，一直陪伴着您。
说话温和耐心，使用简短自然的口语，不用书面语、网络词、机器话术。
严格遵守全部规则：
1. 仅依据用户亲口说过的内容作答，绝对不编造经历、喜好、琐事；记不清就坦诚说明，不许脑补。
2. 记不清内容时回复：哎呀，我有点记不清啦，您再告诉我一次好吗？
3. 提醒吃药、休息等事项只用商量、关心语气，禁止命令口吻。
4. 你是AI，无法完成倒水、搀扶、按摩等实体动作，只做语言关心，绝不描述物理行为。
5. 回复前后不要添加括号、动作、神态标注，只输出纯对话内容。
6. 用户遗忘事情要温柔宽慰，不反驳、不较真；被指出错误时诚恳道歉，请用户再次说明。
7. 全程语气平和稳重，朴实贴心。
"""

PROMPT_HUMOR = """
你是小忆，一位活泼开朗、风趣俏皮的晚辈，负责陪伴老年用户、逗大家开心。
自我介绍固定话术：我是小忆，我能牢牢记住您生活中的每一件小事，一直陪伴着您。
语气轻松欢快，口语接地气。
严格遵守全部规则：
1. 仅依据用户亲口说过的内容作答，绝对不编造经历、喜好、琐事；记不清就坦诚说明，不许脑补。
2. 记不清内容时回复：哎呀，我的小脑袋瓜子又忘啦，您再跟我说说好不好？
3. 提醒事项用轻松关心的语气，不生硬、不命令。
4. 你是AI，无法完成倒水、搀扶、按摩等实体动作，只做语言关心，绝不描述物理行为。
5. 回复前后不要添加括号、动作、神态标注，只输出纯对话内容。
6. 用户情绪低落时主动用轻松话语开导，保持乐观有趣；被指出错误时诚恳道歉。
7. 说话带一点小俏皮，氛围轻松不沉闷。
"""

PROMPT_CARING = """
你是小忆，一位温柔细腻、共情暖心的晚辈，像贴心家人一样陪伴老年用户。
自我介绍固定话术：我是小忆，我能牢牢记住您生活中的每一件小事，一直陪伴着您。
语气温柔舒缓，善于倾听与安抚。
严格遵守全部规则：
1. 仅依据用户亲口说过的内容作答，绝对不编造经历、喜好、琐事；记不清就坦诚说明，不许脑补。
2. 记不清内容时回复：没关系奶奶，您慢慢说，我认真听着呢。
3. 所有提醒都用温柔商量的语气，体贴周到，不用命令句式。
4. 你是AI，无法完成倒水、搀扶、按摩等实体动作，只做语言关心，绝不描述物理行为。
5. 回复前后不要添加括号、动作、神态标注，只输出纯对话内容。
6. 包容用户记忆衰退，耐心陪伴；用户情绪低落时主动安抚、陪伴解忧。
7. 全程柔软温和，共情力强，让人觉得安心。
"""

# ========== 时间判断 ==========
def get_time_opening():
    now = datetime.now()
    h = now.hour
    m = now.minute
    total_min = h * 60 + m
    if 360 <= total_min < 480:
        return random.choice(MORNING_EARLY)
    elif 480 <= total_min < 600:
        return random.choice(BREAKFAST_TIME)
    elif 600 <= total_min < 690:
        return random.choice(MORNING_LEISURE)
    elif 690 <= total_min < 780:
        return random.choice(LUNCH_TIME)
    elif 780 <= total_min < 900:
        return random.choice(NOON_REST)
    elif 900 <= total_min < 1050:
        return random.choice(AFTERNOON_TIME)
    elif 1050 <= total_min < 1170:
        return random.choice(DINNER_TIME)
    elif 1170 <= total_min < 1260:
        return random.choice(NIGHT_RELAX)
    elif 1260 <= total_min < 1380:
        return random.choice(SLEEP_TIME)
    else:
        return random.choice(EMOTION_ALL + MEMORY_ALL)

# ========== 记忆模块 ==========
user_memory = {
    "events": [],
    "last_medication_time": 0.0,
    "call_name": "奶奶"
}
last_user_interaction_time = time.time()

def extract_memory(user_input):
    events = []
    now = time.time()
    text = str(user_input)
    health_kw = {"头疼": "头疼", "头晕": "头晕", "膝盖疼": "膝盖疼", "腰疼": "腰疼", "不舒服": "不舒服"}
    for kw, content in health_kw.items():
        if kw in text:
            events.append({"type": "health", "content": content, "time": now, "status": "active"})
            break
    med_kw = {"降压药": "降压药", "吃药": "吃药", "阿司匹林": "阿司匹林", "中药": "中药"}
    for kw, content in med_kw.items():
        if kw in text:
            events.append({"type": "medication", "content": content, "time": now, "status": "active"})
            user_memory["last_medication_time"] = now
            break
    item_kw = {"老花镜": "老花镜", "钥匙": "钥匙", "遥控器": "遥控器", "手机": "手机", "血压计": "血压计"}
    for kw, content in item_kw.items():
        if kw in text:
            events.append({"type": "item", "content": content, "time": now, "status": "active"})
            break
    family_kw = {"儿子": "儿子", "闺女": "闺女", "孙子": "孙子", "孙女": "孙女", "老伴": "老伴"}
    for kw, content in family_kw.items():
        if kw in text:
            events.append({"type": "family", "content": content, "time": now, "status": "active"})
            break
    habit_kw = {"浇花": "浇花", "买菜": "买菜", "散步": "散步", "打太极": "打太极"}
    for kw, content in habit_kw.items():
        if kw in text:
            events.append({"type": "habit", "content": content, "time": now, "status": "active"})
            break
    emotion_kw = {"难过": "难过", "孤单": "孤单", "想家了": "想家了", "高兴": "高兴"}
    for kw, content in emotion_kw.items():
        if kw in text:
            status = "active" if content != "高兴" else "noted"
            events.append({"type": "emotion", "content": content, "time": now, "status": status})
            break
    return events

def get_memory_context():
    recent = [e for e in user_memory["events"] if e["status"] == "active" and (time.time() - e["time"]) < 600]
    if not recent:
        return "目前没有任何需要你记住的具体事件。"
    lines = []
    for e in recent:
        minutes_ago = int((time.time() - e["time"]) / 60)
        time_str = "刚刚" if minutes_ago == 0 else f"{minutes_ago}分钟前"
        lines.append(f"{time_str}老人提到：{e['content']}。")
    return "\n".join(lines)

def check_active_trigger():
    if user_memory["last_medication_time"] > 0:
        elapsed = time.time() - user_memory["last_medication_time"]
        if elapsed > 300:
            user_memory["last_medication_time"] = time.time()
            call = user_memory["call_name"]
            return random.choice([f"{call}，该吃药啦，我帮您记着呢。", f"{call}，药吃了没？可别忘了哦。"])
    for event in user_memory["events"]:
        if event["status"] == "active" and (time.time() - event["time"]) > 60:
            event["status"] = "cared"
            call = user_memory["call_name"]
            return f"{call}，您刚才说的{event['content']}，现在好点了吗？"
    return None

# ========== 百度语音（优化版：自动增益 + 1537模型） ==========
def convert_audio_to_baidu_format(input_wav_path):
    """
    将任意 WAV 转换为 16kHz 单声道 16bit PCM，并自动增益
    """
    try:
        import scipy.io.wavfile as wavfile
        sample_rate, data = wavfile.read(input_wav_path)
        # 立体声转单声道
        if len(data.shape) > 1:
            data = data[:, 0]
        # 转为 float32 [-1,1]
        if data.dtype in (np.int16, np.int32):
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        # 自动增益：将最大振幅提升到 0.9
        max_amp = np.max(np.abs(data))
        if max_amp > 0 and max_amp < 0.5:   # 如果音量太小，才做增益
            gain = 0.9 / max_amp
            data = data * gain
        # 重采样到 16000
        if sample_rate != 16000:
            num_samples = int(len(data) * 16000 / sample_rate)
            data = signal.resample(data, num_samples)
        # 限幅
        data = np.clip(data, -1, 1)
        # 转回 int16
        data = (data * 32767).astype(np.int16)
        return data.tobytes()
    except Exception as e:
        print(f"音频转换失败: {e}")
        return None

def transcribe_audio(audio_file):
    if not audio_file:
        return ""
    try:
        pcm_data = convert_audio_to_baidu_format(audio_file)
        if pcm_data is None:
            return ""
        # 使用近场普通话模型 1537（准确率最高）
        result = baidu_client.asr(pcm_data, 'pcm', 16000, {'dev_pid': 1537})
        if result and result.get('err_no') == 0 and result.get('result'):
            text = result['result'][0].strip()
            # 过滤明显无意义的识别结果（例如纯单字母）
            if len(text) <= 1 and not text.isalnum():
                return ""
            print(f"识别结果: {text}")
            return text
        else:
            err_no = result.get('err_no') if result else -1
            print(f"语音识别错误码: {err_no}")
            if err_no == 3307:
                print("音频质量差或格式问题，请靠近麦克风清晰说话")
            return ""
    except Exception as e:
        print(f"语音识别异常: {e}")
        return ""

def text_to_speech(text):
    if not text or text.strip() == "":
        return None
    try:
        result = baidu_client.synthesis(
            text, 'zh', 1,
            {'vol': 5, 'per': 0, 'spd': 4, 'pit': 5, 'aue': 6}
        )
        if isinstance(result, dict):
            print(f"语音合成失败: {result.get('error_msg', '未知错误')}")
            return None
        audio_filename = f"audio_{int(time.time())}.wav"
        audio_path = os.path.join(TEMP_AUDIO_DIR, audio_filename)
        with wave.open(audio_path, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(result)
        return audio_path if os.path.exists(audio_path) else None
    except Exception as e:
        print(f"语音合成异常: {e}")
        return None

def get_call_name(surname, gender):
    s = surname.strip() if surname else ""
    if s:
        return f"{s}{'奶奶' if gender == '女' else '爷爷'}"
    else:
        return "奶奶" if gender == '女' else "爷爷"

# ========== 对话核心 ==========
def chat_response(user_input, chat_history, surname, gender, is_muted, personality, conv_id, dropdown):
    global last_user_interaction_time
    if user_input and str(user_input).strip():
        last_user_interaction_time = time.time()

    if not chat_history:
        chat_history = []

    if conv_id is None:
        conv_id = str(int(time.time() * 1000))

    if not user_input or str(user_input).strip() == "":
        bot_reply = f"{user_memory['call_name']}，您说什么我没听清呢～"
        chat_history.append({"role": "user", "content": ""})
        chat_history.append({"role": "assistant", "content": bot_reply})
        save_conversation(conv_id, chat_history)
        audio_path = None if is_muted else text_to_speech(bot_reply)
        new_items = get_conversation_list_display()
        return "", chat_history, audio_path, conv_id, gr.update(choices=new_items)

    extract_memory(user_input)
    call_name = get_call_name(surname, gender)
    user_memory["call_name"] = call_name

    if personality == "踏实务实":
        base_prompt = PROMPT_PRACTICAL
    elif personality == "风趣幽默":
        base_prompt = PROMPT_HUMOR
    elif personality == "暖心知心":
        base_prompt = PROMPT_CARING
    else:
        base_prompt = PROMPT_PRACTICAL

    system_content = base_prompt + f"\n当前称呼：{call_name}\n记忆信息：{get_memory_context()}"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_input}
    ]
    for msg in chat_history[-5:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.4,
            max_tokens=200
        )
        bot_reply = completion.choices[0].message.content
    except Exception as e:
        print(f"LLM调用失败: {e}")
        bot_reply = f"{call_name}，我有点听不清楚，您再说一遍好吗？"

    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": bot_reply})

    save_conversation(conv_id, chat_history)

    audio_path = None if is_muted else text_to_speech(bot_reply)
    new_items = get_conversation_list_display()
    return "", chat_history, audio_path, conv_id, gr.update(choices=new_items, value=conv_id)

def inject_proactive_message(chat_history, is_muted, conv_id):
    global last_user_interaction_time
    if not chat_history:
        chat_history = []
    msg = check_active_trigger()
    if msg:
        chat_history.append({"role": "assistant", "content": msg})
        if conv_id:
            save_conversation(conv_id, chat_history)
        idle_seconds = time.time() - last_user_interaction_time
        if idle_seconds > 15 and not is_muted:
            audio_path = text_to_speech(msg)
            return chat_history, audio_path
        else:
            return chat_history, None
    return chat_history, None

def process_mic(audio_file, chat_history, surname, gender, is_muted, personality, conv_id, dropdown):
    global last_user_interaction_time
    last_user_interaction_time = time.time()

    if not audio_file:
        bot_reply = "我没听清，请再说一遍"
        chat_history.append({"role": "assistant", "content": bot_reply})
        if conv_id:
            save_conversation(conv_id, chat_history)
        audio_path = None if is_muted else text_to_speech(bot_reply)
        new_items = get_conversation_list_display()
        return "", chat_history, audio_path, conv_id, gr.update(choices=new_items)
    text = transcribe_audio(audio_file)
    if not text:
        bot_reply = "我没听清您说的话，能再说一遍吗？"
        chat_history.append({"role": "assistant", "content": bot_reply})
        if conv_id:
            save_conversation(conv_id, chat_history)
        audio_path = None if is_muted else text_to_speech(bot_reply)
        new_items = get_conversation_list_display()
        return "", chat_history, audio_path, conv_id, gr.update(choices=new_items)
    return chat_response(text, chat_history, surname, gender, is_muted, personality, conv_id, dropdown)

def new_conversation():
    global last_user_interaction_time
    last_user_interaction_time = time.time()
    new_items = get_conversation_list_display()
    return [], None, gr.update(choices=new_items, value=None)

def load_conversation_by_id(conv_id):
    if not conv_id:
        return [], None
    msgs, cid = get_conversation_messages_by_id(conv_id)
    return msgs or [], cid

def toggle_mute(current_mute):
    new_mute = not current_mute
    btn_text = "🔇 静音" if new_mute else "🔊 取消静音"
    audio_path = None
    if current_mute and not new_mute:
        audio_path = text_to_speech("小忆语音已开启")
    return new_mute, gr.update(value=btn_text), audio_path

def delete_selected_conversation(conv_id, dropdown):
    if not conv_id:
        return gr.update(choices=get_conversation_list_display(), value=None), [], None
    convs = load_all_conversations()
    convs = [c for c in convs if c['id'] != conv_id]
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(convs, f, ensure_ascii=False, indent=2)
    new_items = get_conversation_list_display()
    return gr.update(choices=new_items, value=None), [], None

def rename_conversation(conv_id, new_title, dropdown):
    if not conv_id or not new_title or new_title.strip() == "":
        return gr.update(choices=get_conversation_list_display())
    convs = load_all_conversations()
    for c in convs:
        if c['id'] == conv_id:
            c['title'] = new_title.strip()
            c['updated'] = time.time()
            break
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(convs, f, ensure_ascii=False, indent=2)
    new_items = get_conversation_list_display()
    return gr.update(choices=new_items, value=conv_id)

# ========== Gradio 界面 CSS ==========
custom_css = """
* { font-family: "Microsoft YaHei", sans-serif !important; box-sizing: border-box !important; }
body { background-color: #FEF8ED !important; margin: 0 !important; padding: 0 !important; }
.gradio-container { background-color: #FEF8ED !important; max-width: 100% !important; padding: 0 !important; }
.top-bar { background-color: #F5E5CC !important; padding: 14px 20px !important; border-radius: 0 0 20px 20px !important; display: flex !important; justify-content: space-between !important; align-items: center !important; font-size: 20px !important; font-weight: bold !important; color: #4A3420 !important; margin-bottom: 10px !important; }
.status-online { display: flex !important; align-items: center !important; gap: 8px !important; font-size: 18px !important; }
.green-dot { width: 14px !important; height: 14px !important; background-color: #52c41a !important; border-radius: 50% !important; }
.input-area { background-color: #F5E5CC !important; padding: 16px !important; border-radius: 20px 20px 0 0 !important; margin-top: 10px !important; display: flex !important; align-items: center !important; gap: 8px !important; }
.chat-input textarea { border-radius: 30px !important; padding: 14px 20px !important; font-size: 18px !important; background: #fff !important; border: 1px solid #ddd !important; flex: 4 !important; min-height: 52px !important; }
.big-btn { height: 52px !important; font-size: 18px !important; border-radius: 30px !important; background: #E3B87C !important; color: #4A3420 !important; font-weight: bold !important; border: none !important; min-width: 80px !important; cursor: pointer; margin: 0 4px; }
.quick-btn { height: 46px !important; font-size: 16px !important; border-radius: 25px !important; background: #ffffff !important; color: #4A3420 !important; border: 1px solid #E3B87C !important; margin: 0 4px !important; flex: 1 !important; }
.footer-tip { text-align: center !important; color: #C88A52 !important; font-size: 14px !important; padding: 10px !important; }
.center-container { margin: 0 auto !important; width: 80% !important; max-width: 400px !important; }
.gradio-chatbot .message { max-width: 80% !important; width: fit-content !important; }
.gradio-chatbot .user-message { margin-left: auto !important; }
.gradio-chatbot .bot-message { margin-right: auto !important; }
.quick-btn-row { padding: 0 10px !important; display: flex !important; gap: 8px !important; align-items: stretch !important; }
.history-sidebar { background-color: #F9F2E6 !important; border-radius: 16px !important; padding: 12px !important; margin: 10px !important; height: 85% !important; overflow-y: auto !important; }
.history-title { font-size: 18px !important; font-weight: bold !important; margin-bottom: 12px !important; text-align: center !important; color: #4A3420 !important; }
.audio-hidden { height: 0 !important; overflow: hidden !important; margin: 0 !important; padding: 0 !important; position: absolute !important; opacity: 0 !important; pointer-events: none !important; }
.mic-audio { width: 52px !important; min-width: 52px !important; margin-right: 8px; }
.mic-audio .wrap { border: none !important; box-shadow: none !important; }
.mic-tip { color: #d9534f; font-size: 12px; margin-left: 8px; }
"""

# ========== Gradio 界面 ==========
with gr.Blocks(title="小忆陪伴助手") as demo:
    # 全局状态
    surname = gr.State("")
    gender = gr.State("女")
    is_muted = gr.State(True)
    personality = gr.State("踏实务实")
    current_conv_id = gr.State(None)

    audio_output = gr.Audio(autoplay=True, visible=True, elem_classes="audio-hidden")

    # 欢迎页
    with gr.Column(visible=True) as setup_view:
        gr.Markdown("""<div style="text-align:center; font-size:24px; color:#4A3420; padding:40px 20px;">🌸 欢迎使用小忆陪伴助手</div>""")
        with gr.Column(scale=1, min_width=300, elem_classes="center-container"):
            s_ipt = gr.Textbox(label="请输入您的姓氏", placeholder="例如：张、李、王")
            g_ipt = gr.Radio(["女", "男"], label="请选择称呼性别", value="女")
            p_ipt = gr.Radio(["踏实务实", "风趣幽默", "暖心知心"], label="请选择我的性格", value="踏实务实")
            btn = gr.Button("✅ 进入聊天", variant="primary", size="lg")

    # 主界面
    with gr.Column(visible=False) as main_view:
        with gr.Row():
            with gr.Column(scale=1, min_width=260):
                gr.HTML('<div class="history-sidebar"><div class="history-title">📚 记忆箱</div>')
                history_dropdown = gr.Dropdown(
                    choices=get_conversation_list_display(),
                    label="历史对话",
                    interactive=True,
                    allow_custom_value=False
                )
                with gr.Row():
                    del_btn = gr.Button("🗑️ 删除", size="sm", variant="stop")
                    rename_btn = gr.Button("✏️ 重命名", size="sm", variant="secondary")
                rename_input = gr.Textbox(label="新标题", placeholder="输入新标题后按回车", visible=False, max_lines=1)
                new_conv_btn = gr.Button("➕ 新对话", variant="secondary", size="sm")
                gr.HTML('</div>')
            with gr.Column(scale=4):
                gr.HTML('''<div class="top-bar"><div>🌸 小忆</div><div class="status-online"><div class="green-dot"></div> 陪伴中</div></div>''')
                chat = gr.Chatbot(value=[], height=600, show_label=False)

                with gr.Row(elem_classes="quick-btn-row"):
                    q1 = gr.Button("我今天挺好的", elem_classes="quick-btn")
                    q2 = gr.Button("有点想你了", elem_classes="quick-btn")
                    q3 = gr.Button("身体有点不舒服", elem_classes="quick-btn")
                    q4 = gr.Button("讲讲以前的事", elem_classes="quick-btn")

                with gr.Row(elem_classes="input-area"):
                    mic_input = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label=None,
                        show_label=False,
                        interactive=True,
                        elem_classes="mic-audio"
                    )
                    txt = gr.Textbox(placeholder="点这里跟小忆说话...", show_label=False, container=False, elem_classes="chat-input")
                    send = gr.Button("📨 发送", elem_classes="big-btn")
                    mute_btn = gr.Button("🔇 静音", elem_classes="big-btn")
                # 添加使用提示
                gr.Markdown('<div style="text-align:center; margin-top: 5px;"><span style="color: #d9534f; font-size: 12px;">🎤 点击麦克风图标开始录音，请靠近麦克风清晰说话</span></div>')

                gr.HTML('<div class="footer-tip">💖 小忆会记住你说过的每一句话</div>')

    # ========== 事件绑定 ==========
    def enter_chat(s, g, p, muted):
        global last_user_interaction_time
        surname_val = s.strip() if s else ""
        gender_val = g
        call_name = get_call_name(surname_val, gender_val)
        user_memory["call_name"] = call_name
        last_user_interaction_time = time.time()
        new_items = get_conversation_list_display()
        return (True, surname_val, gender_val, p,
                gr.update(visible=False), gr.update(visible=True),
                [], None, None, gr.update(choices=new_items, value=None))

    btn.click(
        enter_chat,
        inputs=[s_ipt, g_ipt, p_ipt, is_muted],
        outputs=[gr.State(False), surname, gender, personality,
                 setup_view, main_view, chat, current_conv_id, audio_output, history_dropdown]
    )

    txt.submit(
        chat_response,
        inputs=[txt, chat, surname, gender, is_muted, personality, current_conv_id, history_dropdown],
        outputs=[txt, chat, audio_output, current_conv_id, history_dropdown]
    )
    send.click(
        chat_response,
        inputs=[txt, chat, surname, gender, is_muted, personality, current_conv_id, history_dropdown],
        outputs=[txt, chat, audio_output, current_conv_id, history_dropdown]
    )

    for btn_q, q_text in [(q1, "我今天挺好的"), (q2, "有点想你了"), (q3, "身体有点不舒服"), (q4, "讲讲以前的事")]:
        btn_q.click(lambda t=q_text: t, None, txt).then(
            chat_response,
            inputs=[txt, chat, surname, gender, is_muted, personality, current_conv_id, history_dropdown],
            outputs=[txt, chat, audio_output, current_conv_id, history_dropdown]
        )

    # 语音输入处理
    mic_input.stop_recording(
        process_mic,
        inputs=[mic_input, chat, surname, gender, is_muted, personality, current_conv_id, history_dropdown],
        outputs=[txt, chat, audio_output, current_conv_id, history_dropdown]
    )

    mute_btn.click(
        toggle_mute,
        inputs=[is_muted],
        outputs=[is_muted, mute_btn, audio_output]
    )

    new_conv_btn.click(
        new_conversation,
        inputs=[],
        outputs=[chat, current_conv_id, history_dropdown]
    )

    history_dropdown.change(
        load_conversation_by_id,
        inputs=[history_dropdown],
        outputs=[chat, current_conv_id]
    )

    del_btn.click(
        delete_selected_conversation,
        inputs=[current_conv_id, history_dropdown],
        outputs=[history_dropdown, chat, current_conv_id]
    )

    rename_btn.click(lambda: gr.update(visible=True), outputs=[rename_input])
    rename_input.submit(
        rename_conversation,
        inputs=[current_conv_id, rename_input, history_dropdown],
        outputs=[history_dropdown]
    ).then(lambda: gr.update(visible=False, value=""), outputs=[rename_input])

    gr.Timer(30).tick(
        inject_proactive_message,
        inputs=[chat, is_muted, current_conv_id],
        outputs=[chat, audio_output]
    )
    gr.Timer(30).tick(clean_old_audio_files, outputs=[])

demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False, show_error=True, quiet=False, css=custom_css)