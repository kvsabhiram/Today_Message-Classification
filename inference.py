"""
ToDoZee inference server — v11
  - Adapter path: output_v11
  - 19 tasks (Smart Scheduling removed)
  - Updated routing rules
  - System prompt explicitly forbids invented locations
"""

import os
import json
import logging
import re
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from contextlib import asynccontextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel, PeftConfig
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================
#              CONFIG
# ==============================================================

# Paths/config are env-driven so the same image runs anywhere (CI, EC2, local).
# ADAPTER_PATH defaults to ./output_v11 next to this file.
_HERE               = os.path.dirname(os.path.abspath(__file__))
ADAPTER_PATH        = os.environ.get("ADAPTER_PATH", os.path.join(_HERE, "output_v11"))
BASE_MODEL          = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-3B")
MAX_NEW_TOKENS      = int(os.environ.get("MAX_NEW_TOKENS", "120"))
MERGE_ADAPTER       = os.environ.get("MERGE_ADAPTER", "1") == "1"
HOST                = os.environ.get("HOST", "0.0.0.0")
PORT                = int(os.environ.get("PORT", "5011"))

USE_MODEL                    = True
USE_ROUTING_AS_FALLBACK      = True
USE_ENTITY_EXTRACT           = True

CACHE_SIZE          = 500

LOG_DIR             = "./logs"
LOG_FILE            = os.path.join(LOG_DIR, "requests.jsonl")
LOG_ERRORS_FILE     = os.path.join(LOG_DIR, "errors.jsonl")


# ==============================================================
#              REQUEST LOGGER
# ==============================================================

class RequestLogger:
    def __init__(self, log_file: str, errors_file: str):
        self.log_file = log_file
        self.errors_file = errors_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log(self, entry: Dict):
        entry["timestamp"] = datetime.now().isoformat()
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write log: {e}")

    def log_error(self, entry: Dict):
        entry["timestamp"] = datetime.now().isoformat()
        try:
            with open(self.errors_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write error log: {e}")


request_logger: Optional[RequestLogger] = None


# ==============================================================
#              SYSTEM PROMPT (v11 - 19 tasks, no Smart Scheduling)
# ==============================================================

SYSTEM_PROMPT = """You are ToDoZee, an AI assistant for an Indian lifestyle app.

Classify user messages into one of these tasks:
1. Past Life Finder - past life, karma, reincarnation
2. Astrology - horoscope, zodiac, rashi, kundali, lucky color/number, gemstone
3. Daily Motivational Quotes - motivation, inspiration, quotes
4. Weather - temperature, hot/cold, humidity, forecast, AQI, sunny, cloudy
5. Rain Nearby - rain, rainfall, umbrella, monsoon, drizzle
6. Alarm & Reminders - remind, alarm, alert, snooze
7. Price Alerts - price, stock, gold, petrol, rate, currency
8. Chef & Calories Advisor - food, recipe, calories, nutrition, meal plan
9. Auto Meter Justice - auto fare, rickshaw, meter, overcharging
10. Fast Route Suggestion - route, direction, traffic, navigate (locations only if user mentions them)
11. Women Safety Tracking - SOS, emergency, safety, track journey
12. CricketBuzz - cricket, IPL, match, score, player stats, records
13. Instant Chat Link - share chat, chat link, conversation link
14. Referrals - refer friend, referral code, invite, reward
15. Voice Note to Text - transcribe, voice to text, audio to text
16. Unread Message Summary - unread messages, summary, recap
17. Messaging - send message, text someone, WhatsApp (scheduled = subtask)
18. Calling - call someone, phone call, dial (scheduled = subtask)
19. Status Update - post status, view story, WhatsApp/Insta status

Output: Task[<name>] Subtask[<action>] Recipient[] Message[] Time[] Frequency[] From[] To[] Title[]

Rules:
- One line only. Fill relevant fields. Empty [] if not mentioned.
- If no match, all fields empty.
- For Calling/Messaging/Status Update: use "schedule X" subtask when user wants to schedule for future time.
- For From/To (Fast Route, Auto Meter, Women Safety): only fill if location is explicitly in the input. Never invent locations."""

# ==============================================================
#              SUPPORTED TASKS (19)
# ==============================================================

SUPPORTED_TASKS = [
    "Past Life Finder",
    "Astrology",
    "Daily Motivational Quotes",
    "Weather",
    "Rain Nearby",
    "Alarm & Reminders",
    "Price Alerts",
    "Chef & Calories Advisor",
    "Auto Meter Justice",
    "Fast Route Suggestion",
    "Women Safety Tracking",
    "CricketBuzz",
    "Instant Chat Link",
    "Referrals",
    "Voice Note to Text",
    "Unread Message Summary",
    "Messaging",
    "Calling",
    "Status Update",
]

VALID_TASKS_LC = {t.lower() for t in SUPPORTED_TASKS}


# ==============================================================
#              FALLBACK ROUTING (model runs first; this only fires on failure)
# ==============================================================

ROUTING_RULES: List[Tuple[re.Pattern, str, str]] = [
    (re.compile(r'\b(past life|reincarnation|karma|previous birth|purvajanma|rebirth)\b', re.I),
     "Past Life Finder", "past life report"),
    (re.compile(r'\b(horoscope|astrology|kundali|rashi|zodiac|nakshatra|planetary|jyotish|birth chart|gemstone|mangalik)\b', re.I),
     "Astrology", "horoscope"),
    (re.compile(r'\b(quote|motivat|inspir|blessing|devotional|wisdom|prayer|affirmation)\b', re.I),
     "Daily Motivational Quotes", "daily quote"),
    (re.compile(r'\b(weather|temperature|forecast|humid|climate|heatwave|sunny|cloudy|sunset|sunrise|aqi|air quality|uv index|cold today|hot today)\b', re.I),
     "Weather", "current weather"),
    (re.compile(r'\b(rain|raining|rainfall|drizzle|shower|umbrella|monsoon|precipitation)\b', re.I),
     "Rain Nearby", "rain check"),
    (re.compile(r'\b(remind|reminder|alarm|alert me|wake me|don\'t forget|set alarm|notify me|remember to|snooze)\b', re.I),
     "Alarm & Reminders", "set reminder"),
    (re.compile(r'\b(price|stock|gold rate|petrol|diesel|rate|cost|rupee|dollar|crypto|bitcoin|nifty|sensex)\b', re.I),
     "Price Alerts", "price check"),
    (re.compile(r'\b(calori|nutrition|diet|food|meal|recipe|protein|carb|healthy|cook|bmi)\b', re.I),
     "Chef & Calories Advisor", "nutrition info"),
    (re.compile(r'\b(auto fare|rickshaw|meter|cab fare|taxi fare|ola fare|uber fare|auto charge|overcharg)\b', re.I),
     "Auto Meter Justice", "verify fare"),
    (re.compile(r'\b(route|direction|navigate|traffic|fastest|shortest|way to|reach|maps|gps)\b', re.I),
     "Fast Route Suggestion", "find route"),
    (re.compile(r'\b(sos|emergency|safety|track me|safe|guardian|help me|danger|women safety|panic)\b', re.I),
     "Women Safety Tracking", "emergency alert"),
    (re.compile(r'\b(cricket|ipl|match score|wicket|batsman|bowler|innings|runs|t20|odi|kohli|rohit sharma|orange cap|purple cap)\b', re.I),
     "CricketBuzz", "live score"),
    (re.compile(r'\b(share chat|chat link|conversation link|forward chat|export chat|whatsapp link)\b', re.I),
     "Instant Chat Link", "create link"),
    (re.compile(r'\b(refer|referral|invite friend|earn reward|share code|promo code|invite code)\b', re.I),
     "Referrals", "share referral"),
    (re.compile(r'\b(transcribe|voice to text|audio message|convert voice|speech to text|voice note)\b', re.I),
     "Voice Note to Text", "transcribe"),
    (re.compile(r'\b(unread message|message summary|pending message|missed message|summarize message|recap)\b', re.I),
     "Unread Message Summary", "summarize"),
    (re.compile(r'\b(send message|text |whatsapp|sms|message to|hit up|ping |dm |text someone)\b', re.I),
     "Messaging", "send message"),
    (re.compile(r'\b(call |phone call|dial |ring |reach .* by phone|make call|give .* a call)\b', re.I),
     "Calling", "make call"),
    (re.compile(r'\b(post status|update status|whatsapp status|insta story|view story|story view|hide status|delete status)\b', re.I),
     "Status Update", "post status"),
]


def apply_routing(text: str) -> Optional[Tuple[str, str]]:
    for pattern, task, subtask in ROUTING_RULES:
        if pattern.search(text):
            return task, subtask
    return None


# ==============================================================
#              ENTITY EXTRACTION
# ==============================================================

def extract_entities(text: str) -> Dict[str, str]:
    entities = {}

    time_patterns = [
        r'at\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)?)',
        r'(\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM))',
        r'\b(morning|afternoon|evening|night|tonight|tomorrow|today|noon|midnight)\b',
    ]
    for p in time_patterns:
        m = re.search(p, text, re.I)
        if m:
            entities['time'] = m.group(1) if m.lastindex else m.group(0)
            break

    freq_match = re.search(r'(every\s+(?:day|week|month|morning|evening)|daily|weekly|monthly)', text, re.I)
    if freq_match:
        entities['frequency'] = freq_match.group(1)

    cities = r'\b(Mumbai|Delhi|Bangalore|Bengaluru|Hyderabad|Chennai|Kolkata|Pune|Ahmedabad|Jaipur|Lucknow|Vizag|Goa|Kochi|Noida|Gurgaon|Gurugram)\b'
    m = re.search(cities, text, re.I)
    if m:
        entities['title'] = m.group(1)

    name_match = re.search(r'\b(?:to|tell|remind|message|text|call|notify)\s+([A-Z][a-z]+)\b', text)
    if name_match:
        entities['recipient'] = name_match.group(1)

    msg_match = re.search(r'(?:remind me to|don\'t forget to|remember to)\s+(.+?)(?:\s+at\s+|\s+on\s+|$)', text, re.I)
    if msg_match:
        entities['message'] = msg_match.group(1).strip()

    return entities


# ==============================================================
#              HELPERS
# ==============================================================

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids: List[List[int]]):
        self.stop_ids = stop_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for s in self.stop_ids:
            if len(s) <= input_ids.shape[1] and input_ids[0, -len(s):].tolist() == s:
                return True
        return False


def clean_output(raw: str) -> str:
    for s in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        raw = raw.split(s)[0]
    raw = raw.encode('ascii', errors='ignore').decode('ascii')
    match = re.search(r'Task\[.*?Title\[[^\]]*\]', raw, re.DOTALL)
    if match:
        return match.group(0).replace('\n', ' ')
    lines = raw.strip().split('\n')
    for line in lines:
        if line.strip().startswith('Task['):
            return line.strip()
    return raw.strip().split('\n')[0] if raw.strip() else ""


def parse_fields(output: str) -> Dict[str, str]:
    return {m.group(1).lower(): m.group(2) for m in re.finditer(r'(\w+)\[([^\]]*)\]', output)}


def build_output(task: str, subtask: str, **fields) -> str:
    out = f"Task[{task}] Subtask[{subtask}]"
    for f in ['Recipient', 'Message', 'Time', 'Frequency', 'From', 'To', 'Title']:
        out += f" {f}[{fields.get(f.lower(), '')}]"
    return out


def is_model_output_valid(parsed: Dict[str, str]) -> bool:
    if not parsed:
        return False
    if 'task' not in parsed:
        return False
    task = parsed.get('task', '').strip().lower()
    if task == "":
        return True
    return task in VALID_TASKS_LC


# ==============================================================
#              CACHE
# ==============================================================

class SimpleCache:
    def __init__(self, maxsize=500):
        self.cache, self.order, self.maxsize = {}, [], maxsize

    def get(self, key: str) -> Optional[Dict]:
        return self.cache.get(hashlib.md5(key.lower().strip().encode()).hexdigest())

    def set(self, key: str, value: Dict):
        h = hashlib.md5(key.lower().strip().encode()).hexdigest()
        if h not in self.cache:
            if len(self.cache) >= self.maxsize:
                self.cache.pop(self.order.pop(0), None)
            self.cache[h] = value
            self.order.append(h)


# ==============================================================
#              CLASSIFIER
# ==============================================================

class TaskClassifier:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.stop_ids = []
        self.cache = SimpleCache(CACHE_SIZE)
        self._load()

    def _load(self):
        logger.info(f"Loading model from {ADAPTER_PATH}")

        try:
            cfg = PeftConfig.from_pretrained(ADAPTER_PATH)
            base = cfg.base_model_name_or_path
        except:
            base = BASE_MODEL

        self.tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base, device_map="cuda:0", trust_remote_code=True,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )

        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        if MERGE_ADAPTER:
            logger.info("Merging adapter...")
            model = model.merge_and_unload()

        model.eval()
        self.model = model

        for s in ["<|im_end|>", "<|endoftext|>", "\n"]:
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            if ids:
                self.stop_ids.append(ids)

        logger.info("Model loaded successfully!")

    def _run_model(self, text: str) -> Tuple[str, Dict[str, str], str]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{text}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs['input_ids'].shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.2,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            stopping_criteria=StoppingCriteriaList([StopOnTokens(self.stop_ids)]),
        )

        raw = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=False)
        cleaned = clean_output(raw)
        parsed = parse_fields(cleaned)
        return cleaned, parsed, raw

    @torch.inference_mode()
    def classify(self, text: str) -> Dict:
        text = text.strip()
        start_time = datetime.now()

        cached = self.cache.get(text)
        if cached:
            result = {**cached, 'cached': True}
            self._log_request(text, result, start_time, source="cache")
            return result

        result = {
            'input': text,
            'output': '',
            'parsed': {},
            'routed': False,
            'cached': False,
        }

        # STEP 1 — MODEL FIRST
        raw_output = ""
        if USE_MODEL:
            try:
                cleaned, parsed, raw_output = self._run_model(text)
                if is_model_output_valid(parsed):
                    if USE_ENTITY_EXTRACT and parsed.get('task'):
                        entities = extract_entities(text)
                        for k, v in entities.items():
                            if k in parsed and not parsed[k]:
                                parsed[k] = v
                    result['output'] = cleaned
                    result['parsed'] = parsed
                    self.cache.set(text, result)
                    self._log_request(text, result, start_time,
                                      source="model", raw_output=raw_output)
                    return result
                else:
                    logger.warning(
                        f"Model output invalid for input '{text[:60]}': '{cleaned[:80]}' — "
                        f"falling back to routing"
                    )
            except Exception as e:
                logger.error(f"Model inference failed: {e} — falling back to routing")

        # STEP 2 — ROUTING FALLBACK
        if USE_ROUTING_AS_FALLBACK:
            routed = apply_routing(text)
            if routed:
                task, subtask = routed
                entities = extract_entities(text) if USE_ENTITY_EXTRACT else {}
                result['output'] = build_output(task, subtask, **entities)
                result['parsed'] = parse_fields(result['output'])
                result['routed'] = True
                self.cache.set(text, result)
                self._log_request(text, result, start_time, source="routing_fallback")
                return result

        # STEP 3 — Empty fallback
        result['output'] = build_output("", "")
        result['parsed'] = parse_fields(result['output'])
        self.cache.set(text, result)
        self._log_request(text, result, start_time, source="empty", raw_output=raw_output)
        return result

    def _log_request(self, text: str, result: Dict, start_time: datetime,
                     source: str, raw_output: str = ""):
        if request_logger is None:
            return

        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

        log_entry = {
            "input": text,
            "output": result.get('output', ''),
            "task": result.get('parsed', {}).get('task', ''),
            "subtask": result.get('parsed', {}).get('subtask', ''),
            "recipient": result.get('parsed', {}).get('recipient', ''),
            "message": result.get('parsed', {}).get('message', ''),
            "time": result.get('parsed', {}).get('time', ''),
            "frequency": result.get('parsed', {}).get('frequency', ''),
            "from": result.get('parsed', {}).get('from', ''),
            "to": result.get('parsed', {}).get('to', ''),
            "source": source,
            "routed": result.get('routed', False),
            "cached": result.get('cached', False),
            "latency_ms": round(elapsed_ms, 2),
        }

        if raw_output:
            log_entry["raw_model_output"] = raw_output

        request_logger.log(log_entry)
        logger.info(
            f"[{source.upper()}] "
            f"Input: \"{text[:80]}{'...' if len(text) > 80 else ''}\" -> "
            f"Task: {log_entry['task'] or 'none'} | "
            f"Subtask: {log_entry['subtask'] or 'none'} | "
            f"{elapsed_ms:.0f}ms"
        )


# ==============================================================
#              FASTAPI
# ==============================================================

classifier: Optional[TaskClassifier] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier, request_logger
    logger.info("Starting ToDoZee API Server (v11)...")
    request_logger = RequestLogger(LOG_FILE, LOG_ERRORS_FILE)
    classifier = TaskClassifier()
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="ToDoZee Task Classifier API",
    description="API for classifying user messages into 19 task categories (v11, model-first)",
    version="11.0",
    lifespan=lifespan,
)


class ClassifyRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts: List[str]

class ClassifyResponse(BaseModel):
    input: str
    output: str
    task: str
    subtask: str
    parsed: Dict[str, str]
    routed: bool
    cached: bool

class BatchResponse(BaseModel):
    results: List[ClassifyResponse]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    tasks_supported: int

class TasksResponse(BaseModel):
    tasks: List[str]
    count: int


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if classifier else "unhealthy",
        model_loaded=classifier is not None,
        tasks_supported=len(SUPPORTED_TASKS),
    )


@app.get("/tasks", response_model=TasksResponse)
async def list_tasks():
    return TasksResponse(tasks=SUPPORTED_TASKS, count=len(SUPPORTED_TASKS))


@app.post("/classify", response_model=ClassifyResponse)
async def classify_endpoint(request: ClassifyRequest):
    if not classifier:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        result = classifier.classify(request.text)
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        if request_logger:
            request_logger.log_error({
                "input": request.text,
                "error": str(e),
                "error_type": type(e).__name__,
            })
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

    return ClassifyResponse(
        input=result['input'],
        output=result['output'],
        task=result['parsed'].get('task', ''),
        subtask=result['parsed'].get('subtask', ''),
        parsed=result['parsed'],
        routed=result['routed'],
        cached=result['cached'],
    )


@app.post("/batch", response_model=BatchResponse)
async def batch_classify(request: BatchRequest):
    if not classifier:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    if len(request.texts) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 texts per batch")

    results = []
    for text in request.texts:
        if text.strip():
            try:
                result = classifier.classify(text)
                results.append(ClassifyResponse(
                    input=result['input'],
                    output=result['output'],
                    task=result['parsed'].get('task', ''),
                    subtask=result['parsed'].get('subtask', ''),
                    parsed=result['parsed'],
                    routed=result['routed'],
                    cached=result['cached'],
                ))
            except Exception as e:
                logger.error(f"Batch item failed: {text[:50]} - {e}")
                if request_logger:
                    request_logger.log_error({
                        "input": text,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "context": "batch",
                    })
                results.append(ClassifyResponse(
                    input=text, output=f"Error: {str(e)}",
                    task="", subtask="", parsed={},
                    routed=False, cached=False,
                ))

    return BatchResponse(results=results)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)

# import os
# import json
# import logging
# import re
# import hashlib
# from datetime import datetime
# from typing import List, Dict, Optional, Tuple
# from contextlib import asynccontextmanager

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
# from peft import PeftModel, PeftConfig
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # ==============================================================
# #              CONFIG
# # ==============================================================

# ADAPTER_PATH        = "todo_zee_qwen/output_v9_final"
# BASE_MODEL          = "Qwen/Qwen2.5-3B"
# MAX_NEW_TOKENS      = 120
# MERGE_ADAPTER       = True
# USE_ROUTING         = True
# USE_ENTITY_EXTRACT  = True
# CACHE_SIZE          = 500

# # Logging config
# LOG_DIR             = "./logs"
# LOG_FILE            = os.path.join(LOG_DIR, "requests.jsonl")    # Every request/response
# LOG_ERRORS_FILE     = os.path.join(LOG_DIR, "errors.jsonl")      # Failed requests only


# # ==============================================================
# #              REQUEST LOGGER
# # ==============================================================

# class RequestLogger:
#     """Logs every input/output pair to a JSONL file for debugging and analysis."""

#     def __init__(self, log_file: str, errors_file: str):
#         self.log_file = log_file
#         self.errors_file = errors_file
#         os.makedirs(os.path.dirname(log_file), exist_ok=True)

#     def log(self, entry: Dict):
#         """Append one JSON line to the log file."""
#         entry["timestamp"] = datetime.now().isoformat()
#         try:
#             with open(self.log_file, "a", encoding="utf-8") as f:
#                 f.write(json.dumps(entry, ensure_ascii=False) + "\n")
#         except Exception as e:
#             logger.error(f"Failed to write log: {e}")

#     def log_error(self, entry: Dict):
#         """Append one JSON line to the errors file."""
#         entry["timestamp"] = datetime.now().isoformat()
#         try:
#             with open(self.errors_file, "a", encoding="utf-8") as f:
#                 f.write(json.dumps(entry, ensure_ascii=False) + "\n")
#         except Exception as e:
#             logger.error(f"Failed to write error log: {e}")


# # Global logger instance
# request_logger: Optional[RequestLogger] = None


# # ==============================================================
# #              SYSTEM PROMPT
# # ==============================================================

# SYSTEM_PROMPT = """You are ToDoZee, an AI assistant for an Indian lifestyle app.

# Classify user messages into one of these tasks:
# 1. Past Life Finder - past life, karma, reincarnation
# 2. Astrology - horoscope, zodiac, rashi, kundali
# 3. Daily Motivational Quotes - motivation, inspiration, quotes
# 4. Weather - temperature, forecast, climate
# 5. Rain Nearby - rain, rainfall, umbrella, monsoon
# 6. Alarm & Reminders - remind, alarm, alert, schedule
# 7. Price Alerts - price, stock, gold, petrol, rate
# 8. Chef & Calories Advisor - food, recipe, calories, nutrition
# 9. Auto Meter Justice - auto fare, rickshaw, meter
# 10. Fast Route Suggestion - route, direction, traffic, navigate
# 11. Women Safety Tracking - SOS, emergency, safety, track
# 12. CricketBuzz - cricket, IPL, match, score
# 13. Smart Scheduling - schedule meeting, calendar, appointment
# 14. Instant Chat Link - share chat, chat link
# 15. Referrals - refer friend, referral code
# 16. Voice Note to Text - transcribe, voice to text
# 17. Unread Message Summary - unread messages, summary
# 18. Messaging - send message, text someone, WhatsApp
# 19. Calling - call someone, phone call, dial

# Output: Task[<n>] Subtask[<action>] Recipient[] Message[] Time[] Frequency[] From[] To[] Title[]

# Rules: One line only. Fill relevant fields. Empty [] if not mentioned. If no match, all fields empty."""

# # ==============================================================
# #              SUPPORTED TASKS
# # ==============================================================

# SUPPORTED_TASKS = [
#     "Past Life Finder",
#     "Astrology",
#     "Daily Motivational Quotes",
#     "Weather",
#     "Rain Nearby",
#     "Alarm & Reminders",
#     "Price Alerts",
#     "Chef & Calories Advisor",
#     "Auto Meter Justice",
#     "Fast Route Suggestion",
#     "Women Safety Tracking",
#     "CricketBuzz",
#     "Smart Scheduling",
#     "Instant Chat Link",
#     "Referrals",
#     "Voice Note to Text",
#     "Unread Message Summary",
#     "Messaging",
#     "Calling",
# ]

# # ==============================================================
# #              ROUTING RULES
# # ==============================================================

# ROUTING_RULES: List[Tuple[re.Pattern, str, str]] = [
#     (re.compile(r'\b(past life|reincarnation|karma|previous birth|purvajanma|rebirth)\b', re.I),
#      "Past Life Finder", "past life report"),
#     (re.compile(r'\b(horoscope|astrology|kundali|rashi|zodiac|nakshatra|planetary|jyotish|birth chart)\b', re.I),
#      "Astrology", "horoscope"),
#     (re.compile(r'\b(quote|motivat|inspir|blessing|devotional|wisdom|prayer|affirmation)\b', re.I),
#      "Daily Motivational Quotes", "daily quote"),
#     (re.compile(r'\b(weather|temperature|forecast|sunny|cloudy|humid|climate|heatwave)\b', re.I),
#      "Weather", "current weather"),
#     (re.compile(r'\b(rain|raining|rainfall|drizzle|shower|umbrella|monsoon|precipitation)\b', re.I),
#      "Rain Nearby", "rain check"),
#     (re.compile(r'\b(remind|reminder|alarm|alert me|wake me|don\'t forget|set alarm|notify me|remember to)\b', re.I),
#      "Alarm & Reminders", "set reminder"),
#     (re.compile(r'\b(price|stock|gold rate|petrol|diesel|rate|cost|rupee|dollar|crypto|bitcoin|nifty|sensex)\b', re.I),
#      "Price Alerts", "price check"),
#     (re.compile(r'\b(calori|nutrition|diet|food|meal|recipe|protein|carb|healthy|cook|bmi)\b', re.I),
#      "Chef & Calories Advisor", "nutrition info"),
#     (re.compile(r'\b(auto fare|rickshaw|meter|cab fare|taxi fare|ola fare|uber fare|auto charge)\b', re.I),
#      "Auto Meter Justice", "verify fare"),
#     (re.compile(r'\b(route|direction|navigate|traffic|fastest|shortest|way to|reach|maps|gps)\b', re.I),
#      "Fast Route Suggestion", "find route"),
#     (re.compile(r'\b(sos|emergency|safety|track me|safe|guardian|help me|danger|women safety|panic)\b', re.I),
#      "Women Safety Tracking", "emergency alert"),
#     (re.compile(r'\b(cricket|ipl|match score|wicket|batsman|bowler|innings|runs|t20|odi)\b', re.I),
#      "CricketBuzz", "live score"),
#     (re.compile(r'\b(schedule meeting|book meeting|calendar|appointment|slot|booking|set meeting)\b', re.I),
#      "Smart Scheduling", "schedule event"),
#     (re.compile(r'\b(share chat|chat link|conversation link|forward chat|export chat|whatsapp link)\b', re.I),
#      "Instant Chat Link", "create link"),
#     (re.compile(r'\b(refer|referral|invite friend|earn reward|share code|promo code|invite code)\b', re.I),
#      "Referrals", "share referral"),
#     (re.compile(r'\b(transcribe|voice to text|audio message|convert voice|speech to text|voice note)\b', re.I),
#      "Voice Note to Text", "transcribe"),
#     (re.compile(r'\b(unread message|message summary|pending message|missed message|summarize message)\b', re.I),
#      "Unread Message Summary", "summarize"),
#     (re.compile(r'\b(send message|text |whatsapp|sms|message to|hit up|ping |dm |text someone)\b', re.I),
#      "Messaging", "send message"),
#     (re.compile(r'\b(call |phone call|dial |ring |reach .* by phone|make call|give .* a call)\b', re.I),
#      "Calling", "make call"),
# ]


# def apply_routing(text: str) -> Optional[Tuple[str, str]]:
#     for pattern, task, subtask in ROUTING_RULES:
#         if pattern.search(text):
#             return task, subtask
#     return None


# # ==============================================================
# #              ENTITY EXTRACTION
# # ==============================================================

# def extract_entities(text: str) -> Dict[str, str]:
#     entities = {}

#     # Time
#     time_patterns = [
#         r'at\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)?)',
#         r'(\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM))',
#         r'\b(morning|afternoon|evening|night|tonight|tomorrow|today|noon|midnight)\b',
#     ]
#     for p in time_patterns:
#         m = re.search(p, text, re.I)
#         if m:
#             entities['time'] = m.group(1) if m.lastindex else m.group(0)
#             break

#     # Frequency
#     freq_match = re.search(r'(every\s+(?:day|week|month|morning|evening)|daily|weekly|monthly)', text, re.I)
#     if freq_match:
#         entities['frequency'] = freq_match.group(1)

#     # Cities
#     cities = r'\b(Mumbai|Delhi|Bangalore|Bengaluru|Hyderabad|Chennai|Kolkata|Pune|Ahmedabad|Jaipur|Lucknow|Vizag|Goa|Kochi|Noida|Gurgaon|Gurugram)\b'
#     m = re.search(cities, text, re.I)
#     if m:
#         entities['title'] = m.group(1)

#     # Recipient
#     name_match = re.search(r'\b(?:to|tell|remind|message|text|call|notify)\s+([A-Z][a-z]+)\b', text)
#     if name_match:
#         entities['recipient'] = name_match.group(1)

#     # Message
#     msg_match = re.search(r'(?:remind me to|don\'t forget to|remember to)\s+(.+?)(?:\s+at\s+|\s+on\s+|$)', text, re.I)
#     if msg_match:
#         entities['message'] = msg_match.group(1).strip()

#     return entities


# # ==============================================================
# #              HELPERS
# # ==============================================================

# class StopOnTokens(StoppingCriteria):
#     def __init__(self, stop_ids: List[List[int]]):
#         self.stop_ids = stop_ids

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#         for s in self.stop_ids:
#             if len(s) <= input_ids.shape[1] and input_ids[0, -len(s):].tolist() == s:
#                 return True
#         return False


# def clean_output(raw: str) -> str:
#     for s in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
#         raw = raw.split(s)[0]
#     raw = raw.encode('ascii', errors='ignore').decode('ascii')
#     match = re.search(r'Task\[.*?Title\[[^\]]*\]', raw, re.DOTALL)
#     if match:
#         return match.group(0).replace('\n', ' ')
#     lines = raw.strip().split('\n')
#     for line in lines:
#         if line.strip().startswith('Task['):
#             return line.strip()
#     return raw.strip().split('\n')[0] if raw.strip() else ""


# def parse_fields(output: str) -> Dict[str, str]:
#     return {m.group(1).lower(): m.group(2) for m in re.finditer(r'(\w+)\[([^\]]*)\]', output)}


# def build_output(task: str, subtask: str, **fields) -> str:
#     out = f"Task[{task}] Subtask[{subtask}]"
#     for f in ['Recipient', 'Message', 'Time', 'Frequency', 'From', 'To', 'Title']:
#         out += f" {f}[{fields.get(f.lower(), '')}]"
#     return out


# # ==============================================================
# #              CACHE
# # ==============================================================

# class SimpleCache:
#     def __init__(self, maxsize=500):
#         self.cache, self.order, self.maxsize = {}, [], maxsize

#     def get(self, key: str) -> Optional[Dict]:
#         return self.cache.get(hashlib.md5(key.lower().strip().encode()).hexdigest())

#     def set(self, key: str, value: Dict):
#         h = hashlib.md5(key.lower().strip().encode()).hexdigest()
#         if h not in self.cache:
#             if len(self.cache) >= self.maxsize:
#                 self.cache.pop(self.order.pop(0), None)
#             self.cache[h] = value
#             self.order.append(h)


# # ==============================================================
# #              CLASSIFIER
# # ==============================================================

# class TaskClassifier:
#     def __init__(self):
#         self.model = None
#         self.tokenizer = None
#         self.stop_ids = []
#         self.cache = SimpleCache(CACHE_SIZE)
#         self._load()

#     def _load(self):
#         logger.info(f"Loading model from {ADAPTER_PATH}")

#         try:
#             cfg = PeftConfig.from_pretrained(ADAPTER_PATH)
#             base = cfg.base_model_name_or_path
#         except:
#             base = BASE_MODEL

#         self.tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
#         if not self.tokenizer.pad_token:
#             self.tokenizer.pad_token = self.tokenizer.eos_token

#         model = AutoModelForCausalLM.from_pretrained(
#             base, device_map="cuda:0", trust_remote_code=True,
#             torch_dtype=torch.bfloat16, attn_implementation="sdpa"
#         )

#         model = PeftModel.from_pretrained(model, ADAPTER_PATH)
#         if MERGE_ADAPTER:
#             logger.info("Merging adapter...")
#             model = model.merge_and_unload()

#         model.eval()
#         self.model = model

#         for s in ["<|im_end|>", "<|endoftext|>", "\n"]:
#             ids = self.tokenizer.encode(s, add_special_tokens=False)
#             if ids:
#                 self.stop_ids.append(ids)

#         logger.info("Model loaded successfully!")

#     @torch.inference_mode()
#     def classify(self, text: str) -> Dict:
#         text = text.strip()
#         start_time = datetime.now()

#         # Check cache
#         cached = self.cache.get(text)
#         if cached:
#             result = {**cached, 'cached': True}
#             # Log cached hit
#             self._log_request(text, result, start_time, source="cache")
#             return result

#         result = {'input': text, 'output': '', 'parsed': {}, 'routed': False, 'cached': False}

#         # Try routing
#         if USE_ROUTING:
#             routed = apply_routing(text)
#             if routed:
#                 task, subtask = routed
#                 entities = extract_entities(text) if USE_ENTITY_EXTRACT else {}
#                 result['output'] = build_output(task, subtask, **entities)
#                 result['parsed'] = parse_fields(result['output'])
#                 result['routed'] = True
#                 self.cache.set(text, result)
#                 self._log_request(text, result, start_time, source="routing")
#                 return result

#         # Model inference
#         messages = [
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": text},
#         ]

#         try:
#             prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         except:
#             prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"

#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
#         input_len = inputs['input_ids'].shape[1]

#         outputs = self.model.generate(
#             **inputs,
#             max_new_tokens=MAX_NEW_TOKENS,
#             do_sample=False,
#             repetition_penalty=1.2,
#             eos_token_id=self.tokenizer.eos_token_id,
#             pad_token_id=self.tokenizer.pad_token_id,
#             stopping_criteria=StoppingCriteriaList([StopOnTokens(self.stop_ids)]),
#         )

#         raw = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=False)
#         cleaned = clean_output(raw)
#         parsed = parse_fields(cleaned)

#         # Entity extraction
#         if USE_ENTITY_EXTRACT and parsed.get('task'):
#             entities = extract_entities(text)
#             for k, v in entities.items():
#                 if k in parsed and not parsed[k]:
#                     parsed[k] = v

#         result['output'] = cleaned
#         result['parsed'] = parsed
#         self.cache.set(text, result)

#         self._log_request(text, result, start_time, source="model", raw_output=raw)

#         return result

#     def _log_request(self, text: str, result: Dict, start_time: datetime,
#                      source: str, raw_output: str = ""):
#         """Log input/output to file."""
#         if request_logger is None:
#             return

#         elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

#         log_entry = {
#             "input": text,
#             "output": result.get('output', ''),
#             "task": result.get('parsed', {}).get('task', ''),
#             "subtask": result.get('parsed', {}).get('subtask', ''),
#             "recipient": result.get('parsed', {}).get('recipient', ''),
#             "message": result.get('parsed', {}).get('message', ''),
#             "time": result.get('parsed', {}).get('time', ''),
#             "frequency": result.get('parsed', {}).get('frequency', ''),
#             "source": source,             # "model", "routing", or "cache"
#             "routed": result.get('routed', False),
#             "cached": result.get('cached', False),
#             "latency_ms": round(elapsed_ms, 2),
#         }

#         # Include raw model output for debugging (only for model calls)
#         if raw_output:
#             log_entry["raw_model_output"] = raw_output

#         request_logger.log(log_entry)
#         logger.info(
#             f"[{source.upper()}] "
#             f"Input: \"{text[:80]}{'...' if len(text) > 80 else ''}\" → "
#             f"Task: {log_entry['task'] or 'none'} | "
#             f"Subtask: {log_entry['subtask'] or 'none'} | "
#             f"{elapsed_ms:.0f}ms"
#         )


# # ==============================================================
# #              FASTAPI
# # ==============================================================

# # Global classifier
# classifier: Optional[TaskClassifier] = None


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Load model on startup."""
#     global classifier, request_logger
#     logger.info("Starting ToDoZee API Server...")
#     request_logger = RequestLogger(LOG_FILE, LOG_ERRORS_FILE)
#     logger.info(f"Request logs: {LOG_FILE}")
#     logger.info(f"Error logs: {LOG_ERRORS_FILE}")
#     classifier = TaskClassifier()
#     yield
#     logger.info("Shutting down...")


# app = FastAPI(
#     title="ToDoZee Task Classifier API",
#     description="API for classifying user messages into 19 task categories",
#     version="9.0",
#     lifespan=lifespan,
# )


# # Request/Response models
# class ClassifyRequest(BaseModel):
#     text: str

# class BatchRequest(BaseModel):
#     texts: List[str]

# class ClassifyResponse(BaseModel):
#     input: str
#     output: str
#     task: str
#     subtask: str
#     parsed: Dict[str, str]
#     routed: bool
#     cached: bool

# class BatchResponse(BaseModel):
#     results: List[ClassifyResponse]

# class HealthResponse(BaseModel):
#     status: str
#     model_loaded: bool
#     tasks_supported: int

# class TasksResponse(BaseModel):
#     tasks: List[str]
#     count: int


# @app.get("/health", response_model=HealthResponse)
# async def health_check():
#     """Health check endpoint."""
#     return HealthResponse(
#         status="healthy" if classifier else "unhealthy",
#         model_loaded=classifier is not None,
#         tasks_supported=19
#     )


# @app.get("/tasks", response_model=TasksResponse)
# async def list_tasks():
#     """List all supported tasks."""
#     return TasksResponse(
#         tasks=SUPPORTED_TASKS,
#         count=len(SUPPORTED_TASKS)
#     )


# @app.post("/classify", response_model=ClassifyResponse)
# async def classify_endpoint(request: ClassifyRequest):
#     """Classify a single input text."""
#     if not classifier:
#         raise HTTPException(status_code=503, detail="Model not loaded")

#     if not request.text.strip():
#         raise HTTPException(status_code=400, detail="Text cannot be empty")

#     try:
#         result = classifier.classify(request.text)
#     except Exception as e:
#         logger.error(f"Classification failed: {e}")
#         if request_logger:
#             request_logger.log_error({
#                 "input": request.text,
#                 "error": str(e),
#                 "error_type": type(e).__name__,
#             })
#         raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

#     return ClassifyResponse(
#         input=result['input'],
#         output=result['output'],
#         task=result['parsed'].get('task', ''),
#         subtask=result['parsed'].get('subtask', ''),
#         parsed=result['parsed'],
#         routed=result['routed'],
#         cached=result['cached'],
#     )


# @app.post("/batch", response_model=BatchResponse)
# async def batch_classify(request: BatchRequest):
#     """Classify multiple inputs."""
#     if not classifier:
#         raise HTTPException(status_code=503, detail="Model not loaded")

#     if not request.texts:
#         raise HTTPException(status_code=400, detail="Texts list cannot be empty")

#     if len(request.texts) > 100:
#         raise HTTPException(status_code=400, detail="Maximum 100 texts per batch")

#     results = []
#     for text in request.texts:
#         if text.strip():
#             try:
#                 result = classifier.classify(text)
#                 results.append(ClassifyResponse(
#                     input=result['input'],
#                     output=result['output'],
#                     task=result['parsed'].get('task', ''),
#                     subtask=result['parsed'].get('subtask', ''),
#                     parsed=result['parsed'],
#                     routed=result['routed'],
#                     cached=result['cached'],
#                 ))
#             except Exception as e:
#                 logger.error(f"Batch item failed: {text[:50]} — {e}")
#                 if request_logger:
#                     request_logger.log_error({
#                         "input": text,
#                         "error": str(e),
#                         "error_type": type(e).__name__,
#                         "context": "batch",
#                     })
#                 results.append(ClassifyResponse(
#                     input=text,
#                     output=f"Error: {str(e)}",
#                     task="",
#                     subtask="",
#                     parsed={},
#                     routed=False,
#                     cached=False,
#                 ))

#     return BatchResponse(results=results)


# # ==============================================================
# #              MAIN
# # ==============================================================

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=5011)



