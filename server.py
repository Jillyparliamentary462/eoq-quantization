#!/usr/bin/env python3
"""Web chat with streaming + tok/s for EOQ-quantized model."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import time
import torch
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from core.utils import quantize_absmax, dequantize

model = None
tokenizer = None
history = []
model_info = {}

HTML = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EOQ Chat</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0a0a0a;color:#e0e0e0;height:100vh;display:flex;flex-direction:column}
header{background:#111;border-bottom:1px solid #222;padding:14px 24px;flex-shrink:0;display:flex;justify-content:space-between;align-items:center}
header h1{font-size:17px;font-weight:600;color:#fff}
.badge{display:inline-block;background:#1a3a2a;color:#4ade80;font-size:11px;padding:2px 8px;border-radius:10px;margin-left:8px;font-weight:500}
.stats{font-size:12px;color:#666;display:flex;gap:16px}
.stats .val{color:#aaa;font-weight:500}
#chat{flex:1;overflow-y:auto;padding:24px;display:flex;flex-direction:column;gap:14px}
.msg{max-width:80%;padding:12px 16px;border-radius:12px;line-height:1.6;font-size:14px;white-space:pre-wrap;word-wrap:break-word}
.msg.user{align-self:flex-end;background:#1a4a8a;color:#fff;border-bottom-right-radius:4px}
.msg.assistant{align-self:flex-start;background:#1a1a1a;border:1px solid #2a2a2a;border-bottom-left-radius:4px;position:relative}
.msg.system{align-self:center;background:#1a1a0a;border:1px solid #333;color:#888;font-size:12px;max-width:90%}
.tok-badge{position:absolute;bottom:-18px;right:8px;font-size:10px;color:#555;font-weight:500}
#input-area{flex-shrink:0;border-top:1px solid #222;padding:14px 24px;background:#111;display:flex;gap:10px}
#input{flex:1;background:#1a1a1a;border:1px solid #333;border-radius:8px;padding:12px 16px;color:#fff;font-size:14px;font-family:inherit;outline:none;resize:none;min-height:44px;max-height:120px}
#input:focus{border-color:#4a9eff}
#send{background:#2563eb;color:#fff;border:none;border-radius:8px;padding:0 24px;font-size:14px;font-weight:500;cursor:pointer;height:44px;min-width:80px}
#send:hover{background:#1d4ed8}
#send:disabled{background:#333;color:#666;cursor:not-allowed}
#clear{background:#222;color:#aaa;border:1px solid #333;border-radius:8px;padding:0 14px;font-size:13px;cursor:pointer;height:44px}
#clear:hover{background:#2a2a2a}
.cursor{display:inline-block;width:2px;height:14px;background:#4a9eff;animation:pulse .6s infinite;vertical-align:text-bottom;margin-left:2px}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0}}
</style>
</head>
<body>

<header>
  <div>
    <h1>EOQ Chat <span class="badge" id="quant-badge">Q4</span></h1>
  </div>
  <div class="stats">
    <span>Model: <span class="val" id="model-name">loading...</span></span>
    <span>Size: <span class="val" id="model-size">-</span></span>
    <span>Last: <span class="val" id="last-speed">-</span></span>
  </div>
</header>

<div id="chat">
  <div class="msg system">Modelo carregado. Digite sua mensagem.</div>
</div>

<div id="input-area">
  <button id="clear" onclick="clearChat()">Limpar</button>
  <textarea id="input" placeholder="Digite sua mensagem..." rows="1" onkeydown="handleKey(event)"></textarea>
  <button id="send" onclick="sendMessage()">Enviar</button>
</div>

<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
let generating = false;

fetch('/info').then(r=>r.json()).then(d=>{
  document.getElementById('model-name').textContent = d.model;
  document.getElementById('model-size').textContent = d.size;
  document.getElementById('quant-badge').textContent = 'Q'+d.bits;
});

input.addEventListener('input', function(){
  this.style.height='auto';
  this.style.height=Math.min(this.scrollHeight,120)+'px';
});

function handleKey(e){
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMessage()}
}

function scrollBottom(){chat.scrollTop=chat.scrollHeight}

async function sendMessage(){
  const text=input.value.trim();
  if(!text||generating) return;
  generating=true;
  sendBtn.disabled=true;
  input.value='';
  input.style.height='auto';

  // User message
  const userDiv=document.createElement('div');
  userDiv.className='msg user';
  userDiv.textContent=text;
  chat.appendChild(userDiv);

  // Assistant message (will stream into)
  const aiDiv=document.createElement('div');
  aiDiv.className='msg assistant';
  aiDiv.innerHTML='<span class="cursor"></span>';
  chat.appendChild(aiDiv);
  scrollBottom();

  let fullText='';
  let tokenCount=0;
  const t0=performance.now();

  try{
    const res=await fetch('/chat',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:text})
    });

    const reader=res.body.getReader();
    const decoder=new TextDecoder();
    let buffer='';

    while(true){
      const{done,value}=await reader.read();
      if(done) break;
      buffer+=decoder.decode(value,{stream:true});

      const lines=buffer.split('\n');
      buffer=lines.pop();

      for(const line of lines){
        if(!line.startsWith('data: ')) continue;
        const payload=line.slice(6).trim();
        if(payload==='[DONE]') continue;
        try{
          const d=JSON.parse(payload);
          if(d.token){
            fullText+=d.token;
            tokenCount++;
            aiDiv.textContent=fullText;
            const cur=document.createElement('span');
            cur.className='cursor';
            aiDiv.appendChild(cur);
            scrollBottom();
          }
          if(d.stats){
            aiDiv.textContent=fullText;
            const badge=document.createElement('div');
            badge.className='tok-badge';
            badge.textContent=d.stats.tokens_per_sec.toFixed(1)+' tok/s | '+d.stats.total_tokens+' tokens | '+d.stats.elapsed.toFixed(1)+'s';
            aiDiv.appendChild(badge);
            document.getElementById('last-speed').textContent=d.stats.tokens_per_sec.toFixed(1)+' tok/s';
          }
        }catch(e){}
      }
    }
  }catch(e){
    removeTyping();
    if(!fullText) aiDiv.textContent='Erro: '+e.message;
  }finally{
    generating=false;
    sendBtn.disabled=false;
    input.focus();
    scrollBottom();
    // Remove lingering cursor
    const cur=aiDiv.querySelector('.cursor');
    if(cur) cur.remove();
  }
}

async function clearChat(){
  await fetch('/clear',{method:'POST'});
  chat.innerHTML='<div class="msg system">Historico limpo.</div>';
}

input.focus();
</script>
</body>
</html>"""


class ChatHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path == "/info":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(model_info).encode())
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML.encode("utf-8"))

    def do_POST(self):
        global history

        if self.path == "/clear":
            history = []
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            return

        if self.path == "/chat":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            user_msg = body.get("message", "")

            history.append({"role": "user", "content": user_msg})

            prompt = tokenizer.apply_chat_template(
                history, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt")
            input_len = inputs["input_ids"].shape[1]

            # Streaming
            streamer = TextIteratorStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            gen_kwargs = dict(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                streamer=streamer,
            )

            # Run generation in background thread
            t0 = time.time()
            thread = threading.Thread(target=lambda: model.generate(**gen_kwargs))
            thread.start()

            # Stream SSE response
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            full_response = ""
            token_count = 0

            for text in streamer:
                if text:
                    full_response += text
                    token_count += 1
                    sse = f"data: {json.dumps({'token': text})}\n\n"
                    try:
                        self.wfile.write(sse.encode())
                        self.wfile.flush()
                    except BrokenPipeError:
                        break

            thread.join()
            elapsed = time.time() - t0
            tps = token_count / elapsed if elapsed > 0 else 0

            # Send final stats
            stats = {
                "stats": {
                    "total_tokens": token_count,
                    "elapsed": round(elapsed, 2),
                    "tokens_per_sec": round(tps, 1),
                }
            }
            try:
                self.wfile.write(f"data: {json.dumps(stats)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except BrokenPipeError:
                pass

            history.append({"role": "assistant", "content": full_response})
            print(f"  [{token_count} tokens in {elapsed:.1f}s = {tps:.1f} tok/s]")
            return

        self.send_response(404)
        self.end_headers()


def main():
    global model, tokenizer, model_info

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    print(f"Loading {args.model} (Q{args.bits})...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, trust_remote_code=True
    )

    count = 0
    for name, param in model.named_parameters():
        if param.ndim >= 2 and param.numel() >= 256:
            with torch.no_grad():
                qt = quantize_absmax(param.data, args.bits, 128)
                param.data.copy_(dequantize(qt))
                count += 1
    model.eval()
    load_time = time.time() - t0

    model_info = {
        "model": args.model.split("/")[-1],
        "bits": args.bits,
        "tensors": count,
        "size": "287 MB" if args.bits == 4 else f"~{args.bits * 40}MB",
        "load_time": f"{load_time:.1f}s",
    }

    print(f"  {count} tensors quantized in {load_time:.1f}s")
    print(f"\n  http://localhost:{args.port}\n")

    server = HTTPServer(("0.0.0.0", args.port), ChatHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutdown.")
        server.shutdown()


if __name__ == "__main__":
    main()
