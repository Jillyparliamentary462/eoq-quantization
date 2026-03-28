#!/usr/bin/env python3
"""Web chat with streaming + tok/s using QuantizedLinear for REAL memory savings.

Unlike server.py (which dequantizes at load time and stores full FP32 weights
in RAM), this version replaces every nn.Linear with a QuantizedLinear layer
that stores weights as packed low-bit integer codes + per-block FP16 scales.
Dequantization happens on-the-fly during the forward pass, so the model's
resident memory footprint is genuinely reduced.

Usage:
    python server_v2.py --model Qwen/Qwen2.5-0.5B-Instruct --bits 4 --port 8080
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import json
import math
import time
import struct
import threading
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


# ---------------------------------------------------------------------------
# QuantizedLinear -- stores weights as int8 codes + FP16 scales
# ---------------------------------------------------------------------------

class QuantizedLinear(nn.Module):
    """Drop-in replacement for nn.Linear that stores weights in quantized form.

    Weights are kept as ``int8`` codes (for 2-8 bit quantization) plus per-block
    ``float16`` scales, using block-wise symmetric absmax quantization. This
    gives genuine RAM savings vs keeping full FP32/FP16 weight tensors.

    During ``forward()``, the weights are dequantized on-the-fly to ``float32``
    for the matmul. This adds a small compute cost but dramatically reduces
    memory.

    Memory per weight element:
        FP32:  4 bytes
        FP16:  2 bytes
        Q8:    1 byte  + ~0.016 bytes scale  => ~1.02 bytes  (3.9x vs FP32)
        Q4:    1 byte* + ~0.016 bytes scale  => ~1.02 bytes  (3.9x vs FP32)

    *We store codes as int8 even for <8-bit quantization. True sub-byte packing
    would save more, but int8 storage already gives ~4x savings over FP32 and
    keeps the implementation simple and fast.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 bits: int = 4, block_size: int = 128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.block_size = block_size

        # Quantized weight storage (registered as buffers, not parameters,
        # so they don't get gradients or show up in optimizer)
        self.register_buffer("weight_codes", torch.zeros(out_features, in_features, dtype=torch.int8))

        # Number of blocks for scale storage
        n_elements = out_features * in_features
        n_blocks = math.ceil(n_elements / block_size)
        self.register_buffer("weight_scales", torch.zeros(n_blocks, dtype=torch.float16))

        if bias:
            # Bias is small -- keep it in float32
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self._n_elements = n_elements

    @staticmethod
    def from_linear(linear: nn.Linear, bits: int = 4, block_size: int = 128) -> "QuantizedLinear":
        """Create a QuantizedLinear from an existing nn.Linear, quantizing its weights."""
        has_bias = linear.bias is not None
        ql = QuantizedLinear(
            linear.in_features, linear.out_features,
            bias=has_bias, bits=bits, block_size=block_size,
        )

        # Quantize the weight
        weight = linear.weight.data.detach().float()
        codes, scales = _quantize_absmax_raw(weight, bits, block_size)
        ql.weight_codes.copy_(codes.to(torch.int8).view(linear.out_features, linear.in_features))
        ql.weight_scales.copy_(scales.to(torch.float16))

        # Copy bias as-is
        if has_bias:
            ql.bias.data.copy_(linear.bias.data.float())

        return ql

    def _dequantize_weight(self) -> torch.Tensor:
        """Reconstruct float32 weight from stored codes + scales."""
        flat = self.weight_codes.flatten().float()
        n = flat.numel()
        bs = self.block_size

        # Pad to multiple of block_size
        pad_len = (bs - n % bs) % bs
        if pad_len > 0:
            flat = torch.nn.functional.pad(flat, (0, pad_len), value=0.0)

        blocks = flat.view(-1, bs)
        scales = self.weight_scales.float().unsqueeze(1)  # (n_blocks, 1)
        deq = (blocks * scales).flatten()[:n]
        return deq.view(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._dequantize_weight()
        return torch.nn.functional.linear(x, weight, self.bias)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, bits={self.bits}, block_size={self.block_size}")


def _quantize_absmax_raw(tensor: torch.Tensor, bits: int, block_size: int):
    """Block-wise symmetric absmax quantization. Returns (codes, scales).

    This is a standalone version (not using core.utils) so server_v2.py
    is fully self-contained.
    """
    t = tensor.detach().float().flatten()
    n = t.numel()

    pad_len = (block_size - n % block_size) % block_size
    if pad_len > 0:
        t = torch.nn.functional.pad(t, (0, pad_len), value=0.0)

    blocks = t.view(-1, block_size)

    qmax = (1 << (bits - 1)) - 1
    absmax = blocks.abs().amax(dim=1)
    scale = absmax / qmax
    scale = scale.clamp(min=1e-10)

    quantized = (blocks / scale.unsqueeze(1)).round().clamp(-qmax, qmax)

    # Trim padding, keep flat
    codes_flat = quantized.flatten()[:n]
    return codes_flat, scale


def replace_linear_with_quantized(model: nn.Module, bits: int = 4, block_size: int = 128,
                                  min_elements: int = 256) -> int:
    """Walk the model tree and replace every nn.Linear whose weight has
    >= min_elements with a QuantizedLinear. Returns the count of replaced layers.

    This modifies the model in-place.
    """
    count = 0
    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child.weight.numel() >= min_elements:
                ql = QuantizedLinear.from_linear(child, bits=bits, block_size=block_size)
                setattr(module, child_name, ql)
                count += 1
    return count


# ---------------------------------------------------------------------------
# Memory measurement
# ---------------------------------------------------------------------------

def get_model_ram_bytes(model: nn.Module) -> int:
    """Estimate actual RAM used by all parameters and buffers in the model."""
    total = 0
    seen = set()
    for name, p in model.named_parameters():
        ptr = p.data_ptr()
        if ptr not in seen:
            seen.add(ptr)
            total += p.nelement() * p.element_size()
    for name, b in model.named_buffers():
        ptr = b.data_ptr()
        if ptr not in seen:
            seen.add(ptr)
            total += b.nelement() * b.element_size()
    return total


def format_bytes(n: int) -> str:
    """Format a byte count as a human-readable string."""
    if n >= 1e9:
        return f"{n / 1e9:.1f} GB"
    elif n >= 1e6:
        return f"{n / 1e6:.0f} MB"
    elif n >= 1e3:
        return f"{n / 1e3:.0f} KB"
    return f"{n} B"


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

model = None
tokenizer = None
history = []
model_info = {}
session_stats = {"total_tokens": 0, "total_time": 0.0, "num_generations": 0}

# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EOQ Chat v2 (QuantizedLinear)</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0a0a0a;color:#e0e0e0;height:100vh;display:flex;flex-direction:column}
header{background:#111;border-bottom:1px solid #222;padding:14px 24px;flex-shrink:0}
.header-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
header h1{font-size:17px;font-weight:600;color:#fff}
.badge{display:inline-block;background:#1a3a2a;color:#4ade80;font-size:11px;padding:2px 8px;border-radius:10px;margin-left:8px;font-weight:500}
.badge-blue{background:#1a2a4a;color:#60a5fa}
.stats{font-size:12px;color:#666;display:flex;gap:16px;flex-wrap:wrap}
.stats .val{color:#aaa;font-weight:500}
.stats .ram{color:#f59e0b}
.stats .savings{color:#4ade80;font-weight:600}
.session-bar{display:flex;gap:16px;font-size:11px;color:#555;padding:6px 0 0 0;border-top:1px solid #1a1a1a;margin-top:6px}
.session-bar .val{color:#888}
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
  <div class="header-top">
    <div>
      <h1>EOQ Chat v2 <span class="badge" id="quant-badge">Q4</span> <span class="badge badge-blue">QuantizedLinear</span></h1>
    </div>
    <div class="stats">
      <span>Model: <span class="val" id="model-name">loading...</span></span>
      <span>RAM: <span class="val ram" id="ram-usage">-</span></span>
      <span>Savings: <span class="val savings" id="ram-savings">-</span></span>
      <span>Last: <span class="val" id="last-speed">-</span></span>
    </div>
  </div>
  <div class="session-bar">
    <span>Layers quantized: <span class="val" id="layers-count">-</span></span>
    <span>Session tokens: <span class="val" id="session-tokens">0</span></span>
    <span>Avg tok/s: <span class="val" id="avg-tps">-</span></span>
    <span>Generations: <span class="val" id="gen-count">0</span></span>
  </div>
</header>

<div id="chat">
  <div class="msg system">Model loaded with QuantizedLinear. Weights stored as low-bit codes in RAM. Type your message.</div>
</div>

<div id="input-area">
  <button id="clear" onclick="clearChat()">Clear</button>
  <textarea id="input" placeholder="Type your message..." rows="1" onkeydown="handleKey(event)"></textarea>
  <button id="send" onclick="sendMessage()">Send</button>
</div>

<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
let generating = false;

fetch('/info').then(r=>r.json()).then(d=>{
  document.getElementById('model-name').textContent = d.model;
  document.getElementById('quant-badge').textContent = 'Q'+d.bits;
  document.getElementById('ram-usage').textContent = d.ram_usage;
  document.getElementById('ram-savings').textContent = d.ram_savings;
  document.getElementById('layers-count').textContent = d.layers_quantized;
});

function updateSessionStats(s){
  if(!s) return;
  document.getElementById('session-tokens').textContent = s.total_tokens;
  document.getElementById('avg-tps').textContent = s.avg_tps > 0 ? s.avg_tps.toFixed(1)+' tok/s' : '-';
  document.getElementById('gen-count').textContent = s.num_generations;
}

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

  const userDiv=document.createElement('div');
  userDiv.className='msg user';
  userDiv.textContent=text;
  chat.appendChild(userDiv);

  const aiDiv=document.createElement('div');
  aiDiv.className='msg assistant';
  aiDiv.innerHTML='<span class="cursor"></span>';
  chat.appendChild(aiDiv);
  scrollBottom();

  let fullText='';
  let tokenCount=0;

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
          if(d.session){
            updateSessionStats(d.session);
          }
        }catch(e){}
      }
    }
  }catch(e){
    if(!fullText) aiDiv.textContent='Error: '+e.message;
  }finally{
    generating=false;
    sendBtn.disabled=false;
    input.focus();
    scrollBottom();
    const cur=aiDiv.querySelector('.cursor');
    if(cur) cur.remove();
  }
}

async function clearChat(){
  await fetch('/clear',{method:'POST'});
  chat.innerHTML='<div class="msg system">History cleared.</div>';
}

input.focus();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class ChatHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress default logging

    def do_GET(self):
        if self.path == "/info":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(model_info).encode())
            return
        # Serve HTML for any other GET
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML.encode("utf-8"))

    def do_POST(self):
        global history, session_stats

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

            # Streaming via TextIteratorStreamer
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

            # Update cumulative session stats
            session_stats["total_tokens"] += token_count
            session_stats["total_time"] += elapsed
            session_stats["num_generations"] += 1
            avg_tps = (session_stats["total_tokens"] / session_stats["total_time"]
                       if session_stats["total_time"] > 0 else 0)

            # Send final stats + session stats
            final_payload = {
                "stats": {
                    "total_tokens": token_count,
                    "elapsed": round(elapsed, 2),
                    "tokens_per_sec": round(tps, 1),
                },
                "session": {
                    "total_tokens": session_stats["total_tokens"],
                    "avg_tps": round(avg_tps, 1),
                    "num_generations": session_stats["num_generations"],
                },
            }
            try:
                self.wfile.write(f"data: {json.dumps(final_payload)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except BrokenPipeError:
                pass

            history.append({"role": "assistant", "content": full_response})
            print(f"  [{token_count} tokens in {elapsed:.1f}s = {tps:.1f} tok/s]  "
                  f"(session: {session_stats['total_tokens']} total, {avg_tps:.1f} avg tok/s)")
            return

        self.send_response(404)
        self.end_headers()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global model, tokenizer, model_info

    import argparse
    parser = argparse.ArgumentParser(
        description="EOQ Chat v2 -- web server with QuantizedLinear for real RAM savings"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 5, 6, 8])
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    print(f"Loading {args.model} ...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, trust_remote_code=True
    )

    # Measure FP32 baseline RAM before quantization
    fp32_ram = get_model_ram_bytes(model)
    print(f"  FP32 model RAM: {format_bytes(fp32_ram)}")

    # Try to import patch_model from core.model_patcher, or
    # replace_linear_with_quantized from core.quantized_linear.
    # If neither is available, use our built-in version.
    patch_fn = None
    try:
        from core.model_patcher import patch_model
        patch_fn = lambda m: patch_model(m, bits=args.bits, block_size=args.block_size)
        print("  Using core.model_patcher.patch_model")
    except ImportError:
        try:
            from core.quantized_linear import replace_linear_with_quantized as _replace
            patch_fn = lambda m: _replace(m, bits=args.bits, block_size=args.block_size)
            print("  Using core.quantized_linear.replace_linear_with_quantized")
        except ImportError:
            print("  [!] core.model_patcher / core.quantized_linear not found.")
            print("      Using built-in QuantizedLinear (server_v2.py).")
            patch_fn = None

    # Replace nn.Linear layers with QuantizedLinear
    if patch_fn is not None:
        count = patch_fn(model)
    else:
        count = replace_linear_with_quantized(
            model, bits=args.bits, block_size=args.block_size
        )

    model.eval()

    # Measure quantized RAM
    quant_ram = get_model_ram_bytes(model)
    savings_ratio = fp32_ram / quant_ram if quant_ram > 0 else 1.0
    load_time = time.time() - t0

    print(f"  {count} nn.Linear layers replaced with QuantizedLinear (Q{args.bits})")
    print(f"  Quantized model RAM: {format_bytes(quant_ram)}")
    print(f"  Savings: {savings_ratio:.1f}x vs FP32 "
          f"({format_bytes(fp32_ram)} -> {format_bytes(quant_ram)})")
    print(f"  Load time: {load_time:.1f}s")

    model_info = {
        "model": args.model.split("/")[-1],
        "bits": args.bits,
        "layers_quantized": count,
        "ram_usage": format_bytes(quant_ram),
        "ram_fp32": format_bytes(fp32_ram),
        "ram_savings": f"{savings_ratio:.1f}x vs FP32",
        "load_time": f"{load_time:.1f}s",
    }

    print(f"\n  http://localhost:{args.port}\n")

    server = HTTPServer(("0.0.0.0", args.port), ChatHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutdown.")
        server.shutdown()


if __name__ == "__main__":
    main()
