"""Chat app — Foundry Local SDK v1.2.x + streaming UI at http://localhost:8080.

Run:  python chat_app.py   (inside the venv from the runbook)
Deps: foundry-local-sdk  (macOS/Linux)  or  foundry-local-sdk-winml  (Windows)
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ALIAS = "phi-3.5-mini"
PORT = 8080

STATE = {"stage": "starting", "detail": "", "ready": False, "model_id": None}
_chat = None
_lock = threading.Lock()


def init_model():
    global _chat
    from foundry_local_sdk import Configuration, FoundryLocalManager
    try:
        STATE.update(stage="runtime", detail="Initializing Foundry Local runtime...")
        FoundryLocalManager.initialize(Configuration(app_name="foundry-local-pot"))
        mgr = FoundryLocalManager.instance
        eps = ", ".join(e.name for e in mgr.discover_eps()) or "none"
        STATE.update(stage="catalog", detail=f"EPs: {eps}. Resolving '{MODEL_ALIAS}'...")
        model = mgr.catalog.get_model(MODEL_ALIAS)
        if model is None:
            opts = [m.alias for m in mgr.catalog.list_models()][:8]
            raise RuntimeError(f"'{MODEL_ALIAS}' not in catalog; try one of {opts}")
        if not model.is_cached:
            STATE.update(stage="download", detail="Downloading model (~2 GB first run)...")
            model.download(progress_callback=lambda p: STATE.update(
                detail=f"Downloading model... {p:.0f}%"))
        else:
            STATE.update(stage="download", detail="Model found in cache.")
        STATE.update(stage="load", detail="Loading model into GPU memory...")
        model.load()
        _chat = model.get_chat_client()
        STATE.update(stage="ready", detail=f"Model {model.id} ready.", ready=True,
                     model_id=model.id)
    except Exception as e:  # surfaced in the UI instead of a dead page
        STATE.update(stage="error", detail=f"{type(e).__name__}: {e}")


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Foundry Local chat</title>
<style>
 body{font-family:Segoe UI,Arial,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;background:#f7f8fa}
 h2{color:#1f3864} #stages li{margin:.3rem 0;color:#888} #stages li.done{color:#1a7f37}
 #stages li.now{color:#1f3864;font-weight:600} #chat{display:none}
 #log{border:1px solid #ddd;background:#fff;border-radius:8px;padding:1rem;min-height:260px;white-space:pre-wrap}
 .u{color:#1f3864;font-weight:600}.a{color:#222} textarea{width:100%;height:70px;margin-top:.5rem}
 button{background:#1f3864;color:#fff;border:0;border-radius:6px;padding:.5rem 1.2rem;cursor:pointer}
 .err{color:#b91c1c;font-weight:600}
</style></head><body>
<h2>Foundry Local: on-device chat</h2>
<ol id="stages">
 <li data-s="runtime">Start local runtime</li><li data-s="catalog">Resolve model for hardware</li>
 <li data-s="download">Download / verify cache</li><li data-s="load">Load into GPU</li>
 <li data-s="ready">Ready</li>
</ol><p id="detail"></p>
<div id="chat"><div id="log"></div>
<textarea id="q" placeholder="Ask something... (Ctrl+Enter to send)"></textarea><br>
<button onclick="send()">Send</button></div>
<script>
const order=["runtime","catalog","download","load","ready"];
async function poll(){const r=await fetch('/status');const s=await r.json();
 document.getElementById('detail').textContent=s.detail;
 if(s.stage==='error'){document.getElementById('detail').className='err';return;}
 const i=order.indexOf(s.stage);
 document.querySelectorAll('#stages li').forEach((li,k)=>{li.className=k<i?'done':(k===i?'now':'');});
 if(s.ready){document.getElementById('chat').style.display='block';
  document.querySelectorAll('#stages li').forEach(li=>li.className='done');return;}
 setTimeout(poll,700);} poll();
document.getElementById('q').addEventListener('keydown',e=>{if(e.ctrlKey&&e.key==='Enter')send();});
async function send(){const q=document.getElementById('q');const log=document.getElementById('log');
 const text=q.value.trim();if(!text)return;q.value='';
 log.innerHTML+='<div class="u">You: '+text.replace(/</g,'&lt;')+'</div><div class="a">AI: ';
 const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({message:text})});
 const rd=r.body.getReader();const dec=new TextDecoder();
 while(true){const{done,value}=await rd.read();if(done)break;
  log.innerHTML+=dec.decode(value).replace(/</g,'&lt;');log.scrollTop=log.scrollHeight;}
 log.innerHTML+='</div>';}
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet console
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/status":
            self._send(200, "application/json", json.dumps(STATE).encode())
        else:
            self._send(200, "text/html; charset=utf-8", PAGE.encode())

    def do_POST(self):
        if self.path != "/chat" or not STATE["ready"]:
            self._send(503, "text/plain", b"model not ready")
            return
        n = int(self.headers.get("Content-Length", 0))
        msg = json.loads(self.rfile.read(n))["message"]
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with _lock:  # one generation at a time on a single local model
            for chunk in _chat.complete_streaming_chat(
                [{"role": "user", "content": msg}]
            ):
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    try:
                        self.wfile.write(delta.encode())
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionAbortedError):
                        return


if __name__ == "__main__":
    threading.Thread(target=init_model, daemon=True).start()
    print(f"Open http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
