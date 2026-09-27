"""Serve a browser dashboard for the local education pipeline."""

import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from src.pipeline import OUTPUT_DIR, run_pipeline


HOST = "127.0.0.1"
PORT = 8000
REPORT_PATH = os.path.join(OUTPUT_DIR, "pipeline_report.json")

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Edu AI Pipeline</title><style>
:root{color-scheme:light;--paper:#f4f3ed;--ink:#202521;--muted:#6e756e;--line:#d9ddd4;--green:#286a4a;--lime:#d5ef67;--coral:#d36b50;--white:#fffefa}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 "Segoe UI",sans-serif}
header{min-height:68px;padding:14px max(22px,calc((100vw - 1120px)/2));display:flex;justify-content:space-between;align-items:center;background:var(--white);border-bottom:1px solid var(--line)}
.brand{display:flex;align-items:center;gap:11px;font-weight:700}.mark{width:30px;height:30px;display:grid;place-items:center;background:var(--lime);border-radius:6px;color:var(--green)}.muted,.status{color:var(--muted);font-size:13px}
main{max-width:1120px;margin:auto;padding:40px 22px 60px}.intro{display:flex;align-items:end;justify-content:space-between;gap:20px;margin-bottom:28px}.eyebrow{margin:0 0 8px;color:var(--green);font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:1px}
h1,h2,p{margin-top:0}h1,h2{font-family:Georgia,serif;font-weight:500}h1{margin-bottom:5px;font-size:clamp(30px,5vw,44px);line-height:1.1}h2{margin-bottom:0;font-size:23px}.sub{margin:0;color:var(--muted)}
button{font:inherit}.action{min-height:42px;padding:0 15px;border:0;border-radius:5px;background:var(--green);color:#fff;font-weight:650;cursor:pointer}.action:hover{background:#1d573a}.action:disabled{opacity:.6;cursor:progress}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);background:var(--white)}.metric{padding:19px 21px;border-right:1px solid var(--line)}.metric:last-child{border:0}.label{display:block;margin-bottom:10px;color:var(--muted);font-size:13px}.value{font:32px/1 Georgia,serif}
.heading{display:flex;justify-content:space-between;align-items:baseline;gap:12px;margin:36px 0 13px}.grid{display:grid;grid-template-columns:1.15fr .85fr;gap:20px}.panel{min-width:0;padding:21px;border-top:3px solid var(--green);background:var(--white)}h3{margin:0 0 13px;font-size:15px}.summary{margin:0 0 18px;color:#424a43;font-size:14px}.summary:last-child{margin:0}.tag{display:block;margin-bottom:5px;color:var(--green);font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.7px}.concepts{display:flex;flex-wrap:wrap;gap:8px}.concept{padding:5px 8px;border:1px solid var(--line);border-radius:4px;font-size:12px}.concept b{margin-left:4px;color:var(--green)}
.videos{border-top:1px solid var(--line)}.row{display:grid;grid-template-columns:minmax(150px,1.1fr) .8fr 1fr;gap:16px;padding:15px 0;border-bottom:1px solid var(--line)}.name{font-size:13px;font-weight:700;overflow-wrap:anywhere}.detail{color:var(--muted);font-size:12px}.detail strong{display:block;margin-bottom:4px;color:var(--ink)}.request{margin:3px 0;color:#515951}.links{display:flex;gap:18px;margin-top:23px}.links a{color:var(--green);font-size:13px;font-weight:650;text-underline-offset:3px}.error{padding:12px 15px;border-left:3px solid var(--coral);background:#fff1ed;color:#743b2c}
@media(max-width:720px){main{padding:28px 16px 45px}.intro{align-items:start;flex-direction:column}.metrics{grid-template-columns:repeat(2,1fr)}.metric:nth-child(2){border-right:0}.metric:nth-child(-n+2){border-bottom:1px solid var(--line)}.grid{grid-template-columns:1fr}.heading{align-items:start;flex-direction:column;gap:4px}.row{grid-template-columns:1fr 1fr}.row>:last-child{grid-column:1/-1}}
</style></head><body>
<header><div class="brand"><span class="mark" aria-hidden="true">E</span>Edu AI Pipeline</div><span class="status" id="status">Local workspace</span></header>
<main><section class="intro"><div><p class="eyebrow">Knowledge extraction / Demo</p><h1>Pipeline overview</h1><p class="sub">Transcript insights, knowledge fusion, and audience signals.</p></div><button class="action" id="run">&#8635; &nbsp;Run pipeline</button></section>
<div id="message" role="status"></div><section class="metrics" aria-label="Pipeline metrics">
<div class="metric"><span class="label">Videos processed</span><span class="value" id="video-count">-</span></div><div class="metric"><span class="label">Segments extracted</span><span class="value" id="segment-count">-</span></div><div class="metric"><span class="label">Fused concepts</span><span class="value" id="node-count">-</span></div><div class="metric"><span class="label">Knowledge graph links</span><span class="value" id="edge-count">-</span></div></section>
<div class="heading"><h2>Unified knowledge</h2><span class="muted">Concepts shared across the transcript library</span></div><section class="grid">
<article class="panel"><h3>Generated summaries</h3><p class="summary"><span class="tag">English</span><span id="summary-en">Loading report...</span></p><p class="summary"><span class="tag">Sinhala glossary demo</span><span id="summary-si"></span></p></article>
<article class="panel"><h3>Most-mentioned concepts</h3><div class="concepts" id="concepts"></div></article></section>
<div class="heading"><h2>Video signals</h2><span class="muted">Feedback mining and simulated audio analysis</span></div><section class="videos" id="videos"></section>
<nav class="links"><a href="/outputs/pipeline_report.json">View JSON report</a><a href="/outputs/unified_knowledge_graph.graphml">Download knowledge graph</a></nav></main>
<script>
const el=id=>document.getElementById(id), esc=value=>String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
function render(data){const videos=data.stage1_extraction||[], concepts=data.stage2_unified_knowledge_base||[], content=data.stage2_generated_content||{};
el("video-count").textContent=videos.length;el("segment-count").textContent=videos.reduce((n,v)=>n+(v.segments||[]).length,0);el("node-count").textContent=data.stage2_graph_stats?.node_count??0;el("edge-count").textContent=data.stage2_graph_stats?.edge_count??0;
el("summary-en").textContent=content.en?.summary||"No English summary available.";el("summary-si").textContent=content.si?.summary||"No Sinhala summary available.";
el("concepts").innerHTML=concepts.slice(0,12).map(c=>`<span class="concept">${esc(c.concept)}<b>${esc(c.mention_count)}</b></span>`).join("")||"No concepts found.";
const feedback=data.stage3_feedback||{}, audio=data.stage4_audio_recommendations||{};el("videos").innerHTML=videos.map(v=>{const f=feedback[v.video_id]||{},a=audio[v.video_id]||{},requests=(f.content_requests||[]).map(x=>`<p class="request">${esc(x)}</p>`).join("")||'<p class="request">No content requests detected.</p>';return `<article class="row"><div class="name">${esc(v.video_id)}</div><div class="detail"><strong>${(v.segments||[]).length} segments</strong>Sentiment: ${esc(f.overall_sentiment_score??"n/a")}<br>Audio: ${esc(a.audio_emotion||"n/a")}</div><div class="detail"><strong>Audience requests</strong>${requests}</div></article>`}).join("");
el("status").textContent="Report loaded from local workspace";el("message").replaceChildren();}
async function load(){const r=await fetch("/api/report");if(!r.ok)throw Error(await r.text());render(await r.json())}
el("run").addEventListener("click",async()=>{const b=el("run");b.disabled=true;b.textContent="Running pipeline...";el("status").textContent="Processing sample transcripts";try{const r=await fetch("/api/run",{method:"POST"});if(!r.ok)throw Error(await r.text());render(await r.json())}catch(e){el("message").innerHTML=`<p class="error">${esc(e.message)}</p>`}finally{b.disabled=false;b.innerHTML="&#8635; &nbsp;Run pipeline"}});
load().catch(e=>{el("summary-en").textContent="Could not load the report.";el("message").innerHTML=`<p class="error">${esc(e.message)}</p>`});
</script></body></html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    def send_body(self, status, body, content_type):
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def send_report(self):
        with open(REPORT_PATH, encoding="utf-8") as report_file:
            self.send_body(200, report_file.read(), "application/json; charset=utf-8")

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self.send_body(200, PAGE, "text/html; charset=utf-8")
        elif path == "/api/report":
            if not os.path.exists(REPORT_PATH):
                run_pipeline()
            self.send_report()
        elif path in ("/outputs/pipeline_report.json", "/outputs/unified_knowledge_graph.graphml"):
            file_path = os.path.join(OUTPUT_DIR, os.path.basename(path))
            if not os.path.isfile(file_path):
                self.send_body(404, "Output file not found", "text/plain; charset=utf-8")
                return
            with open(file_path, encoding="utf-8") as output_file:
                kind = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
                self.send_body(200, output_file.read(), f"{kind}; charset=utf-8")
        else:
            self.send_body(404, "Not found", "text/plain; charset=utf-8")

    def do_POST(self):
        if urlparse(self.path).path != "/api/run":
            self.send_body(404, "Not found", "text/plain; charset=utf-8")
            return
        try:
            run_pipeline()
            self.send_report()
        except Exception as error:
            self.send_body(500, str(error), "text/plain; charset=utf-8")

    def log_message(self, format_string, *args):
        print(f"{self.address_string()} - {format_string % args}")


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    print(f"Edu AI Pipeline dashboard: http://localhost:{PORT}")
    print("Press Ctrl+C to stop the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server.")
    finally:
        server.server_close()