APP_CSS = """
<style>
:root {--ink:#132033;--muted:#68758a;--line:#e4e8ef;--accent:#335cff;--soft:#f5f7fb;}
.stApp {background:#f7f8fa;color:var(--ink);}
.block-container {max-width:1280px;padding-top:2.1rem;padding-bottom:4rem;}
h1,h2,h3 {letter-spacing:-.035em;color:var(--ink);}
.v2k-header {background:white;border:1px solid var(--line);border-radius:18px;padding:24px 28px;margin-bottom:22px;box-shadow:0 8px 26px rgba(25,39,68,.055)}
.v2k-brand {font-size:1.65rem;font-weight:760;letter-spacing:-.04em;margin:0}.v2k-sub {color:var(--muted);margin:.35rem 0 0;font-size:1rem}
.v2k-card {background:white;border:1px solid var(--line);border-radius:16px;padding:20px 22px;box-shadow:0 5px 18px rgba(25,39,68,.04);margin-bottom:14px}
.v2k-kicker {color:#53617a;font-size:.72rem;letter-spacing:.11em;text-transform:uppercase;font-weight:750;margin-bottom:8px}
.v2k-status {display:inline-flex;align-items:center;gap:7px;padding:6px 10px;border:1px solid var(--line);border-radius:999px;background:#fff;font-size:.78rem;color:#41506a;margin-right:7px}
.v2k-dot {width:7px;height:7px;border-radius:50%;background:#27a66f;display:inline-block}.v2k-dot.warn{background:#d89b2b}
.model-row {border:1px solid var(--line);border-radius:13px;padding:13px 14px;margin:9px 0;background:#fbfcfe}.model-role{font-size:.76rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}.model-name{font-weight:700;margin-top:4px}.model-meta{font-size:.8rem;color:var(--muted);margin-top:2px}
div[data-testid="stFileUploader"] section {min-height:150px;border:1.5px dashed #aeb9cb;border-radius:14px;background:#fbfcff}div[data-testid="stFileUploader"] section:hover{border-color:var(--accent);background:#f7f9ff}
.stButton>button[kind="primary"] {border-radius:10px;font-weight:680;background:var(--accent)}
.st-key-generate_cta {margin-top:12px}
.st-key-generate_cta .stButton>button {
  min-height:42px;height:42px;padding:0 18px;border-radius:10px;
  font-size:.92rem;font-weight:680;letter-spacing:.005em;
  box-shadow:0 4px 12px rgba(51,92,255,.14);
  transition:transform .15s ease,box-shadow .15s ease,background .15s ease;
}
.st-key-generate_cta .stButton>button:not(:disabled):hover {
  transform:translateY(-1px);box-shadow:0 6px 15px rgba(51,92,255,.2);background:#294ee8;
}
.st-key-generate_cta .stButton>button:disabled {
  background:#e8ecf3;color:#8b96a8;border-color:#dfe4ec;box-shadow:none;opacity:1;
}
.stDownloadButton>button {min-height:46px;border-radius:11px;font-weight:700}
div[data-baseweb="select"]>div, .stTextInput input {border-radius:10px!important}
div[data-testid="stMetric"] {background:white;border:1px solid var(--line);padding:14px;border-radius:13px}
@media(max-width:820px){
  .block-container{padding:1rem}.v2k-header{padding:20px}.v2k-status{margin-bottom:6px}
  div[data-testid="stHorizontalBlock"]{flex-wrap:wrap!important}
  div[data-testid="stColumn"]{min-width:100%!important;flex:1 1 100%!important}
}
</style>
"""
