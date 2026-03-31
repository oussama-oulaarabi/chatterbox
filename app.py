import datetime as dt, hashlib, json, os, platform, re, shutil, subprocess, sys, time, traceback, uuid
from pathlib import Path
import gradio as gr, numpy as np, soundfile as sf, torch

APP_ROOT = Path(os.environ.get("APP_ROOT", Path(__file__).resolve().parent))
DATA_ROOT = APP_ROOT / "data"
INPUT_ROOT, OUTPUT_ROOT, LOG_ROOT, STATE_ROOT = DATA_ROOT/"inputs", DATA_ROOT/"outputs", DATA_ROOT/"logs", DATA_ROOT/"state"
RUNTIME_LOG_DIR = APP_ROOT / "runtime_logs"
for p in [INPUT_ROOT, OUTPUT_ROOT, LOG_ROOT, STATE_ROOT, RUNTIME_LOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

SUPPORTED_LANGUAGES = ["ar","da","de","el","en","es","fi","fr","he","hi","it","ja","ko","ms","nl","no","pl","pt","ru","sv","sw","tr","zh"]
DEFAULT_PROFILE = {"language":"ja","voice":"ja_female_neutral","speed":0.92,"pitch":-1.0,"temperature":0.55,"top_p":0.9,"pause_scale":1.2,"sentence_silence_ms":320,"comma_silence_ms":140,"emotion_strength":0.35,"clarity":0.95,"breathiness":0.25,"chunk_soft_limit":220,"chunk_hard_limit":280,"retry_per_chunk":2,"target_sr":24000}
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = None

def now_utc(): return dt.datetime.utcnow().isoformat() + "Z"
def log_line(msg:str):
    with open(RUNTIME_LOG_DIR/"debug_app.log","a",encoding="utf-8") as f: f.write(f"[{now_utc()}] {msg}\n")
def write_json(path:Path,obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
def sha256_file(path:Path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
def shell_capture(cmd):
    p=subprocess.run(cmd,capture_output=True,text=True)
    return {"cmd":cmd,"rc":p.returncode,"stdout":p.stdout[-12000:],"stderr":p.stderr[-12000:]}
def write_environment_snapshot():
    snap={"timestamp_utc":now_utc(),"python":sys.version,"platform":platform.platform(),"device":DEVICE,"gpu_name":GPU_NAME,"nvidia_smi":shell_capture(["bash","-lc","nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"]),"rclone_version":shell_capture(["bash","-lc","rclone version || true"]),"git_head":shell_capture(["bash","-lc","cd \"$APP_ROOT/src\" 2>/dev/null && git rev-parse HEAD || true"]),"env_subset":{k:os.environ.get(k,"") for k in ["APP_ROOT","SERVER_PORT","RCLONE_REMOTE","FORK_REPO_URL","FORK_COMMIT_SHA","HF_MODEL_REPO_ID","HF_MODEL_REVISION","VAST_INSTANCE_ID","DESTROY_INSTEAD_OF_STOP","DESTROY_ON_UPLOAD_FAILURE","UPLOAD_RETRY_ATTEMPTS"]}}
    write_json(RUNTIME_LOG_DIR/"environment_snapshot.json", snap)
def schedule_vast_action(action:str, reason:str):
    instance_id=os.environ.get("VAST_INSTANCE_ID","").strip() or os.environ.get("CONTAINER_ID","").strip()
    if not instance_id:
        log_line(f"Skip {action}; no instance id. reason={reason}"); return
    script=APP_ROOT/f"{action}_after_run.sh"
    script.write_text(f"#!/usr/bin/env bash\nset -e\nsleep 10\npython3 -m pip -q install vastai >/dev/null 2>&1 || true\nvastai {action} instance {instance_id} || true\n",encoding="utf-8")
    script.chmod(0o755); log_line(f"Scheduling {action} for instance {instance_id}; reason={reason}")
    subprocess.Popen(["bash", str(script)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
def load_model():
    global MODEL
    if MODEL is not None: return MODEL
    log_line("Loading Chatterbox multilingual model")
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    MODEL = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
    write_json(RUNTIME_LOG_DIR/"model_load.json",{"timestamp_utc":now_utc(),"import_path":"chatterbox.mtl_tts.ChatterboxMultilingualTTS","factory":"ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)","device":DEVICE,"gpu_name":GPU_NAME})
    return MODEL
def sanitize_name(x:str)->str:
    x=re.sub(r"[^a-zA-Z0-9._-]+","_",(x or "").strip()); return x.strip("_") or f"item_{uuid.uuid4().hex[:8]}"
def split_sections(text:str):
    parts=re.split(r"\[(HOOK|CONTEXT|DEV|CLIMAX|END)\]",text)
    if len(parts)<3: return [("DEV",text.strip())]
    out=[]
    for i in range(1,len(parts),2):
        body=parts[i+1].strip() if i+1 < len(parts) else ""
        if body: out.append((parts[i],body))
    return out or [("DEV",text.strip())]
def strip_markers_for_model(text:str)->str:
    return re.sub(r"\[(HOOK|CONTEXT|DEV|CLIMAX|END)\]"," ",text).replace("  "," ").strip()
def section_profile(base,tag):
    p=dict(base)
    if tag=="HOOK":
        p["speed"]=max(0.84,p["speed"]-0.04); p["emotion_strength"]=min(0.55,p["emotion_strength"]+0.05)
    elif tag=="CONTEXT":
        p["speed"]=min(1.00,p["speed"]+0.02); p["clarity"]=min(1.0,p["clarity"]+0.02)
    elif tag=="CLIMAX":
        p["speed"]=max(0.86,p["speed"]-0.03); p["emotion_strength"]=min(0.60,p["emotion_strength"]+0.07)
    elif tag=="END":
        p["speed"]=max(0.82,p["speed"]-0.05)
    return p
def sentence_split_generic(text:str):
    chunks=re.split(r"(?<=[。！？….!?])",text); chunks=[x.strip() for x in chunks if x.strip()]; return chunks or [text]
def chunk_text(text,soft_limit=220,hard_limit=280):
    text=text.strip()
    if len(text)<=hard_limit: return [text]
    out=[]; buf=""
    for sent in sentence_split_generic(text):
        candidate=(buf+" "+sent).strip() if buf else sent
        if len(candidate)<=soft_limit: buf=candidate; continue
        if buf: out.append(buf); buf=sent
        else:
            while len(sent)>hard_limit: out.append(sent[:hard_limit]); sent=sent[hard_limit:]
            buf=sent
    if buf: out.append(buf)
    return out
def estimate_seconds(text): return round((max(1,len(text))/11.0) * (0.55 if DEVICE=="cuda" else 3.0), 1)
def concat_with_silence(wavs,sr,sentence_silence_ms=320):
    if not wavs: return np.zeros((1,),dtype=np.float32), sr
    silence=np.zeros((int(sr*sentence_silence_ms/1000.0),),dtype=np.float32); out=[]
    for i,w in enumerate(wavs):
        out.append(np.asarray(w).astype(np.float32).reshape(-1))
        if i < len(wavs)-1: out.append(silence)
    return np.concatenate(out), sr
def run_rclone_copy(src_dir:Path, remote_path:str):
    p=subprocess.run(["rclone","copy",str(src_dir),remote_path,"--create-empty-src-dirs","-P"],capture_output=True,text=True)
    return p.returncode,p.stdout[-12000:],p.stderr[-12000:]
def retry_rclone_copy(src_dir:Path, remote_path:str, label:str, attempts:int, log_obj:dict):
    hist=[]
    for attempt in range(1,attempts+1):
        rc,out,err=run_rclone_copy(src_dir,remote_path)
        hist.append({"attempt":attempt,"rc":rc,"stdout":out,"stderr":err})
        if rc==0:
            log_obj[label]={"ok":True,"attempts":hist}; return True
        time.sleep(min(10,attempt*2))
    log_obj[label]={"ok":False,"attempts":hist}; return False
def build_jobs_table(jobs):
    return [[i,j["name"],j["language"],len(strip_markers_for_model(j["script"])),bool(j.get("clone_wav")),estimate_seconds(strip_markers_for_model(j["script"]))] for i,j in enumerate(jobs,1)]
def add_job(name, language, script, clone_wav, jobs_state):
    jobs=list(jobs_state or [])
    if not (name or "").strip(): raise gr.Error("Name is required.")
    if not (script or "").strip(): raise gr.Error("Script is required.")
    if language not in SUPPORTED_LANGUAGES: raise gr.Error("Unsupported language.")
    safe_name=sanitize_name(name); clone_path=clone_wav if clone_wav else None
    jobs.append({"name":safe_name,"language":language,"script":script.strip(),"clone_wav":clone_path})
    log_line(f"Added job {safe_name} lang={language} clone={bool(clone_path)}")
    return jobs, build_jobs_table(jobs), f"{len(jobs)} job(s) in queue.", "", None
def remove_last_job(jobs_state):
    jobs=list(jobs_state or [])
    if jobs: removed=jobs.pop(); log_line(f"Removed job {removed.get('name')}")
    return jobs, build_jobs_table(jobs), f"{len(jobs)} job(s) in queue."
def clear_jobs():
    log_line("Cleared queue"); return [], [], "Queue cleared."
def synthesize_one(item_name,text,voice_wav,profile,run_dir:Path,per_job_log_dir:Path):
    model=load_model(); sections=split_sections(text); all_audio=[]; logs=[]
    for tag, section_text in sections:
        p=section_profile(profile,tag); chunks=chunk_text(section_text,int(p["chunk_soft_limit"]),int(p["chunk_hard_limit"]))
        for chunk_idx,chunk in enumerate(chunks,1):
            last_err=None
            for attempt in range(int(p["retry_per_chunk"])+1):
                try:
                    t0=time.time(); wav=model.generate(chunk,language_id=p["language"],audio_prompt_path=voice_wav if voice_wav else None); gen_s=round(time.time()-t0,3)
                    if torch.is_tensor(wav): wav=wav.detach().float().cpu().numpy()
                    all_audio.append(np.asarray(wav).reshape(-1)); logs.append({"section":tag,"chunk_index":chunk_idx,"attempt":attempt+1,"status":"ok","language":p["language"],"chars":len(chunk),"generation_seconds":gen_s}); break
                except Exception as e:
                    last_err=traceback.format_exc(); logs.append({"section":tag,"chunk_index":chunk_idx,"attempt":attempt+1,"status":"retry" if attempt < int(p["retry_per_chunk"]) else "failed","error":str(e),"traceback":last_err[-6000:],"language":p["language"],"chars":len(chunk)}); time.sleep(1.0+attempt)
            else:
                write_json(per_job_log_dir/f"{item_name}_crash.json",{"error":last_err}); raise RuntimeError(f"Chunk failed permanently: {tag} #{chunk_idx}\n{last_err}")
    sr=getattr(model,"sr",int(profile["target_sr"])); audio,sr=concat_with_silence(all_audio,sr=sr,sentence_silence_ms=int(profile["sentence_silence_ms"])); out_wav=run_dir/f"{item_name}.wav"; sf.write(out_wav,audio,sr); return out_wav,logs
def generate_all(jobs_state, default_language, voice_name, profile_json, auto_upload, destroy_after_run, progress=gr.Progress(track_tqdm=False)):
    write_environment_snapshot(); jobs=list(jobs_state or [])
    if not jobs: raise gr.Error("Add at least one job first.")
    started=time.time(); started_utc=now_utc(); run_id=dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8]
    run_dir,inp_dir,log_dir,per_job_log_dir=OUTPUT_ROOT/run_id,INPUT_ROOT/run_id,LOG_ROOT/run_id,(LOG_ROOT/run_id/"jobs")
    for p in [run_dir,inp_dir,log_dir,per_job_log_dir]: p.mkdir(parents=True, exist_ok=True)
    profile=dict(DEFAULT_PROFILE); profile["language"]=default_language; profile["voice"]=voice_name
    incoming=json.loads(profile_json.strip()) if profile_json.strip() else {}; profile.update(incoming)
    write_json(log_dir/"profile_effective.json",profile); write_json(log_dir/"queued_jobs_initial.json",jobs)
    staged_jobs=[]
    for idx,job in enumerate(jobs,1):
        clone_name=""; staged_clone=None
        if job.get("clone_wav"):
            clone_src=Path(job["clone_wav"]); clone_name=clone_src.name; staged_clone=inp_dir/f"{idx:03d}_{sanitize_name(clone_src.name)}"; shutil.copy2(clone_src, staged_clone)
        staged_jobs.append({"name":sanitize_name(job["name"]),"language":job.get("language") or default_language,"script":job["script"],"clone_wav":str(staged_clone) if staged_clone else None,"clone_wav_name":clone_name})
    write_json(inp_dir/"jobs.json",staged_jobs); write_json(inp_dir/"profile.json",profile)
    rows=[]; generated_paths=[]
    for idx,job in enumerate(staged_jobs,1):
        progress((idx-1)/max(1,len(staged_jobs)), desc=f"Generating {job['name']} [{job['language']}] ({idx}/{len(staged_jobs)})")
        jp=dict(profile); jp["language"]=job["language"]
        out_wav, section_logs=synthesize_one(job["name"],job["script"],job["clone_wav"],jp,run_dir,per_job_log_dir)
        generated_paths.append(str(out_wav))
        write_json(per_job_log_dir/f"{job['name']}.json",{"sections":section_logs,"chars":len(strip_markers_for_model(job["script"])),"clone_wav_name":job["clone_wav_name"],"language":job["language"],"output_wav":str(out_wav)})
        rows.append({"name":job["name"],"language":job["language"],"chars":len(strip_markers_for_model(job["script"])),"eta_s":estimate_seconds(job["script"]),"clone_wav":job["clone_wav_name"],"file":str(out_wav)})
    upload_retry_attempts=int(os.environ.get("UPLOAD_RETRY_ATTEMPTS","3") or "3")
    destroy_on_upload_failure=os.environ.get("DESTROY_ON_UPLOAD_FAILURE","true").lower()=="true"
    upload_log={"uploaded":False,"upload_retry_attempts":upload_retry_attempts,"destroy_on_upload_failure":destroy_on_upload_failure}
    if auto_upload:
        remote=os.environ.get("RCLONE_REMOTE","").strip()
        if remote:
            ok_inputs=retry_rclone_copy(inp_dir,f"{remote}/{run_id}/inputs","inputs",upload_retry_attempts,upload_log)
            ok_outputs=retry_rclone_copy(run_dir,f"{remote}/{run_id}/outputs","outputs",upload_retry_attempts,upload_log)
            ok_logs=retry_rclone_copy(log_dir,f"{remote}/{run_id}/logs","logs",upload_retry_attempts,upload_log)
            upload_log["uploaded"]=bool(ok_inputs and ok_outputs and ok_logs)
            if not upload_log["uploaded"]:
                salvage={}
                salvage_ok=retry_rclone_copy(log_dir,f"{remote}/{run_id}/logs_failure_salvage","logs_failure_salvage",max(1,upload_retry_attempts),salvage)
                upload_log["failure_log_salvage"]=salvage.get("logs_failure_salvage",{})
                upload_log["failure_log_salvage"]["ok"]=bool(salvage_ok)
        else:
            upload_log["error"]="RCLONE_REMOTE is empty"
    write_json(log_dir/"upload.json",upload_log)
    req_path=APP_ROOT/"requirements.lock.txt"
    metadata={"run_id":run_id,"repo_url":os.environ.get("FORK_REPO_URL",""),"repo_commit_sha":os.environ.get("FORK_COMMIT_SHA",""),"hf_model_repo_id":os.environ.get("HF_MODEL_REPO_ID",""),"hf_model_revision":os.environ.get("HF_MODEL_REVISION",""),"requirements_lock_sha256":sha256_file(req_path) if req_path.exists() else "","gpu_name":GPU_NAME,"device":DEVICE,"started_utc":started_utc,"ended_utc":now_utc(),"rclone_remote":os.environ.get("RCLONE_REMOTE",""),"count_items":len(staged_jobs),"profile":profile,"generated_files":generated_paths,"jobs":staged_jobs}
    write_json(log_dir/"RUN_METADATA.json",metadata)
    elapsed=round(time.time()-started,2)
    if destroy_after_run:
        if upload_log.get("uploaded"):
            schedule_vast_action("destroy" if os.environ.get("DESTROY_INSTEAD_OF_STOP","true").lower()=="true" else "stop", "upload_success")
        elif destroy_on_upload_failure:
            schedule_vast_action("destroy","upload_failed_after_retries")
    summary={"run_id":run_id,"gpu":GPU_NAME,"items":len(staged_jobs),"elapsed_s":elapsed,"uploaded":upload_log.get("uploaded",False),"generated":generated_paths,"languages":sorted(list({j["language"] for j in staged_jobs})),"destroy_on_upload_failure":destroy_on_upload_failure}
    write_json(log_dir/"summary.json",summary)
    md=f"### Run summary\n- **Run ID**: `{run_id}`\n- **GPU**: `{GPU_NAME}`\n- **Items**: `{len(staged_jobs)}`\n- **Languages**: `{', '.join(summary['languages'])}`\n- **Elapsed**: `{elapsed}s`\n- **Uploaded**: `{upload_log.get('uploaded', False)}`\n- **Destroy on upload failure**: `{destroy_on_upload_failure}`\n"
    return rows, generated_paths, json.dumps(summary, ensure_ascii=False, indent=2), md

CSS=".gradio-container {max-width: 1550px !important;} .header-card {border-radius: 18px; padding: 18px; background: linear-gradient(135deg, #111827, #1f2937); color: white; margin-bottom: 12px;} .small-muted {opacity: .78; font-size: .92rem;}"
example_profile=json.dumps(DEFAULT_PROFILE, ensure_ascii=False, indent=2)
with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
    jobs_state=gr.State([])
    gr.HTML(f'<div class="header-card"><h1>Chatterbox — Destroy After Failed Upload With Log Salvage</h1><div class="small-muted">GPU: {GPU_NAME} • Device: {DEVICE}</div></div>')
    with gr.Row():
        with gr.Column(scale=2):
            job_name=gr.Textbox(label="Name", placeholder="episode_001")
            job_language=gr.Dropdown(choices=SUPPORTED_LANGUAGES, value="ja", label="Job language")
            job_script=gr.Textbox(label="Script", lines=16, placeholder="[HOOK] ... [CONTEXT] ... [DEV] ... [CLIMAX] ... [END] ...")
            job_clone=gr.File(label="Clone WAV", file_count="single", type="filepath")
            with gr.Row():
                add_btn=gr.Button("Add", variant="primary"); remove_btn=gr.Button("Remove Last"); clear_btn=gr.Button("Clear All")
            queue_status=gr.Markdown("0 job(s) in queue.")
        with gr.Column(scale=1):
            default_language=gr.Dropdown(choices=SUPPORTED_LANGUAGES, value="ja", label="Default language")
            voice_name=gr.Textbox(value="ja_female_neutral", label="Voice name")
            profile_json=gr.Code(value=example_profile, language="json", label="Profile JSON")
            auto_upload=gr.Checkbox(value=True, label="Upload inputs/outputs/logs with rclone")
            destroy_after_run=gr.Checkbox(value=True, label="Destroy after success or failed upload retries")
            gen_btn=gr.Button("Generate All", variant="primary")
    jobs_table=gr.Dataframe(headers=["#","name","language","chars","has_clone","eta_s"], datatype=["number","str","str","number","bool","number"], label="Queued jobs", interactive=False)
    result_table=gr.Dataframe(headers=["name","language","chars","eta_s","clone_wav","file"], datatype=["str","str","number","number","str","str"], label="Generated items", interactive=False)
    gallery_audio=gr.File(label="Output WAV files", file_count="multiple")
    summary_json=gr.Code(label="Summary JSON", language="json")
    summary_md=gr.Markdown()
    add_btn.click(fn=add_job, inputs=[job_name,job_language,job_script,job_clone,jobs_state], outputs=[jobs_state,jobs_table,queue_status,job_name,job_clone], queue=False)
    remove_btn.click(fn=remove_last_job, inputs=[jobs_state], outputs=[jobs_state,jobs_table,queue_status], queue=False)
    clear_btn.click(fn=clear_jobs, inputs=[], outputs=[jobs_state,jobs_table,queue_status], queue=False)
    gen_btn.click(fn=generate_all, inputs=[jobs_state,default_language,voice_name,profile_json,auto_upload,destroy_after_run], outputs=[result_table,gallery_audio,summary_json,summary_md], queue=True)
if __name__=="__main__":
    log_line("App started"); demo.queue(default_concurrency_limit=1); demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("SERVER_PORT","7860")), share=False)
