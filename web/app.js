let specs=null,current=null,runId=null,timer=null,lastFrame=-1;
let playbackTimer=null,playbackFrame=0,playbackTotal=0,playbackPlaying=false;
let zoom=1,panX=0,panY=0,dragState=null;
let deep=null,deepActive=false,deepManifest=null;
let neuralTarget={kind:0,custom:false};
const FRAME_INTERVAL_MS=140;
const $=s=>document.querySelector(s);
const stories={
 black_hole:["Start with an ordinary sky.","Now bend every line of sight around the lens.","The image is the answer to hundreds of thousands of independent questions.","Then pull back: one observer was only the beginning."],
 pbh:["The Universe is extremely young.","Increase one small density perturbation.","Below the threshold it disperses; above it, collapse accelerates.","Pull back and map the boundary using many universes."],
 fluid:["Begin with smooth flow.","Put an obstacle in the stream.","Watch coherent vortices form and interact.","Then reveal the enormous lattice being updated underneath."],
 cosmic_web:["Begin almost uniform.","Let gravity amplify tiny fluctuations.","Clusters, filaments and voids emerge.","Pull back: repeat the experiment with different initial universes."],
 galaxy_collision:["Two calm galaxies approach.","Their gravity creates long tidal tails.","The final shape remembers the orbit.","Then reveal an atlas of many possible encounters."],
 reaction_diffusion:["Start from a tiny disturbance.","Only local rules are applied.","Complex global structure appears.","Pull back and reveal the entire parameter space."],
 crystal:["One microscopic seed.","Six primary arms emerge from the seed.","Each branch creates smaller branches of its own.","Zoom in: the same growth rule repeats at several scales."],
 neural_wall:["This network is learning.","Its reconstruction becomes sharper.","Actually… something was left out.","We were training a wall of networks and searching for the best one."],
 fusion_plasma:["A coherent wave circles the magnetic bottle.","Heating drives nonlinear structure through the plasma.","The edge becomes turbulent while the field tries to confine it.","Pull back: we computed a complete reactor operating map."],
 weather_ensemble:["Begin from today’s global observations.","The atmosphere carries vorticity and moisture around the planet.","Tiny uncertainties grow as the forecast races five days ahead.","Pull back: one forecast was only one possible future."],
 molecular_dynamics:["Begin with one loose molecular chain.","Every particle attracts, repels and pulls on its bonded neighbours.","The chain rearranges while temperature and solvent compete.","Pull back: we ran a virtual laboratory of independent trajectories."]
};

async function init(){specs=await (await fetch('/api/specs')).json();renderGallery();applyBackends();loadLibrary();}

// ---- compute backend ------------------------------------------------
function applyBackends(){
  let b=specs.backends||{},sel=$('#backend');
  let gpu=b.gpu||{available:false,detail:'unknown'};
  let opt=[...sel.options].find(o=>o.value==='gpu');
  if(opt){
    opt.disabled=!gpu.available;
    opt.textContent=gpu.available?'GPU':'GPU (unavailable)';
    opt.title=gpu.detail||'';
  }
  let hybrid=[...sel.options].find(o=>o.value==='hybrid'), h=b.hybrid||{available:false,detail:'unknown'};
  if(hybrid){
    hybrid.disabled=!h.available;
    hybrid.textContent=h.available?'GPU + CPU pipeline':'GPU + CPU pipeline (unavailable)';
    hybrid.title=h.detail||'';
  }
  sel.title=`CPU: ${(b.cpu||{}).detail||''}
GPU: ${gpu.detail||''}`;
  $('#backendPill').textContent=gpu.available?`GPU READY · ${gpu.detail}`:'CPU ONLY · NO CUDA DEVICE';
}

// ---- saved run library ----------------------------------------------
let library=[];
async function loadLibrary(){
  try{library=await (await fetch('/api/runs?limit=120')).json();}catch(e){library=[];}
  renderLibrary();renderStageRuns();
}
function runLabel(r){
  let d=new Date((r.created||0)*1000);
  return isNaN(d)?r.id:d.toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});
}
function renderLibrary(){
  let el=$('#libList');if(!el)return;el.innerHTML='';
  if(!library.length){el.innerHTML='<p style="color:#7d93b6;font-size:14px">No saved runs yet. Run a simulation and it will appear here.</p>';return;}
  library.forEach(r=>{
    let name=(specs.demos[r.demo]||{}).name||r.demo;
    let card=document.createElement('article');card.className='runCard';
    let gpu=/cupy|cuda/i.test(r.backend||'');
    card.innerHTML=`<img loading="lazy" src="${r.thumb}" alt="">
      <div class=runMeta><b>${name}</b><span>${runLabel(r)} · ${r.frames} frames</span>
      <div class=runTags><i>${r.profile||'?'}</i><i class="${gpu?'gpu':''}">${r.backend||'?'}</i>${r.zoom?'<i>zoom</i>':''}</div></div>`;
    card.onclick=()=>openRun(r);
    el.appendChild(card);
  });
}
function renderStageRuns(){
  let el=$('#stageRuns');if(!el)return;el.innerHTML='';
  let mine=library.filter(r=>r.demo===current);
  if(!mine.length)return;
  let head=document.createElement('span');head.className='chip';head.style.cursor='default';
  head.innerHTML='<em>Replay:</em>';el.appendChild(head);
  mine.slice(0,12).forEach(r=>{
    let c=document.createElement('button');c.className='chip'+(r.id===runId?' active':'');
    c.innerHTML=`${runLabel(r)} <em>${r.backend||''}</em>`;
    c.onclick=()=>openRun(r);el.appendChild(c);
  });
}

// Replay a finished run straight from disk: no recomputation, and every
// control (playback, reveal, deep zoom) behaves as it does after a live run.
function openRun(r){
  if(r.demo!==current&&openDemo(r.demo)===false)return;
  resetRunState();
  runId=r.id;lastFrame=r.frames-1;playbackTotal=r.frames;
  deepManifest=r.zoom||null;
  $('#frameSeek').max=Math.max(0,r.frames-1);
  $('#status').textContent='REPLAY';$('#status').style.color='#ffc46b';
  $('#metric2').textContent=`elapsed ${(r.elapsed||0).toFixed(1)} s`;
  $('#metric3').textContent=`backend ${r.backend||'—'}`;
  Object.entries(r.params||{}).forEach(([k,v])=>{
    let inp=$('#p_'+k);if(inp){inp.value=v;let out=$('#v_'+k);if(out)out.textContent=v;}
  });
  if(r.profile)$('#profile').value=r.profile;
  setPlaybackControls(true);
  $('#reveal').disabled=!r.has_reveal;
  startPlayback(0);
  renderStageRuns();
}
$('#libRefresh').onclick=loadLibrary;
function renderGallery(){let g=$('#gallery');g.innerHTML='';Object.entries(specs.demos).forEach(([id,d],i)=>{let c=document.createElement('article');c.className='card';c.innerHTML=`<div class=num>${String(i+1).padStart(2,'0')}</div><h3>${d.name}</h3><p>${d.tagline}</p><div class=go>OPEN DEMO →</div>`;c.onclick=()=>openDemo(id);g.appendChild(c);});}
function hideReveal(){let sw=$('.screenWrap');sw.classList.remove('revealing');$('#scaleReveal').classList.remove('show');$('#screen').style.opacity=1;}
function stopPlayback(){if(playbackTimer)clearInterval(playbackTimer);playbackTimer=null;playbackPlaying=false;updatePlaybackButton();}
function updatePlaybackButton(){$('#playPause').textContent=playbackPlaying?'Pause':'Play';}
function setPlaybackControls(enabled){$('#playPause').disabled=!enabled;$('#playbackRate').disabled=!enabled;$('#frameSeek').disabled=!enabled;$('#reveal').disabled=!enabled;$('#deepZoom').disabled=!(enabled&&deepManifest);updatePlaybackButton();}
function frameAvailable(){return Boolean(runId&&lastFrame>=0);}
function updateViewport(){let wrap=$('.screenWrap');wrap.style.setProperty('--view-zoom',zoom);wrap.style.setProperty('--view-pan-x',`${panX}px`);wrap.style.setProperty('--view-pan-y',`${panY}px`);wrap.classList.toggle('isZoomed',zoom>1);let enabled=frameAvailable();$('#zoomIn').disabled=!enabled;$('#zoomOut').disabled=!enabled||zoom<=1;$('#zoomReset').disabled=!enabled||zoom===1;$('#zoomReset').textContent=`${zoom.toFixed(zoom%1?1:0)}×`;}
function resetViewport(){zoom=1;panX=0;panY=0;updateViewport();}
function changeZoom(amount){if(deepActive||!frameAvailable())return;zoom=Math.max(1,Math.min(8,Math.round((zoom+amount)*10)/10));let wrap=$('.screenWrap');let limitX=wrap.clientWidth*(zoom-1)/2;let limitY=wrap.clientHeight*(zoom-1)/2;panX=Math.max(-limitX,Math.min(limitX,panX));panY=Math.max(-limitY,Math.min(limitY,panY));updateViewport();}
function showFrame(frame,total=playbackTotal){if(!runId||total<1)return;frame=Math.max(0,Math.min(total-1,Math.trunc(frame)));playbackFrame=frame;hideReveal();$('#screen').src=`/runs/${runId}/frames/frame_${String(frame).padStart(4,'0')}.jpg?t=${Date.now()}`;let p=(frame+1)/total;$('#bar').style.width=(p*100)+'%';$('#frameSeek').value=frame;$('#metric1').textContent=`frame ${frame+1}/${total}`;$('#story').textContent=stories[current][Math.min(3,Math.floor(p*4))];updateViewport();}
function startPlayback(frame=playbackFrame){if(!runId||playbackTotal<1)return;stopPlayback();showFrame(frame,playbackTotal);playbackPlaying=true;updatePlaybackButton();let rate=Number($('#playbackRate').value);playbackTimer=setInterval(()=>showFrame((playbackFrame+1)%playbackTotal,playbackTotal),Math.max(40,FRAME_INTERVAL_MS/rate));}
function resetRunState(){if(timer)clearInterval(timer);timer=null;stopTargetCamera();stopPlayback();lastFrame=-1;playbackFrame=0;playbackTotal=0;runId=null;deepManifest=null;exitDeep();hideReveal();$('#frameSeek').max=0;$('#frameSeek').value=0;setPlaybackControls(false);resetViewport();}
function stopTargetCamera(){if(neuralTarget.stream){neuralTarget.stream.getTracks().forEach(track=>track.stop());neuralTarget.stream=null;}}
function addNeuralTargetTools(host,defaultKind){
  stopTargetCamera();
  neuralTarget={kind:Number(defaultKind),custom:false};
  const el=document.createElement('div');el.className='targetTools';
  el.innerHTML=`<div class="targetHeading"><span>RGB training target</span><small>Use a colour preset, draw, upload a photo, or take a local webcam picture.</small></div><div class="targetBody"><div class="presetButtons"><button type="button" data-kind="0">Nebula</button><button type="button" data-kind="1">RGB waves</button><button type="button" data-kind="2">Flower</button><button type="button" data-kind="3">Ribbon</button></div><div class="drawing"><canvas id="targetCanvas" width="128" height="128" aria-label="Draw a target image"></canvas><video id="targetCamera" class="hidden" autoplay muted playsinline></video><div><button type="button" id="clearDrawing">Clear</button><label class="targetUpload">Upload<input id="targetUpload" type="file" accept="image/*"></label><button type="button" id="openCamera">Camera</button><button type="button" id="captureCamera" class="hidden">Capture</button><span id="targetMode">Preset target</span></div></div></div>`;
  host.appendChild(el);
  const canvas=el.querySelector('#targetCanvas'),ctx=canvas.getContext('2d');
  const clear=()=>{ctx.fillStyle='#000';ctx.fillRect(0,0,canvas.width,canvas.height);};
  function choose(kind){neuralTarget={kind:Number(kind),custom:false};el.querySelectorAll('[data-kind]').forEach(b=>b.classList.toggle('selected',Number(b.dataset.kind)===neuralTarget.kind));el.querySelector('#targetMode').textContent=`Preset ${neuralTarget.kind+1} selected`;}
  clear();choose(defaultKind);
  el.querySelectorAll('[data-kind]').forEach(button=>button.onclick=()=>choose(button.dataset.kind));
  el.querySelector('#clearDrawing').onclick=()=>{clear();choose(neuralTarget.kind);};
  function useCustom(label){neuralTarget.custom=true;el.querySelectorAll('[data-kind]').forEach(b=>b.classList.remove('selected'));el.querySelector('#targetMode').textContent=label;}
  el.querySelector('#targetUpload').onchange=event=>{const file=event.target.files&&event.target.files[0];if(!file)return;const image=new Image();image.onload=()=>{clear();ctx.drawImage(image,0,0,canvas.width,canvas.height);useCustom('Photo selected — RGB target');URL.revokeObjectURL(image.src);};image.src=URL.createObjectURL(file);};
  const video=el.querySelector('#targetCamera'),capture=el.querySelector('#captureCamera');
  el.querySelector('#openCamera').onclick=async()=>{try{stopTargetCamera();neuralTarget.stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'user',width:{ideal:640},height:{ideal:480}},audio:false});video.srcObject=neuralTarget.stream;video.classList.remove('hidden');capture.classList.remove('hidden');el.querySelector('#targetMode').textContent='Camera live — capture when ready';}catch(error){el.querySelector('#targetMode').textContent='Camera unavailable — upload a photo instead';}};
  capture.onclick=()=>{if(!video.videoWidth)return;clear();ctx.drawImage(video,0,0,canvas.width,canvas.height);stopTargetCamera();video.classList.add('hidden');capture.classList.add('hidden');useCustom('Camera photo selected — RGB target');};
  let drawing=false,last=null;
  const point=event=>{const r=canvas.getBoundingClientRect();return {x:(event.clientX-r.left)*canvas.width/r.width,y:(event.clientY-r.top)*canvas.height/r.height};};
  function paint(event){const p=point(event);ctx.strokeStyle='#fff';ctx.lineCap='round';ctx.lineJoin='round';ctx.lineWidth=9;ctx.beginPath();ctx.moveTo(last.x,last.y);ctx.lineTo(p.x,p.y);ctx.stroke();last=p;useCustom('Your drawing selected — RGB target');}
  canvas.addEventListener('pointerdown',event=>{drawing=true;last=point(event);canvas.setPointerCapture(event.pointerId);paint(event);});
  canvas.addEventListener('pointermove',event=>{if(drawing)paint(event);});
  canvas.addEventListener('pointerup',()=>{drawing=false;last=null;});canvas.addEventListener('pointercancel',()=>{drawing=false;last=null;});
}
function openDemo(id){let spec=specs&&specs.demos?specs.demos[id]:null;
  // A saved run can name a demo this build no longer ships; refuse to open
  // it rather than throwing on a missing spec and leaving a dead stage.
  if(!spec){alert('This build has no demo called "'+id+'".');return false;}
  resetRunState();current=id;$('#gallery').classList.add('hidden');$('#library').classList.add('hidden');$('#stage').classList.remove('hidden');let d=specs.demos[id];$('#stageTitle').textContent=d.name;$('#stageTag').textContent=d.tagline;$('#stageEyebrow').textContent='VISUAL STORY · '+id.replaceAll('_',' ');let s=$('#sliders');s.innerHTML='';Object.entries(d.params).forEach(([k,p])=>{if(id==='neural_wall'&&k==='target')return;let el=document.createElement('div');el.className='control';el.innerHTML=`<div class=row><span>${k.replaceAll('_',' ')}</span><b id="v_${k}">${p.value}</b></div><input id="p_${k}" type=range min="${p.min}" max="${p.max}" step="${p.step}" value="${p.value}">`;s.appendChild(el);el.querySelector('input').oninput=e=>$('#v_'+k).textContent=e.target.value;});if(id==='neural_wall')addNeuralTargetTools(s,d.params.target.value);$('#screen').src='';renderStageRuns();$('#story').textContent=stories[id][0];$('#status').textContent='READY';$('#status').style.color='';$('#bar').style.width='0';$('#metric1').textContent='frame —';$('#metric2').textContent='elapsed —';$('#metric3').textContent='backend —';return true;}
$('#back').onclick=()=>{resetRunState();$('#stage').classList.add('hidden');$('#gallery').classList.remove('hidden');$('#library').classList.remove('hidden');loadLibrary();};
$('#run').onclick=async()=>{if(!current)return;resetRunState();let ps={};Object.keys(specs.demos[current].params).forEach(k=>{if(current==='neural_wall'&&k==='target')ps[k]=neuralTarget.kind;else ps[k]=Number($('#p_'+k).value);});let req={profile:$('#profile').value,frames:Number($('#frames').value),params:ps,backend:$('#backend').value};if(current==='neural_wall'&&neuralTarget.custom)req.target_image=$('#targetCanvas').toDataURL('image/png');let response=await fetch('/api/run/'+current,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(req)});if(!response.ok){let detail='Request rejected';try{let body=await response.json();detail=body.detail||detail;}catch(_){ }$('#status').textContent='FAILED TO START';$('#story').textContent=String(detail);return;}runId=(await response.json()).id;$('#status').textContent='COMPUTING';$('#status').style.color='#67f0d0';timer=setInterval(poll,300);};
async function poll(){if(!runId)return;let m=await (await fetch('/api/run/'+runId+'?t='+Date.now())).json();let total=Number($('#frames').value);if(m.frame!==undefined&&m.frame>=0){lastFrame=m.frame;playbackTotal=total;$('#frameSeek').max=Math.max(0,total-1);showFrame(m.frame,total);$('#metric2').textContent=`elapsed ${(m.elapsed||0).toFixed(1)} s`;$('#metric3').textContent=`backend ${m.backend||'—'}`;}if(m.status==='complete'){clearInterval(timer);timer=null;deepManifest=m.zoom||null;playbackTotal=Number(m.frames)||total;$('#frameSeek').max=Math.max(0,playbackTotal-1);$('#status').textContent='COMPLETE';loadLibrary();$('#metric3').textContent=`backend ${m.backend||'—'}`;setPlaybackControls(true);startPlayback(0);}if(m.status==='failed'){clearInterval(timer);timer=null;$('#status').textContent='FAILED';$('#story').textContent=m.error||'Simulation failed';}}
async function showReveal(){if(!runId)return;stopPlayback();let m=await (await fetch('/api/run/'+runId)).json();if(m.reveal){let sw=$('.screenWrap');sw.classList.add('revealing');$('#scaleReveal').classList.add('show');$('#screen').style.opacity=.15;setTimeout(()=>{$('#screen').src=`/runs/${runId}/${m.reveal}?t=${Date.now()}`;$('#screen').style.opacity=1;$('#story').textContent=stories[current][3];},220);}}
const viewport=$('.screenWrap');
viewport.addEventListener('wheel',event=>{if(deepActive)return;if(!frameAvailable())return;event.preventDefault();let rect=viewport.getBoundingClientRect();viewport.style.setProperty('--view-origin-x',`${(event.clientX-rect.left)/rect.width*100}%`);viewport.style.setProperty('--view-origin-y',`${(event.clientY-rect.top)/rect.height*100}%`);changeZoom(event.deltaY<0?.5:-.5);},{passive:false});
viewport.addEventListener('pointerdown',event=>{if(deepActive)return;if(zoom<=1||!frameAvailable())return;dragState={x:event.clientX,y:event.clientY,panX,panY};viewport.setPointerCapture(event.pointerId);viewport.classList.add('isPanning');});
viewport.addEventListener('pointermove',event=>{if(deepActive||!dragState)return;let limitX=viewport.clientWidth*(zoom-1)/2;let limitY=viewport.clientHeight*(zoom-1)/2;panX=Math.max(-limitX,Math.min(limitX,dragState.panX+event.clientX-dragState.x));panY=Math.max(-limitY,Math.min(limitY,dragState.panY+event.clientY-dragState.y));updateViewport();});
viewport.addEventListener('pointerup',()=>{dragState=null;viewport.classList.remove('isPanning');});
viewport.addEventListener('pointercancel',()=>{dragState=null;viewport.classList.remove('isPanning');});
$('#playPause').onclick=()=>playbackPlaying?stopPlayback():startPlayback(playbackFrame);
$('#playbackRate').onchange=()=>{if(playbackPlaying)startPlayback(playbackFrame);};
$('#frameSeek').oninput=e=>{stopPlayback();showFrame(Number(e.target.value),playbackTotal);};
$('#zoomIn').onclick=()=>changeZoom(.5);
$('#zoomOut').onclick=()=>changeZoom(-.5);
$('#zoomReset').onclick=resetViewport;
function exitDeep(){if(deep&&deep.drag)deep.onUp({});deepActive=false;$('#deepCanvas').classList.add('hidden');$('#deepBadge').classList.add('hidden');$('#screen').classList.remove('hidden');$('#deepZoom').textContent='Deep zoom';}
function enterDeep(){
  if(!runId||!deepManifest)return;
  if(!deep){deep=new DeepZoom($('#deepCanvas'));deep.onstatus=(f,l,max)=>{
    let z=f<1000?f.toFixed(1):(f<1e6?(f/1e3).toFixed(1)+'k':(f<1e9?(f/1e6).toFixed(1)+'M':(f/1e9).toFixed(1)+'B'));
    $('#deepBadge').textContent=`DEEP ZOOM ${z}× · LEVEL ${l}`;};}
  stopPlayback();hideReveal();
  deepActive=true;
  $('#screen').classList.add('hidden');
  $('#deepCanvas').classList.remove('hidden');
  $('#deepBadge').classList.remove('hidden');
  $('#deepZoom').textContent='Exit deep zoom';
  $('#story').textContent='Every level is re-rendered from the model, not enlarged.';
  deep.load(`/runs/${runId}/zoom`,deepManifest,`/api/zoom_tile/${runId}`);
}
$('#deepZoom').onclick=()=>deepActive?exitDeep():enterDeep();
window.addEventListener('resize',()=>{if(deepActive&&deep)deep.resize();});
$('#reveal').onclick=showReveal;
init();
