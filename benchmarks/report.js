const fmt=n=>n==null?'—':`${n.toFixed(n<10?3:2)} s`;
const stage=(r,n)=>r.timings?.[n]?.seconds||0;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const colours={initialization:'#b7c4d2',simulation:'#ff9b58',ensemble:'#ee6ca9',deep_zoom:'#d594ff',render:'#54e4ff',visualization:'#9a7cff',jpeg_encode:'#59e7ae',frame_copy:'#59e7ae',frame_write:'#e7d659',other:'#405466'};
Promise.all([
  fetch('latest.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('No benchmark data yet');return r.json()}),
  fetch('galaxy-solvers/latest.json',{cache:'no-store'}).then(r=>r.ok?r.json():null).catch(()=>null),
  fetch('galaxy3d/latest.json',{cache:'no-store'}).then(r=>r.ok?r.json():null).catch(()=>null),
  fetch('plasma-guardian/latest.json',{cache:'no-store'}).then(r=>r.ok?r.json():null).catch(()=>null)
]).then(([data,solverData,galaxy3dData,guardianData])=>{
  for(const extra of [solverData,galaxy3dData,guardianData]){
    if(!extra)continue;
    const demos=new Set(extra.results.map(row=>row.demo));
    data.results=[...data.results.filter(row=>!demos.has(row.demo)),...extra.results];
    data.notes=[...new Set([...(data.notes||[]),...(extra.notes||[])])];
  }
  if(solverData)data.created_iso=`${data.created_iso}; galaxy solvers ${solverData.created_iso}`;
  if(galaxy3dData)data.created_iso=`${data.created_iso}; 3D galaxy ${galaxy3dData.created_iso}`;
  if(guardianData)data.created_iso=`${data.created_iso}; plasma guardian ${guardianData.created_iso}`;
  const done=data.results.filter(r=>r.status==='complete'), unsupported=data.results.filter(r=>r.status==='unsupported');
  const gpu=data.environment.backends.gpu;
  document.querySelector('#summary').textContent=`${done.length} measured runs from ${data.environment.hostname}, using the fixed “${data.profile}” workload (${data.frames} frames). ${gpu.detail}. Generated ${data.created_iso}.`;
  const fastest=done.reduce((a,b)=>!a||b.wall_seconds<a.wall_seconds?b:a,null);
  const encode=done.reduce((s,r)=>s+stage(r,'jpeg_encode'),0), wall=done.reduce((s,r)=>s+r.wall_seconds,0);
  document.querySelector('#cards').innerHTML=[['Measured runs',done.length],['Demos',new Set(data.results.map(r=>r.demo)).size],['Fastest run',fastest?fmt(fastest.wall_seconds):'—'],['JPEG share',wall?`${(100*encode/wall).toFixed(1)}%`:'—']].map(([a,b])=>`<article class="card"><b>${b}</b><span>${a}</span></article>`).join('');
  const cpu=new Map(done.filter(r=>r.backend_requested==='cpu').map(r=>[`${r.demo}|${r.method}`,r]));
  document.querySelector('#rows').innerHTML=data.results.map(r=>{
    const base=cpu.get(`${r.demo}|${r.method}`), speed=base&&r.wall_seconds?base.wall_seconds/r.wall_seconds:null;
    const cls=r.backend_requested==='cpu'?'cpu':r.backend_requested==='gpu'?'gpu':'hybrid';
    return `<tr><td><strong>${esc(r.demo.replaceAll('_',' '))}</strong></td><td>${esc(r.method)}</td><td><span class="pill ${cls}">${esc(r.backend_requested)}</span></td><td class="number">${fmt(r.wall_seconds)}</td><td class="number">${fmt(stage(r,'simulation'))}</td><td class="number">${fmt(stage(r,'render')+stage(r,'visualization'))}</td><td class="number">${fmt(stage(r,'jpeg_encode'))}</td><td class="number">${fmt(stage(r,'frame_write'))}</td><td class="number ${speed>1?'good':''}">${speed?`${speed.toFixed(2)}×`:'—'}</td><td class="${r.status==='complete'?'good':r.status==='failed'?'bad':'muted'}" title="${esc(r.error)}">${esc(r.status)}</td></tr>`}).join('');
  document.querySelector('#breakdowns').innerHTML=done.map(r=>{
    const known=['initialization','simulation','ensemble','deep_zoom','render','visualization','jpeg_encode','frame_copy','frame_write'].map(n=>[n,stage(r,n)]), sum=known.reduce((s,x)=>s+x[1],0), other=Math.max(0,r.wall_seconds-sum), vals=[...known,['other',other]], total=vals.reduce((s,x)=>s+x[1],0)||1;
    return `<article class="breakdown"><h3>${esc(r.demo.replaceAll('_',' '))} · ${esc(r.method)} · ${esc(r.backend_requested)} · ${fmt(r.wall_seconds)}</h3><div class="bar">${vals.map(([n,v])=>`<span class="${n}" style="width:${100*v/total}%;background:${colours[n]}" title="${n}: ${fmt(v)}"></span>`).join('')}</div><div class="bar-labels">${vals.filter(x=>x[1]>.001).map(([n,v])=>`<span>${n.replace('_',' ')} ${fmt(v)}</span>`).join('')}</div></article>`}).join('');
  document.querySelector('#notes').innerHTML=[...data.notes,`${unsupported.length} matrix entries are architecturally unsupported and were not run.`].map(n=>`<li>${esc(n)}</li>`).join('');
}).catch(error=>{document.querySelector('#summary').textContent=`${error.message}. Run tools/benchmark_demos.py first.`});
