from pathlib import Path
import argparse, json, math
import numpy as np
from PIL import Image, ImageDraw
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from leonardo_demos.render import add_title,add_progress,save_frame,font,palette,mosaic

def sphere(r,rho,size=620):
    y,x=np.mgrid[-1:1:complex(size),-1:1:complex(size)]; R=np.sqrt(x*x+y*y); vals=np.interp(R,r,np.log10(np.maximum(rho,1e-5)),left=np.log10(max(rho[0],1e-5)),right=0); return Image.fromarray(palette(vals,'cosmic'))

def main(src,out):
    src,out=Path(src),Path(out); (out/'frames').mkdir(parents=True,exist_ok=True); files=sorted(src.glob('state_*.npz')); meta=json.loads((src/'metadata.json').read_text()) if (src/'metadata.json').exists() else {}
    if not files: raise SystemExit('No state_*.npz files found')
    previews=[]
    for i,p in enumerate(files):
        z=np.load(p); r=z['r']; rho=z['rho_over_rhob']; im=Image.new('RGB',(1280,720),(2,5,14)); im.paste(sphere(r,rho).resize((720,720)),(0,0)); d=ImageDraw.Draw(im,'RGBA'); d.text((760,195),f"research state {i+1}/{len(files)}",font=font(22,True),fill='white'); d.text((760,235),f"max rho/rho_b = {float(np.max(rho)):.3g}",font=font(18),fill=(180,220,255)); im=add_title(im,'Primordial black-hole research solver','Imported numerical state · headless playback adapter',badge='RESEARCH DATA'); add_progress(im,(i+1)/len(files),'INITIAL PERTURBATION','LATE-TIME STATE'); save_frame(im,out/'frames'/f'frame_{i:04d}.jpg'); previews.append(im.copy())
    chosen=previews[-16:] if len(previews)>=16 else previews; cols=max(1,int(math.ceil(math.sqrt(len(chosen))))); rev=mosaic(chosen,cols,title='Saved numerical states from the research solver'); save_frame(rev,out/'reveal.jpg'); (out/'meta.json').write_text(json.dumps({'status':'complete','demo':'pbh','frames':len(files),'frame':len(files)-1,'backend':'research-import','reveal':'reveal.jpg','source_meta':meta},indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('src');ap.add_argument('--out',required=True);a=ap.parse_args();main(a.src,a.out)
