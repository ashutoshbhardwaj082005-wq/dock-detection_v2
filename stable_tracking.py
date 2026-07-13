# stable_tracking.py
# NOTE: Skeleton implementation showing the complete structure.
# Replace MODEL_PATH and VIDEO_PATH as needed.

import cv2
import math
from ultralytics import YOLO

MODEL_PATH = r"C:\Users\70005759\Downloads\best (2).pt"
VIDEO_PATH = r"X:\IT\VisionX\ASRS Video_2026-07-06.mp4"
OUTPUT_PATH = "output_tracking.mp4"

model = YOLO(MODEL_PATH)

def iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    aa=(ax2-ax1)*(ay2-ay1); ba=(bx2-bx1)*(by2-by1)
    return inter/(aa+ba-inter+1e-6)

def norm_class(name):
    if "Forklift" in name:
        return "Forklift"
    return name

tracks={}
next_id=1

cap=cv2.VideoCapture(VIDEO_PATH)
w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps=cap.get(cv2.CAP_PROP_FPS)
out=cv2.VideoWriter(OUTPUT_PATH,cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))

while True:
    ok,frame=cap.read()
    if not ok: break
    res=model.track(frame,persist=True,tracker="bytetrack.yaml",conf=0.25,verbose=False)[0]
    if res.boxes is not None:
        for b in res.boxes:
            x1,y1,x2,y2=map(int,b.xyxy[0])
            cls=model.names[int(b.cls)]
            base=norm_class(cls)
            cx=(x1+x2)//2; cy=(y1+y2)//2
            best=None; bestscore=-1
            for sid,obj in tracks.items():
                if obj["base"]!=base: continue
                d=math.hypot(cx-obj["c"][0],cy-obj["c"][1])
                ds=max(0,1-d/200)
                s=0.6*iou((x1,y1,x2,y2),obj["b"])+0.4*ds
                if s>bestscore:
                    bestscore=s; best=sid
            if best is None or bestscore<0.35:
                sid=next_id; next_id+=1
            else:
                sid=best
            tracks[sid]={"base":base,"b":(x1,y1,x2,y2),"c":(cx,cy)}
            label=f"{base}_{sid}"
            if base=="Forklift":
                label+= " (With Pallet)" if "with" in cls.lower() else " (Without Pallet)"
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,label,(x1,max(20,y1-10)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    out.write(frame)
    cv2.imshow("Stable Tracking",frame)
    if cv2.waitKey(1)&0xFF==27: break

cap.release(); out.release(); cv2.destroyAllWindows()
print("Saved:",OUTPUT_PATH)
