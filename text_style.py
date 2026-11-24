"""Estimate a region's original solid ink color before any erasing."""
import cv2
import numpy as np
from PIL import Image,ImageDraw

def prose_erase_mask(image_size,area,region,regions,keep,bubble=None):
    """Erase the source paragraph, not the smaller Chinese replacement bounds.

    The user permits a small background repair under verified prose. Exclude
    every neighboring region and all deliberately preserved pixels.
    """
    mask=Image.new('L',image_size)
    ImageDraw.Draw(mask).rectangle((area[0],area[1],area[2]-1,area[3]-1),fill=255)
    lines=region.get('source_lines',[])
    sx,sy,ex,ey=region['box']
    complete=bool(lines and max(c for a,b,c,d in lines)-min(a for a,b,c,d in lines)>=(ex-sx)*.8 and max(d for a,b,c,d in lines)-min(b for a,b,c,d in lines)>=(ey-sy)*.8)
    if bubble is None and complete and not region.get('direction_uncertain'):
        # Follow the detected source lines instead of whitening their empty corners.
        mask=Image.new('L',image_size);draw=ImageDraw.Draw(mask)
        sx,sy,_,_=region['box']
        for a,b,c,d in lines:
            draw.rectangle((sx+a-2,sy+b-2,sx+c+1,sy+d+1),fill=255)
    if bubble:
        safe=Image.new('L',image_size)
        safe.paste(Image.fromarray(bubble['safe'].astype(np.uint8)*255),bubble['box'][:2])
        mask=Image.fromarray(np.minimum(np.asarray(mask),np.asarray(safe)))
    draw=ImageDraw.Draw(mask)
    for other in regions:
        if other is region:continue
        boxes=[other['box']]
        # Cleaning must not depend on which neighbor was rendered first.
        # Source ownership is stable; previous output bounds are not.
        for a,b,c,d in boxes:draw.rectangle((a,b,c-1,d-1),fill=0)
    return Image.fromarray(np.where(np.asarray(keep.crop(area))>0,0,np.asarray(mask.crop(area))).astype(np.uint8))

def original_ink(image,mask,box):
    rgb=np.asarray(image.crop(box).convert('RGB')); ink=np.asarray(mask.crop(box))>64
    if not rgb.size:return (0,0,0)
    pixels=rgb.astype(np.float32)
    border=np.concatenate([pixels[0],pixels[-1],pixels[:,0],pixels[:,-1]])
    outside=pixels[~ink]
    background=np.median(outside if len(outside)>20 else border,axis=0)
    distance=np.max(np.abs(pixels-background),axis=2)
    candidates=ink & (distance>38)
    if candidates.sum()<5:candidates=distance>60
    if candidates.sum()<5:return (0,0,0) if background.mean()>128 else (255,255,255)
    # Use dominant hue for colored glyphs; exclude nearly neutral outline pixels.
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    colored=candidates & (hsv[:,:,1]>85) & (hsv[:,:,2]>55)
    if colored.sum()>=max(8,candidates.sum()*.2):
        hue=hsv[:,:,0];hist=np.bincount((hue[colored]//10).astype(int),minlength=18)
        dominant=int(hist.argmax());delta=np.minimum(abs(hue.astype(int)-dominant*10-5),180-abs(hue.astype(int)-dominant*10-5))
        selected=colored & (delta<=12)
        if selected.sum()>=colored.sum()*.55:
            # Most saturated pixels reduce antialias/background contamination.
            selected &= hsv[:,:,1]>=np.percentile(hsv[:,:,1][selected],45)
            return tuple(int(v) for v in np.median(rgb[selected],axis=0))
    values=rgb[candidates].mean(axis=1)
    dark=int((values<85).sum());light=int((values>185).sum())
    if dark or light:return (0,0,0) if dark>=light else (255,255,255)
    return tuple(int(v) for v in np.median(rgb[candidates],axis=0))

def recover_flat_mask(image,box):
    """Recover regular ink on uniform backgrounds, without painting whole rectangles."""
    rgb=np.asarray(image.crop(box).convert('RGB')).astype(float)
    if rgb.size==0:return None
    border=np.concatenate([rgb[0],rgb[-1],rgb[:,0],rgb[:,-1]])
    bg=np.median(border,axis=0)
    if (np.max(abs(border-bg),axis=1)<20).mean()<.8:return None
    ink=(np.max(abs(rgb-bg),axis=2)>42).astype(np.uint8)
    n,labels,stats,_=cv2.connectedComponentsWithStats(ink,8)
    good=[]
    for i,(x,y,w,h,area) in enumerate(stats[1:],1):
        if area>=3 and area<ink.size*.3 and w<ink.shape[1]*.92 and h<ink.shape[0]*.92:good.append(i)
    recovered=np.isin(labels,good).astype(np.uint8)*255
    if not good or (recovered>0).mean()>.45:return None
    return Image.fromarray(recovered)
