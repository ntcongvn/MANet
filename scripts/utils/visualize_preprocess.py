import numpy as np
import os
import nrrd
import imageio
from copy import deepcopy 
import cv2
from skimage import measure
from PIL import Image


def Visualize_LungData(img_path,save_dir="/content/"):
  lungdata=nrrd.read(img_path)[0]
  slice_datas=[]
  for z in np.arange(lungdata.shape[0]):
    slice_data=lungdata[z]
    slice_data=slice_data[:,:, np.newaxis]
    slice_data_rbg=np.concatenate([slice_data,slice_data,slice_data],axis=2)
    slice_datas.append(slice_data_rbg)
    
  filename=os.path.splitext(os.path.basename(img_path))[0]
  with imageio.get_writer(save_dir+filename+'.gif', mode='I') as writer:
    for slice_data in slice_datas:
        writer.append_data(slice_data)
  print("Save at "+save_dir+filename+'.gif')

def Visualize_LungNoduleMask(mask_path,save_dir="/content/"):
  maskdata=nrrd.read(mask_path)[0]
  maskdata[maskdata>0]=255
  slice_datas=[]
  for z in np.arange(maskdata.shape[0]):
    slice_data=maskdata[z]
    slice_data=slice_data[:,:, np.newaxis]
    slice_data_rbg=np.concatenate([slice_data,slice_data,slice_data],axis=2)
    slice_datas.append(slice_data_rbg)
    
  filename=os.path.splitext(os.path.basename(mask_path))[0]
  with imageio.get_writer(save_dir+filename+'.gif', mode='I') as writer:
    for slice_data in slice_datas:
        writer.append_data(slice_data)
  print("Save at "+save_dir+filename+'.gif')

def Visualize_LungNoduleWithMask(img_path,mask_path,save_dir="/content/"):
  imgdata=nrrd.read(img_path)[0]
  maskdata=nrrd.read(mask_path)[0]
  maskdata[maskdata>0]=255

  slice_datas=[]
  for z in np.arange(imgdata.shape[0]):
    slice_data=imgdata[z]
    slice_data=slice_data[:,:, np.newaxis]
    slice_data_rbg=np.concatenate([slice_data,slice_data,slice_data],axis=2)

    slice_mask=maskdata[z]

    slice_data_rbg[slice_mask>0]=[255,0,0]
    slice_datas.append(slice_data_rbg)
    
  filename=os.path.splitext(os.path.basename(img_path))[0]
  with imageio.get_writer(save_dir+filename[:-5]+'clean_mask.gif', mode='I') as writer:
    for slice_data in slice_datas:
        writer.append_data(slice_data)
  print(save_dir+filename[:-5]+'clean_mask.gif')

def Visualize_ALL(img_path,mask_path,save_dir="/content/"):
  imgdata=nrrd.read(img_path)[0]
  maskdata=nrrd.read(mask_path)[0]
  maskdata[maskdata>0]=255

  slice_datas=[]
  for z in np.arange(imgdata.shape[0]):
    slice_data=imgdata[z]
    slice_data=slice_data[:,:, np.newaxis]
    slice_data_rbg=np.concatenate([slice_data,slice_data,slice_data],axis=2)

    slice_colum1=deepcopy(slice_data_rbg)

    slice_mask=maskdata[z]
    slice_data_rbg[slice_mask>0]=[255,0,0]
    slice_colum3=slice_data_rbg

    slice_mask=slice_mask[:,:, np.newaxis]
    slice_mask_rbg=np.concatenate([slice_mask,slice_mask,slice_mask],axis=2)
    slice_colum2=slice_mask_rbg

    slice_datas.append(np.concatenate([slice_colum1,slice_colum2,slice_colum3],axis=1))
    
  filename=os.path.splitext(os.path.basename(img_path))[0]
  with imageio.get_writer(save_dir+filename[:-5]+'all.gif', mode='I') as writer:
    for slice_data in slice_datas:
        writer.append_data(slice_data)
  print(save_dir+filename[:-5]+'all.gif')


def FindCountor(img):
  im=deepcopy(img)
  im[im>0]=1
  out = np.zeros_like(im)
  #print(np.sum(im))
  if np.sum(im)==0:
    return out
  idx = measure.find_contours(im,fully_connected='high')[0]
  out = np.zeros_like(im)
  for i in np.arange(idx.shape[0]):
    out[int(idx[i,0]),int(idx[i,1])] = 255
  return out

def AddTitle(img,title):
  #print(img.shape)      #h w c
  margin=30
  newimg=np.zeros([img.shape[0]+margin,img.shape[1]+10,img.shape[2]],dtype=np.uint8)
  newimg[:,:,:]=[255,255,255]
  newimg = cv2.putText(newimg, title, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
  newimg[margin-5:-5,5:-5,:]=img

  return newimg
def Visualize_ALL_NP(img_path,grountruth_path,mask_path,save_dir="/content/"):
  alpha=0.4
  imgdata=nrrd.read(img_path)[0]
  maskdata=np.load(mask_path)
  gtdata=nrrd.read(grountruth_path)[0]
  maskdata[maskdata>0]=255
  gtdata[gtdata>0]=255
  i_d,i_h,i_w=imgdata.shape
  maskdata=maskdata[:i_d,:i_h,:i_w]
  slice_datas=[]

  


  print(imgdata.shape)
  print(maskdata.shape)
  for z in np.arange(imgdata.shape[0]):
    #Image
    slice_data=imgdata[z]
    slice_data=slice_data[:,:, np.newaxis]
    slice_data_rbg=np.concatenate([slice_data,slice_data,slice_data],axis=2)
    slice_colum1=deepcopy(slice_data_rbg)
    slice_colum1=AddTitle(slice_colum1,"CT Image")
    #np.zeros([], dtype=np.int32)

    #groundtruth
    slice_gt=gtdata[z]   
    slice_gt_contour=FindCountor(slice_gt)
    copyslice_gt=deepcopy(slice_gt)
    copyslice_gt=copyslice_gt[:,:, np.newaxis]
    copyslice_gt=np.concatenate([copyslice_gt,copyslice_gt,copyslice_gt],axis=2)
    copyslice_gt[slice_gt>0]=[255,0,0]
    prepare_predict1=deepcopy(slice_data_rbg)
    prepare_predict1=cv2.addWeighted(copyslice_gt, alpha, prepare_predict1, 1 - alpha,		0)
    prepare_predict1[slice_gt==0]=[0,0,0]
    prepare_predict2=deepcopy(slice_data_rbg)
    prepare_predict2[slice_gt>0]=[0,0,0]
    prepare_predict=prepare_predict1+prepare_predict2
    prepare_predict[slice_gt_contour>0]=[255,0,0]   
    slice_colum2=prepare_predict
    slice_colum2=AddTitle(slice_colum2,"Ground truth")

    #predict
    slice_mask=maskdata[z]   
    slice_contour=FindCountor(slice_mask)
    copyslice_mask=deepcopy(slice_mask)
    copyslice_mask=copyslice_mask[:,:, np.newaxis]
    copyslice_mask=np.concatenate([copyslice_mask,copyslice_mask,copyslice_mask],axis=2)
    copyslice_mask[slice_mask>0]=[0,0,255]
    prepare_predict1=deepcopy(slice_data_rbg)
    prepare_predict1=cv2.addWeighted(copyslice_mask, alpha, prepare_predict1, 1 - alpha,		0)
    prepare_predict1[slice_mask==0]=[0,0,0]
    prepare_predict2=deepcopy(slice_data_rbg)
    prepare_predict2[slice_mask>0]=[0,0,0]
    prepare_predict=prepare_predict1+prepare_predict2
    prepare_predict[slice_contour>0]=[0,0,255]   
    slice_colum3=prepare_predict
    slice_colum3=AddTitle(slice_colum3,"Model prediction")

    """
    contour=deepcopy(slice_data_rbg)
    copyslice_contour=deepcopy(slice_contour)
    copyslice_contour=copyslice_contour[:,:, np.newaxis]
    copyslice_contour=np.concatenate([copyslice_contour,copyslice_contour,copyslice_contour],axis=2)
    copyslice_contour[slice_contour>0]=[0,0,255]

    copyslice_gt_contour=deepcopy(slice_gt_contour)
    copyslice_gt_contour=copyslice_gt_contour[:,:, np.newaxis]
    copyslice_gt_contour=np.concatenate([copyslice_gt_contour,copyslice_gt_contour,copyslice_gt_contour],axis=2)
    copyslice_gt_contour[slice_gt_contour>0]=[255,0,0]
    
    contour_mask=cv2.addWeighted(copyslice_contour, alpha, copyslice_gt_contour, 1 - alpha,		0)
    contour[slice_contour>0]=[0,0,0]   
    contour[slice_gt_contour>0]=[0,0,0] 
    contour=contour+contour_mask
    slice_colum4=contour
    """

    #Contour
    contour=deepcopy(slice_data_rbg)
    contour[slice_contour>0]=[0,0,255]   
    contour[slice_gt_contour>0]=[255,0,0]   
    slice_colum4=contour
    slice_colum4=AddTitle(slice_colum4,"Contour comparison")

    #slice_mask=slice_mask[:,:, np.newaxis]
    #slice_mask_rbg=np.concatenate([slice_mask,slice_mask,slice_mask],axis=2)
    #slice_colum2=slice_mask_rbg
    row1=np.concatenate([slice_colum1,slice_colum2],axis=1)
    row2=np.concatenate([slice_colum3,slice_colum4],axis=1)
    slice_datas.append(np.concatenate([row1,row2],axis=0))
    
  filename=os.path.splitext(os.path.basename(img_path))[0]
  count=0
  with imageio.get_writer(save_dir+filename[:-5]+'all.gif', mode='I') as writer:
    for slice_data in slice_datas:
        file_im=save_dir+filename[:-5]+'all_'+str(count)+'.gif'
        im = Image.fromarray(slice_data)
        im.save(file_im)
        writer.append_data(slice_data)
        count=count+1
  print(save_dir+filename[:-5]+'all.gif')