import sys
import os

def get_convertpath(luna16subfolders):
  convert_id_path={}
  dirsubs=["subset0","subset1","subset2","subset3","subset4","subset5","subset6","subset7","subset8","subset9"]
  
  for dirsub in dirsubs:
    dir_path=os.path.join(luna16subfolders,dirsub)
    pids = [f[:-4] for f in os.listdir(dir_path) if f.endswith('.mhd')]
    for pid in pids:
      convert_id_path[pid]=dirsub+"/"+pid
  return convert_id_path
