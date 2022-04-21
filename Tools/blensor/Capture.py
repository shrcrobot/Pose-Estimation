import bpy
import blensor
import os
import os.path
import shutil
import random
import math
import time

bpy.data.objects.remove(bpy.data.objects[1])

rootdir = '/home/shrc/singleclass'
# rootdir = '/home/shrc/EightClass'
# rootdir = '/home/shrc/EightClass-Testset'
pts_c = 1024

pi=math.pi

scanner = bpy.data.objects["Camera"];

if not os.path.exists(rootdir+"/tmp"):
    os.makedirs(rootdir+"/tmp")

def capture(filepath):
    flag = bpy.ops.import_scene.obj(filepath=filepath)
    if flag.pop()!="FINISHED":
        print("fail!")
        return
    obj = bpy.data.objects[os.path.splitext(os.path.basename(filepath))[0]]
    dim_old = []
    dim_old.append(obj.dimensions.x)
    dim_old.append(obj.dimensions.y)
    dim_old.append(obj.dimensions.z)
    scalar = 2/max(dim_old)
    obj.scale=obj.scale*scalar
    obj.rotation_euler=(0,0,0)
    scanner.rotation_euler = (pi/2, 0,0)
    scanner.location = (0, -max(dim_old)*scalar*3.5, dim_old[2]*scalar/2)
    obj.rotation_mode = 'XYZ'
    # random generation
    for idx in range(0,20):
        semi = random.randint(0,1)
        if semi == 0:
            t = random.randint(270,450)
            f = random.randint(0,360)
            j = 0
        else:
            t = 0
            f = random.randint(90,270)
            j = random.randint(0,360)
        t_r = t/180*pi
        f_r = f/180*pi
        j_r = j/180*pi
        obj.rotation_euler=(t_r,f_r,j_r)
        # time.sleep(0.1)
        blensor.blendodyne.scan_advanced(scanner,  rotation_speed = 10.0,
                                simulation_fps=24, angle_resolution = 0.05,
                                max_distance = 120, evd_file= rootdir+"/tmp/scan.pcd",
                                noise_mu=0.0, noise_sigma=0, start_angle =-90.0,
                                end_angle=90.0, evd_last_scan=False, 
                                add_blender_mesh = False, 
                                add_noisy_blender_mesh = False)
        if os.path.exists(rootdir+"/tmp/scan00000.pcd"):
                # file name class _ axis _ angle _ (location of scanner in image)
            filename=os.path.splitext(os.path.basename(filepath))[0]+"_"+str(t)+"_"+str(f)+"_"+str(j)+"_"+str(0)+"_"+str(round(-dim_old[2]*scalar/2*100)/100.0)+"_"+str(round(-max(dim_old)*scalar*3.5*100)/100.0)+".pcd"
            shutil.move(rootdir+"/tmp/scan00000.pcd",rootdir+"/generated/"+filename)
            print(filename)
        cmd = "/home/shrc/pcd_prep/build/main "+rootdir+"/generated/"+filename +" "+ str(pts_c)
        os.system(cmd)
    bpy.data.objects.remove(obj)
    return

cnt = 0
cls = {}
for i in os.listdir(rootdir):
    temp_dir = os.path.join(rootdir, i)
    if os.path.isdir(temp_dir) and i!="generated" and i!="tmp":
        cls[i]=cnt
        cnt=cnt+1
        pcd_files=os.listdir(temp_dir)
        for pcd in pcd_files:
            if os.path.splitext(pcd)[1]==".obj":
                print(os.path.join(temp_dir,pcd))
                capture(os.path.join(temp_dir,pcd))
    else:
        print("Not a dir")