# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os, sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm
import torch
import pickle
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points

def split_from_maya(string):
    arr = string.split('[')[1:]
    neck_vertex_indices = []
    for ele in arr:
        ele = ele.split(']')[0]
        if(':' in ele):
            start = int(ele.split(':')[0])
            end = int(ele.split(':')[1]) + 1
            neck_vertex_indices += list(range(start, end))
        else:
            neck_vertex_indices += [int(ele)]
    return neck_vertex_indices

def keep_face_area(vertices, faces, face_area):
    face_area.sort()
    face_area = set(face_area)
    vertices_map = [0] * vertices.shape[0]
    acc = 0
    for i in range(len(vertices)):
        if(i in face_area):
            vertices_map[i] = i - acc
        else:
            acc += 1
    keep_face_indices = []
    for face in faces:
        if(face[0] in face_area and face[1] in face_area and face[2] in face_area):
            keep_face_indices.append([vertices_map[face[0]], vertices_map[face[1]], vertices_map[face[2]]])
    return keep_face_indices

def main(args):
    # if args.rasterizer_type != 'standard':
    #     args.render_orig = False
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    # load test images 
    testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector, sample_step=args.sample_step)

    new_param_dict = {
        'betas': [],
        'global_orient': [],
        'neck_pose': [],
        'leye_pose': [],
        'reye_pose': [],
        'jaw_pose': [],
        'expression': [],
        'orig_cam': []
    }
    sequence_param = {
        'shape': [],
        'exp': [],
        'pose': []
    }

    # run DECA
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca = DECA(config = deca_cfg, device=device)

    vertex_except_eyeball = None
    fl_keep_faces = None
    if(args.removeEyeball):
        string = "'voca_mesh:Mesh.vtx[2279:2281]', 'voca_mesh:Mesh.vtx[2284:2285]', 'voca_mesh:Mesh.vtx[2362:2363]', 'voca_mesh:Mesh.vtx[2366]', 'voca_mesh:Mesh.vtx[2368]', 'voca_mesh:Mesh.vtx[2392:2398]', 'voca_mesh:Mesh.vtx[2409:2410]', 'voca_mesh:Mesh.vtx[2413:2415]', 'voca_mesh:Mesh.vtx[2458:2459]', 'voca_mesh:Mesh.vtx[2463:2464]', 'voca_mesh:Mesh.vtx[2468:2469]', 'voca_mesh:Mesh.vtx[2479]', 'voca_mesh:Mesh.vtx[3616]', 'voca_mesh:Mesh.vtx[3702]', 'voca_mesh:Mesh.vtx[3930]', 'voca_mesh:Mesh.vtx[837:838]', 'voca_mesh:Mesh.vtx[840]', 'voca_mesh:Mesh.vtx[846:847]', 'voca_mesh:Mesh.vtx[1001:1002]', 'voca_mesh:Mesh.vtx[1007]', 'voca_mesh:Mesh.vtx[1010]', 'voca_mesh:Mesh.vtx[1063:1065]', 'voca_mesh:Mesh.vtx[1068]', 'voca_mesh:Mesh.vtx[1075]', 'voca_mesh:Mesh.vtx[1085:1086]', 'voca_mesh:Mesh.vtx[1116:1117]', 'voca_mesh:Mesh.vtx[1127:1129]', 'voca_mesh:Mesh.vtx[1228:1229]', 'voca_mesh:Mesh.vtx[1241:1242]', 'voca_mesh:Mesh.vtx[1284]', 'voca_mesh:Mesh.vtx[1287]', 'voca_mesh:Mesh.vtx[1321]', 'voca_mesh:Mesh.vtx[3824]', 'voca_mesh:Mesh.vtx[3862]', 'voca_mesh:Mesh.vtx[3929]'"
        near_eyeball_vertices = split_from_maya(string)
        flame_mask = pickle.load(open(deca_cfg.model.flame_mask_path, 'rb'), encoding='latin1')
        eye_region_vertex = set(flame_mask['left_eyeball']).union(set(flame_mask['right_eyeball'])).union(set(near_eyeball_vertices))
        vertex_except_eyeball = list(set(range(5023)).difference(eye_region_vertex))
        fl_keep_faces = keep_face_area(np.arange(5023), deca.flame.faces, vertex_except_eyeball)
        fl_keep_faces = np.array(fl_keep_faces)
        vertex_except_eyeball = np.array(vertex_except_eyeball)

    if(args.loadCUSTOM_PKL):
        codedictcustoms = []
        # jaw_exp_list = np.load(r'F:\TTSDataset\exp_jaw.npy')[0, ...]
        jaw_exp_list = np.load(r'/mnt/workplace/FaceFormer/output/exp_jaw.npy')[0, ...]
        cano_len = jaw_exp_list.shape[0]
        for i in range(cano_len):
            a = jaw_exp_list[i:i+1, 50:]
            b = jaw_exp_list[i:i+1, :50]
            # a[0, 0] += 0.02
            # a[0, :] *= 1.5
            codedictcustoms.append([b, a])

    blink_coeff = torch.tensor(np.array([ 0.1237,  0.3986,  0.1954,  0.4408, -0.5949, -1.0346,  0.5870, -1.4986,
         2.5118,  1.6563, -0.7887,  1.2971,  1.3368,  0.2926,  2.5543, -0.3052,
        -2.5140, -0.1137,  1.0665,  2.4840,  0.6413, -2.6480,  2.7470,  0.5609,
        -1.8226,  1.7931, -0.8398,  0.3199, -0.8713,  0.2080, -0.7218, -0.0051,
         0.9509, -0.8668,  1.6583,  0.4251,  0.3105, -0.5553,  0.0682,  0.0976,
         0.4672, -0.3140,  1.2987,  0.5105, -0.4981,  0.6391, -0.8752,  0.4839,
         0.0454,  0.3147]).astype(np.float32)).to(device)
    
    blink_index = random.randint(4*30, 6*30)
    blink_weights = np.zeros(cano_len)
    while(blink_index<cano_len-6):
        blink_weights[blink_index:blink_index+6] = np.array([0.1, 0.4, 0.7, 0.6, 0.4, 0.1])
        blink_index += random.randint(5*30, 8*30)
        if(blink_index<cano_len-6):
            blink_weights[blink_index:blink_index+6] = np.array([0.1, 0.4, 0.7, 0.6, 0.4, 0.1])
            blink_index += random.randint(1*30, 3*30)

    if(args.loadCUSTOM_PKL):
        length = cano_len
        trunk = cano_len//(len(testdata)-1) + 1
        fwd = np.arange(len(testdata)-1)
        bck = np.arange(len(testdata)-1, 0, -1)
        idx = np.array([]).astype(int)
        for t in range(trunk):
            if(t%2==0):
                idx = np.concatenate([idx, fwd])
            else:
                idx = np.concatenate([idx, bck])
    else:
        length = len(testdata)
        idx = np.arange(length)

    for i in tqdm(range(length)):
        name = '{:04d}'.format(i)
        images = testdata[idx[i]]['image'].to(device)[None,...]
        with torch.no_grad():
            codedict = deca.encode(images)
            custom_dict = None
            if(args.loadCUSTOM_PKL):
                custom_dict = {'pose': torch.concat([codedict['pose'][:, :3], torch.tensor(codedictcustoms[i][1][:, :]).to(device)], dim=-1),
                            'exp': torch.tensor(codedictcustoms[i][0][:, :50]).to(device) + blink_weights[i]*blink_coeff}
            opdict, visdict = deca.decode(codedict, custom_dict=custom_dict) #tensor
            if args.render_orig:
                tform = testdata[idx[i]]['tform'][None, ...]
                tform = torch.inverse(tform).transpose(1,2).to(device)
                original_image = testdata[idx[i]]['original_image'][None, ...].to(device)
                _, orig_visdict = deca.decode(codedict, render_orig=True, original_image=original_image, tform=tform, custom_dict=custom_dict, 
                                              remove_eyeball=args.removeEyeball, vertex_except_eyeball=vertex_except_eyeball, fl_keep_faces=fl_keep_faces)    
                orig_visdict['inputs'] = original_image            

        if args.saveParam:
            new_param_dict['betas'].append(codedict['shape'][0, :10].cpu().numpy())
            new_param_dict['global_orient'].append(codedict['pose'][0, :3].reshape(-1).cpu().numpy())
            new_param_dict['neck_pose'].append(np.zeros([3], dtype=np.float32))
            new_param_dict['leye_pose'].append(np.zeros([3], dtype=np.float32))
            new_param_dict['reye_pose'].append(np.zeros([3], dtype=np.float32))
            new_param_dict['jaw_pose'].append(codedict['pose'][0, 3:].reshape(-1).cpu().numpy())
            new_param_dict['expression'].append(codedict['exp'][0].cpu().numpy())
            sx = codedict['new_cam'][:, 0]
            sy = sx
            tx = codedict['new_cam'][:, 1]
            ty = codedict['new_cam'][:, 2]
            new_param_dict['orig_cam'].append(torch.stack([sx, sy, tx, ty]).T[0].cpu().numpy())

        if args.saveSequenceParam:
            sequence_param['shape'].append(codedict['shape'].cpu().numpy())
            sequence_param['exp'].append(codedict['exp'].cpu().numpy())
            sequence_param['pose'].append(codedict['pose'].cpu().numpy())

        if args.saveDepth or args.saveKpt or args.saveObj or args.saveMat or args.saveImages:
            os.makedirs(os.path.join(savefolder, name), exist_ok=True)
        # -- save results
        if args.saveDepth:
            depth_image = deca.render.render_depth(opdict['trans_verts']).repeat(1,3,1,1)
            visdict['depth_images'] = depth_image
            cv2.imwrite(os.path.join(savefolder, name, name + '_depth.jpg'), util.tensor2image(depth_image[0]))
        if args.saveKpt:
            np.savetxt(os.path.join(savefolder, name, name + '_kpt2d.txt'), opdict['landmarks2d'][0].cpu().numpy())
            np.savetxt(os.path.join(savefolder, name, name + '_kpt3d.txt'), opdict['landmarks3d'][0].cpu().numpy())
        if args.saveObj:
            deca.save_obj(os.path.join(savefolder, name, name + '.obj'), opdict)
        if args.saveMat:
            opdict = util.dict_tensor2npy(opdict)
            savemat(os.path.join(savefolder, name, name + '.mat'), opdict)
        if args.saveVis:
            cv2.imwrite(os.path.join(savefolder, name + '_vis.jpg'), deca.visualize(visdict))
            if args.render_orig:
                cv2.imwrite(os.path.join(savefolder, name + '_vis_original_size.jpg'), deca.visualize(orig_visdict))
        if args.saveImages:
            for vis_name in ['rendered_images', 'shape_images']:
                if vis_name not in visdict.keys():
                    continue
                # image = util.tensor2image(visdict[vis_name][0])
                # cv2.imwrite(os.path.join(savefolder, name, name + '_' + vis_name +'.jpg'), util.tensor2image(visdict[vis_name][0]))
                if args.render_orig:
                    # image = util.tensor2image(orig_visdict[vis_name][0])
                    cv2.imwrite(os.path.join(savefolder, name, 'orig_' + name + '_' + vis_name +'.jpg'), util.tensor2image(orig_visdict[vis_name][0]))

        if(args.loadCUSTOM_PKL):
            if(cano_len==i+1):
                break

    if args.saveParam:
        util.save_pkl(os.path.join(savefolder, 'param_dict.pkl'), new_param_dict)

    if(args.saveSequenceParam):
        util.save_pkl(os.path.join(savefolder, 'sequence_param_dict.pkl'), sequence_param)

    print(f'-- please check the results in {savefolder}')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-i', '--inputpath', default=r'/mnt/workplace/pixrefer-tf2.bak/2009_crop', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder', default='TestVideo/results', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu' )
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    parser.add_argument('--sample_step', default=10, type=int,
                        help='sample images from video data for every step' )
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check decalib/detectors.py for details' )
    # rendering option
    parser.add_argument('--rasterizer_type', default='pytorch3d', type=str,
                        help='rasterizer type: pytorch3d or standard' )
    parser.add_argument('--render_orig', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to render results in original image size, currently only works when rasterizer_type=standard')
    # save
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--saveVis', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output' )
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save 2D and 3D keypoints' )
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save depth image' )
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj, detail mesh will end with _detail.obj. \
                            Note that saving objs could be slow' )
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat' )
    parser.add_argument('--saveImages', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images' )
    parser.add_argument('--saveSequenceParam', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images' )
    parser.add_argument('--saveParam', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save parameters as pkl file' )
    parser.add_argument('--removeEyeball', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to remove eyeball when render' )
    parser.add_argument('--loadCUSTOM_PKL', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether load custom exp_jaw' )
    main(parser.parse_args())