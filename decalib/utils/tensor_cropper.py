'''
crop
for torch tensor
Given image, bbox(center, bboxsize)
return: cropped image, tform(used for transform the keypoint accordingly)
only support crop to squared images
'''
import torch
from kornia.geometry.transform.imgwarp import (
    warp_perspective, get_perspective_transform, warp_affine
)

def points2bbox(points, points_scale=None):
    if points_scale:
        assert points_scale[0]==points_scale[1]
        points = points.clone()
        points[:,:,:2] = (points[:,:,:2]*0.5 + 0.5)*points_scale[0]
    min_coords, _ = torch.min(points, dim=1)
    xmin, ymin = min_coords[:, 0], min_coords[:, 1]
    max_coords, _ = torch.max(points, dim=1)
    xmax, ymax = max_coords[:, 0], max_coords[:, 1]
    center = torch.stack([xmax + xmin, ymax + ymin], dim=-1) * 0.5

    width = (xmax - xmin)
    height = (ymax - ymin)
    # Convert the bounding box to a square box
    size = torch.max(width, height).unsqueeze(-1)
    return center, size

def augment_bbox(center, bbox_size, scale=[1.0, 1.0], trans_scale=0.):
    batch_size = center.shape[0]
    trans_scale = (torch.rand([batch_size, 2], device=center.device)*2. -1.) * trans_scale
    center = center + trans_scale*bbox_size # 0.5
    scale = torch.rand([batch_size,1], device=center.device) * (scale[1] - scale[0]) + scale[0]
    size = bbox_size*scale
    return center, size

def crop_tensor(image, center, bbox_size, crop_size, interpolation = 'bilinear', align_corners=False):
    ''' for batch image
    Args:
        image (torch.Tensor): the reference tensor of shape BXHxWXC.
        center: [bz, 2]
        bboxsize: [bz, 1]
        crop_size;
        interpolation (str): Interpolation flag. Default: 'bilinear'.
        align_corners (bool): mode for grid_generation. Default: False. See
          https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for details
    Returns:
        cropped_image
        tform
    '''
    dtype = image.dtype
    device = image.device
    batch_size = image.shape[0]
    # points: top-left, top-right, bottom-right, bottom-left    
    src_pts = torch.zeros([4,2], dtype=dtype, device=device).unsqueeze(0).expand(batch_size, -1, -1).contiguous()

    src_pts[:, 0, :] = center - bbox_size*0.5  # / (self.crop_size - 1)
    src_pts[:, 1, 0] = center[:, 0] + bbox_size[:, 0] * 0.5
    src_pts[:, 1, 1] = center[:, 1] - bbox_size[:, 0] * 0.5
    src_pts[:, 2, :] = center + bbox_size * 0.5
    src_pts[:, 3, 0] = center[:, 0] - bbox_size[:, 0] * 0.5
    src_pts[:, 3, 1] = center[:, 1] + bbox_size[:, 0] * 0.5

    DST_PTS = torch.tensor([[
        [0, 0],
        [crop_size - 1, 0],
        [crop_size - 1, crop_size - 1],
        [0, crop_size - 1],
    ]], dtype=dtype, device=device).expand(batch_size, -1, -1)
    # estimate transformation between points
    dst_trans_src = get_perspective_transform(src_pts, DST_PTS)
    # simulate broadcasting
    # dst_trans_src = dst_trans_src.expand(batch_size, -1, -1)

    # warp images 
    cropped_image = warp_affine(
        image, dst_trans_src[:, :2, :], (crop_size, crop_size),
        flags=interpolation, align_corners=align_corners)

    tform = torch.transpose(dst_trans_src, 2, 1)
    # tform = torch.inverse(dst_trans_src)
    return cropped_image, tform

class Cropper(object):
    def __init__(self, crop_size, scale=[1,1], trans_scale = 0.):
        self.crop_size = crop_size
        self.scale = scale
        self.trans_scale = trans_scale

    def crop(self, image, points, points_scale=None):
        # points to bbox
        center, bbox_size = points2bbox(points.clone(), points_scale)
        # argument bbox. TODO: add rotation?
        center, bbox_size = augment_bbox(center, bbox_size, scale=self.scale, trans_scale=self.trans_scale)
        # crop
        cropped_image, tform = crop_tensor(image, center, bbox_size, self.crop_size)
        return cropped_image, tform
    
    def transform_points(self, points, tform, points_scale=None, normalize = True):
        points_2d = points[:,:,:2]
        
        #'input points must use original range'
        if points_scale:
            assert points_scale[0]==points_scale[1]
            points_2d = (points_2d*0.5 + 0.5)*points_scale[0]

        batch_size, n_points, _ = points.shape
        trans_points_2d = torch.bmm(
                        torch.cat([points_2d, torch.ones([batch_size, n_points, 1], device=points.device, dtype=points.dtype)], dim=-1), 
                        tform
                        ) 
        trans_points = torch.cat([trans_points_2d[:,:,:2], points[:,:,2:]], dim=-1)
        if normalize:
            trans_points[:,:,:2] = trans_points[:,:,:2]/self.crop_size*2 - 1
        return trans_points

def transform_points(points, tform, points_scale=None, out_scale=None):
    points_2d = points[:,:,:2]
        
    #'input points must use original range'
    if points_scale:
        assert points_scale[0]==points_scale[1]
        points_2d = (points_2d*0.5 + 0.5)*points_scale[0]
    # import ipdb; ipdb.set_trace()

    batch_size, n_points, _ = points.shape
    trans_points_2d = torch.bmm(
                    torch.cat([points_2d, torch.ones([batch_size, n_points, 1], device=points.device, dtype=points.dtype)], dim=-1), 
                    tform
                    ) 
    if out_scale: # h,w of output image size
        trans_points_2d[:,:,0] = trans_points_2d[:,:,0]/out_scale[1]*2 - 1
        trans_points_2d[:,:,1] = trans_points_2d[:,:,1]/out_scale[0]*2 - 1
    trans_points = torch.cat([trans_points_2d[:,:,:2], points[:,:,2:]], dim=-1)
    return trans_points


def get_transform_cam(cam_input, tform, points_scale=None, out_scale=None):

    D = torch.tensor([[[0.5,  0.0,   0.0,   0.0],
                      [0.0,  0.5,   0.0,   0.0],
                      [0.0,  0.0,   0.5,   0.0],
                      [0.5,  0.5,   0.5,   1.0]]], 
                      device=tform.device, dtype=tform.dtype)

    C = torch.tensor([[[points_scale[0], 0.0,                0.0,               0.0],
                      [0.0,             points_scale[1],    0.0,                0.0],
                      [0.0,             0.0,             points_scale[1],       0.0],
                      [0.0,             0.0,                0.0,                1.0]]], 
                      device=tform.device, dtype=tform.dtype)

    tform2 = torch.zeros([1, 4, 4], device=tform.device, dtype=tform.dtype)
    tform2[0, 0, 0] = tform[0, 0, 0]
    tform2[0, 1, 1] = tform[0, 0, 0]
    tform2[0, 2, 2] = tform[0, 0, 0]
    tform2[0, 3, 3] = 1.0
    tform2[0, 3, 0] = tform[0, 2, 0]
    tform2[0, 3, 1] = tform[0, 2, 1]

    A = torch.tensor([[[2.0/out_scale[0], 0.0,               0.0,               0.0],
                      [0.0,               2.0/out_scale[1],  0.0,               0.0],
                      [0.0,               0.0,               2.0/out_scale[1],  0.0],
                      [-1.0,              -1.0,             -1.0,               1.0]]], 
                      device=tform.device, dtype=tform.dtype)

    cam_input_matrix = torch.tensor([[[cam_input[0, 0],                         0.0,                          0.0,             0.0],
                                    [0.0,                                   cam_input[0, 0],                  0.0,             0.0],
                                    [0.0,                                       0.0,                        cam_input[0, 0],   0.0],
                                    [cam_input[0, 1]*cam_input[0, 0],  -cam_input[0, 2]*cam_input[0, 0],      0.0,             1.0]]], 
                                    device=tform.device, dtype=tform.dtype)

    cam_matrix = cam_input_matrix @ D @ C @ tform2 @ A
    cam = torch.zeros([1, 3], device=tform.device, dtype=tform.dtype)
    cam[0, 0] = cam_matrix[0, 0, 0]
    cam[0, 1] = cam_matrix[0, 3, 0]/cam[0, 0]
    cam[0, 2] = cam_matrix[0, 3, 1]/cam[0, 0]
    return cam

# def get_transform_cam(cam_input, tform, points_scale=None, out_scale=None):
#     s4 = cam_input[0, 0]; s3 = 2.0/out_scale[0]; s2 = tform[0, 0, 0]; s1 = points_scale[0]; s0 = 0.5
#     x4 = cam_input[0, 1]; x3 = -1.0;             x2 = tform[0, 2, 0]; x1 = 0.0;             x0 = 0.5
#     y4 = cam_input[0, 2]; y3 = -1.0;             y2 = tform[0, 2, 1]; y1 = 0.0;             y0 = 0.5

#     cam = torch.zeros([1, 3], device=tform.device, dtype=tform.dtype)
#     cam[0, 0] = s0 * s1 * s2 * s3 * s4
#     cam[0, 1] = (s0 * (s1 * (s2 * (s3 * x4 + x3) + x2) + x1) + x0) / cam[0, 0]
#     cam[0, 2] = (s0 * (s1 * (s2 * (s3 * y4 + y3) + y2) + y1) + y0) / cam[0, 0]
#     return cam


def transform_points_3d(points, tform, points_scale=None, out_scale=None):

    #'input points must use original range'
    if points_scale:
        assert points_scale[0]==points_scale[1]
        points = (points*0.5 + 0.5)*points_scale[0]
    # import ipdb; ipdb.set_trace()

    batch_size, n_points, _ = points.shape

    tform_3d = torch.zeros([batch_size, 4, 4], dtype=tform.dtype, device=tform.device)
    tform_3d[:, :2, :2] = tform[:, :2, :2]
    tform_3d[:, 2, :] = torch.Tensor([0, 0, tform[0, 0, 0], 0])
    tform_3d[:, 3, :] = torch.Tensor([tform[0, 2, 0], tform[0, 2, 1], 0, 1])

    trans_points_3d = torch.bmm(
                        torch.cat([points, torch.ones([batch_size, n_points, 1], device=points.device, dtype=points.dtype)], dim=-1), 
                        tform_3d
                        )
    trans_points_3d2 = trans_points_3d.clone()
    trans_points_3d = trans_points_3d[..., :-1]
    
    # Rx = torch.zeros([batch_size, 4, 4], dtype=tform.dtype, device=tform.device)
    # Rx[:, 0, 0] = 1
    # Rx[:, 1, 1] = -1
    # Rx[:, 2, 2] = -1
    # Rx[:, 3, 3] = 1
    # trans_points_3d2 = torch.einsum('bij,bni->bni', Rx, trans_points_3d2)
    # trans_points_3d2 = torch.einsum('bni,bij->bni', trans_points_3d2, Rx)

    trans_points_3d2 = trans_points_3d2[..., :-1]

    if out_scale: # h,w of output image size
        trans_points_3d[:,:,0] = trans_points_3d[:,:,0]/out_scale[1]*2 - 1
        trans_points_3d[:,:,1] = trans_points_3d[:,:,1]/out_scale[1]*2 - 1
        trans_points_3d[:,:,2] = trans_points_3d[:,:,2]/out_scale[1]*2 - 1

        # trans_points_3d[:,:,0] = trans_points_3d[:,:,0]/out_scale[1]*2 - 1
        # trans_points_3d[:,:,1] = trans_points_3d[:,:,1]/out_scale[0]*2 - 1
        # trans_points_3d[:,:,2] = trans_points_3d[:,:,2]/(out_scale[1]+out_scale[0])*4 - 1

    return trans_points_3d, trans_points_3d2