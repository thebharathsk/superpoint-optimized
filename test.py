import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from model import SuperPointOptimized

#draw matches
def draw_matches(img1, img2, kp1, kp2, matches):
    fig = plt.figure()
    
    #create output image
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    w_ = img1.shape[1] + img2.shape[1]
    h_ = max(img1.shape[0], img2.shape[0])
    
    img_matches = np.zeros((h_, w_, 3), 'uint8')    
    img_matches[0:h1, 0:w1,:] = img1
    img_matches[0:h2, w1:w1+w2,:] = img2
    
    #shift the second set of keypoints
    kp2[:, 1] += img1.shape[1]
    
    #plot RGB image
    plt.imshow(img_matches[:,:,::-1])

    for i, (m, n, _) in enumerate(matches):
        y1, x1, _ = kp1[m.long()].cpu().numpy()
        y2, x2, _ = kp2[n.long()].cpu().numpy()

        color = np.random.rand(3)

        plt.plot([x1, x2], [y1, y2], color=color)
    
    for i, k in enumerate(kp1):
        y, x, _ = k.cpu().numpy()
        plt.scatter(x,y, color='r', s=1)

    for i, k in enumerate(kp2):
        y, x, _ = k.cpu().numpy()
        plt.scatter(x,y, color='r', s=1)
    
    plt.savefig('out_matches.png')
    plt.close()

#pre-processing function
def preprocess(img):
    #re-size
    h,w = img.shape[0], img.shape[1]
    s = 320
    
    if h > w:
        h_ = s
        w_ = round((w/h)*s)

    else:
        h_ = round((h/w)*s)
        w_ = s
    
    img = cv2.resize(img, (w_, h_))
    
    #ensure sides are multiples of 8
    img = img[0:(h_//8)*8, 0:(w_//8)*8]
    
    img_cv2 = img.copy()
    
    #RGB to BGR
    img = img[:,:,::-1]
    
    #normalize change data-type and reshape
    img = img.astype(np.float32)/255.0
    img = np.transpose(img, (2,0,1))
    img = img[np.newaxis,...]
    
    #convert to tensor
    img_t = torch.from_numpy(img).float()
    
    return img_t, img_cv2

#post processing function
def postprocess(semi, desc, threshold=0.015):
    h, w = semi.shape[2], semi.shape[3]
    
    #apply softmax
    semi = F.softmax(semi, dim=1)
    
    #ignore last channel
    semi = semi[:,:-1,...]
    
    #reshape
    semi = semi.view(-1, 8, 8, h, w)
    semi = semi.permute(0, 3, 1, 4, 2)
    semi = semi.contiguous().view(-1, 8*h, 8*w).squeeze()
    
    #apply threshold
    kps = torch.nonzero(semi > threshold)
    kps_prob = semi[kps[:,0], kps[:,1]]
    
    #combine coordinates and confidence scores
    kps_data = torch.cat([kps, kps_prob.view(-1, 1)], dim=1)
    
    #sample descriptors
    grid = torch.zeros(1, 1, kps.shape[0], 2).to(semi.device)
    grid[:,:,:,0] = kps[:,1]/semi.shape[1]
    grid[:,:,:,1] = kps[:,0]/semi.shape[0]
    
    grid = 2*grid - 1    
    
    desc = torch.grid_sampler(desc, grid, 0, 0, True).squeeze().permute(1,0)
    
    return kps_data, desc

#apply non-maximal suppression
def NMS(kp, desc, H, W):
    #sort keypoints by confidence
    sort_idx = torch.argsort(kp[:,-1], descending=True)
    
    #create tensors store outputs
    kp_out = torch.zeros((0, kp.shape[-1]), device = kp.device)
    desc_out = torch.zeros((0, desc.shape[-1]), device = kp.device)
    
    #create a grid to store whether it's been valid
    valid = torch.ones((H, W), device = kp.device)*-1
    valid[kp[:,0].long(), kp[:,1].long()] = 1
        
    #iterate through keypoints
    for idx in sort_idx:
        #get y and x coordinates        
        y, x = kp[idx, 0].long(), kp[idx, 1].long()
        
        #if idx is a valid index
        if valid[y, x] == 1:
            #set surrounding pixels to invalid
            valid[y-1:y+2, x-1:x+2] = -1
            
            #add index to output tensors
            kp_out = torch.cat([kp_out, kp[idx].view(1, -1)], dim=0)
            desc_out = torch.cat([desc_out, desc[idx].view(1, -1)], dim=0)
            
    return kp_out, desc_out

#matching descriptors
def matching(desc1, desc2, threshold=0.7):
    #create matrix to store matches
    matches = torch.zeros((0, 3), device=desc1.device)

    #compute a matrix of scores
    score = torch.matmul(desc1, desc2.T)
    
    #find best matches for each descriptor
    best1 = torch.argmax(score, dim=1)
    best2 = torch.argmax(score, dim=0)
    
    #perform 2-way matching
    for idx1, idx2 in enumerate(best1):
        if best2[idx2] == idx1 and score[idx1, idx2] > threshold:
            new_match = torch.Tensor([idx1, idx2, score[idx1, idx2]]).to(desc1.device)
            matches = torch.cat([matches, new_match.view(1,-1)], dim=0)

    return matches
        
if __name__ == "__main__":
    #set parameters
    prob_thresh = 0.015
    dist_thresh = 0.5
    
    #set device
    device = 'cuda'
    
    #load model
    '''
    model = SuperPointOptimized().to(device)

    #load checkpoint
    ckpt = torch.load('./superpoint_opt.pth', map_location=device)

    #load weights
    model.load_state_dict(ckpt)
    '''
    
    model = torch.jit.load('./superpoint_opt.pt').to(device)
    
    #load 2 images
    img1 = cv2.imread('data/img1.jpeg')
    img2 = cv2.imread('data/img2.jpeg')
    
    #pre-process the images
    img1_t, img1_cv2 = preprocess(img1)
    img2_t, img2_cv2 = preprocess(img2)
    
    img1_t = img1_t.to(device)
    img2_t = img2_t.to(device)
    
    #collect outputs
    with torch.no_grad():
        semi1, desc1 = model(img1_t)
        semi2, desc2 = model(img2_t)
    
    #post-processing
    kp1, desc1 = postprocess(semi1, desc1, threshold=prob_thresh)
    kp2, desc2 = postprocess(semi2, desc2, threshold=prob_thresh)
    
    #apply NMS
    kp1, desc1 = NMS(kp1, desc1, img1_t.shape[2], img1_t.shape[3])
    kp2, desc2 = NMS(kp2, desc2, img2_t.shape[2], img2_t.shape[3])
    
    #match features
    matches = matching(desc1, desc2, threshold=dist_thresh)
    
    #save outputs
    draw_matches(img1_cv2, img2_cv2, kp1, kp2, matches)