###########################################################################
# Computer vision - Embedded person tracking demo software by HyperbeeAI. #
# Copyrights © 2023 Hyperbee.AI Inc. All rights reserved. hello@hyperbee.ai #
###########################################################################
import distinctipy
import random
import cv2

N = 36
# generate N visually distinct colours to mark different tracks
colors = distinctipy.get_colors(N, pastel_factor=0.7,rng=random.seed(6))

def draw_BBs(boxes_to_draw, confs_to_draw, frame, IDs = None, colors = colors, show_track_confidences = True):
    
    image_height, image_width, _ = frame.shape
   
    for i, conf in enumerate(confs_to_draw):
        
        if IDs is not None:
            # Unique color for each ID
            BB_ID = IDs[i]
            color = list(colors[BB_ID%36])
        else:
            color = colors[1]

        color = [int(x*255) for x in color]
        color = tuple(color)

        BB = boxes_to_draw[i,:]
        xmin,ymin,xmax,ymax = BB
        xmin = max(xmin.item(),0)
        ymin = max(ymin.item(),0)
        xmax = min(xmax.item(),image_width)
        ymax = min(ymax.item(),image_height)


        topleft = (int(xmin), int(ymin))
        bottomright = (int(xmax), int(ymax))
        
        thickness = 2
        cv2.rectangle(frame, topleft, bottomright, color, thickness)


        # Write text on the frame
        label_conf = str(round(conf.item(),3))
        label_conf = 'C:' + label_conf
        
        x1,y1 = topleft
        x2,y2 = bottomright       
        
        if IDs is not None:
            # If track BB is to be drawn
            label_ID = 'ID:' + str(BB_ID)
            if show_track_confidences:
                frame = cv2.rectangle(frame, (int(x1-thickness/2), y1 - 25),
                                      (int(x2+thickness/2), y1-1), color, -1)
                frame = cv2.putText(frame, label_conf, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                frame = cv2.putText(frame, label_ID, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            else:
                frame = cv2.rectangle(frame, (int(x1-thickness/2), y1 - 15),
                                      (int(x2+thickness/2), y1-1), color, -1)
                frame = cv2.putText(frame, label_ID, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        else:
            frame = cv2.rectangle(frame, (int(x1-thickness/2), y1 - 15),
                                  (int(x2+thickness/2), y1-1), color, -1)
            frame = cv2.putText(frame, label_conf, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
    return frame


def project_BBs_to_original_frame(BBs, padding_left, padding_top, scale):
    
    BBs_new = BBs.clone()
    
    BBs_new[:,0] = BBs_new[:,0] - padding_left
    BBs_new[:,1] = BBs_new[:,1] - padding_top
    BBs_new[:,2] = BBs_new[:,2] - padding_left
    BBs_new[:,3] = BBs_new[:,3] - padding_top
    
    return(BBs_new * scale)