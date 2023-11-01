def bbox_intersection(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of each bounding box
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the IoU by taking the intersection area and dividing it
    # by the sum of the two areas minus the intersection area
    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)
    return iou

def bbox_to_string(trk,float_list):
    if trk == '':
        temp_li = []
        for li in float_list:
            int_list = [int(x) for x in li]
            st = '.'.join(map(str, int_list))
            # st = st+'u'
            temp_li.append(st)
        return 'n'.join(temp_li)
    else:
        traced_boxes = trk.split('n')

        temp_li = []
        for li in float_list:
            int_list = [int(x) for x in li]
            check = True
            for traced_b in traced_boxes:
                traced_split = traced_b.split('.')
                traced_box = [int(x) for x in traced_split]
                
                if bbox_intersection(li,traced_box)>0.3:
                    # print('overlapppppppppppp')
                    check = False
            if check:
                st = '.'.join(map(str, int_list))
                temp_li.append(st)
        return 'n'.join(temp_li)
