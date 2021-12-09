import cv2
import numpy as np
import os

def recognizing_details(path, images, methods, ifThreshold):
    templates_paths = os.listdir(path)
    templates_names = []
    templates_images = []
    for tp in templates_paths:
        ti = cv2.imread(f'{path}/{tp}', 0)

        if ifThreshold:
            _, ti = cv2.threshold(ti, 187, 255, cv2.THRESH_BINARY)

        templates_images.append(ti)
        templates_names.append(os.path.splitext(tp)[0])

    h, w = templates_images[0].shape
    detected_shapes = []
    for meth in methods:
        meth1 = eval(meth)
        for img in images:

            if ifThreshold:
                _, img_copy = cv2.threshold(img.copy(), 187, 255, cv2.THRESH_BINARY)
            else:
                img_copy = img.copy()

            best_matches = []
            for ti in templates_images:
                res = cv2.matchTemplate(img, ti, meth1)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                if meth1 in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    location = min_loc
                    value = min_val
                else:
                    location = max_loc
                    value = max_val

                best_matches.append([value, location])

            if meth1 in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                best_match_value = 1000000000000000
            else:
                best_match_value = 0

            for i in range(len(best_matches)):
                if meth1 in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    if best_matches[i][0] < best_match_value:
                        best_match_value = best_matches[i][0]
                        i_max = i
                else:
                    if best_matches[i][0] > best_match_value:
                        best_match_value = best_matches[i][0]
                        i_max = i

            bottom_right = (best_matches[i_max][1][0] + w, best_matches[i_max][1][1] + h)
            cv2.rectangle(img_copy, best_matches[i_max][1], bottom_right, 0, 3)
            detected_shapes.append(templates_names[i_max])
            #print(templates_names[i_max])
            #cv2.putText(img_copy, templates_names[i_max], (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 3, cv2.LINE_AA)
            #cv2.imshow("Detail", img_copy)
            #cv2.waitKey(0)

    return detected_shapes

#Reading pictures, Gaussian Blur and converting to gray
img = cv2.imread("cards/cards1.jpeg")
#img_blur = cv2.GaussianBlur(img, (3,3), 0)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Gray", cv2.pyrDown(cv2.pyrDown(img_gray)))
#cv2.waitKey(0)

#Thresholding
_, img_threshold = cv2.threshold(img_gray,187,255, cv2.THRESH_BINARY) #187
kernel = np.ones((3,3), np.uint8)
img_threshold2 = cv2.erode(img_threshold, kernel)
cv2.imshow("Erozja", cv2.pyrDown(cv2.pyrDown(img_threshold2)))
cv2.imshow("Threshold", cv2.pyrDown(cv2.pyrDown(img_threshold)))
cv2.waitKey(0)

#Finding contours
contours, hierarchy = cv2.findContours(img_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
img_contours1 = cv2.drawContours(img.copy(), contours, -1, (0,255,0), 3)
#print("Contours: ", len(hierarchy[0]))
#cv2.imshow("Contours1", cv2.pyrDown(cv2.pyrDown(img_contours1)))
#cv2.waitKey(0)

#Taking contours with area larger than constant
new_hierarchy = []
new_contours = []
for i in range(len(contours)):
    if cv2.contourArea(contours[i]) >=10000:
        new_contours.append(contours[i])
        new_hierarchy.append(hierarchy[0][i])
#print("Contours: ", len(new_hierarchy))
img_contours2 = cv2.drawContours(img.copy(), new_contours, -1, (0,255,0), 3)
#cv2.imshow("Contours2", cv2.pyrDown(cv2.pyrDown(img_contours2)))
#cv2.waitKey(0)

#Taking only contour without parents
deleted_inner_contours = []
deleted_inner_hierarchy = []
for i in range(len(new_hierarchy)):
    if new_hierarchy[i][3] == -1:
        deleted_inner_contours.append(new_contours[i])
        deleted_inner_hierarchy.append(new_hierarchy[i])
#print("Contours: ", len(deleted_inner_contours))
img_contours3 = cv2.drawContours(img.copy(), deleted_inner_contours, -1, (0,255,0), 3)
#cv2.imshow("Contours3", cv2.pyrDown(cv2.pyrDown(img_contours3)))
#cv2.waitKey(0)

#Rectangles with min area
rects = []
tops_of_cards = []
for contour in deleted_inner_contours:
    rect = cv2.minAreaRect(contour)
    rect = (rect[0], (rect[1][0] + 20, rect[1][1] + 20), rect[2])
    box = np.float32(np.int0(cv2.boxPoints(rect)))
    box_to_draw = np.int0(cv2.boxPoints(rect))
    tops_of_cards.append(box)
    rects.append(rect)
    img_rectangle = cv2.drawContours(img.copy(), [box_to_draw], -1, (0,255,0), 3)
    #cv2.imshow("Rectangle", cv2.pyrDown(cv2.pyrDown(img_rectangle)))
    #cv2.waitKey(0)

############
matrixes = []
cards = []
for i in range(len(rects)):
    #Matrix to perspective transform
    if rects[i][1][0] >= rects[i][1][1]:
        pts_convert = np.float32([[0, 0], [250, 0], [250, 400], [0, 400]])
    else:
        pts_convert = np.float32([[0, 400], [0, 0], [250, 0], [250, 400]])
    matrix = cv2.getPerspectiveTransform(tops_of_cards[i], pts_convert)
    matrixes.append(matrix)

    #Perspective transform
    img_prspective = cv2.warpPerspective(img_gray, matrix, (250, 400))
    img_perspective2 = cv2.warpPerspective(img_blur, matrix, (250, 400))
    #cv2.imshow("Perspective Transform", img_perspective2)
    #cv2.waitKey(0)

    #Thresholding card
    _, output_img = cv2.threshold(img_prspective, 187, 255, cv2.THRESH_BINARY)

    #Searching contours in small images
    contour_in_small_images, hierarchy_in_small_images = cv2.findContours(output_img,  cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    #Searching contour with largest area
    contour_area = 0
    for j in range(len(contour_in_small_images)):
        if cv2.contourArea(contour_in_small_images[j]) > contour_area:
            contour_area = cv2.contourArea(contour_in_small_images[j])
            j_max = j
    bigest_contour = contour_in_small_images[j_max]
    img_biggest = cv2.drawContours(img_perspective2.copy(), bigest_contour, -1, (0,255,0), 3)
    #cv2.imshow("Biggest contour in image", img_biggest)
    #cv2.waitKey(0)

    #Clearing background
    for w in range(img_perspective2.shape[0]):
        for h in range(img_perspective2.shape[1]):
            if cv2.pointPolygonTest(bigest_contour, (h,w), True) >= 0:
                pass
            else:
                img_perspective2[w, h] = [0, 0, 0]
    #cv2.imshow("Cleared background", img_perspective2)
    #cv2.waitKey(0)
    img_perspective2 = cv2.cvtColor(img_perspective2, cv2.COLOR_BGR2GRAY)
    cards.append(img_perspective2)

################################################

#Searching figures
path = 'figures'
methods = [cv2.TM_CCOEFF_NORMED]
detected_figures = recognizing_details(path, cards, methods, True)

#######################

#Searching colors
path = 'colors'
methods = ['cv2.TM_CCOEFF_NORMED']
detected_colors = recognizing_details(path, cards, methods, False)


###########################################################

type_of_card = [[detected_figures[i], " of ", detected_colors[i]]
                for i in range(len(detected_colors))]

font = cv2.FONT_HERSHEY_SIMPLEX
for i in range(len(rects)):
    for j in range(3):
        if type_of_card[i][0] in ["King", "Queen", "Jack"]:
            cv2.putText(img, type_of_card[i][j], (int(rects[i][0][0] - 150), int(rects[i][0][1]) - 70 + j * 70), font,
                        2, (0, 255, 0), 5, cv2.LINE_AA)
        elif type_of_card[i][2] in ["Spades", "Clubs"]:
            cv2.putText(img, type_of_card[i][j], (int(rects[i][0][0] - 150), int(rects[i][0][1]) - 70 + j*70), font,
                        2, (0, 255, 0), 5, cv2.LINE_AA)
        else:
            cv2.putText(img, type_of_card[i][j], (int(rects[i][0][0] - 150), int(rects[i][0][1]) - 70 + j*70), font,
                        2, (0, 0, 0), 5, cv2.LINE_AA)
cv2.imshow("cards", cv2.pyrDown(cv2.pyrDown(img)))

#print(type_of_card)
cv2.waitKey(0)
cv2.destroyAllWindows()