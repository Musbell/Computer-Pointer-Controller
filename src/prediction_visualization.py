import cv2


def draw_face_bbox(frame, coord_0, coord_1, coord_2, coord_3):
    cv2.rectangle(frame, (coord_0, coord_1), (coord_2, coord_3), (255, 255, 255), 2)


def display_hp(frame, hp_out_0, hp_out_1, hp_out_2):
    cv2.putText(
        frame,
        "Pose Angles: yaw:{:.2f} | pitch:{:.2f} | roll:{:.2f}".format(hp_out_0, hp_out_1, hp_out_2),
        (100, 1000),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


def draw_landmarks(frame, eye_coords):
    cv2.rectangle(frame, (eye_coords[0][0] - 10, eye_coords[0][1] - 10),
                  (eye_coords[0][2] + 10, eye_coords[0][3] + 10), (255, 255, 255), 2)
    cv2.rectangle(frame, (eye_coords[1][0] - 10, eye_coords[1][1] - 10),
                  (eye_coords[1][2] + 10, eye_coords[1][3] + 10), (255, 255, 255), 2)


def draw_gaze(face_frame, gaze_vector, left_eye, right_eye, eye_coords):
    x, y, w = int(gaze_vector[0] * 12), int(gaze_vector[1] * 12), 160
    le = cv2.line(left_eye, (x - w, y - w), (x + w, y + w), (255, 255, 255), 2)
    cv2.line(le, (x - w, y + w), (x + w, y - w), (255, 255, 255), 2)
    re = cv2.line(right_eye, (x - w, y - w), (x + w, y + w), (255, 255, 255), 2)
    cv2.line(re, (x - w, y + w), (x + w, y - w), (255, 255, 255), 2)
    face_frame[eye_coords[0][1]:eye_coords[0][3], eye_coords[0][0]:eye_coords[0][2]] = le
    face_frame[eye_coords[1][1]:eye_coords[1][3], eye_coords[1][0]:eye_coords[1][2]] = re
