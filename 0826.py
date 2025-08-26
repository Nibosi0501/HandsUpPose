import cv2
from ultralytics import YOLO
import numpy as np
from typing import List, Optional
from ultralytics.engine.results import Keypoints as YoloKeypoints
import torch

# ランドマーク間の接続定義
JOINTS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (6, 8), (7, 9),
    (8, 10), (11, 12), (11, 13), (12, 14),
    (13, 15), (14, 16), (5, 11), (6, 12)
]

# キーポイントの名称定義
KEYPOINTS_NAMES = [
    "nose", "eye(L)", "eye(R)", "ear(L)", "ear(R)",
    "shoulder(L)", "shoulder(R)", "elbow(L)", "elbow(R)",
    "wrist(L)", "wrist(R)", "hip(L)", "hip(R)", "knee(L)",
    "knee(R)", "ankle(L)", "ankle(R)",
]

CONF = 0.0  # 信頼度の閾値

def check_keypoints(keypoints: YoloKeypoints, *keypoints_idx: int) -> bool:
    """
    指定されたキーポイントインデックスが有効であり,
    その信頼度が十分で座標が適切であるかを確認します.

    Parameters
    ----------
    keypoints : YoloKeypoints
        検出された人物のキーポイントデータ
    keypoints_idx : int
        チェックするキーポイントのインデックス

    Returns
    -------
    bool
        全てのキーポイントが有効であれば True, 無効であれば False.
    """
    # keypoints_indexが0~16の範囲内かチェック
    if not all(0 <= keypoint_idx <= 16 for keypoint_idx in keypoints_idx):
        print("Error: Invalid keypoints index.")
        return False

    for keypoint_idx in keypoints_idx:
        if keypoints.conf[0][keypoint_idx] is None:
            #print("Error: Invalid confidence.")
            return False
    
    for keypoint_idx in keypoints_idx:
        if keypoints.xy[0][keypoint_idx][0] == 0 and keypoints.xy[0][keypoint_idx][1] == 0:
            #print("Error: Invalid keypoints.")
            return False

    return True

def is_HandsUp(keypoints: YoloKeypoints) -> bool:
    """
    Hands Up, バンザイ（両手が鼻より上）を判定します.

    Parameters
    ----------
    keypoints : YoloKeypoints
        検出された人物のキーポイントデータ

    Returns
    -------
    bool
        HandsUp(バンザイ)を検出した場合は True, それ以外は False.
    """
    
    wrist_left_idx = 9
    wrist_right_idx = 10
    nose_idx = 0

    if not check_keypoints(keypoints, wrist_left_idx, wrist_right_idx, nose_idx):
        return False

    wrists_left = keypoints.xy[0][wrist_left_idx]
    wrists_right = keypoints.xy[0][wrist_right_idx]
    nose = keypoints.xy[0][nose_idx]
    
    if (wrists_left[1] < nose[1] and wrists_right[1] < nose[1]):
        return True

    return False

# 両足首の中心座標を取得する関数
def get_ankle_center(keypoints):
    ankle_left_idx = 15
    ankle_right_idx = 16
    if not check_keypoints(keypoints, ankle_left_idx, ankle_right_idx):
        return None
    
    ankle_left = keypoints.xy[0][ankle_left_idx]
    ankle_right = keypoints.xy[0][ankle_right_idx]

    # 両足首の中点座標を計算
    ankle_center = ((ankle_left[0] + ankle_right[0]) / 2, (ankle_left[1] + ankle_right[1]) / 2)
    return ankle_center


def main() -> None:
    #model = YOLO('yolo11m-pose.pt')
    model = YOLO('yolo11x-pose.mlpackage') # yolo export model=yolo11x-pose.pt format=coreml

    cap = cv2.VideoCapture(1)
    #cap = cv2.VideoCapture("./output3.mov")

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Hands Up(ばんざい)
    handsUp_history: List[Optional[bool]] = []
    handsUp_cnt = 0

    ankle_center_list = []

    
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        #resizeImage = cv2.resize(frame, (720, 1080))

        results = model.track(frame, show=False, persist=True, conf=CONF, verbose=True, save=False, device='cpu')
        #results = model.predict(resizeImage, show=False, conf=CONF, verbose=True, save=False)

        # バウンディングボックス, 信頼度, 追跡ID, キーポイントを取得
        boxes = results[0].boxes.xywh
        confidences = results[0].boxes.conf
        track_ids = results[0].boxes.id

        keypoints_list = results[0].keypoints

        # 検出がない場合, 空のフレームをそのまま描画.
        if boxes is None or confidences is None or track_ids is None:
            annotated_frame = frame.copy()
            #annotated_frame = resizeImage.copy()
        else:
            annotated_frame = frame.copy()
            #annotated_frame = resizeImage.copy()
            # annotated_frameと同じサイズの真っ白な画像を作成
            hoge = np.ones_like(annotated_frame) * 255
            for box, confidence, track_id, keypoints in zip(boxes, confidences, track_ids, keypoints_list):
                # バウンディングボックスの描画
                x, y, w, h = box
                x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
                #cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

                # ID と信頼度をボックスに表示
                #cv2.putText(annotated_frame, f"ID: {track_id} Conf: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)


                track_id = int(track_id) - 1
                # シュート判定用の履歴 ---------------------------------------------------------
                if len(handsUp_history) <= track_id:
                    handsUp_history.extend([None] * (track_id - len(handsUp_history) + 1))
                    #print("handsUp_history: ", len(handsUp_history), handsUp_history)
                # ---------------------------------------------------------------------------

                # Hands Up (バンザイ)判定 ----------------------------------------------------------------
                if handsUp_history[track_id] is None:
                    if is_HandsUp(keypoints):
                        handsUp_history[track_id] = True
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    elif not is_HandsUp(keypoints):
                        handsUp_history[track_id] = False
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                else:
                    if handsUp_history[track_id]:
                        if is_HandsUp(keypoints):
                            pass
                        elif not is_HandsUp(keypoints):
                            pass
                    elif not handsUp_history[track_id]:
                        if is_HandsUp(keypoints):
                            handsUp_cnt += 1
                            print("Hands Up!")
                            #with open("/Volumes/banzai/shot.txt", "w") as file:
                            #        pass  # 何も書き込まない
                        elif not is_HandsUp(keypoints):
                            pass
                
                if is_HandsUp(keypoints):
                    handsUp_history[track_id] = True
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                else:
                    handsUp_history[track_id] = False
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                #print("handsUp_history: ", len(handsUp_history), handsUp_history)
                # ----------------------------------------------------------------------------

                confs = keypoints.conf[0].tolist()  # 推論結果:1に近いほど信頼度が高い
                xys = keypoints.xy[0].tolist()  # 座標

                # ランドマーク間の線を描画
                
                for (start, end) in JOINTS:
                    if confs[start] >= CONF and confs[end] >= CONF:
                        start_point = (int(xys[start][0]), int(xys[start][1]))
                        end_point = (int(xys[end][0]), int(xys[end][1]))
                        if(start_point[0] == 0 and start_point[1] == 0) or (end_point[0] == 0 and end_point[1] == 0):
                            continue
                        cv2.line(annotated_frame, start_point, end_point, (0, 255, 0), 2)

                # キーポイントの描画
                for index, (xy, conf) in enumerate(zip(xys, confs)):
                    if conf < CONF:
                        continue

                    x, y = int(xy[0]), int(xy[1])
                    if x == 0 and y == 0:
                        continue
                    annotated_frame = cv2.rectangle(
                        annotated_frame,
                        (x, y),
                        (x + 3, y + 3),
                        (255, 0, 255),
                        cv2.FILLED,
                        cv2.LINE_AA,
                    )
                    
                
                # 両足首の中点座標を取得する
                ankle_center = get_ankle_center(keypoints)
                if ankle_center is not None:
                    # 中点座標を描画
                    #cv2.circle(annotated_frame, (int(ankle_center[0]), int(ankle_center[1])), 5, (0, 0, 255), -1)
                    cv2.circle(hoge, (int(ankle_center[0]), int(ankle_center[1])), 5, (0, 0, 255), -1)
                    cv2.imwrite("0729.png", hoge)
                    ankle_center_list.append([ankle_center,handsUp_history[track_id]])
                
                # shot_cntの表示
                #cv2.putText(annotated_frame, f"SHOT: {shot_cnt}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        
        # ankle_center_listをfileに保存
        with open("ankle_center_list.txt", "w") as f:
            for center in ankle_center_list:
                # 中点座標をint型に変換して保存
                # f.write(f"{int(center[0])} {int(center[1])}\n")
                f.write(f"{int(center[0][0])} {int(center[0][1])} {int(center[1])}\n")
        
        ankle_center_list = []

        # 履歴の最大長が1000に達した場合, モデルを再初期化し, 履歴とカウンタをリセット.
        if len(handsUp_history) >= 1000:
            print("handsUp is max length: 1000")

            #モデルを再初期化
            #model = YOLO('yolo11n-pose.pt')
            model.predictor = None
            # 履歴をリセット
            handsUp_history = []

            # intの最大値を超えないようにカウンタをリセット
            handsUp_cnt = 0

        # 処理結果をウィンドウに表示
        #cv2.imshow("frame", frame)
        # annotated_frameを1/3にリサイズして表示
        lalala = cv2.resize(annotated_frame, (0, 0), fx=2/3, fy=2/3)
        #cv2.imshow("annotated_frame", annotated_frame)
        cv2.imshow("annotated_frame", lalala)

        # 'q' キーを押すとループを終了
        # ctl + c で終了する方が良いかも.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 使用したリソースを解放
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()