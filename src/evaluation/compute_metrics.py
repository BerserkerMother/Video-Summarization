import h5py
import os

import numpy as np

from .evaluation_metrics import evaluate_summary
from .compute_correlation import evaluate_scores
from .generate_summary import generate_summary

PATH = {
    'ovp': 'eccv16_dataset_ovp_google_pool5.h5',
    'summe': 'eccv16_dataset_summe_google_pool5.h5',
    'tvsum': 'eccv16_dataset_tvsum_google_pool5.h5',
    'youtube': 'eccv16_dataset_youtube_google_pool5.h5'
}


def upsample(scores, n_frames, positions):
    """Upsample scores vector to the original number of frames.
    Input
      scores: (n_steps,)
      n_frames: (1,)
      positions: (n_steps, 1)
    Output
      frame_scores: (n_frames,)
    """
    frame_scores = np.zeros((n_frames), dtype=np.float32)
    if positions.dtype != int:
        positions = positions.astype(np.int32)
    if positions[-1] != n_frames:
        positions = np.concatenate([positions, [n_frames]])
    for i in range(len(positions) - 1):
        pos_left, pos_right = positions[i], positions[i + 1]
        if i == len(scores):
            frame_scores[pos_left:pos_right] = 0
        else:
            frame_scores[pos_left:pos_right] = scores[i]
    return frame_scores


def eval_metrics(data, args):
    eval_method = 'avg'
    dataset_path = os.path.join(args.data, PATH[args.dataset])

    all_scores = []
    keys = list(data.keys())

    for video_name in keys:  # for each video inside that json file ...
        scores = data[video_name]  # read the importance scores from frames
        all_scores.append(scores)

    all_user_summary, all_user_scores, all_shot_bound, \
        all_nframes, all_positions = [], [], [], [], []
    with h5py.File(dataset_path, 'r') as f:
        for key in keys:
            user_summary = np.array(f[key]['user_summary'])
            user_scores = np.array(f[key]["user_scores"])
            sb = np.array(f[key]['change_points'])
            n_frames = np.array(f[key]['n_frames'])
            positions = np.array(f[key]['picks'])

            all_user_summary.append(user_summary)
            all_user_scores.append(user_scores)
            all_shot_bound.append(sb)
            all_nframes.append(n_frames)
            all_positions.append(positions)

    all_summaries = generate_summary(
        all_shot_bound, all_scores, all_nframes, all_positions)

    all_f_scores = []
    all_kendal = []
    all_spear = []
    # compare the resulting summary with the ground truth one, for each video
    for video_index in range(len(all_summaries)):
        summary = all_summaries[video_index]
        scores = all_scores[video_index]
        user_summary = all_user_summary[video_index]
        user_scores = all_user_scores[video_index]
        n_frames = all_nframes[video_index]
        positions = all_positions[video_index]
        scores = upsample(scores, n_frames, positions)

        f_score = evaluate_summary(summary, user_summary, eval_method)
        kendal, spear = evaluate_scores(scores, user_scores)
        all_kendal.append(kendal)
        all_spear.append(spear)
        all_f_scores.append(f_score)
    print(
        f" [f_score: {np.mean(all_f_scores):.4f}, kenadall_tau: {np.mean(all_kendal):.4f}, spearsman_r: {np.mean(all_spear):.4f}]")

    return np.mean(all_f_scores), np.mean(all_kendal), np.mean(all_spear)
