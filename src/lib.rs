// lib.rs

use std::slice;
use libc::{c_float, c_uchar};

extern crate ndarray;
extern crate statrs;


fn find_optimal_threshold_combination(
    preds_1: &[(f32, u8)],
    preds_2: &[(f32, u8)],
    num_thresholds: usize,
    lower_1:f32,
    lower_2:f32,
    upper_1:f32,
    upper_2:f32,
) -> (f32, f32) {
    let mut best_threshold1 = 0.0;
    let mut best_threshold2 = 0.0;
    let mut best_f1_score = -1.0;

    let thresholds1 = linspace(lower_1, upper_1, num_thresholds);
    let thresholds2 = linspace(lower_2, upper_2, num_thresholds);

    for &threshold1 in &thresholds1 {
        for &threshold2 in &thresholds2 {
            let mut y_pred = vec![0u8; preds_1.len()];

            for i in 0..preds_1.len() {
                y_pred[i] = if preds_1[i].0 >= threshold1 {
                    if preds_2[i].0 >= threshold2 { 1 } else { 0 }
                } else {
                    0
                }
            }

            let f1_score = f1_score(&y_pred, &preds_1.iter().map(|x| x.1).collect::<Vec<_>>());

            if f1_score > best_f1_score {
                best_f1_score = f1_score;
                best_threshold1 = threshold1;
                best_threshold2 = threshold2;
            }
        }
    }

    (best_threshold1, best_threshold2)
}

fn f1_score(y_pred: &[u8], y_true: &[u8]) -> f32 {
    let mut true_positive = 0;
    let mut false_positive = 0;
    let mut false_negative = 0;

    for (&yp, &yt) in y_pred.iter().zip(y_true.iter()) {
        if yp == 1 && yt == 1 {
            true_positive += 1;
        } else if yp == 1 && yt == 0 {
            false_positive += 1;
        } else if yp == 0 && yt == 1 {
            false_negative += 1;
        }
    }

    let precision = true_positive as f32 / (true_positive + false_positive) as f32;
    let recall = true_positive as f32 / (true_positive + false_negative) as f32;

    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

fn linspace(start: f32, end: f32, num_points: usize) -> Vec<f32> {
    let step = (end - start) / (num_points - 1) as f32;
    (0..num_points).map(|i| start + step * i as f32).collect()
}

#[no_mangle]
pub extern "C" fn find_optimal_threshold_combination_ffi(
    preds_1_probs_ptr: *const c_float,
    preds_1_labels_ptr: *const c_uchar,
    preds_1_len: usize,
    preds_2_probs_ptr: *const c_float,
    preds_2_labels_ptr: *const c_uchar,
    preds_2_len: usize,
    num_thresholds: usize,
    result_ptr: *mut c_float,
    lower_bound_1: c_float,
    lower_bound_2: c_float,
    upper_bound_1: c_float,
    upper_bound_2: c_float,
) {
    let preds_1_probs_raw = unsafe { slice::from_raw_parts(preds_1_probs_ptr, preds_1_len) };
    let preds_1_labels_raw = unsafe { slice::from_raw_parts(preds_1_labels_ptr, preds_1_len) };
    let preds_1: Vec<(f32, u8)> = preds_1_probs_raw
        .iter()
        .zip(preds_1_labels_raw.iter())
        .map(|(p, l)| (*p, *l))
        .collect();

    let preds_2_probs_raw = unsafe { slice::from_raw_parts(preds_2_probs_ptr, preds_2_len) };
    let preds_2_labels_raw = unsafe { slice::from_raw_parts(preds_2_labels_ptr, preds_2_len) };
    let preds_2: Vec<(f32, u8)> = preds_2_probs_raw
        .iter()
        .zip(preds_2_labels_raw.iter())
        .map(|(p, l)| (*p, *l))
        .collect();

    let lower_1 = lower_bound_1 as f32;
    let lower_2 = lower_bound_2 as f32;
    let upper_1 = upper_bound_1 as f32;
    let upper_2 = upper_bound_2 as f32;
    
    let (best_threshold1, best_threshold2) =
        find_optimal_threshold_combination(&preds_1, &preds_2, num_thresholds, lower_1, lower_2, upper_1, upper_2);

    unsafe {
        *result_ptr = best_threshold1 as c_float;
        *result_ptr.add(1) = best_threshold2 as c_float;
    }
}



