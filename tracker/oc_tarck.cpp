#include <algorithm>
#include <numeric>
#include "oc_tracker.h"
#include "hungarian_algorithm.h"

using namespace OcSort;
static constexpr int32_t kDeltaFrameNum = 3;
static constexpr float kAngleCostWeight = 0.2;
static constexpr float kIouCostWeight = 1 - kAngleCostWeight;

std::pair<float,float> Track::TwoBoxes2Speed(const Bbox2D& box1, const Bbox2D& box2) const {
     // Use cx and cy to calculate unit tranalation speed between boxes
     int32_t cx1 = box1.x + box1.w / 2;
     int32_t cy1 = box1.y + box1.h / 2;
     int32_t cx2 = box2.x + box2.w / 2;
     int32_t cy2 = box2.y + box2.h / 2;
     float speed_x = cx2 - cx1;
     float speed_y = cy2 - cy1;
     float norm = sqrt(speed_x*speed_x + speed_y*speed_y) + 1e-6;
     return std::pair<float, float>(speed_x / norm, speed_y / norm);
}

void XysaTrack::Initialize(const Bbox2D& bbox, const int32_t& track_id, const int32_t& frame_id) {
     // Use cx, cy, aspect ratio and box area as the first four state variables; the other three states are their corresponding volecity except area
     EigenVector<kDimx> X;
     X << bbox.x + bbox.w/2, bbox.y + bbox.h/2, bbox.w*bbox.h, static_cast<float>(bbox.w)/bbox.h, 0, 0, 0;
     // Initalize X's convariance matrix
     EigenMatrix<kDimx, kDimx> P;
     P << 10, 0, 0, 0, 0, 0, 0,
          0, 10, 0, 0, 0, 0, 0,
          0, 0, 10, 0, 0, 0, 0,
          0, 0, 0, 10, 0, 0, 0,
          0, 0, 0, 0, 1e4, 0, 0,
          0, 0, 0, 0, 0, 1e4, 0,
          0, 0, 0, 0, 0, 0, 1e4;
     // Initialize uniform linear transition matrix (X(t-1)--->X(t))
     EigenMatrix<kDimx, kDimx> F;
     F << 1, 0, 0, 0, 1, 0, 0,
          0, 1, 0, 0, 0, 1, 0,
          0, 0, 1, 0, 0, 0, 1,
          0, 0, 0, 1, 0, 0, 0,
          0, 0, 0, 0, 1, 0, 0,
          0, 0, 0, 0, 0, 1, 0,
          0, 0, 0, 0, 0, 0, 1;
     // Initialize project transistion matrix (X(t)--->X(t))
     EigenMatrix<kDimz, kDimx> H;
     H << 1, 0, 0, 0, 0, 0, 0,
          0, 1, 0, 0, 0, 0, 0,
          0, 0, 1, 0, 0, 0, 0,
          0, 0, 0, 1, 0, 0, 0;
     kf_ = KalmanFilter<kDimx, kDimz>(X, P, F, H);
     track_id_ = track_id;
     frame_id_ = frame_id;
     cls_id_ = bbox.cls_id;
     tracklet_len_ = 0;
     det_score_ = bbox.cls_confidence;
     track_state_ = kTrackState::Active;
     track_velocity_ = std::pair<float, float>(0, 0);
     det_box_history_.push_back(std::pair<int32_t, Bbox2D>(frame_id, bbox));
}

void XysaTrack::Predict() {
     EigenMatrix<kDimx, kDimx> Q;  // Enviroment uncertentiy
     const EigenVector<kDimx> X = kf_.GetStateMean();
     if (X[2] + X[6] <= 0) {
     //TO DO kf_.X[6] = 0.0
     }
     Q << 1, 0, 0, 0, 0, 0, 0, 
          0, 1, 0, 0, 0, 0, 0, 
          0, 0, 1, 0, 0, 0, 0, 
          0, 0, 0, 1, 0, 0, 0, 
          0, 0, 0, 0, 1e-2, 0, 0, 
          0, 0, 0, 0, 0, 1e-2, 0, 
          0, 0, 0, 0, 0, 0, 1e-4;
     kf_.Predict(Q);
     predicted_box_history_.push_back(std::pair<int32_t, Bbox2D>(frame_id_, State2Bbox()));
     if (predicted_box_history_.size() > kDeltaFrameNum) {
          predicted_box_history_.pop_front();
     }
}

void XysaTrack::Update(const Bbox2D& bbox, const int32_t& frame_id) {
     EigenVector<kDimz> Z;    // New meassurement
     EigenMatrix<kDimz, kDimz> R;  // New measurment's uncertentiy
     Z << bbox.x + bbox.w/2, bbox.y + bbox.h/2, bbox.w*bbox.h, static_cast<float>(bbox.w)/bbox.h;
     R << 1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 10, 0,
          0, 0, 0, 10;
     kf_.Update(Z, R);
     tracklet_len_++;
     cls_id_ = bbox.cls_id;
     det_score_ = bbox.cls_confidence;
     cls_name_ = bbox.cls_name;
     frame_id_ = frame_id;
     last_matched_det_box_ = bbox;

     // Check https://github.com/noahcao/OC_SORT/blob/master/trackers/ocsort_tracker/ocsort.py#L109
     Bbox2D prev_box = det_box_history_.back().second;
     int32_t least_frame_id = frame_id - kDeltaFrameNum;
     for (auto iter = det_box_history_.end() - 1; iter != det_box_history_.begin() - 1; iter--) {
          if (least_frame_id >= iter->first) {
               prev_box = iter->second;
               break;
          }
     }
     track_velocity_ = TwoBoxes2Speed(prev_box, bbox);
     det_box_history_.push_back(std::pair<int32_t, Bbox2D>(frame_id, bbox));
     if (det_box_history_.size() > kDeltaFrameNum) {
          det_box_history_.pop_front();
     }
}

Bbox2D XysaTrack::State2Bbox() const{
     const EigenVector<kDimx> X = kf_.GetStateMean();
     Bbox2D bbox;
     float w = sqrt(X[2] * X[3]);
     bbox.w = static_cast<int32_t>(w);
     bbox.h = static_cast<int32_t>(X[2] / w);
     bbox.x = static_cast<int32_t>(X[0] - w / 2);
     bbox.y = static_cast<int32_t>(X[1] - bbox.h / 2);
     bbox.cls_id = cls_id_;
     bbox.cls_confidence = det_score_;
     bbox.cls_name = cls_name_;
     return bbox;
}

void Tracker::TwoBoxVectors2Speed(const std::vector<Bbox2D>& row_boxes, const std::vector<Bbox2D> col_boxes, std::vector<std::pair<float, float>>& speeds) {
     speeds.reserve(row_boxes.size()*col_boxes.size());
     for (const auto& row_box : row_boxes) {
          int32_t cx1 = row_box.x + row_box.w / 2;
          int32_t cy1 = row_box.y + row_box.h / 2;
          for (const auto& col_box : col_boxes) {
               int32_t cx2 = col_box.x + col_box.w / 2;
               int32_t cy2 = col_box.y + col_box.h / 2;
               float speed_x = cx2 - cx1;
               float speed_y = cy2 - cy1;
               float norm = sqrt(speed_x*speed_x + speed_y*speed_y) + 1e-6;
               speeds.push_back(std::pair<float, float>(speed_x, speed_y));
          }
     }
}

void Tracker::TwoBoxVectors2Speed(const std::vector<Bbox2D>& row_boxes, const std::vector<const Bbox2D*> col_boxes, std::vector<std::pair<float, float>>& speeds) {
     speeds.reserve(row_boxes.size()*col_boxes.size());
     for (const auto& row_box : row_boxes) {
          int32_t cx1 = row_box.x + row_box.w / 2;
          int32_t cy1 = row_box.y + row_box.h / 2;
          for (const auto& col_box : col_boxes) {
               int32_t cx2 = col_box->x + col_box->w / 2;
               int32_t cy2 = col_box->y + col_box->h / 2;
               float speed_x = cx2 - cx1;
               float speed_y = cy2 - cy1;
               float norm = sqrt(speed_x*speed_x + speed_y*speed_y) + 1e-6;
               speeds.push_back(std::pair<float, float>(speed_x / norm, speed_y / norm));
          }
     }
}

void Tracker::CalculateAnglesCostMatrix(const std::vector<Bbox2D>& row_boxes, const std::vector<const Bbox2D*>& col_boxes, std::vector<std::pair<float, float>>& row_boxes_inertias, std::vector<std::vector<float>>& cost_matrix) {
     // assert cost_matrix.size() == (std::max)(row_boxes.size(), col_boxes.size());
     std::vector<float> normalized_thetas;
     std::vector<std::pair<float, float>> speeds;
     speeds.reserve(row_boxes.size()*col_boxes.size());
     normalized_thetas.reserve(row_boxes.size()*col_boxes.size());
     TwoBoxVectors2Speed(row_boxes, col_boxes, speeds);
     for (int32_t i = 0; i < row_boxes.size(); i++) {
          float inertia_x = row_boxes_inertias[i].first;
          float inertia_y = row_boxes_inertias[i].second;
          // Inertia is invalid
          if (inertia_x == 0 && inertia_y == 0) {
               for (int32_t j = 0; j < col_boxes.size(); j++) {
                    normalized_thetas.push_back(0);
               }    
          } else {
               for (int32_t j = 0; j < col_boxes.size(); j++) {
                    std::pair<float, float> speed = speeds[i * col_boxes.size() + j];
                    float cos_value = inertia_x * speed.first + inertia_y * speed.second;
                    // float arcsin_value = 0.5 * M_PI - acos(cos_value); // convert arccos to arcsin
                    // normalized_thetas.push_back(arcsin_value / M_PI); // normalize to -0.5~0.5
                    normalized_thetas.push_back((0.5 - acos(cos_value)) / M_PI); // normalize thea to -0.5~0.5
               }
          }
     }
     // Check https://github.com/noahcao/OC_SORT/blob/master/trackers/ocsort_tracker/association.py#L267
     for (int32_t i = 0; i < normalized_thetas.size(); i++) {
          normalized_thetas[i] *= col_boxes[i % col_boxes.size()]->cls_confidence;
     }
     for (int32_t i = 0; i < cost_matrix.size(); i++) {
          for (int32_t j = 0; j < cost_matrix.size(); j++) {
               if (cost_matrix[i][j] == kMaxAssignmentCost) continue;
               cost_matrix[i][j] -= kAngleCostWeight*normalized_thetas[i * col_boxes.size() + j];
          }
     }
}

void Tracker::CalculateIouCostMatrix(const std::vector<Bbox2D>& row_boxes, const std::vector<const Bbox2D*>& col_boxes, std::vector<std::vector<float>>& cost_matrix) {
     int32_t cost_matrix_size = (std::max)(row_boxes.size(), col_boxes.size());
     cost_matrix.resize(cost_matrix_size, std::vector<float>(cost_matrix_size, kMaxAssignmentCost));
     for (int32_t i = 0; i < row_boxes.size(); i++) {
          for (int32_t j = 0; j < col_boxes.size(); j++) {
               int32_t inter_left   = std::max(row_boxes[i].x, col_boxes[j]->x);
               int32_t inter_right  = std::min(row_boxes[i].x + row_boxes[i].w, col_boxes[j]->x + col_boxes[j]->w);
               int32_t inter_top    = std::max(row_boxes[i].y, col_boxes[j]->y);
               int32_t inter_bottom = std::min(row_boxes[i].y + row_boxes[i].h, col_boxes[j]->y + col_boxes[j]->h);
               if (inter_left > inter_right || inter_top > inter_bottom) {
                    continue;
               }
               int32_t area_inter = (inter_right - inter_left) * (inter_bottom - inter_top);
               int32_t area_i = row_boxes[i].h * row_boxes[i].w;
               int32_t area_j = col_boxes[j]->h * col_boxes[j]->w;
               float iou = static_cast<float>(area_inter) / (area_i + area_j - area_inter);
               if (iou < cost_thresh_iou_) {
                    continue;
               }
               cost_matrix[i][j] = kMaxAssignmentCost - iou;
          }
     }
}

void Tracker::LinearOcmAssignment(const std::vector<std::shared_ptr<Track>>& cur_tracklets, const std::vector<const Bbox2D*>& det_boxes, std::vector<int32_t>& tracklets_assignment_res, std::vector<int32_t>& dets_assignment_res) {
     if (cur_tracklets.size() == 0 || det_boxes.size() == 0) {
          tracklets_assignment_res.resize(cur_tracklets.size(), -1);
          dets_assignment_res.resize(det_boxes.size(), -1);
          return;
     }

     std::vector<Bbox2D> tracklet_predicted_boxes;
     std::vector<Bbox2D> tracklet_observed_boxes;
     std::vector<std::pair<float, float>> tracklet_inertias;
     tracklet_predicted_boxes.reserve(cur_tracklets.size());
     tracklet_observed_boxes.reserve(cur_tracklets.size());
     tracklet_inertias.reserve(cur_tracklets.size());
     for (const auto& tracklet : cur_tracklets) {
          tracklet_predicted_boxes.push_back(tracklet->GetLatestPredictedBox());
          tracklet_observed_boxes.push_back(tracklet->GetAroundDetBox(cur_frame_id_));
          tracklet_inertias.push_back(tracklet->GetTrackInertia());
     }

     std::vector<std::vector<float>> cost_matrix;
     if (cur_frame_id_ == 43 || cur_frame_id_ == 228 || cur_frame_id_ == 413) {
          int a = 2;
     }
     CalculateIouCostMatrix(tracklet_predicted_boxes, det_boxes, cost_matrix);
     CalculateAnglesCostMatrix(tracklet_observed_boxes, det_boxes, tracklet_inertias, cost_matrix);

     tracklets_assignment_res.resize(cost_matrix.size(), -1);
     dets_assignment_res.resize(cost_matrix.size(), -1);
     HungarianAlgorithm<float> solver(cost_matrix);
     solver.Solve(tracklets_assignment_res, dets_assignment_res);
     for (int32_t i = 0; i < cost_matrix.size(); i++) {
          if (tracklets_assignment_res[i] >= 0) {
               if (tracklets_assignment_res[i] >= det_boxes.size() || cost_matrix[i][tracklets_assignment_res[i]] == kMaxAssignmentCost) {
                    tracklets_assignment_res[i] = -1;
               }
          }
          if (dets_assignment_res[i] >= 0) {
               if (dets_assignment_res[i] >= cur_tracklets.size() || cost_matrix[dets_assignment_res[i]][i] == kMaxAssignmentCost) {
                    dets_assignment_res[i] = -1;
               }
          }
     }
}

void Tracker::LinearAssignment(const std::vector<std::shared_ptr<Track>>& cur_tracklets, const std::vector<const Bbox2D*>& det_boxes, const kAssignmentObj& obj , std::vector<int32_t>& tracklets_assignment_res, std::vector<int32_t>& dets_assignment_res) {
     if (cur_tracklets.size() == 0 || det_boxes.size() == 0) {
          tracklets_assignment_res.resize(cur_tracklets.size(), -1);
          dets_assignment_res.resize(det_boxes.size(), -1);
          return;
     }
     std::vector<Bbox2D> tracklet_boxes;
     std::vector<std::vector<float>> cost_matrix;
     tracklet_boxes.reserve(cur_tracklets.size());
     if (obj == kAssignmentObj::PredictedBoxes) {
          for (const auto& tracklet : cur_tracklets) {
               tracklet_boxes.push_back(tracklet->State2Bbox());
          }
     } 
     if (obj == kAssignmentObj::PredictedBoxes) {
          for (const auto& tracklet : cur_tracklets) {
               tracklet_boxes.push_back(tracklet->GetAroundDetBox(cur_frame_id_));
          }
     }
     CalculateIouCostMatrix(tracklet_boxes, det_boxes, cost_matrix);

     tracklets_assignment_res.resize(cost_matrix.size(), -1);
     dets_assignment_res.resize(cost_matrix.size(), -1);
     HungarianAlgorithm<float> solver(cost_matrix);
     solver.Solve(tracklets_assignment_res, dets_assignment_res);
     for (int32_t i = 0; i < cost_matrix.size(); i++) {
          if (tracklets_assignment_res[i] >= 0) {
               if (tracklets_assignment_res[i] >= det_boxes.size() || cost_matrix[i][tracklets_assignment_res[i]] == kMaxAssignmentCost) {
                    tracklets_assignment_res[i] = -1;
               }
          }
          if (dets_assignment_res[i] >= 0) {
               if (dets_assignment_res[i] >= cur_tracklets.size() || cost_matrix[dets_assignment_res[i]][i] == kMaxAssignmentCost) {
                    dets_assignment_res[i] = -1;
               }
          }
     }
}

void Tracker::Update(const std::vector<Bbox2D>& det_boxes, std::vector<const Bbox2D*>& unmatched_boxes) {
     // 1. Predict the active and lost tracklets' next positions
     std::vector<std::shared_ptr<Track>> merged_tracklets(active_tracks_);
     merged_tracklets.insert(merged_tracklets.begin(), lost_tracks_.begin(), lost_tracks_.end());
     for (const auto& tracklet : merged_tracklets) {
          tracklet->Predict();
     }

     // 2. Split the current frame's meassurments into two sets
     std::vector<const Bbox2D*> high_conf_dets;
     std::vector<const Bbox2D*> low_conf_dets;
     for (const auto& det_box : det_boxes) {
          if (det_box.cls_confidence >= split_thresh_score_) {
               high_conf_dets.push_back(&det_box);
          } else {
               low_conf_dets.push_back(&det_box);
          }
     }

     // 3.1 First-pass assignment, mathcing between merged tracklets' predicted boxes with high-conf dets; metric is iou plus angle difference
     std::vector<int32_t> tracklets_assignment_res;
     std::vector<int32_t> dets_assignment_res;
     std::vector<std::shared_ptr<Track>> unmatched_tracklets;
     std::vector<const Bbox2D*> unmatched_high_conf_dets;
     LinearOcmAssignment(merged_tracklets, high_conf_dets, tracklets_assignment_res, dets_assignment_res);
     for (int32_t i = 0; i < merged_tracklets.size(); i++) {
          Track* tracklet = merged_tracklets[i].get();
          if (tracklets_assignment_res[i] >= 0) {
               const Bbox2D* bbox = high_conf_dets[tracklets_assignment_res[i]];
               if (tracklet->GetTrackState() == kTrackState::Active) {
                    tracklet->Update(*bbox, cur_frame_id_);
               } else {
                    tracklet->Update(*bbox, cur_frame_id_);
                    // remove from lost vec & push_back into active vec
                    tracklet->MarkTrackState(kTrackState::Active);
                    active_tracks_.push_back(merged_tracklets[i]);
                    // lost_tracks_.erase(lost_tracks_.begin() + (i - active_tracks_.size())); // It will cause a bug
               }
          } else {
               unmatched_tracklets.push_back(merged_tracklets[i]);
          }
     }
     for (auto it = lost_tracks_.begin(); it != lost_tracks_.end();) {
          if (it->get()->GetTrackState() != kTrackState::Lost) {
               it = lost_tracks_.erase(it);
          } else {
               it++;
          }
     }
     for (int32_t i = 0; i < high_conf_dets.size(); i++) {
          if (dets_assignment_res[i] < 0) {
               unmatched_high_conf_dets.push_back(high_conf_dets[i]);
          }
     }

     // 3.2 Second-pass assignment, matching between the first-pass unmatched tracklets' predicted boxes with low-conf dets; metric is iou only
     std::vector<std::shared_ptr<Track>> unmatched_twice_tracklet;
     std::vector<const Bbox2D*> unmatched_low_conf_dets;
     tracklets_assignment_res.clear();
     dets_assignment_res.clear();
     LinearAssignment(unmatched_tracklets, low_conf_dets, kAssignmentObj::PredictedBoxes, tracklets_assignment_res, dets_assignment_res);
     for (int32_t i = 0; i < unmatched_tracklets.size(); i++) {
          Track* tracklet = unmatched_tracklets[i].get();
          if (tracklets_assignment_res[i] >= 0) {
               const Bbox2D* bbox = low_conf_dets[tracklets_assignment_res[i]];
               if (tracklet->GetTrackState() == kTrackState::Active) {
                    tracklet->Update(*bbox, cur_frame_id_);
               } else {
                    tracklet->Update(*bbox, cur_frame_id_);
                    // remove from lost vec & push_back into active vec
                    tracklet->MarkTrackState(kTrackState::Active);
                    active_tracks_.push_back(unmatched_tracklets[i]);
               }
          } else {
               unmatched_twice_tracklet.push_back(unmatched_tracklets[i]);
          }
     }
     for (auto it = lost_tracks_.begin(); it != lost_tracks_.end();) {
          if (it->get()->GetTrackState() != kTrackState::Lost) {
               it = lost_tracks_.erase(it);
          } else {
               it++;
          }
     }
     for (int32_t i = 0; i < unmatched_tracklets.size(); i++) {
          if (dets_assignment_res[i] < 0) {
               
          }
     }
     for (int32_t i = 0; i < low_conf_dets.size(); i++) {
          if (dets_assignment_res[i] < 0) {
               unmatched_low_conf_dets.push_back(low_conf_dets[i]);
          }
     }

     // 3.3 Thrid-pass assignment, matching between second-pass unmatched lost-tracklets' observed boxes and first-pass unmatched high-conf dets; metric is iou only
     tracklets_assignment_res.clear();
     dets_assignment_res.clear();
     std::vector<const Bbox2D*> unmatched_twice_high_conf_dets;
     LinearAssignment(unmatched_twice_tracklet, unmatched_high_conf_dets, kAssignmentObj::ObservedBoxes, tracklets_assignment_res, dets_assignment_res);
     for (int32_t i = 0; i < unmatched_twice_tracklet.size(); i++) {
          Track* tracklet = unmatched_twice_tracklet[i].get();
          if (tracklets_assignment_res[i] >= 0) {
               const Bbox2D* bbox = low_conf_dets[tracklets_assignment_res[i]];
               if (tracklet->GetTrackState() == kTrackState::Active) {
                    tracklet->Update(*bbox, cur_frame_id_);
               } else {
                    tracklet->Update(*bbox, cur_frame_id_);
                    // remove from lost vec & push_back into active vec
                    tracklet->MarkTrackState(kTrackState::Active);
                    active_tracks_.push_back(unmatched_twice_tracklet[i]);
               }
          } else {
               if (tracklet->GetTrackState() == kTrackState::Active) {
                    // remove from active vec & push_back into lost vec
                    tracklet->MarkTrackState(kTrackState::Lost);
                    lost_tracks_.push_back(unmatched_twice_tracklet[i]);
               }
          }
     }
     for (int32_t i = 0; i < unmatched_high_conf_dets.size(); i++) {
          if (dets_assignment_res[i] < 0) {
               unmatched_twice_high_conf_dets.push_back(unmatched_high_conf_dets[i]);
          }
     }
     for (auto it = active_tracks_.begin(); it != active_tracks_.end();) {
          if (it->get()->GetTrackState() != kTrackState::Active) {
               it = active_tracks_.erase(it);
          } else {
               it++;
          }
     }
     for (auto it = lost_tracks_.begin(); it != lost_tracks_.end();) {
          if (it->get()->GetTrackState() != kTrackState::Lost) {
               it = lost_tracks_.erase(it);
          } else {
               it++;
          }
     }
     merged_tracklets.clear();
     unmatched_tracklets.clear();

     // 4.1 Mark this frame's final unmatched high-conf dets as new tracklets
     for (const auto& bbox : unmatched_twice_high_conf_dets) {
          // new_tracks_.push_back(std::make_shared<Track>(*bbox, all_tracks_num_++, cur_frame_id_));
          std::shared_ptr<Track> ptr(new XysaTrack());
          ptr->Initialize(*bbox, all_tracks_num_++, cur_frame_id_);
          active_tracks_.push_back(ptr);
     }

     // 4.2 Remove the lost tracklets whose lost time is longer than the threshold
     for (auto it = lost_tracks_.begin(); it != lost_tracks_.end();) {
          if (cur_frame_id_ - it->get()->GetLastFrameId() > max_lost_frame_num_) {
               it->get()->MarkTrackState(kTrackState::Removed);
               it = lost_tracks_.erase(it);
          } else {
               it++;
          }
     }
     unmatched_boxes = unmatched_low_conf_dets;
     cur_frame_id_++;
}