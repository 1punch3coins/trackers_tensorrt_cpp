#include <iostream>
#include <vector>
#include <array>
#include <memory>

#include "../det_structs.h"
#include "kalman_filter.h"

namespace OcSort {
    constexpr size_t kDimx = 7;
    constexpr size_t kDimz = 4;
    constexpr int32_t kDeltaFrameNum = 3;
    constexpr float kMaxAssignmentCost = 1.0;

    enum kTrackState {
        Active,
        Lost,
        Removed
    };

    enum kAssignmentObj{
        PredictedBoxes,
        ObservedBoxes
    };

    class Track {
    public:
        Track() {};

    public:
        virtual Bbox2D State2Bbox() const = 0;
        virtual void Initialize(const Bbox2D& bbox, const int32_t& track_id, const int32_t& frame_id) = 0;
        virtual void Predict() = 0;
        virtual void Update(const Bbox2D& bbox, const int32_t& frame_id) = 0;

    public:
        int32_t GetTrackId() const {
            return track_id_;
        };
        int32_t GetLastFrameId() const {
            return frame_id_;
        };
        int32_t GetTrackletLength() const {
            return tracklet_len_;
        };
        kTrackState GetTrackState() const {
            return track_state_;
        }
        void MarkTrackState(const kTrackState& state) {
            track_state_ = state;
        }
    
    public:
        Bbox2D GetLatestPredictedBox() const{
            return predicted_box_history_.back().second;
        }
        Bbox2D GetAroundDetBox(int32_t frame_id) const{
            Bbox2D det_box = det_box_history_.back().second;
            int32_t least_frame_id = frame_id - kDeltaFrameNum;
            for (auto iter = det_box_history_.end() - 1; iter != det_box_history_.begin() - 1; iter--) {
                if (least_frame_id >= iter->first) {
                    det_box = iter->second;
                    break;
                }
            }
            return det_box;
        }
        std::pair<float,float> GetTrackInertia() const {
            return track_velocity_;
        }
    
    protected:
        std::pair<float,float> TwoBoxes2Speed(const Bbox2D& box1, const Bbox2D& box2) const;

    protected:
        int32_t track_id_;
        int32_t frame_id_;
        int32_t tracklet_len_;
        std::pair<float, float> track_velocity_;
        kTrackState track_state_;
        KalmanFilter<kDimx, kDimz> kf_;

    protected:
        int32_t cls_id_;
        std::string cls_name_;
        float det_score_;
        Bbox2D last_matched_det_box_;
        std::deque<std::pair<int32_t, Bbox2D>> det_box_history_;
        std::deque<std::pair<int32_t, Bbox2D>> predicted_box_history_;
    };

    class XysaTrack : public Track{
    public:
        Bbox2D State2Bbox() const override;
        void Initialize(const Bbox2D& bbox, const int32_t& track_id, const int32_t& frame_id) override;
        void Predict() override;
        void Update(const Bbox2D& bbox, const int32_t& frame_id) override;
    };

    class XywhTrack : public Track{
    public:
        Bbox2D State2Bbox() const override;
        void Initialize(const Bbox2D& bbox, const int32_t& track_id, const int32_t& frame_id) override;
        void Predict() override;
        void Update(const Bbox2D& bbox, const int32_t& frame_id) override;
    };

    class Tracker {
    public:
        Tracker(const float& split_thresh_score, const float& cost_thresh_iou, const int32_t& max_lost_frame_num):
            split_thresh_score_(split_thresh_score),
            cost_thresh_iou_(cost_thresh_iou),
            max_lost_frame_num_(max_lost_frame_num),
            cur_frame_id_(0),
            all_tracks_num_(0)
        {}
        Tracker():
            split_thresh_score_(0.3),
            add_thresh_score_(0.4),
            cost_thresh_iou_(0.3),
            max_lost_frame_num_(50),
            cur_frame_id_(0),
            all_tracks_num_(0)
        {}

    public:
        void Initialize();
        void Update(const std::vector<Bbox2D>& det_boxes, std::vector<const Bbox2D*>& unmatched_boxes);

    public:
        const std::vector<std::shared_ptr<Track>>* GetActiveTracklets() const {
            return &active_tracks_;
        }

    private:
        void TwoBoxVectors2Speed(const std::vector<Bbox2D>& row_boxes, const std::vector<Bbox2D> col_boxes, std::vector<std::pair<float, float>>& speeds);
        void TwoBoxVectors2Speed(const std::vector<Bbox2D>& row_boxes, const std::vector<const Bbox2D*> col_boxes, std::vector<std::pair<float, float>>& speeds);
        void CalculateAnglesCostMatrix(const std::vector<Bbox2D>& row_boxes, const std::vector<const Bbox2D*>& col_boxes, std::vector<std::pair<float, float>>& row_boxes_inertias, std::vector<std::vector<float>>& cost_matrix);
        void CalculateIouCostMatrix(const std::vector<Bbox2D>& row_boxes, const std::vector<const Bbox2D*>& col_boxes, std::vector<std::vector<float>>& cost_matrix);
        void LinearOcmAssignment(const std::vector<std::shared_ptr<Track>>& cur_tracklets, const std::vector<const Bbox2D*>& det_boxes, std::vector<int32_t>& tracklets_assignment_res, std::vector<int32_t>& dets_assignment_res);
        void LinearAssignment(const std::vector<std::shared_ptr<Track>>& cur_tracklets, const std::vector<const Bbox2D*>& det_boxes, const kAssignmentObj& obj , std::vector<int32_t>& tracklets_assignment_res, std::vector<int32_t>& dets_assignment_res);

    private:
        int32_t cur_frame_id_;
        int32_t all_tracks_num_;

    private:
        float split_thresh_score_;      // The threshold of cls score to split high_conf and low_conf detection boxes
        float add_thresh_score_;        // The threshold of cls score determining whether to add a det box into new_tracks_ vector
        float cost_thresh_iou_;         // The threshold of iou detereming whether two boxes would ever be matched
        int32_t max_lost_frame_num_;    // The threshold of frame_num to decide the lost boxes' max alive time

    private:
        std::vector<std::shared_ptr<Track>> active_tracks_;
        std::vector<std::shared_ptr<Track>> lost_tracks_;
        std::vector<std::shared_ptr<Track>> removed_tracks_;
    };
}