use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PipelineStage {
    Created,
    ProbingVideo,
    PlanningFrames,
    ExtractingFrames,
    ExtractingFeatures,
    Matching,
    Reconstructing,
    ValidatingReconstruction,
    /// 位姿驅動的自動標註：由相機位姿與稀疏點雲投影出逐影格標註。
    ///
    /// 為什麼放在訓練之前：本階段只依賴重建結果（位姿＋稀疏點雲），
    /// 而訓練是全流程最長的一段。放在前面意味著使用者若中途取消訓練，
    /// 資料集已經產出——對「以資料集為主要產物」的用途（建築辨識）尤其重要。
    Annotating,
    TrainingSplats,
    Exporting,
    Completed,
    Failed,
    Cancelled,
}

impl PipelineStage {
    pub fn can_transition_to(self, next: Self) -> bool {
        use PipelineStage::*;
        if matches!(next, Failed | Cancelled) && !matches!(self, Completed | Failed | Cancelled) {
            return true;
        }
        matches!(
            (self, next),
            (Created, ProbingVideo)
                | (ProbingVideo, PlanningFrames)
                | (PlanningFrames, ExtractingFrames)
                | (ExtractingFrames, ExtractingFeatures)
                | (ExtractingFeatures, Matching)
                | (Matching, Reconstructing)
                | (Reconstructing, ValidatingReconstruction)
                | (ValidatingReconstruction, Annotating)
                | (Annotating, TrainingSplats)
                | (TrainingSplats, Exporting)
                | (Exporting, Completed)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::PipelineStage::*;

    #[test]
    fn allows_only_forward_pipeline_transitions() {
        assert!(Created.can_transition_to(ProbingVideo));
        assert!(Matching.can_transition_to(Reconstructing));
        assert!(TrainingSplats.can_transition_to(Failed));
        assert!(!Matching.can_transition_to(PlanningFrames));
        assert!(!Completed.can_transition_to(Failed));
    }

    /// 自動標註階段的轉移：重建驗證後進入，完成後接訓練。
    #[test]
    fn annotation_stage_sits_between_validation_and_training() {
        assert!(ValidatingReconstruction.can_transition_to(Annotating));
        assert!(Annotating.can_transition_to(TrainingSplats));
        // 不得跳過或回頭
        assert!(!ValidatingReconstruction.can_transition_to(TrainingSplats));
        assert!(!Annotating.can_transition_to(ValidatingReconstruction));
        assert!(!TrainingSplats.can_transition_to(Annotating));
        // 仍可被取消／失敗
        assert!(Annotating.can_transition_to(Cancelled));
        assert!(Annotating.can_transition_to(Failed));
    }
}
