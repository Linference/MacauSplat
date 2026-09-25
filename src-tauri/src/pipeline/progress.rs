use super::PipelineStage;

pub fn stage_progress_range(stage: PipelineStage) -> (f32, f32) {
    use PipelineStage::*;
    match stage {
        Created | ProbingVideo => (0.0, 5.0),
        PlanningFrames => (5.0, 7.0),
        ExtractingFrames => (7.0, 20.0),
        ExtractingFeatures => (20.0, 32.0),
        Matching => (32.0, 45.0),
        Reconstructing => (45.0, 58.0),
        ValidatingReconstruction => (58.0, 60.0),
        // 自動標註：只依賴重建結果，通常只需數秒（讀位姿、擬合包圍盒、投影、寫檔）
        Annotating => (60.0, 62.0),
        TrainingSplats => (62.0, 98.0),
        Exporting => (98.0, 100.0),
        Completed => (100.0, 100.0),
        Failed | Cancelled => (0.0, 0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::PipelineEvent;

    #[test]
    fn maps_stage_progress_to_monotonic_global_ranges() {
        assert_eq!(
            PipelineEvent::mapped(PipelineStage::ExtractingFrames, 0.0, "").progress,
            7.0
        );
        assert_eq!(
            PipelineEvent::mapped(PipelineStage::ExtractingFrames, 1.0, "").progress,
            20.0
        );
        // 加入標註階段後，訓練區間由 60~98 順延為 62~98，
        // 因此 0.5 的位置由 79.0 變為 80.0
        assert_eq!(
            PipelineEvent::mapped(PipelineStage::TrainingSplats, 0.5, "").progress,
            80.0
        );
        // 新增的標註階段介於驗證與訓練之間，且不破壞單調性
        assert_eq!(
            PipelineEvent::mapped(PipelineStage::Annotating, 0.0, "").progress,
            60.0
        );
        assert_eq!(
            PipelineEvent::mapped(PipelineStage::Annotating, 1.0, "").progress,
            62.0
        );
        assert_eq!(
            PipelineEvent::mapped(PipelineStage::Failed, 1.0, "").progress,
            0.0
        );
        assert_eq!(
            PipelineEvent::mapped(PipelineStage::Cancelled, 1.0, "").progress,
            0.0
        );
    }
}
